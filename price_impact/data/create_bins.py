import logging
import os
import time
from logging import Logger
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from price_impact import data_dir, tick_base_dir, tickers
from price_impact.utils import get_logger

bin_path = data_dir/'bins'

futures_headers = ['date', 'time', 'price', 'volume', 'market_flag',
                   'sales_condition', 'exclude_flag', 'unfiltered_price']


logging.getLogger().setLevel(logging.INFO)
logger: Logger = get_logger(__name__)


class TickBinner:
    """
    Class to bin tick level data from shared drive.
    1) bin_to_local() - Copies data from shared drive to a local directory, bins the tick data at a specified frequency.
    2) orgainise_local() - Cleans the copied data locally. Creates specific vol and price files, and fills NA's.
    """

    def __init__(self, freq: str, tick_ticker: str, tickdir: Path, bin_path: Path):
        """
        :param str freq: Frequency to bin the tick data to, eg. '1m', '15s'. Uses pd.DataFrame.resample
        :param str tick_ticker: Ticker to bin, as specified by the TickData documentation, eg. ES, NQ
        :param Path tickdir: Directory of tick data, at the ticker level, ie './FUT/E/ES'
        :param int processes: Uses multiprocessing, number of parallel processes to use.
        :param Path bin_path: Local path to copy data t
        """

        self.freq = freq
        self.tick_ticker = tick_ticker
        self.tickdir = tickdir
        self.bin_path = bin_path

        open_hours_df = pd.read_csv(Path(os.path.dirname(__file__))/'exchange_open_info.csv')
        open_hours_dict = open_hours_df.set_index('tick_code')[['TIME_ZONE_NUM', 'bst_offset']].to_dict()

        self.base_tz = open_hours_dict['TIME_ZONE_NUM'][self.tick_ticker]
        # BST + hours needed
        self.offset_tz = 1+open_hours_dict['bst_offset'][self.tick_ticker]

    def bin_to_local(self, process_pool: Pool):
        """
        Method to bin tick level data to remote dir.
        """
        if not os.path.exists(bin_path/self.freq/self.tick_ticker):
            os.makedirs(bin_path/self.freq/self.tick_ticker)

        zip_files = os.listdir(self.tickdir)[::-1]

        process_pool.map(self.process_one_zipfile, zip_files)

        return

    def orgainise_local(self, process_pool: Pool):
        """
        Main method to call for the 2nd part of the ETL.
        Iterates over all contracts loaded in part 1, and aggregates each contract.
        """
        contracts = os.listdir(self.bin_path/self.freq/self.tick_ticker)
        contracts = [contract for contract in contracts if os.path.isdir(self.bin_path/self.freq/self.tick_ticker/contract)]
        logger.debug('starting contract write')

        process_pool.map(self.combine_contracts, contracts)

        return

    @staticmethod
    def offset_to_tz_str(offset: int) -> str:
        """Creates UTC string for a specific timezone

        :param int offset: Hours with respect to UTC
        :return str: eg. 'UTC-02:00'
        """
        return f'UTC{"-" if offset<= 0 else "+"}{abs(offset):02}:00'

    def combine_contracts(self, contract: str):
        """
        Reads local time fragmented contract files and concats them into one file.
        Changes the time zone of the file given the specified instance variables.
        Writes price and volume file for each contract locally.

        :param str contract: contract name to aggregate, eg. ESM23
        """

        # concat all files and write to single file
        dflist = []
        for file in os.listdir(self.bin_path/self.freq/self.tick_ticker/contract):
            df_path = Path(self.bin_path/self.freq/self.tick_ticker/contract/file)
            if df_path.suffix != '.zip':
                continue
            df = pd.read_pickle(df_path)
            dflist.append(df)

        df = pd.concat(dflist)

        # align market open and do time shifts
        base_tz_str = self.offset_to_tz_str(self.base_tz)
        to_tz_str = self.offset_to_tz_str(self.offset_tz)
        df = df.set_index('datetime').tz_localize(tz=base_tz_str).tz_convert(to_tz_str)

        df['date'] = df.index.date
        df['time'] = df.index.time

        # pivot to wide format and ffill / fillna
        pivot_df = df.pivot(index='date', columns='time', values=['price', 'volume']).astype('float32')
        pivot_df = pivot_df.sort_index()
        pivot_df.index = pd.to_datetime(pivot_df.index)
        pivot_df = pivot_df[pivot_df.index.day_of_week < 5]

        cols = sorted(pivot_df['volume'].columns)

        vol = pivot_df['volume'][cols]
        px = pivot_df['price'][cols]

        vol.fillna(0, inplace=True)
        px = px.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)

        # save data
        vol.to_pickle(bin_path/self.freq/self.tick_ticker/f'{contract}_vol.pkl')
        px.to_pickle(bin_path/self.freq/self.tick_ticker/f'{contract}_px.pkl')

        return

    def bin_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bins tick level data to a specified frequency.
        Also filters the tick data based on exclude filters and data quality.
        Tick volume is signed based on next tick logic.

        :param pd.DataFrame df: Tick Level dataframe.
        :return pd.DataFrame: Binned and filtered data, containing signed volume and price
        """

        df.columns = futures_headers
        df = df[df['exclude_flag'].isna()]
        df = df[df['price'] != 0]
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format="%m/%d/%Y %H:%M:%S.%f")
        df.drop(columns=['date', 'time', 'unfiltered_price', 'market_flag',
                'sales_condition', 'exclude_flag'], inplace=True)
        df.set_index('datetime', inplace=True)

        df = df.astype({'volume': 'int32', 'price': 'float32'})

        # compute signed volumes
        df['prev_price'] = df['price'].shift()
        df = df.assign(price_mult=lambda x: 0)
        df.loc[df['price'] > df['prev_price'], 'price_mult'] = 1
        df.loc[df['price'] < df['prev_price'], 'price_mult'] = -1

        df['volume'] = df['volume']*df['price_mult']
        df.drop(columns=['prev_price', 'price_mult'], inplace=True)

        # bin data
        rs = df.resample(self.freq, label='left')
        binned = rs[['price']].last()
        binned['volume'] = rs['volume'].sum()

        return binned

    def process_one_zipfile(self, zip_file: str):
        """
        Reads one zip file from the shared TickData directory. Extracts all files from within and writes them to 
        local directory at the contracy level

        :param str zip_file: Specific zip file to read.
        """
        logger.debug(f"starting {zip_file}")

        if int(zip_file.split('_')[0]) < 2012:
            logger.debug(f"skipping year {zip_file}")
            return

        with ZipFile(self.tickdir/zip_file, 'r') as zf:

            # make dict to store each individal contract
            df_lists = {name: []
                        for name in
                        list(set([fname.split('_')[0] for fname in zf.namelist()
                                  if not is_summary_file(fname)]))
                        }

            # read each file, ignoring the summary
            for filename in zf.namelist():
                if is_summary_file(filename):
                    continue

                contract_name = filename.split('_')[0]

                # logger.info(filename)
                with zf.open(filename) as f:
                    df = pd.read_csv(f, compression='gzip', header=None, low_memory=False)

                if df.shape[1] != 8:
                    logger.debug(f"Old format for {filename}. Rejecting.")
                    continue

                binned = self.bin_data(df)
                df_lists[contract_name].append(binned)

        for contract, df_list in df_lists.items():
            if not df_list:
                continue

            binned = pd.concat(df_list)
            if binned.shape[0] == 0:
                logger.debug(f'SKIPPING {contract}, empty df ')
                continue

            binned = binned.reset_index()

            os.makedirs(bin_path/self.freq/self.tick_ticker/contract, exist_ok=True)
            binned.to_pickle(bin_path/self.freq/self.tick_ticker/contract/zip_file)

        logger.debug(f"Finished {zip_file}")


def is_summary_file(fname: str) -> bool:
    """In each TickData file there will be a 'SUMMARY' file, we want to ignore this 

    :param str fname: file name
    :return bool: 
    """
    return fname[-8] == 'Y'


def main(freq: str):
    logger.setLevel(logging.INFO)

    processes = 8

    count = 0

    with Pool(processes=processes) as pool:
        for letter in os.listdir(tick_base_dir):

            full_path = tick_base_dir / letter
            if not os.path.isdir(full_path):
                # skip individial files
                continue

            for tick_ticker in os.listdir(full_path):

                if tick_ticker not in tickers:
                    # skip tickers we dont care about
                    continue

                try:
                    t1 = time.time()
                    logger.info(f"Starting {count}/{len(tickers)}. {tick_ticker}")

                    logger.debug(full_path/tick_ticker)
                    tick_binner = TickBinner(freq=freq, tick_ticker=tick_ticker,
                                                tickdir=tick_base_dir/letter/tick_ticker,
                                                bin_path=bin_path)
                    tick_binner.bin_to_local(pool)
                    logger.info(f"Finished binning {count}/{len(tickers)}. {tick_ticker}")
                    
                    tick_binner.orgainise_local(pool)
                    count += 1

                except Exception as e:
                    logger.error(f"Failed {full_path/tick_ticker} \n {e}")

                t2 = time.time()
                logger.info(f"Finished {count}/{len(tickers)}. {tick_ticker} - {(t2-t1)/60:.2f}m")


if __name__ == '__main__':
    main('10s')
