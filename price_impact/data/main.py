import logging
import os
from concurrent.futures import ThreadPoolExecutor
from logging import Logger

import pandas as pd
from price_impact.data import create_bins
from price_impact import data_dir, tickers
from george_dev.price_impact.price_impact.modelling.Futures import FutureContract
from george_dev.price_impact.price_impact.utils import get_logger

logging.getLogger().setLevel(logging.INFO)
logger: Logger = get_logger(__name__)

bin_dir = data_dir/'bins'
synthetic_bin_dir = data_dir/'synthetic'/'bins'


def create_synthetic(freq: str):

    for idx, spot in enumerate(tickers):
        print(idx, end='\r')
        vols = []
        pxs = []

        for file in os.listdir(bin_dir/freq/spot):

            if os.path.isdir(bin_dir/freq/spot/file):
                continue
            # print(file)
            future, metric = file.split('.')[0].split('_')
            future = FutureContract(future)

            if not future.is_active_contract():
                continue

            df = pd.read_pickle(bin_dir/freq/spot/file)

            start_date, end_date = future.previous_contract().backtest_end_date, future.backtest_end_date
            df = df[(start_date < df.index) & (df.index <= end_date)]

            if metric == 'vol':
                vols.append(df)
            else:
                pxs.append(df)

        if not vols:
            print(f"\n no data {spot}")
            continue
        vol_df = pd.concat(vols).sort_index()
        px_df = pd.concat(pxs).sort_index()

        vol_df.to_pickle(synthetic_bin_dir/freq/f"{spot}_vol.pkl")
        px_df.to_pickle(synthetic_bin_dir/freq/f"{spot}_px.pkl")


def make_one_daily_average(spot, freq, sma_len=20):

    out_dir = bin_dir/'..'/'daily'

    os.makedirs(out_dir, exist_ok=True)

    vol_df = pd.read_pickle(synthetic_bin_dir/freq/f"{spot}_vol.pkl")
    px_df = pd.read_pickle(synthetic_bin_dir/freq/f"{spot}_px.pkl")

    px_df = px_df.pct_change(axis=1).std(axis=1)
    px_df = px_df.rolling(sma_len).mean().shift(1)

    vol_df = vol_df.abs().sum(axis=1)
    vol_df = vol_df.rolling(sma_len).mean().shift(1)

    df = pd.concat([vol_df, px_df], axis=1)
    df = df.rename(columns={0: 'volume', 1: 'stdev'})

    df.to_pickle(out_dir/f"{spot}_daily_sma{sma_len}.pkl", protocol=0)
    return


def main(freq: str = '10s', sma_len: int = 20):
    logger.setLevel(logging.DEBUG)
    os.makedirs(synthetic_bin_dir/freq, exist_ok=True)

    logger.info("Running tick aggregation")
    create_bins.main(freq)

    logger.info('creating synthetic')
    create_synthetic(freq)

    def fn(x): return make_one_daily_average(x, freq, sma_len)

    logger.info('Making daily averages')
    with ThreadPoolExecutor(10) as exec:
        exec.map(fn, tickers)


if __name__ == '__main__':
    main(freq='10s', sma_len=20)
