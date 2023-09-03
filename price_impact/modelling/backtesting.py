import datetime as dt
import os
import time
from multiprocessing import Pool
from typing import Dict, List

import pandas as pd
import linreg
from dateutil.relativedelta import relativedelta
from price_impact import exponents, tickers, regression_path
from price_impact.modelling.Futures import Spot
from price_impact.modelling.backtest_analysis import BacktestSettings

from price_impact.utils import PoolExecutor


class DataLoadError(Exception):
    """ Generic exception for data load issues"""
    pass


class ModelFitError(Exception):
    """ Generic exception for data load issues"""
    pass


def start_of_month(date: dt.datetime):
    return date - relativedelta(days=date.day - 1)


def add_hour(df: pd.DataFrame):
    data = df.copy()

    hours = data.columns.get_level_values('time').to_series().apply(lambda x: x.hour).astype('ubyte')
    hours.index.rename('hour', inplace=True)
    data = data.T.set_index(hours, append=True).T
    data.columns.rename('hour', level=1, inplace=True)
    return data


def crossvalidation_generator(date_index: pd.DatetimeIndex, train_len: int, test_len: int):

    assert 12 % test_len == 0

    start_date = date_index.min()
    train_start_date = start_date + relativedelta(months=1 + test_len - start_date.month % test_len)
    end_date = date_index.max()

    test_end_date = start_date

    dates = []

    while test_end_date+relativedelta(months=test_len) <= end_date:

        train_start_date = start_of_month(train_start_date)
        test_start_date = start_of_month(train_start_date+relativedelta(months=train_len))
        test_end_date = start_of_month(train_start_date+relativedelta(months=train_len+test_len))

        dates.append((train_start_date, test_start_date, test_end_date))

        train_start_date = start_of_month(train_start_date+relativedelta(months=test_len))

    return dates


class Backtest:

    def __init__(self, spot: Spot,
                 settings: BacktestSettings,
                 impact_functions: List[float] = exponents,
                 ):

        self.impact_functions = impact_functions
        self.spot = spot
        self.bin_freq = settings.bin_freq
        self.forecast_horizon = settings.forecast_horizon_s
        self.decay = settings.decay
        self.smoothing = settings.smoothing_str
        self.training_months = settings.train_len
        self.test_months = settings.test_len

        self.train_df: pd.DataFrame = None
        self.test_df: pd.DataFrame = None

        self.fit_summary: pd.DataFrame = None
        self.forecasts: pd.DataFrame = None
        self.input_data_dir = regression_path / settings.bin_freq / str(settings.forecast_horizon_s) / "pre_regression"/spot
        self.output_data_dir = regression_path / settings.bin_freq / str(settings.forecast_horizon_s) / "results"/"model_output"

        self.returns_df = pd.read_pickle(self.input_data_dir/f"{spot}-returns.pkl")
        self.crossval_windows = crossvalidation_generator(self.returns_df.index, self.training_months, self.test_months)

        return

    def get_hourly_dfs(self, impact) -> Dict[int, pd.DataFrame]:

        impact_file = f"{self.spot}_decay_{self.decay}-impact_{impact}-scaling_{self.smoothing}-pre_regression.pkl"

        impact_df = add_hour(pd.read_pickle(self.input_data_dir / impact_file))
        ret_df = add_hour(self.returns_df)

        xy = pd.concat({'x': impact_df, 'y': ret_df}, axis=1, names=['xy', 'time', 'hour']).swaplevel(0, 2, axis=1)

        end_hour = (self.spot.trading_end_time + relativedelta(hours=self.spot.bst_offset)).hour
        hours = list(set(xy.columns.get_level_values('hour')))
        hours = [hour for hour in hours if hour < end_hour]

        dfs = {hour: xy[hour].stack(level=0).reset_index('time', drop=True).dropna() for hour in hours}

        return dfs

    def run_backtest(self):

        all_results = []

        for impact in self.impact_functions:

            hourly_dfs = self.get_hourly_dfs(impact)

            for train_start, test_start, test_end in self.crossval_windows:
                for hour, xy in hourly_dfs.items():

                    train_start_idx = xy.index.searchsorted(train_start)
                    test_end_idx = xy.index.searchsorted(test_end)

                    ttf = xy.iloc[train_start_idx:test_end_idx, :]

                    test_start_idx = ttf.index.searchsorted(test_start)

                    train_df = ttf.iloc[:test_start_idx, :].astype('float64')
                    test_df = ttf.iloc[test_start_idx:, :].astype('float64')

                    if train_df.shape[0] == 0:
                        continue
                    if train_df['y'].abs().sum() == 0 or test_df['y'].abs().sum() == 0:
                        continue

                    coef, tr_r2 = linreg.linear_regression(train_df['x'].values,
                                                           train_df['y'].values)

                    te_r2 = linreg.predict_and_score(test_df['x'].values,
                                                     test_df['y'].values, coef)

                    all_results.append({'impact': impact, 'test_start': test_start, 'hour': hour,
                                        'train_r2': tr_r2,
                                        'test_r2': te_r2, 'coef': coef})

        self.save_results(all_results)

    def save_results(self, results: List[Dict]):
        os.makedirs(self.output_data_dir, exist_ok=True)

        file_name = f"{self.spot}-decay_{self.decay}-scaling_{self.smoothing}_train_{self.training_months}-test_{self.test_months}.csv"
        df = pd.DataFrame(results)
        df.to_csv(self.output_data_dir/file_name, index=False)


def backtest_one_spot(spot: str, backtest_settings: BacktestSettings):

    print(f"Starting {spot}")
    t_start = time.time()
    bt = Backtest(Spot(spot), backtest_settings)

    try:
        bt.run_backtest()
    except Exception as e:
        print(f"Failed {spot}: {e}")

    t_end = time.time()
    print(f"Finished {spot}: {t_end-t_start:.2f}")


def main(backtest_settings: BacktestSettings):
    print('running backtest')
    processes = 8
    processes = min(len(tickers), processes)
    bt_fn = PoolExecutor(backtest_one_spot, backtest_settings)

    with Pool(processes=processes) as pool:
        pool.map(bt_fn, tickers)


if __name__ == '__main__':

    settings = BacktestSettings(train_len=6,
                                test_len=3,
                                bin_freq='10s',
                                forecast_horizon_s=10,
                                smoothing_str='sma20',
                                decay=3600)

    for train_len, test_len in [(1, 1), (2, 1), (4, 2)]:
        print(f"{train_len=}, {test_len=}")
        settings = BacktestSettings(train_len=train_len,
                                    test_len=test_len,
                                    bin_freq='10s',
                                    forecast_horizon_s=10,
                                    smoothing_str='sma20',
                                    decay=3600)
        main(settings)
