import pandas as pd
import numpy as np
from pathlib import Path
from price_impact.modelling.backtest_analysis import BacktestResults, BacktestSettings
from price_impact.modelling.Futures import FutureContract, Spot
from price_impact import data_dir
from typing import Tuple
import datetime as dt
from dateutil.relativedelta import relativedelta

bin_path: Path = data_dir / 'synthetic' / 'bins'


def resample_lambdas(lambdas: pd.Series, spot: Spot, freq_m: int = 15) -> pd.Series:
    """
    Takes a series of hourly lambdas, a spot and a frequency. 
    Returns a resampled lambda series at the required frequency.


    :param pd.Series lambdas: Series of hourly lambdas
    :param Spot spot: Spot class, used to get correct correct market hours
    :param int freq_m: Resample frequency (minutes), defaults to 15
    :return _type_: Series of resampled lambdas with datetime index
    """
    trading_hours = pd.DataFrame(pd.date_range(spot.trading_start_time, spot.trading_end_time, freq='H')) + dt.timedelta()
    trading_hours = trading_hours + dt.timedelta(hours=spot.bst_offset)
    trading_hours.set_index(trading_hours[0].dt.hour, inplace=True, drop=True)

    lambdas = pd.merge(trading_hours, lambdas, left_index=True, right_index=True,
                       how='left').set_index(0).rename_axis('time')
    lambdas.index = pd.to_datetime(lambdas.index)

    lambdas = lambdas.resample(f'{freq_m}min').first().fillna(method='ffill', axis=0).iloc[:-1]

    return lambdas


def example_delta_lambdas(future: FutureContract, settings: BacktestSettings,
                          freq_m: int = 15, date='2022-01-01') -> Tuple[float, np.ndarray]:

    backtest_results = BacktestResults(future.spot, settings)

    delta, lambdas = backtest_results.best_params(date)
    lambdas = resample_lambdas(lambdas, future.spot, freq_m)
    lambdas = lambdas.to_numpy().flatten()

    return delta, lambdas


def get_std(spot, train_len, test_start_date, freq: str = '10s'):

    spot = Spot(spot)
    fname = bin_path/freq/f"{spot}_px.pkl"
    px_df = pd.read_pickle(fname)
    train_start = pd.to_datetime(test_start_date) - relativedelta(months=train_len)
    px_df = px_df.query(f"date<'{test_start_date}' and date>='{train_start}'")
    stdev = px_df.pct_change(axis=1).std(axis=1).mean()

    return stdev


def get_vwap(spot: Spot, train_len: int, test_start_date: str, in_freq: str = '10s', out_freq_m: int = 15):

    spot = Spot(spot)
    fname = bin_path/in_freq/f"{spot}_vol.pkl"
    vol_df = pd.read_pickle(fname)
    train_start = pd.to_datetime(test_start_date) - relativedelta(months=train_len)
    vol_df = vol_df.query(f"date<'{test_start_date}' and date>='{train_start}'").abs()

    vol_means = vol_df.mean(axis=0)
    vol_means.index = pd.to_datetime(vol_means.index, format='%H:%M:%S')

    vol_means = vol_means.resample(f'{out_freq_m}T').sum()
    vol_means = vol_means[vol_means != 0]
    vol_means = vol_means/vol_means.sum()
    vol_means.index = vol_means.index.time

    return vol_means.values


if __name__ == '__main__':
    get_vwap('ES', 6, '2022-01-01', '10s', 15)
