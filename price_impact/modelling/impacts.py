import logging
import os
from itertools import product
from logging import Logger
from multiprocessing import Pool
from typing import List, Union

import numpy as np
import pandas as pd
from ewm import ewm_alpha
from price_impact import data_dir, exponents, tickers
from price_impact.modelling.Futures import Spot
from price_impact.utils import PoolExecutor, get_logger

bin_path = data_dir/'synthetic'/'bins'

logger: Logger = get_logger(__name__)


def impact_state(traded_volume_df: pd.DataFrame, monthly_scaling_factor: pd.DataFrame,
                 half_life_s: float, exponent: Union[float, str], lam: Union[float, pd.DataFrame] = 1,
                 bin_seconds: int = 10) -> pd.DataFrame:
    """
    Implementation of week 2, slide 49
    Calculate impact state iteratively using a model like
    I_{t+1} = (1− β∆t)I_t + λ∆_t Q^d
    Where ∆_t Q is the scaled traded volume at time t, β is the decay factor
    We use the pandas ewm function to calculate the exponential moving average, which expects the form
    I_0 = ∆_0 Q
    I_{t+1} = (1 − α)I_t + α∆_t Q
    so we need to set α = 1-exp(−β∆t) and perform the transformation ∆_t Q -> ∆_t Q/α for t>0
    """

    # power kernel
    def volume_kernel(x): return np.sign(x) * np.power(np.abs(x), exponent)

    beta = np.log(2) / half_life_s
    alpha = 1 - beta*bin_seconds

    assert (monthly_scaling_factor.index == traded_volume_df.index).all(
    ), "Volume df and scaling df need same index."

    pre_ewm = traded_volume_df.copy()
    # put vol in terms of ADV
    pre_ewm = pre_ewm.divide(monthly_scaling_factor["volume"], axis="rows")

    # perform kernel transformation (ie linear or sqrt)
    pre_ewm = volume_kernel(pre_ewm)

    # multiply by vol forecas
    pre_ewm = pre_ewm.multiply(monthly_scaling_factor["stdev"],  axis="rows")

    # apply lambda
    pre_ewm = pre_ewm*lam
    pre_ewm.loc[:, :] = ewm_alpha(pre_ewm.to_numpy(), alpha=alpha)

    return pre_ewm


def compute_impact_on_spot(spot: str, freq_s: str, decays:  List[int] = [3600],
                           scaling_smoother: str = 'sma20', forecast_horizon_s: int = 10):
    logger.info(f"Starting impact on {spot}")

    spot = Spot(spot)
    bin_s = int(freq_s[:-1])

    px_df = pd.read_pickle(bin_path/freq_s/f'{spot}_px.pkl')
    volume_df = pd.read_pickle(bin_path/freq_s/f'{spot}_vol.pkl')

    volume_df.index = pd.to_datetime(volume_df.index)
    px_df.index = pd.to_datetime(px_df.index)

    monthly_scaling_factor = pd.read_pickle(data_dir/'daily'/f"{spot}_daily_{scaling_smoother}.pkl")

    if volume_df.isna().all().all() or monthly_scaling_factor.isna().all().all():
        return
    if monthly_scaling_factor.shape[0] != volume_df.shape[0]:
        # not enough train/ scaling data
        return
    if px_df.shape[0] == 0:
        logger.debug("No data after filtering, skipping")
        return

    explanation_horizon_period = forecast_horizon_s/bin_s

    if not explanation_horizon_period.is_integer():
        raise Exception(
            f"forecast_horizon_s must be divisible by bin_s, current values ({forecast_horizon_s, bin_s})")
    else:
        explanation_horizon_period = int(explanation_horizon_period)

    returns = px_df.pct_change(explanation_horizon_period, axis="columns").shift(-explanation_horizon_period, axis="columns")

    path = bin_path/'..'/'regression' / freq_s/f"{forecast_horizon_s}"/'pre_regression'/spot
    os.makedirs(path, exist_ok=True)

    returns_file_name = f"{spot}-returns.pkl"
    returns = returns.astype('float32')
    returns.to_pickle(path/returns_file_name)

    for half_life_s, exponent in product(decays, exponents):
        # print(f"{exponent=}")
        impact_file_name = f"{spot}_decay_{half_life_s}-impact_{exponent}-scaling_{scaling_smoother}-pre_regression.pkl"

        cum_impact = impact_state(traded_volume_df=volume_df, monthly_scaling_factor=monthly_scaling_factor,
                                  half_life_s=half_life_s, exponent=exponent, bin_seconds=bin_s)
        impact = cum_impact.diff(explanation_horizon_period, axis="columns").shift(-explanation_horizon_period, axis="columns")

        impact = impact.astype('float32')
        impact.to_pickle(path/impact_file_name)

    logger.info(f"Finished impact on {spot}")


def make_impacts(freq: str = '10s', **impact_kwargs):
    logger.info('starting impacts')

    processes = min(8, len(tickers))

    fn = PoolExecutor(compute_impact_on_spot, freq, **impact_kwargs)

    with Pool(processes=processes) as pool:
        pool.map(fn, tickers)
    logger.info('finished impacts')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    make_impacts('10s')
