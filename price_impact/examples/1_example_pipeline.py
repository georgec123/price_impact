from price_impact.data import main as data_main
from price_impact.modelling import impacts, backtesting
from price_impact.modelling.backtest_analysis import BacktestSettings


def main():
    sma_len = 20
    sma_str = f'sma{sma_len}'

    backtest_settings = BacktestSettings(
        train_len=6,
        test_len=3,
        bin_freq='10s',
        forecast_horizon_s=10,
        smoothing_str=sma_str,
        decay=3600
    )

    data_main.main(freq=backtest_settings.bin_freq, sma_len=sma_len)
    impacts.make_impacts(freq=backtest_settings.bin_freq,
                         decays=[backtest_settings.decay],
                         scaling_smoother=backtest_settings.smoothing_str,
                         forecast_horizon_s=backtest_settings.forecast_horizon_s
                         )

    backtesting.main(backtest_settings)


if __name__ == '__main__':
    main()
