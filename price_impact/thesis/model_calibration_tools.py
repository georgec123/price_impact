import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from price_impact.modelling.backtest_analysis import BacktestResults, BacktestSettings
from price_impact.modelling.Futures import Spot


SAMPLE_SPOTS = ['ES', 'NQ', 'CL', 'GL', 'TU', 'GC']


def intraday_analysis_plot(ax: plt.Axes, backtest_results: BacktestResults, date: str, legend: bool = True):

    spot = backtest_results.spot
    delta, _ = backtest_results.best_params(date)

    df = backtest_results.results_df.query(f"test_start=='{date}' and impact=={delta}").pivot(
        index='hour', columns='variable', values='value')

    ax2 = ax.twinx()

    df['coef'].plot(ax=ax2, label=r'$\lambda$', color='r')
    df[['train_r2', 'test_r2']].plot(ax=ax)

    ax.set_ylim(bottom=0, top=1)
    ax2.set_ylim(bottom=0)
    ax.set_title(f"{spot} {spot.description}\n $\delta={delta}$")

    if legend:
        ax.legend([r'Train $R^2$', r'Test $R^2$'], loc=3)
        ax2.legend(loc=4)
    else:
        ax.get_legend().remove()

    return ax, ax2


def lambda_heatplots(backtest_settings: BacktestSettings, delta=None):

    colormap = mpl.colormaps['magma']
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    for idx, spot in enumerate(SAMPLE_SPOTS):
        spot = Spot(spot)
        backtest_results = BacktestResults(spot, backtest_settings)

        if delta is None:
            best = backtest_results.statistics_dict['best']
        else:
            best = delta

        df = backtest_results.results_df

        # create 3d surface plot of coefficient of impact
        surface = df.query(f"variable=='coef' and impact=={best}").copy()
        surface['test_start'] = pd.to_datetime(surface['test_start'])

        surface = surface.drop_duplicates().pivot(columns='hour', index='test_start', values='value')

        i, j = idx//3, idx % 3

        if j == 2:
            cbar_kws = {'label': r'$\lambda$'}
        else:
            cbar_kws = {}

        ax = axs[i][j]
        sns.heatmap(surface.T, annot=False, ax=axs[i][j],
                    cbar_kws=cbar_kws,
                    cmap=colormap)
        ax.set_title(fr"{spot}. $\delta={best}$")

        if not i:
            ax.set_xlabel(None)
            ax.get_xaxis().set_visible(False)

        if i == 1:
            ax.set_xlabel("Test Start Date")
            labels = [item.get_text()[:7] for item in ax.get_xticklabels()]
            ax.set_xticklabels(labels)
        else:
            ax.set_xlabel(None)
        if j == 0:
            ax.set_ylabel("Hour")
        else:
            ax.set_ylabel(None)

        ax.grid(False)

    fig.tight_layout()

    return fig, axs


def plot_delta_changes(backtest_settings: BacktestSettings):
    colormap = mpl.colormaps['magma']
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    for idx, spot in enumerate(SAMPLE_SPOTS):
        spot = Spot(spot)
        backtest_results = BacktestResults(spot, backtest_settings)

        df = backtest_results.results_df
        df['test_start'] = pd.to_datetime(df['test_start'])

        r2_df = df[df['variable'] == 'train_r2']
        plot_df = r2_df.pivot_table(index='test_start', columns='impact', values='value', aggfunc='mean')

        i, j = idx//3, idx % 3
        ax = axs[i][j]
        ax = plot_df.plot(ax=ax, alpha=1, cmap=colormap)

        ax.grid(True)

        ax.set_ylim(bottom=0, top=1)
        ax.set_title(spot)

        if not i:
            ax.set_xlabel(None)
            ax.get_xaxis().set_visible(False)
        else:
            ax.tick_params(axis='x', labelrotation=45)
            ax.set_xlabel(r'Test Start')

        if not j:
            ax.set_ylabel(r'Train $R^2$')

        if not idx:
            ax.legend(ncols=3, title=r'$\delta$', fontsize=8)
        else:
            ax.get_legend().remove()

    fig.suptitle(r"Train $R^2$ for different $\delta$ over time")

    return fig, ax
