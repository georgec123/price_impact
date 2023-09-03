import os
from pathlib import Path
from typing import Tuple

import pandas as pd
import plotly.graph_objects as go
from attrs import define
from price_impact import regression_path


@define
class BacktestSettings:
    train_len: int
    test_len: int
    bin_freq: str = '10s'
    forecast_horizon_s: int = 10
    smoothing_str: str = 'sma20'
    decay: int = 3600


class BacktestResults:

    def __init__(self, spot: str,  settings: BacktestSettings, regression_path: Path = regression_path):
        self.spot = spot
        self.results_path = regression_path / settings.bin_freq / str(settings.forecast_horizon_s) / 'results'
        self.smoothing = settings.smoothing_str
        self.decay = settings.decay
        self.train_months, self.test_months = settings.train_len, settings.test_len

        self.summary_file_path = self.results_path / 'model_output' / \
            f"{self.spot}-decay_{self.decay}-scaling_{self.smoothing}_train_{self.train_months}-test_{self.test_months}.csv"

        self._results_df: pd.DataFrame = pd.read_csv(
            self.summary_file_path).melt(
            id_vars=['impact', 'test_start', 'hour'],
            value_vars=['train_r2', 'test_r2', 'coef'])
        self._r2_df: pd.DataFrame = None
        self._r2_df_average: pd.DataFrame = None

        self._statistics_dict: dict = None

    @property
    def results_df(self):

        return self._results_df

    @property
    def r2_df_average(self):
        if self._r2_df_average is not None:
            return self._r2_df_average

        return self.results_df.pivot_table(index=['variable', 'test_start'], columns='impact', values='value', aggfunc='mean')

    def best_params(self, test_start) -> Tuple[float, pd.Series]:
        """
        Gets best delta and lambdas for a given futures contract


        :param FutureContract future: Future to get parameters for
        :return Tuple[float, np.ndarray]: delta and series of hourly lambdas
        """

        best_params = self.get_best_impact_coefs()
        best_impact = best_params.query(f"test_start=='{test_start}'")['impact'].iloc[0]

        lambdas = self.results_df.query(f"test_start=='{test_start}' and variable=='coef' "
                                        f"and impact=={best_impact}")
        lambdas = lambdas[['hour', 'value']].set_index('hour')
        return float(best_impact), lambdas

    def get_best_impact_coefs(self, test_train: str = 'train') -> pd.DataFrame:

        if test_train not in ['test', 'train']:
            raise ValueError(f"unsupported parameter for test_train, {test_train}")

        gb = self.r2_df_average
        best_params = gb.query(f"variable=='{test_train}_r2'").apply(lambda x: gb.columns[x.argmax()], axis=1).to_frame()
        best_params = best_params.reset_index().rename(columns={0: "impact"}).drop(columns='variable')

        return best_params

    @property
    def statistics_dict(self):
        if self._statistics_dict is not None:
            return self._statistics_dict

        gb = self.r2_df_average
        results_dict = {}

        # pick 'best parameter' from the train results
        results_dict['best_method'] = 'train_r2'
        best_train_params = self.get_best_impact_coefs('train')
        best_test_params = self.get_best_impact_coefs('test')

        # compare train best params with test params
        best_train_params = pd.merge(left=gb.melt(ignore_index=False).reset_index(),
                                     right=best_train_params,
                                     how='inner',
                                     left_on=['test_start', 'impact'],
                                     right_on=['test_start', 'impact'])

        best_test_params = pd.merge(left=gb.melt(ignore_index=False).reset_index(),
                                    right=best_test_params,
                                    how='inner',
                                    left_on=['test_start', 'impact'],
                                    right_on=['test_start', 'impact'])

        best_tr_impacts = best_train_params.pivot(
            index=['test_start', 'impact'],
            columns='variable', values='value').reset_index().sort_values(
            by=['test_start']).reset_index(
            drop=True)

        best_te_impacts = best_test_params.pivot(
            index=['test_start', 'impact'],
            columns='variable', values='value').reset_index().sort_values(
            by=['test_start']).reset_index(
            drop=True)

        b_tr_b_te = pd.merge(best_tr_impacts, best_te_impacts, on=['test_start'],
                             suffixes=('_tr', '_te')).sort_values(by='test_start')

        results_dict['avg_tr_fr_tr'] = b_tr_b_te['train_r2_tr'].mean()
        results_dict['avg_tr_fr_te'] = b_tr_b_te['train_r2_te'].mean()
        results_dict['avg_te_fr_tr'] = b_tr_b_te['test_r2_tr'].mean()
        results_dict['avg_te_fr_te'] = b_tr_b_te['test_r2_te'].mean()

        # measure of stability between test and train given model chosen on train r2
        ttr = b_tr_b_te['test_r2_tr'] / b_tr_b_te['train_r2_tr']
        results_dict['test_train_ratio'] = ttr.mean()

        # measure of regret of chosing the model based on training
        regret = b_tr_b_te['test_r2_te'] - b_tr_b_te['test_r2_tr']
        results_dict['avg_max_test_regret'] = regret.mean()

        vc = b_tr_b_te['impact_tr'].value_counts()
        results_dict['pct_best'] = vc.iloc[0]/vc.sum()
        results_dict['best'] = vc.index[0]

        pct_curr_best_is_next_best = (b_tr_b_te['impact_tr'].shift() == b_tr_b_te['impact_tr']).iloc[1:].mean()
        results_dict['pct_curr_best_is_next_best'] = pct_curr_best_is_next_best

        self._statistics_dict = results_dict

        return self._statistics_dict

    def create_plot_r2(self) -> Tuple[go.Figure, go.Figure]:

        plot_df = self.r2_df_average.query("variable=='test_r2'").sort_values(by=['test_start'])
        rank_df = plot_df.rank(axis=1, ascending=False)

        fig = go.Figure(layout=go.Layout(height=600, width=1000, xaxis_title="Contract",
                        yaxis_title="Test R2", title=f"R2 - Futures of {self.spot}", yaxis_range=[0, 1]))
        rank_fig = go.Figure(layout=go.Layout(height=600, width=1000, xaxis_title="Contract",
                             yaxis_title="Rank - lower is better", title=f"R2 rank - Futures of {self.spot}"))

        idx = pd.to_datetime(plot_df.index.get_level_values('test_start'))

        for col in plot_df.columns:
            rank_fig.add_trace(go.Scatter(x=idx, y=rank_df[col], name=col, mode="lines"))
            fig.add_trace(go.Scatter(x=idx, y=plot_df[col], name=col, mode="lines"))

        return fig, rank_fig

    def create_plot_surface(self, exponent=None) -> go.Figure:
        df = self.results_df
        lambdas = df[df['variable'] == 'coef']
        if exponent is None:
            best = self.statistics_dict['best']
        else:
            best = exponent

        # create 3d surface plot of coefficient of impact
        surface = lambdas[lambdas['impact'] == best].copy()

        surface = surface.drop_duplicates().pivot(columns='hour', index='test_start', values='value')

        fig = go.Figure(layout=go.Layout(height=600, width=1000))
        fig.add_trace(
            go.Surface(
                z=surface.values.tolist(),
                y=surface.index.values, x=surface.columns, colorscale="Viridis"))
        fig.update_layout(title=fr"Coefficient of impact - {self.spot}. f(x) = lambda x^{best}")
        fig.update_scenes(yaxis_title_text='Contract Expiry',
                          xaxis_title_text='Hour',
                          zaxis_title_text=r"lambda")

        return fig

    def create_and_save_plots(self):

        out_dir = self.results_path/'summary' / 'plots' / \
            f"{self.smoothing}_{self.decay}" / f"train_{self.train_months}-test_{self.test_months}"
        os.makedirs(out_dir, exist_ok=True)

        fig, rank_fig = self.create_plot_r2()

        fig.write_html(out_dir/f"{self.spot}_r2.html")
        rank_fig.write_html(out_dir/f"{self.spot}_rank.html")

        lambda_surface = self.create_plot_surface()
        lambda_surface.write_html(out_dir/f"{self.spot}_lambda_surface.html")

        return fig, rank_fig, lambda_surface
