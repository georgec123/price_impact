import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint
from typing import Union
import warnings

   
class PropagatorImpactModel:

    def __init__(self, half_life_s: float, delta: float, trading_hours: float,
                 stdev: float, lambdas: Union[float, np.ndarray], freq_m: int = 15):
        
        self.half_life_s = half_life_s
        self.delta = delta
        self.trading_hours = trading_hours
        self.stdev = stdev
        self.lambdas = lambdas
        self.freq_m = freq_m

        self.volume_bin_size: int = 60*self.freq_m
        self.num_bins: int = int(self.trading_hours*60*60/self.volume_bin_size)

        if isinstance(self.lambdas, np.ndarray):
            self.lambdas_vary: bool = True
        else:
            self.lambdas_vary: bool = False
            self.lambdas = np.ones(shape=self.num_bins)*self.lambdas

        self.beta = np.log(2) / self.half_life_s
        self.beta_dt = self.beta*self.volume_bin_size

    def get_cum_impact(self, v: np.ndarray):

        def impact_kernel(x): return np.sign(x)*np.power(np.abs(x), self.delta)
        dq = impact_kernel(v)*self.lambdas
        ewm = np.frompyfunc(lambda left, right: (1-self.beta_dt)*left + right, 2, 1)
        cum_impact = ewm.accumulate(dq, dtype='object').astype('float32')*self.stdev
        return cum_impact

    def func_to_optimise(self, v: np.ndarray, spread: float = 0):

        cum_impact = self.get_cum_impact(v)

        return v@cum_impact + np.abs(v).sum()*spread/2

    def twap(self, total_volume: float) -> np.ndarray:
        return np.ones(shape=self.num_bins, dtype='float64')*total_volume/self.num_bins

    def get_optimal_volume(self, spread_bps: float, total_volume: float,
                           do_bounds: bool = False,
                           divisor=1, tol=1e-9, x0=None, method='SLSQP',
                           options=None):

        cons = []
        bounds = None

        method = method.upper()
        if do_bounds:
            if method in ['SLSQP', 'TRUST-CONSTR']:
                bounds = [(0, None) for _ in range(self.num_bins)]
            elif method == 'COBYLA':
                cons += [{'type': 'ineq', 'fun': lambda x: min(x)}]
            else:
                raise Exception("unsupported method")

        if x0 is None:
            x0 = self.twap(total_volume)

        spread = spread_bps/100/100

        def func(v): return self.func_to_optimise(v, spread=spread)/divisor
        def total_volume_constraint(dq): return np.sum(dq) - total_volume

        cons += [LinearConstraint(np.ones_like(x0), total_volume, total_volume)]

        result = minimize(func, x0=x0, method=method, constraints=cons,
                          options=options, bounds=bounds, tol=tol)

        if not result.success:
            warnings.warn(f"error fitting: {result.message}")

        return result

    def plot_volume(self, volume_schedule: np.ndarray, spread_bps: float = 0):

        total_volume: float = volume_schedule.sum()

        # plot results
        x = np.linspace(0, self.num_bins, self.num_bins)

        intended_impact = self.get_cum_impact(volume_schedule)
        twap_intended_impact = self.get_cum_impact(self.twap(total_volume))

        fig, ax = plt.subplots(1, figsize=(10, 6))
        ax.bar(x, volume_schedule, label='optimal trades')

        ax2 = ax.twinx()
        ax2.plot(intended_impact*100*100, label='Intended impact', color='r')

        ax2.plot(twap_intended_impact*100*100, label='TWAP Intended impact', color='k')
        ax2.legend(loc=3)

        ax3 = ax2.twinx()
        ax3.plot(self.lambdas, color='g', label=r'$\lambda$')
        ax3.set_ylim(bottom=0)

        ax3.legend(loc=2)

        if self.lambdas_vary:
            lambda_str = r"varying $ \lambda$ "
        else:
            lambda_str = r"constant $ \lambda$ "

        title = fr"Optimal Execution - Q: {total_volume:.3f}. Beta {self.beta:.5f}, delta: {self.delta:.2f} "
        title += fr"- {lambda_str}. spread {spread_bps}bps."+"\n"
        title += fr"$I_{{t+\Delta t}} = (1-\beta \Delta t)I_t + \lambda (\Delta Q)^{{\delta}}$    "
        title += fr"$\Delta t = {{{self.freq_m}}}m$"
        fig.suptitle(title)

        twap_cost = twap_intended_impact@self.twap(total_volume)*100*100
        optimal_cost = intended_impact@volume_schedule*100*100
        print(f"twap cost: {twap_cost:.2f}bps")
        print(f"optimal cost: {optimal_cost:.2f}bps")
        print(f"{100*(optimal_cost/twap_cost -1):.3f}%")

        return fig, ax
