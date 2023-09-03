from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from attrs import define
from george_dev.price_impact.price_impact.execution.impact_functions import ImpactFunction
from scipy.optimize import minimize


class TransientImpact(ABC):

    def __init__(self, impact_func: ImpactFunction, N: int = 100, T: float = 1,
                 total_volume=1, spread: float = 0, alpha: np.ndarray = None):
        self.N = N
        self.T = T
        self.Delta = T/N
        self.impact_func = impact_func
        self.total_volume = total_volume

        self.G = self.toeplitz()
        self.twap = np.repeat(self.Delta*self.total_volume, self.N)
        self.spread = spread

        if alpha is None:
            self.alpha = np.zeros(shape=N, dtype=np.float64)
        else:
            if len(alpha.shape) != 1:
                raise ValueError(f"Alpha signal must be of dimension 1, not {len(alpha.shape)}")
            if alpha.shape[0] != N:
                raise ValueError(f"Alpha must be same length as N. {N}")

            self.alpha = alpha

    def toeplitz(self):
        N = self.N
        lower_ones = 1-np.triu(np.ones(shape=(N, N), dtype=np.float64))
        G_lower = np.fromfunction(self.G_ij_lower, (N, N))
        G_lower = np.nan_to_num(G_lower, nan=0.0)
        G_lower *= lower_ones

        G_diag = np.diag(np.repeat(self.G_ii_diagonal(), N))
        G = G_lower + G_diag

        return G

    def G_ij_lower(self, i, j):
        lower_ones = 1-np.triu(np.ones(shape=(self.N, self.N), dtype=np.float64))

        i *= lower_ones
        j *= lower_ones

        return self._G_ij_lower(i, j)

    @abstractmethod
    def _G_ij_lower(self, i, j):
        pass

    @abstractmethod
    def G_ii_diagonal(self):
        pass

    @abstractmethod
    def kernel_string(self):
        pass

    def expect_cost_compute(self, v):
        return v @ ((self.G @ self.impact_func(v).T) - self.alpha)

    def spread_cost(self, v, spread):
        return np.abs(v).sum()*self.Delta*spread

    def total_volume_constraint(self, x):
        return np.sum(x) - self.total_volume/self.Delta

    def optimal_execution(self, divisor: float = None, **kwargs):

        if self.spread:
            def cost_function(v): return self.expect_cost_compute(v) + self.spread_cost(v, self.spread)
        else:
            def cost_function(v): return self.expect_cost_compute(v)

        if divisor is not None:
            min_function = lambda x: cost_function(x)/divisor
        else:
            min_function = cost_function

        cons = {'type': 'eq', 'fun': self.total_volume_constraint}
        result = minimize(min_function, x0=self.twap,
                          method="SLSQP", constraints=cons,
                          tol=1e-10, **kwargs)

        return result

    def plot_volume(self, volume_schedule: np.ndarray):

        x_axis = np.ones(self.N, dtype=int).cumsum()
        fig, ax = plt.subplots(1)
        ax = ax.bar(x_axis, volume_schedule*self.Delta)
        fig.suptitle(f"${self.impact_func.function_string}{self.kernel_string}$ \t $X={self.total_volume}$ "
                     f"\t Spread {self.spread:.2f}bps")


class PowerDecayTransient(TransientImpact):
    def __init__(self, power_decay: float, *args, **kwargs):

        self.power_decay = power_decay
        super().__init__(*args, **kwargs)

    def _G_ij_lower(self, i, j):
        lower_ones = 1-np.triu(np.ones(shape=(self.N, self.N), dtype=np.float64))

        dec = self.power_decay

        out = (1 / (1 - dec) * (1 - dec)) * (self.Delta) ** (2 - dec)
        out = out*((i - j + lower_ones)**(2 - dec) - 2*(i - j)**(2 - dec) + (i - j - lower_ones)**(2 - dec))
        return out

    def G_ii_diagonal(self):
        dec = self.power_decay
        return (1 / ((1 - dec) * (2 - dec))) * self.Delta ** (2 - dec)

    @property
    def kernel_string(self):
        return f"(t-s)^{{-{self.power_decay}}}"


class ExpDecayTransient(TransientImpact):
    def __init__(self, beta: float, *args, **kwargs):

        self.beta = beta
        super().__init__(*args, **kwargs)

    def _G_ij_lower(self, i, j):

        beta = self.beta

        out = 4*np.exp(-beta*self.Delta*(i-j))/beta/beta
        out = out*(np.sinh(beta*self.Delta/2)**2)

        return out

    def G_ii_diagonal(self):
        beta = self.beta

        diag = (self.Delta + (np.exp(-beta*self.Delta)-1)/beta)/beta

        return diag

    @property
    def kernel_string(self):
        return f"e^{{-{self.beta}(t-s)}}"
