from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from attrs import define


class ImpactFunction(ABC):

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def function_string():
        pass


@define
class PowerImpact(ImpactFunction):

    delta: float
    lam: float = 1

    def __call__(self, x):
        return self.lam * np.sign(x) * np.abs(x)**self.delta

    @property
    def function_string(self):
        return f"{self.lam}x^{{{self.delta}}}"

    def plot(self, span=2, N=101):
        x = np.linspace(-1*span, span, N)
        y = self.__call__(x)
        plt.plot(x, y)


@define
class ConcaveConvexImpact():
    delta: float
    d: float
    v_critical: float = 1
    lam: float = 1

    def __call__(self, x):

        abs_x = np.abs(x)
        V = self.v_critical

        out = self.lam*(abs_x/(abs_x+V))**self.delta
        out = out + self.d*(abs_x*(abs_x+V))/V/V

        return np.sign(x)*out

    @property
    def function_string(self):
        fn_str = rf" {self.lam}\left( \frac{{|v|}}{{|v|+{self.v_critical}}} \right)^{{{self.delta}}} " + \
            rf"+ {self.d} \frac{{|v|(|v| + {self.v_critical})}}{{{self.v_critical}^2}} "
        return fn_str


@define
class ConcaveConvexKatoImpact():
    delta_1: float
    delta_2: float
    lam: float = 1
    x_threshold: float = 1
    alpha: float = None
    gamma: float = None

    def __attrs_post_init__(self):

        self.alpha = self.lam*(self.delta_1/self.delta_2)*(self.x_threshold**(self.delta_1-self.delta_2))
        self.gamma = self.lam*self.x_threshold**self.delta_1 - self.alpha*self.x_threshold**self.delta_2
        return

    def __call__(self, x):
        output = np.copy(x)

        def lower_func(x): return self.lam*np.sign(x)*(np.abs(x)**self.delta_1)
        def upper_func(x): return np.sign(x)*(self.alpha*(np.abs(x)**self.delta_2) + self.gamma)

        lower_mask = np.abs(output) < self.x_threshold

        output[lower_mask] = lower_func(output[lower_mask])
        output[~lower_mask] = upper_func(output[~lower_mask])

        return output

    def plot(self, span=2, N=101):
        x = np.linspace(-1*span*self.x_threshold, span*self.x_threshold, N)
        y = self.__call__(x)
        plt.plot(x, y)

    @property
    def function_string(self):
        return f"CCKI - {self.lam}x^{{{self.delta_1}}}"
