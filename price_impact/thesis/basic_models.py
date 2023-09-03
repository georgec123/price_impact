import numpy as np


class ObizhaevaWang():

    def __init__(self, X_0: float, rho: float, T: float):

        self.X_0 = X_0
        self.rho = rho
        self.T = T

    def optimal_trades(self, n: int):

        bulk_trade = self.X_0/(2+self.rho*self.T)
        linear_trade = (self.X_0 - 2*bulk_trade)/(n-2)

        trades = np.full(fill_value=linear_trade, shape=n, dtype=np.float64)
        trades[0] = bulk_trade
        trades[-1] = bulk_trade

        return trades


class AlmgrenChriss():

    def __init__(self, X_0: float, kappa: float, T: float):

        self.X_0 = X_0
        self.kappa = kappa
        self.T = T

    def optimal_trades(self, n: int):

        times = np.linspace(0, self.T, n+1)
        inventory = np.sinh(self.kappa*(times - self.T)) / np.sinh(self.kappa*self.T)
        trades = np.diff(inventory)

        return trades
