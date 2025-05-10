from abc import ABC, abstractmethod
from copy import copy
import numpy as np
from sklearn.base import BaseEstimator
from sklearn import metrics
from sklearn.linear_model import LinearRegression

class LatexLinearEstimator(BaseEstimator, ABC):
    def __init__(self):
        super().__init__()
        self.linear_model = None

    def __copy__(self):
        m = type(self)().set_my_params(*self.get_my_params())
        return m

    @abstractmethod
    def get_my_params(self):
        pass

    def set_base_params(self, a, b):
        self.linear_model = LinearRegression().fit([[0], [1]], [0, 1])
        self.linear_model.intercept_ = a
        self.linear_model.coef_ = np.array([b])

    @abstractmethod
    def set_my_params(self, a, b):
        pass

    def set_linear_model(self, m):
        self.linear_model = m

    def get_base_params(self):
        if self.linear_model is None:
            raise ValueError("Model not fitted")
        return np.array([self.linear_model.intercept_, self.linear_model.coef_[0]]).ravel()

    @abstractmethod
    def get_equation_string(self, r=8, latex=True):
        pass

    def round_params(self, places=4):
        self.set_my_params(*np.round(self.get_my_params(), places))
        return self

    def calc_r2(self, X, y):
        y_pred = self.predict(X)
        try:
            return metrics.r2_score(y.ravel(), y_pred)
        except ValueError:
            return -np.inf

    def round_threshold(self, X, y, step=0.01):
        r = 8
        baseline = copy(self).round_params(r).calc_r2(X, y)
        r -= 1
        try:
            while baseline - copy(self).round_params(r).calc_r2(X, y) < step and r >= 0:
                r -= 1
        except ValueError:
            pass
        r += 1
        return r