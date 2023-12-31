import numpy as np
from sklearn import metrics
from sklearn.base import BaseEstimator
from abc import ABC, abstractmethod
from copy import copy

from sklearn.linear_model import LinearRegression

from DataWrangling import clean_xy


def f_f(f: float) -> str:
    if f > 0:
        return f"+{f:g}"
    return f"{f:g}"


def f_p(f: float) -> str:
    if f > 0:
        return f"{f:g}"
    return f"({f:g})"


class LatexLinearEstimator(BaseEstimator, ABC):
    def __init__(self):
        super().__init__()
        self.linear_model = None

    def __getitem__(self, item):
        return self

    def __copy__(self):
        m = type(self[0])().set_my_params(*self.get_my_params())
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
        self.linear_model = LinearRegression()

    def set_linear_model(self, m):
        self.linear_model = m

    def get_base_params(self):
        return np.array([self.linear_model.intercept_, self.linear_model.coef_[0]]).ravel()

    @abstractmethod
    def get_equation_string(self, r=8, latex=True):
        raise NotImplementedError("Base Class")

    def round_params(self, places=4, deep=False):
        self.set_my_params(*np.round(self.get_my_params(), places))
        return self

    def calc_r2(self, X, y):
        X, y = clean_xy(X,y)
        y_pred = self.predict(X)
        return metrics.r2_score(y, y_pred)

    def round_threshold(self, X, y, step=0.01):
        r = 8
        baseline = copy(self[0]).round_params(r).calc_r2(X, y)
        r -= 1

        try:
            while baseline - copy(self[0]).round_params(r).calc_r2(X, y) < step:
                r -= 1

        except:
            pass
        r += 1
        return r
