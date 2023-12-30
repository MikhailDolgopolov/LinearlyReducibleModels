from copy import copy

import pandas as pd
from sklearn import metrics
import warnings

from sklearn.base import BaseEstimator

from Estimators.SimpleEstimators import *


class GeneralLinearizationEstimator(LatexLinearEstimator):
    def round_params(self, places=4, deep=False):
        if deep:
            for estimator in self.ordered:
                estimator[0].round_params(places, False)
        else:
            self.ordered[0][0].round_params(places, False)

        return self

    def get_my_params(self):

        return self[0].get_my_params()

    def set_my_params(self, a, b):
        self[0].set_my_params(a,b)

    def get_equation_string(self, r=8, latex=True):
        return self[0].get_equation_string(8)

    def __init__(self, acceptable_r2_level=0.9):
        super().__init__()
        self.acceptable_r2_level = acceptable_r2_level
        estimators = ["LinearEstimator", "ExponentialEstimator", "HorizontalShiftHyperbolaEstimator",
                      "LogarithmicEstimator",
                      "OriginBoundHyperbolaEstimator", "PowerFunctionEstimator", "VerticalShiftHyperbolaEstimator"]

        self.estimators = {eval(name)(): -100 for name in estimators}
        self.ordered: list[tuple[LatexLinearEstimator,float]] = []

    def __getitem__(self, item:int)->LatexLinearEstimator:
        if self.ordered[item][1]==-100:
            raise Exception("Not fitted")
        return self.ordered[item][0]

    def fit(self, X, y):
        self.X_, self.y_ = X, y
        for est in self.estimators.keys():
            try:
                est.fit(X, y)
                self.estimators[est] = est.calc_r2( X, y)
            except:
                warnings.warn(f"{est} could not be fitted. You could try removing zeros.")

        best = max(self.estimators, key=self.estimators.get)

        if self.estimators[best] == -100:
            raise Warning("No fitting model found")

        self.ordered = list(sorted(self.estimators.items(), key=lambda x: x[1], reverse=True))

        if max(self.estimators.values()) < self.acceptable_r2_level:
            print(f"Requested goodness of fit ({self.acceptable_r2_level}) not found.")

        print(f"the best estimator is {best} with coefficient of determination {self.estimators[best]}.")
        return self

    def predict(self, X):
        check_is_fitted(self)
        return max(self.estimators, key=self.estimators.get).predict(X)

    def pick_best_concise_model(self, X, y, error=0.01):
        good_estimators=[]
        for i in range(len(self.ordered)):
            try:
                good_estimators.append(self[i])
            except:
                pass

        px = 0.9
        def th(x):
            return -0.1*(x-px)+error

        thresholds = [th(r2) for r2 in [est.calc_r2(X,y) for est in good_estimators]]
        places = [good_estimators[i].round_threshold(X,y, thresholds[i]) for i in range(len(good_estimators))]
        print(places)
        rounded = [copy(good_estimators[i]).round_params(places[i]) for i in range(len(good_estimators))]
        rating = {rounded[i]:rounded[i].calc_r2(X, y)*(1-places[i]*0.02)
                  for i in range(len(rounded))}
        return [i[0] for i in list(sorted(rating.items(), key=lambda x: x[1], reverse=True))]



