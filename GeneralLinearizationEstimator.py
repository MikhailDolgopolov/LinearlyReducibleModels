import warnings
from copy import copy

import numpy as np
from sklearn.utils.validation import check_is_fitted

from LatexLinearEstimator import LatexLinearEstimator
from SimpleEstimators import (LinearEstimator, ExponentialEstimator,
                              HorizontalShiftHyperbolaEstimator, LogarithmicEstimator,
                              OriginBoundHyperbolaEstimator, PowerFunctionEstimator,
                              VerticalShiftHyperbolaEstimator)


class GeneralLinearizationEstimator(LatexLinearEstimator):
    def __init__(self, acceptable_r2_level=0.5):
        super().__init__()
        self.acceptable_r2_level = acceptable_r2_level
        self.estimator_classes = [LinearEstimator, ExponentialEstimator,
                                  HorizontalShiftHyperbolaEstimator, LogarithmicEstimator,
                                  OriginBoundHyperbolaEstimator, PowerFunctionEstimator,
                                  VerticalShiftHyperbolaEstimator]
        self.estimators = {}
        self.ordered = []
        self.best_estimator_ = None

    def fit(self, X, y):
        self.X_, self.y_ = X, y
        self.estimators = {}
        for cls in self.estimator_classes:
            est = cls()
            try:
                est.fit(X, y)
                r2 = est.calc_r2(X, y)
                self.estimators[est] = r2
            except ValueError as e:
                warnings.warn(f"{cls.__name__} failed to fit: {str(e)}")

        if not self.estimators:
            raise RuntimeError("No models could be fitted")

        self.ordered = list(sorted(self.estimators.items(), key=lambda x: x[1], reverse=True))
        self.best_estimator_ = self.ordered[0][0]

        if self.ordered[0][1] < self.acceptable_r2_level:
            warnings.warn(f"Best R² ({self.ordered[0][1]:.4f}) below threshold ({self.acceptable_r2_level})")

        return self

    def __getitem__(self, item: int) -> LatexLinearEstimator:
        if not self.ordered:
            raise ValueError("Estimator not fitted yet")
        if self.ordered[item][1] == -np.inf:
            raise ValueError(f"Estimator at index {item} was not fitted")
        return self.ordered[item][0]

    def predict(self, X):
        check_is_fitted(self, 'best_estimator_')
        return self.best_estimator_.predict(X)

    def get_my_params(self):
        check_is_fitted(self, 'best_estimator_')
        return self.best_estimator_.get_my_params()

    def set_my_params(self, a, b):
        check_is_fitted(self, 'best_estimator_')
        self.best_estimator_.set_my_params(a, b)
        return self

    def get_equation_string(self, r=8, latex=True):
        check_is_fitted(self, 'best_estimator_')
        return self.best_estimator_.get_equation_string(r, latex)

    def round_params(self, places=4, deep=False):
        check_is_fitted(self, 'best_estimator_')
        if deep:
            for est, _ in self.ordered:
                est.round_params(places)
        else:
            self.best_estimator_.round_params(places)
        return self

    def pick_best_concise_model(self, X, y, error=0.005, simplicity_weight=4):
        check_is_fitted(self, 'best_estimator_')
        good_estimators = [est for est, r2 in self.ordered if r2 > -np.inf]

        if not good_estimators:
            return []

        def th(r2, pivot=0.9, slope=-0.04):
            return slope * (r2 - pivot) + error

        thresholds = [th(r2) for r2 in [est.calc_r2(X, y) for est in good_estimators]]
        places = [est.round_threshold(X, y, thresh) for est, thresh in zip(good_estimators, thresholds)]
        rounded = [copy(est).round_params(p) for est, p in zip(good_estimators, places)]
        rating = {est: est.calc_r2(X, y) * (1 - p * error * simplicity_weight)
                  for est, p in zip(rounded, places)}

        return [est for est, _ in sorted(rating.items(), key=lambda x: x[1], reverse=True)]