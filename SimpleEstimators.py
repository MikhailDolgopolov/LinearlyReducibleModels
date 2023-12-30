import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from DataWrangling import clean_xy
from Estimators.LatexLinearEstimator import LatexLinearEstimator, f_f, f_p


class LinearEstimator(LatexLinearEstimator):

    def set_my_params(self, a, b):
        super().set_base_params(a, b)
        return self

    def get_my_params(self):
        return self.get_base_params()

    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        super().set_linear_model(LinearRegression().fit(X, y))
        return self

    def predict(self, X):
        return self.linear_model.predict(X)

    def get_equation_string(self, r=8, latex=True):
        pms = np.round(self.get_my_params(), r)
        return f"y = {pms[0]:g}{f_f(pms[1])}x"


class ExponentialEstimator(LatexLinearEstimator):
    def set_my_params(self, a, b):
        print("I want to set ", (a,b))
        al=a
        if abs(a) > 0.0000001:
            al = np.log(abs(a))*np.sign(a)
        bl=b
        if abs(b) > 0.0000001:
            bl = np.log(abs(b))*np.sign(a)

        print("In base it's ", (al,bl))
        super().set_base_params(al, bl)
        return self

    def get_equation_string(self, r=8, latex=True):
        a, b = np.round(self.get_my_params(), r)
        mult = r"\cdot" if latex else "*"
        return f"{a:g} {mult} {f_p(b)}^x"

    def get_my_params(self):
        pms = super().get_base_params()
        return np.exp(pms[0]) * np.sign(pms[1]), np.exp(pms[1] * np.sign(pms[1]))

    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        self.X_, self.y_ = clean_xy(X, np.sign(y) * np.log(np.abs(y)))
        self.set_linear_model(LinearRegression().fit(self.X_, self.y_))
        return self

    def predict(self, X):
        y_pred = self.linear_model.predict(X)
        s = np.sign(self.get_my_params()[0])
        return np.exp(y_pred * s) * s


class HorizontalShiftHyperbolaEstimator(LatexLinearEstimator):

    def get_my_params(self):
        return super().get_base_params()

    def set_my_params(self, a, b):
        super().set_base_params(a, b)
        return self

    def get_equation_string(self, r=8, latex=True):
        a, b = np.round(self.get_my_params(), r)
        return r"y = \frac{1}{" + f"{b}x" + f"{f_f(a)}" + "}"

    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        # Check that X and y have correct shape
        X = check_X_y(X, y.ravel())
        self.X_, self.y_ = clean_xy(X, 1 / y)
        self.set_linear_model(LinearRegression().fit(self.X_, self.y_))

        # Return the classifier
        return self

    def predict(self, X):
        X = check_array(X)
        return 1 / self.linear_model.predict(X)


class LogarithmicEstimator(LatexLinearEstimator):

    def get_my_params(self):
        return super().get_base_params()

    def set_my_params(self, a, b):
        super().set_base_params(a, b)
        return self

    def get_equation_string(self, r=8, latex=True):
        a, b = np.round(self.get_my_params(), r)
        mult = r"\cdot" if latex else "*"
        return f"{b}{mult}" + r"\ln x" + f"{f_f(a)}"

    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y.ravel())
        self.X_, self.y_ = clean_xy(np.log(X), y)
        self.set_linear_model(LinearRegression().fit(self.X_, self.y_))
        return self

    def predict(self, X):
        X = np.log(X)

        return self.linear_model.predict(X)


class OriginBoundHyperbolaEstimator(LatexLinearEstimator):

    def get_my_params(self):
        return super().get_base_params()

    def set_my_params(self, a, b):
        super().set_base_params(a, b)
        return self

    def get_equation_string(self, r=8, latex=True):
        a, b = np.round(self.get_my_params(), r)
        return r"y = \frac{x}{" + f"{b}x" + f"{f_f(a)}" + "}"

    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        # Check that X and y have correct shape

        if np.count_nonzero(X == 0) > 0 or np.count_nonzero(y == 0) > 0:
            raise ValueError("Neither inputs should contain 0's")

        self.X_, self.y_ = clean_xy(X, np.divide(X.squeeze(), y))
        self.set_linear_model(LinearRegression().fit(self.X_, self.y_))

        # Return the classifier
        return self

    def predict(self, X):
        base_result = self.linear_model.predict(X)
        return X.reshape(-1) / base_result


class PowerFunctionEstimator(LatexLinearEstimator):

    def __init__(self):
        super().__init__()

    def set_my_params(self, a, b):
        super().set_base_params(np.log(a), b)
        # 
        return self

    def get_my_params(self):
        pms = self.get_base_params()
        return np.exp(pms[0]), pms[1]

    def get_equation_string(self, r=8, latex=True):
        a, b = np.round(self.get_my_params(), r)
        return f"y = {a:g}x^{{{b:g}}}"

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y.ravel())
        self.X_, self.y_ = clean_xy(np.log(X), np.log(y))
        self.set_linear_model(LinearRegression().fit(self.X_, self.y_))
        return self

    def predict(self, X):
        X = np.log(X)

        return np.exp(self.linear_model.predict(X))


class VerticalShiftHyperbolaEstimator(LatexLinearEstimator):
    def get_my_params(self):
        return super().get_base_params()

    def set_my_params(self, a, b):
        super().set_base_params(a, b)
        return self

    def get_equation_string(self, r=8, latex=True):
        a, b = np.round(self.get_my_params(), r)
        return f"{a:g}{f_f(b)}/x"

    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        X, y = check_X_y(X, y.ravel())
        self.X_, self.y_ = clean_xy(1 / X, y)
        self.set_linear_model(LinearRegression().fit(self.X_, self.y_))

        # Return the classifier
        return self

    def predict(self, X):
        X = 1 / check_array(X)
        return self.linear_model.predict(X)
