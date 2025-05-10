import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_X_y, check_array
from LatexLinearEstimator import LatexLinearEstimator
from formatting import f_f, f_p


class LinearEstimator(LatexLinearEstimator):
    """Linear model: y = a + b * x"""

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X_, self.y_ = X, y
        super().set_linear_model(LinearRegression().fit(X, y))
        return self

    def predict(self, X):
        X = check_array(X)
        return self.linear_model.predict(X)

    def get_equation_string(self, r=8, latex=True):
        pms = np.round(self.get_my_params(), r)
        return f"y = {pms[0]:g}{f_f(pms[1])}x"

    def get_my_params(self):
        # Return intercept (a) and slope (b) from the base class
        return self.get_base_params()

    def set_my_params(self, a, b):
        # Set intercept (a) and slope (b) using the base class method
        super().set_base_params(a, b)
        return self


class ExponentialEstimator(LatexLinearEstimator):
    """Exponential model: y = a * b^x. Assumes y > 0."""

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if np.any(y <= 0):
            raise ValueError("y must be positive for exponential model")
        self.X_, self.y_ = X, np.log(y)
        super().set_linear_model(LinearRegression().fit(self.X_, self.y_))
        return self

    def predict(self, X):
        X = check_array(X)
        return np.exp(self.linear_model.predict(X))

    def get_my_params(self):
        # Convert linear params to exponential form: y = a * b^x
        intercept, slope = super().get_base_params()
        return np.exp(intercept), np.exp(slope)

    def set_my_params(self, a, b):
        # Set params by transforming to linear form: ln(y) = ln(a) + x * ln(b)
        if a <= 0 or b <= 0:
            raise ValueError("Parameters a and b must be positive")
        super().set_base_params(np.log(a), np.log(b))
        return self

    def get_equation_string(self, r=8, latex=True):
        a, b = np.round(self.get_my_params(), r)
        mult = r"\cdot" if latex else "*"
        return f"{a:g} {mult} {f_p(b)}^x"


class HorizontalShiftHyperbolaEstimator(LatexLinearEstimator):
    """Hyperbola with horizontal shift: y = 1 / (b * x + a)"""

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if np.any(y == 0):
            raise ValueError("y cannot contain zeros")
        self.X_, self.y_ = X, 1 / y
        super().set_linear_model(LinearRegression().fit(self.X_, self.y_))
        return self

    def predict(self, X):
        X = check_array(X)
        return 1 / self.linear_model.predict(X)

    def get_my_params(self):
        # Return parameters a and b directly from base class
        return self.get_base_params()

    def set_my_params(self, a, b):
        # Set parameters a and b directly
        super().set_base_params(a, b)
        return self

    def get_equation_string(self, r=8, latex=True):
        a, b = np.round(self.get_my_params(), r)
        return r"y = \frac{1}{" + f"{b:g}x{f_f(a)}" + "}" if latex else f"y = 1/({b:g}x{f_f(a)})"


class LogarithmicEstimator(LatexLinearEstimator):
    """Logarithmic model: y = b * ln(x) + a"""

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if np.any(X <= 0):
            raise ValueError("X must be positive")
        self.X_, self.y_ = np.log(X), y
        super().set_linear_model(LinearRegression().fit(self.X_, self.y_))
        return self

    def predict(self, X):
        X = check_array(X)
        if np.any(X <= 0):
            raise ValueError("X must be positive")
        return self.linear_model.predict(np.log(X))

    def get_my_params(self):
        # Return intercept (a) and slope (b) directly
        return self.get_base_params()

    def set_my_params(self, a, b):
        # Set intercept (a) and slope (b) directly
        super().set_base_params(a, b)
        return self

    def get_equation_string(self, r=8, latex=True):
        a, b = np.round(self.get_my_params(), r)
        mult = r"\cdot" if latex else "*"
        return f"{b:g}{mult}" + (r"\ln x" if latex else "ln(x)") + f"{f_f(a)}"


class OriginBoundHyperbolaEstimator(LatexLinearEstimator):
    """Hyperbola bound to origin: y = x / (b * x + a)"""

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if np.any(X == 0) or np.any(y == 0):
            raise ValueError("X and y cannot contain zeros")
        self.X_, self.y_ = X, X.squeeze() / y
        super().set_linear_model(LinearRegression().fit(self.X_, self.y_))
        return self

    def predict(self, X):
        X = check_array(X)
        return X.ravel() / self.linear_model.predict(X)

    def get_my_params(self):
        # Return parameters a and b directly
        return self.get_base_params()

    def set_my_params(self, a, b):
        # Set parameters a and b directly
        super().set_base_params(a, b)
        return self

    def get_equation_string(self, r=8, latex=True):
        a, b = np.round(self.get_my_params(), r)
        return r"y = \frac{x}{" + f"{b:g}x{f_f(a)}" + "}" if latex else f"y = x/({b:g}x{f_f(a)})"


class PowerFunctionEstimator(LatexLinearEstimator):
    """Power model: y = a * x^b"""

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if np.any(X <= 0) or np.any(y <= 0):
            raise ValueError("X and y must be positive")
        self.X_, self.y_ = np.log(X), np.log(y)
        super().set_linear_model(LinearRegression().fit(self.X_, self.y_))
        return self

    def predict(self, X):
        X = check_array(X)
        if np.any(X <= 0):
            raise ValueError("X must be positive")
        return np.exp(self.linear_model.predict(np.log(X)))

    def get_my_params(self):
        # Convert linear params to power form: y = a * x^b
        intercept, slope = super().get_base_params()
        return np.exp(intercept), slope

    def set_my_params(self, a, b):
        # Set params by transforming to linear form: ln(y) = ln(a) + b * ln(x)
        if a <= 0:
            raise ValueError("Parameter a must be positive")
        super().set_base_params(np.log(a), b)
        return self

    def get_equation_string(self, r=8, latex=True):
        a, b = np.round(self.get_my_params(), r)
        return f"y = {a:g}x^{{{b:g}}}" if latex else f"y = {a:g}x^{b:g}"


class VerticalShiftHyperbolaEstimator(LatexLinearEstimator):
    """Hyperbola with vertical shift: y = b / x + a"""

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if np.any(X == 0):
            raise ValueError("X cannot contain zeros")
        self.X_, self.y_ = 1 / X, y
        super().set_linear_model(LinearRegression().fit(self.X_, self.y_))
        return self

    def predict(self, X):
        X = check_array(X)
        if np.any(X == 0):
            raise ValueError("X cannot contain zeros")
        return self.linear_model.predict(1 / X)

    def get_my_params(self):
        # Return intercept (a) and slope (b) directly
        return self.get_base_params()

    def set_my_params(self, a, b):
        # Set intercept (a) and slope (b) directly
        super().set_base_params(a, b)
        return self

    def get_equation_string(self, r=8, latex=True):
        a, b = np.round(self.get_my_params(), r)
        return r"y = \frac{" + f"{b:g}" + r"}{x}" + f"{f_f(a)}" if latex else f"y = {b:g}/x{f_f(a)}"