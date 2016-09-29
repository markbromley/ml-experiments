import math
import numpy
import matplotlib.pyplot as plt

class SimpleLinearRegression(object):
    '''
    A basic implementation of simple linear regression.
    Calculates coefficients b0 and b1, such that f(x) = b1x + b0.
    Also provides estimate for coefficient of determination (e.g. r^2).
    '''
    ROUNDING_LENGTH = 3

    def __init__(self, x, y):
        assert len(x) == len(y), "Input vector must be same length as target vector"
        self._x = sorted(x)
        self._y = sorted(y)
        self._x_mean = self._get_mean(self._x)
        self._y_mean = self._get_mean(self._y)

    def _get_b1(self):
        numerator = 0
        for idx, val in enumerate(self._x):
            numerator += (self._x[idx] - self._x_mean) * (self._y[idx] - self._y_mean)

        denominator = 0
        for idx, val in enumerate(self._x):
            denominator += math.pow((val - self._x_mean),2)
        return float(numerator) / float(denominator)

    def _get_b0(self):
        return (self._y_mean - (self._get_b1() * self._x_mean))

    def _get_mean(self, x):
        return (sum(float(i) for i in x)) / float(len(x))

    def _get_coeff_determination(self):
        return float(self._get_ssr()) / float(self._get_ssto())

    def _get_ssr(self):
        ssr = 0
        for x in self._x:
            ssr += math.pow(self.predict(x) - self._y_mean, 2)
        return ssr

    def _get_ssto(self):
        ssto = 0
        for y in self._y:
            ssto += math.pow(y - self._y_mean, 2)
        return ssto

    def predict(self, x):
        '''
        Returns a predicted value by the model, given an input value.
        '''
        return (self._get_b1() * float(x)) + self._get_b0()

    def visualise(self):
        '''
        Visualises the original data and the model using matplotlib.
        '''
        # Titles
        fig = plt.figure(0)
        title = "Simple Linear Regression"
        fig.canvas.set_window_title(title)
        plt.title(title)
        plt.xlabel("X values")
        plt.ylabel("Y values")

        # Original data
        plt.scatter(self._x, self._y)

        # The regression line
        pred_y = [self.predict(x) for x in self._x]
        plt.plot(self._x, pred_y, label="Model prediction")

        # Coeff of determination
        val = ("R^2 = {}").format(str(round(self._get_coeff_determination(),self.ROUNDING_LENGTH)))
        plt.text(self._x[-1],self.predict(self._x[-1]),val)
        plt.show()

    @property
    def b1(self):
        ''' B1 coefficient.'''
        return self._get_b1()

    @property
    def b0(self):
        '''B0 coefficient.'''
        return self._get_b0()

    @property
    def r2(self):
        ''' Coefficient of Determination.''' 
        return self._get_coeff_determination()

if __name__ == "__main__":
    # Fake data
    x = [1,0,3,9,1]
    y = [1,2,3,4,4]

    # Model
    slr = SimpleLinearRegression(x, y)
    print("B0 Coefficient: {}".format(str(round(slr.b0,slr.ROUNDING_LENGTH))))
    print("B1 Coefficient: {}".format(str(round(slr.b1,slr.ROUNDING_LENGTH))))
    print("Coefficient of Determination (R^2): {}".format(str(round(slr.r2,slr.ROUNDING_LENGTH))))
    slr.visualise()

