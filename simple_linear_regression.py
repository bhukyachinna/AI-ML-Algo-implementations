"""
- Basic version OLS method
- This Implementation only works for single input feature.
"""

import numpy as np

class SimpleLinearRegression:
    def __init__(self) -> None:
        self.m = None
        self.b = None

    def fit(self, X_train, y_train):
        numerator = 0
        denomenator = 0
        x_mean = np.mean(X_train)
        y_mean = np.mean(y_train)

        for i in range(X_train.shape[0]):
            numerator += (X_train[i] - x_mean) * (y_train[i] - y_mean)
            denomenator += (X_train[i] - x_mean) ** 2
        
        self.m = numerator / denomenator
        self.b = y_mean - (self.m * x_mean)

    def predict(self, X_test):
        y_hat = 0
        y_hat = self.m * X_test + self.b

        return y_hat
