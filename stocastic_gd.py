"""
- This is the implementation of Stocastic gradient descent algorithm.
"""

import numpy as np

class SGDRegressor:
    def __init__(self, epoch = 100, lr = 0.01) -> None:
        self.coef_ = None
        self.intercept_ = None
        self.epoch = epoch
        self.lr = lr

    def fit(self, X_train, y_train):
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[0])

        for i in range(self.epoch):
            for j in range(X_train.shape[0]):
                # Here updating coef_, intercept_ n times per epoch
                idx = np.random.randint(0, X_train.shape[0])

                y_hat = np.dot(X_train[idx], self.coef_) + self.intercept_
                intercept_derivative = -2 * (y_train[idx] - y_hat)
                self.intercept_ = self.intercept_ - (self.lr * intercept_derivative)

                coef_derivative = -2 * np.dot((y_train[i] - y_hat), X_train[idx])
                self.coef_ = self.coef_ - (self.lr * coef_derivative)

    def preditct(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_