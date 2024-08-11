"""
- This is implementation of batch gradient descent algorithm.
"""

import numpy as np

class BGDRegressor:
    def __init__(self, epoch = 100, lr = 0.01) -> None:
        self.coef_ = None
        self.intercept_ = None
        self.epoch = epoch
        self.lr = lr

    def fit(self, X_train, y_train):
        self.coef_ = np.ones(X_train.shape[1])
        self.intercept_ = 0

        for i in range(self.epoch):
            # Here updating coef_, intercept_ one time per epoch
            y_hat = np.dot(X_train, self.coef_) + self.intercept_
            intercept_derivative = -2 * np.mean(y_train, y_hat)
            self.intercept_ = self.intercept_ - (self.lr * intercept_derivative)

            coef_derivative = -2 * np.dot((y_train - y_hat), X_train) / X_train.shape[0]
            self.coef_ = self.coef_ - (self.lr * coef_derivative)

    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_