import numpy as np

"""
- This model is Generic linear regression works for n-dimenssion features
- Make sure to insert a column of 1's in dataset before splitting X
"""

class MultipleLinearRegressor:
    def __init__(self) -> None:
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X_train, y_train):
        beta = np.linalg.inv(np.dot(X_train.T, X_train)).dot(X_train.T).dot(y_train)
        self.coef_ = beta[1:]
        self.intercept_ = beta[0]

    def predict(self, X_test):
        y_hat = np.dot(X_test, self.coef_) + self.intercept_
        return y_hat
    