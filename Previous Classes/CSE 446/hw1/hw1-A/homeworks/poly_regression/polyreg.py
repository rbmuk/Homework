"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

from typing import Tuple

import numpy as np

from utils import problem


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=5)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        """Constructor
        """
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        # Fill in with matrix with the correct shape
        self.weight: np.ndarray = np.ndarray((degree+1, 1))  # type: ignore
        # You can add additional fields
        self.mean: np.ndarray = np.ndarray((degree, 1))
        self.std: np.ndarray = np.ndarray((degree, 1))

    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        n = len(X)
        """
        Expands the given X into an (n, degree) array of polynomial features of degree degree.

        Args:
            X (np.ndarray): Array of shape (n, 1).
            degree (int): Positive integer defining maximum power to include.

        Returns:
            np.ndarray: A (n, degree) numpy array, with each row comprising of
                X, X * X, X ** 3, ... up to the degree^th power of X.
                Note that the returned matrix will not include the zero-th power.

        """
        V = np.array([np.power(X.squeeze(), i) for i in range(1, degree+1)])
        # if X is 1x1, then squeeze makes it a 1d array, so the above gives a 
        # length d array instead of a 1xd ndarray, so we need to reshape it
        # One might think that this is where ndmin would help, but
        # it prepends 1s instead of postpends !!!!
        if (X.shape[0] == 1):
              return V.reshape(-1, 1).T
        return np.array([np.power(X.squeeze(), i) for i in range(1, degree+1)]).T


    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model, and saves learned weight in self.weight

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.
            y (np.ndarray): Array of shape (n, 1) with targets.

        Note:
            You will need to apply polynomial expansion and data standardization first.
        """
        X_ = PolynomialRegression.polyfeatures(X, self.degree)
        self.mean = np.mean(X_, axis=0)
        self.std = np.std(X_, axis=0)
        X_ = (X_ - self.mean)/self.std
        n, d = X_.shape
        X_ = np.c_[np.ones((n, 1)), X_]

        reg_matrix = self.reg_lambda * np.eye(d+1)
        reg_matrix[0,0] = 0
        
        self.weight = np.linalg.solve(X_.T @ X_ + reg_matrix, X_.T @ y)

    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to predict values for each instance in X.

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.

        Returns:
            np.ndarray: Array of shape (n, 1) with predictions.
        """
        X_ = PolynomialRegression.polyfeatures(X, self.degree)
        X_ = (X_-self.mean)/self.std
        n = X_.shape[0]
        X_ = np.c_[np.ones((n, 1)), X_]

        return X_ @ self.weight


@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)

    Returns:
        float: mean squared error between a and b.
    """
    return np.mean((a-b)**2)

@problem.tag("hw1-A", start_line=5)
def learningCurve(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    Xtest: np.ndarray,
    Ytest: np.ndarray,
    reg_lambda: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute learning curves.

    Args:
        Xtrain (np.ndarray): Training observations, shape: (n, 1)
        Ytrain (np.ndarray): Training targets, shape: (n, 1)
        Xtest (np.ndarray): Testing observations, shape: (n, 1)
        Ytest (np.ndarray): Testing targets, shape: (n, 1)
        reg_lambda (float): Regularization factor
        degree (int): Polynomial degree

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            1. errorTrain -- errorTrain[i] is the training mean squared error using model trained by Xtrain[0:(i+1)]
            2. errorTest -- errorTest[i] is the testing mean squared error using model trained by Xtrain[0:(i+1)]

    Note:
        - For errorTrain[i] only calculate error on Xtrain[0:(i+1)], since this is the data used for training.
            THIS DOES NOT APPLY TO errorTest.
        - errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """
    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)
    # Fill in errorTrain and errorTest arrays
    for i in range(1, n):
        X = Xtrain[0:(i+1)]
        y = Ytrain[0:(i+1)]
        model = PolynomialRegression(degree=degree, reg_lambda=reg_lambda)
        model.fit(X, y)
        errorTrain[i] = mean_squared_error(model.predict(X), y)
        errorTest[i] = mean_squared_error(model.predict(Xtest), Ytest)
    return errorTrain, errorTest