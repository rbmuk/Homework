from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import itertools as it

from utils import load_dataset, problem


def f_true(x: np.ndarray) -> np.ndarray:
    """True function, which was used to generate data.
    Should be used for plotting.

    Args:
        x (np.ndarray): A (n,) array. Input.

    Returns:
        np.ndarray: A (n,) array.
    """
    return 6 * np.sin(np.pi * x) * np.cos(4 * np.pi * x ** 2)


@problem.tag("hw3-A")
def poly_kernel(x_i: np.ndarray, x_j: np.ndarray, d: int) -> np.ndarray:
    """Polynomial kernel.

    Given two indices a and b it should calculate:
    K[a, b] = (x_i[a] * x_j[b] + 1)^d

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        d (int): Degree of polynomial.

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    return (np.outer(x_i, x_j) + np.ones((len(x_i), len(x_j)))) ** d


@problem.tag("hw3-A")
def rbf_kernel(x_i: np.ndarray, x_j: np.ndarray, gamma: float) -> np.ndarray:
    """Radial Basis Function (RBF) kernel.

    Given two indices a and b it should calculate:
    K[a, b] = exp(-gamma*(x_i[a] - x_j[b])^2)

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        gamma (float): Gamma parameter for RBF kernel. (Inverse of standard deviation)

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    return np.exp(-gamma * ((x_i.reshape(-1, 1) - x_j.reshape(1, -1))**2))


@problem.tag("hw3-A")
def train(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
) -> np.ndarray:
    """Trains and returns an alpha vector, that can be used to make predictions.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.

    Returns:
        np.ndarray: Array of shape (n,) containing alpha hat as described in the pdf.
    """
    K = kernel_function(x, x, kernel_param)
    return np.linalg.solve(K + _lambda * np.identity(len(x)), y)

def predict(X_train, X_test, kernel_function, kernel_param, alpha):
    return kernel_function(X_test, X_train, kernel_param) @ alpha

@problem.tag("hw3-A", start_line=1)
def cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    num_folds: int,
) -> float:
    """Performs cross validation.

    In a for loop over folds:
        1. Set current fold to be validation, and set all other folds as training set.
        2, Train a function on training set, and then get mean squared error on current fold (validation set).
    Return validation loss averaged over all folds.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        float: Average loss of trained function on validation sets across all folds.
    """
    fold_size = len(x) // num_folds
    loss = []
    for k in range(num_folds):
        X_validation = x[k*fold_size:(k+1)*fold_size]
        X_train = np.concatenate((x[:k*fold_size], x[(k+1)*fold_size:]))
        y_validation = y[k*fold_size:(k+1)*fold_size]
        y_train = np.concatenate((y[:k*fold_size], y[(k+1)*fold_size:]))

        alpha = train(X_train, y_train, kernel_function, kernel_param, _lambda)
        y_pred = predict(X_train, X_validation, kernel_function, kernel_param, alpha)
        loss.append(np.mean((y_validation-y_pred)**2))
    return np.mean(loss)

@problem.tag("hw3-A")
def rbf_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, float]:
    """
    Parameter search for RBF kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambda, loop over them and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda from some distribution and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be len(x) for LOO.

    Returns:
        Tuple[float, float]: Tuple containing best performing lambda and gamma pair.

    Note:
        - You do not really need to search over gamma. 1 / (median(dist(x_i, x_j)^2) for all unique pairs x_i, x_j in x
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
    """
    gamma = 1/np.median(((x.reshape(-1, 1) - x.reshape(1, -1)) ** 2)[np.triu_indices(len(x), k=1)])
    best_lambda: float
    best_loss = np.inf
    for i in np.linspace(-5, -1):
        _lambda = 10**i
        loss = cross_validation(x, y, rbf_kernel, gamma, _lambda, num_folds)
        if (loss < best_loss):
            best_lambda = _lambda
            best_loss = loss
    return [best_lambda, gamma]

@problem.tag("hw3-A")
def poly_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, int]:
    """
    Parameter search for Poly kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambdas and ds.
            Have nested loop over all possibilities and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda, d from some distributions and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, int]: Tuple containing best performing lambda and d pair.

    Note:
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
            and d from distribution [5, 6, ..., 24, 25]
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
            and possible ds to [5, 6, ..., 24, 25]
    """
    best_d: int
    best_lambda: float
    best_loss = np.inf
    for d, i in it.product(range(5, 26), np.linspace(-5, -1)):
        _lambda = 10**i
        loss = cross_validation(x, y, poly_kernel, d, _lambda, num_folds)
        if (loss < best_loss):
            best_d = d
            best_lambda = _lambda
            best_loss = loss
    return [best_lambda, best_d]

@problem.tag("hw3-A", start_line=1)
def main():
    """
    Main function of the problem

    It should:
        A. Using x_30, y_30, rbf_param_search and poly_param_search report optimal values for lambda (for rbf), gamma, lambda (for poly) and d.
            Note that x_30, y_30 has been loaded in for you. You do not need to use (x_300, y_300) or (x_1000, y_1000).
        B. For both rbf and poly kernels, train a function using x_30, y_30 and plot predictions on a fine grid

    Note:
        - In part b fine grid can be defined as np.linspace(0, 1, num=100)
        - When plotting you might find that your predictions go into hundreds, causing majority of the plot to look like a flat line.
            To avoid this call plt.ylim(-6, 6).
    """
    (x_30, y_30), (x_300, y_300), (x_1000, y_1000) = load_dataset("kernel_bootstrap")
    rbf_lambda, rbf_gamma = rbf_param_search(x_30, y_30, len(x_30))
    poly_lambda, poly_d = poly_param_search(x_30, y_30, len(x_30))

    alpha = train(x_30, y_30, rbf_kernel, rbf_gamma, rbf_lambda)
    plt.plot(x_30, y_30, linestyle='None', marker='.', color='r', markersize=10.0, label='original data')
    unit_interval = np.linspace(0, 1, num=100)
    plt.plot(unit_interval, f_true(unit_interval), label='true function')
    plt.plot(unit_interval, predict(x_30, unit_interval, rbf_kernel, rbf_gamma, alpha), label='prediction')
    plt.title(f'RBF gamma: {rbf_gamma}, lambda: {rbf_lambda}')
    plt.legend(loc='lower right')
    plt.show()

    alpha = train(x_30, y_30, poly_kernel, poly_d, poly_lambda)
    plt.plot(x_30, y_30, linestyle='None', marker='.', color='r', markersize=10.0, label='original data')
    plt.plot(unit_interval, f_true(unit_interval), label='true function')
    plt.plot(unit_interval, predict(x_30, unit_interval, poly_kernel, poly_d, alpha), label='prediction')
    plt.title(f'Poly d: {poly_d}, lambda: {poly_lambda}')
    plt.ylim(-6, 6)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
