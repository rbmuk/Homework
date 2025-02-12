from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float, eta: float
) -> Tuple[np.ndarray, float]:
    """Single step in ISTA algorithm.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        bias (float): Bias returned from the step before.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.
        eta (float): Step-size. Determines how far the ISTA iteration moves for each step.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
    
    """
    b_prime = bias - 2*eta*np.sum(X @ weight - y + bias)
    new_weight = weight - 2 * eta * (X.T @ (X @ weight + bias - y))
    new_weight = np.where(np.abs(new_weight) >= 2 * eta * _lambda, new_weight, np.zeros(weight.shape[0]))
    new_weight -= np.clip(new_weight, -2 * eta * _lambda, 2 * eta * _lambda)
    return [new_weight, b_prime]


@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized SSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    return np.sum((X @ weight - y + bias)**2) + _lambda * np.linalg.norm(weight, ord=1)


@problem.tag("hw2-A", start_line=5)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    eta: float = 0.00001,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
    start_bias: float = None
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight and bias.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        eta (float): Step size.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.
        start_bias (float, optional): Bias for hot-starting model.
            If None, defaults to zero. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float representing the bias.

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You will also have to keep an old copy of bias for convergence criterion function.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
        start_bias = 0
    old_w: Optional[np.ndarray] = start_weight
    old_b: float = start_bias
    
    start_weight, start_bias = step(X, y, start_weight, start_bias, _lambda, eta)
    while (not convergence_criterion(start_weight, old_w, start_bias, old_b, convergence_delta)):
        old_w, old_b = np.copy(start_weight), start_bias
        start_weight, start_bias = step(X, y, start_weight, start_bias, _lambda, eta)
        assert(old_w is not start_weight)
    return [start_weight, start_bias]

@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, bias: float, old_b: float, convergence_delta: float
) -> bool:
    """Function determining whether weight and bias has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compare it to convergence delta.
    It should also calculate the maximum absolute change between the bias and old_b, and compare it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of gradient descent.
        old_w (np.ndarray): Weight from previous iteration of gradient descent.
        bias (float): Bias from current iteration of gradient descent.
        old_b (float): Bias from previous iteration of gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight and bias has not converged yet. True otherwise.
    """
    return np.linalg.norm(weight-old_w, ord=np.inf) < convergence_delta and np.abs(bias-old_b) < convergence_delta


@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    X = np.loadtxt('data/uniform/X.csv', delimiter=',')
    y = np.loadtxt('data/uniform/y.csv', delimiter=',')
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    curr_lambda = 2 * np.max(np.abs(X.T @ (y-np.mean(y))))
    lambdas, nonzeros = [curr_lambda], [0]
    FDR, TPR = [0], [0]
    weight = np.zeros(X.shape[1])
    while (np.count_nonzero(weight) < 950):
            curr_lambda /= 2
            lambdas.append(curr_lambda)
            weight = train(X, y, convergence_delta=1e-2, _lambda=curr_lambda)[0]
            nonzeros.append(np.count_nonzero(weight))
            FDR.append(np.sum([1 if weight[i] != 0 else 0 for i in range(100, weight.shape[0])])/nonzeros[len(nonzeros)-1])
            TPR.append(np.sum([1 if weight[i] != 0 else 0 for i in range(0, 100)])/100)
    plt.figure(figsize=(15,9), dpi=100)
    plt.subplot(1, 2, 1)
    plt.xscale('log')
    plt.plot(lambdas, nonzeros, 'r-')
    plt.xlabel('lambda')
    plt.ylabel('nnz(w)')
    plt.title('Regularization Path')

    plt.subplot(1, 2, 2)
    plt.plot(FDR, TPR, 'r-')
    plt.xlabel('FDR')
    plt.ylabel('TPR')
    plt.title('TPR/FDR')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
