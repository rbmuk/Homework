if __name__ == "__main__":
    from ISTA import train  # type: ignore
else:
    from .ISTA import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem

def plot(columns, lambdas: list, weights: list[np.ndarray], variable: str, loc):
    plt.subplot(2, 3, loc)
    plt.xscale('log')
    plt.plot(lambdas, [weight[columns.get_loc(variable)-1] for weight in weights], 'r-')
    plt.xlabel('lambda')
    plt.ylabel(f'Weight of {variable}')

@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")

    X = df_train.drop('ViolentCrimesPerPop', axis=1).values
    y = df_train['ViolentCrimesPerPop'].values
    print(df_train.columns.get_loc('agePct12t29'))

    _lambda = 2 * np.max(np.abs(X.T @ (y - np.mean(y))))
    weight: np.ndarray
    lambdas, weights = [_lambda], [np.zeros(X.shape[0])]
    while (_lambda > 0.01):
        _lambda /= 2
        lambdas.append(_lambda)
        weight = train(X, y, convergence_delta=1e-4, _lambda=_lambda)[0]
        weights.append(weight)
    
    plt.figure(figsize=(15, 9), dpi=100)

    # Lambdas and nonzeros
    # plt.xscale('log')
    # plt.plot(lambdas, np.array([np.count_nonzero(weight) for weight in weights]), 'r-')
    # plt.xlabel('lambda')
    # plt.ylabel('nnz(w)')

    # Regularization Path
    columns = df_train.columns
    plot(columns, lambdas, weights, 'agePct12t29', 1)
    plot(columns, lambdas, weights, 'pctWSocSec', 2)

    plt.show()

if __name__ == "__main__":
    main()