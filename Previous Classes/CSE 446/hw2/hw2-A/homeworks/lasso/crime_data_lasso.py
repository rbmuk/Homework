if __name__ == "__main__":
    from ISTA import train  # type: ignore
else:
    from .ISTA import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem

def plot(x: list, y: list, xlabel: str, ylabel: str, numcols: int, numrows: int, loc: int) -> None:
    plt.subplot(numcols, numrows, loc)
    plt.xscale('log')
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def plot_reg(columns, lambdas: list, weight_bias, variable: str, numcols: int, numrows: int, loc: int) -> None:
    # plt.subplot(numcols, numrows, loc)
    plt.xscale('log')
    plt.plot(lambdas, [weight[columns.get_loc(variable)-1] for weight, bias in weight_bias], label=variable)
    plt.xlabel('lambda')
    plt.ylabel(f'Weight of {variable}')

@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")

    X = df_train.drop('ViolentCrimesPerPop', axis=1).values
    columns = df_train.columns
    y = df_train['ViolentCrimesPerPop'].values

    _lambda = 2 * np.max(np.abs(X.T @ (y - np.mean(y))))
    lambdas, weight_bias = [_lambda], [(np.zeros(X.shape[1]), 0)]
    while (_lambda >= 0.01):
        _lambda /= 2
        lambdas.append(_lambda)
        weight_bias.append(train(X, y, convergence_delta=1e-4, _lambda=_lambda))
    
    plt.figure(figsize=(15, 9), dpi=100)

    # Lambdas and nonzeros
    # plt.xscale('log')
    # plt.plot(lambdas, np.array([np.count_nonzero(weight) for weight in weights]), 'r-')
    # plt.xlabel('lambda')
    # plt.ylabel('nnz(w)')

    # Regularization Path
    plot_reg(columns, lambdas, weight_bias, 'agePct12t29', 2, 3, 1)
    plot_reg(columns, lambdas, weight_bias, 'pctWSocSec', 2, 3, 2)
    plot_reg(columns, lambdas, weight_bias, 'pctUrban', 2, 3, 3)
    plot_reg(columns, lambdas, weight_bias, 'agePct65up', 2, 3, 4)
    plot_reg(columns, lambdas, weight_bias, 'householdsize', 2, 3, 6)

    # weight_fixed_lambda = weight_bias[-1][0]
    # ten_largest_vars = columns[(np.argsort(weight_fixed_lambda)+1)[-8:]]
    # for i in range(8):
    #     plot_reg(columns, lambdas, weight_bias, ten_largest_vars[i], 2, 4, i+1)

    # MSE (train) and MSE (test)
    # mse_train = [np.sum((X @ weight + bias - y) ** 2) for weight, bias in weight_bias]
    # X_test = df_test.drop('ViolentCrimesPerPop', axis=1).values
    # y_test = df_test['ViolentCrimesPerPop'].values
    # mse_test = [np.sum((X_test @ weight + bias - y_test) ** 2) for weight, bias in weight_bias]
    # plot(lambdas, mse_train, 'lambda', 'MSE (train)', 1, 2, 1)
    # plot(lambdas, mse_test, 'lambda', 'MSE (test)', 1, 2, 2)
    plt.legend()
    plt.show()

    # Finding the largest and smallest important features for lambda = 30
    # weight = train(X, y, convergence_delta=1e-4, _lambda=30)[0]
    # max_arg, argmin = np.argmax(weight)+1, np.argmin(weight)+1
    # print(columns[max_arg], columns[argmin])

if __name__ == "__main__":
    main()