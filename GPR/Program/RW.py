import numpy as np

"""Rasmussen and Williams' algorithm"""

def gp_regression(X, y, K, x_star, sigma_f, l):
    """
    :parameter
        X: inputs training data
        y: target value
        K: covariance function used when processing GP Regression
        sigma_f: a constant, use as a control variable to limit the range of variance of GPR
        l: a positive constant, use to control the length scale of |x1 - x2|

    :return
        mu: mean value of target function
        var: variance value of target function
    """

    """compute mean"""
    L = np.linalg.cholesky(K(X, X, sigma_f, l))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L , y))
    mu = K(X, x_star, sigma_f, l).T @ alpha

    """compute variance"""
    v = np.linalg.solve(L, K(X, x_star, sigma_f, l))
    var = K(x_star, x_star, sigma_f, l) - v.T @ v

    return mu, var