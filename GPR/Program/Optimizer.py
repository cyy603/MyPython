import numpy as np
import RW
import Plotter
import math
from scipy.spatial.distance import cdist
import scipy.optimize as opt
import pprint
"""

Using marginal likelihood to optimize the performance of hyperparameter

"""

def optimize_f(x, *args):
    """calucalte the value of the object (negative log marginal likelihood)"""
    X, y, kernel, s_star = args
    var_f, l, var_noise = x

    L = np.linalg.cholesky(kernel(X, X, var_f, l) + var_noise * np.eye(X.shpae[0]))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

    return - (- 0.5 * y.T @ alpha - 0.5 * np.trace(np.log(L)) - X.shape[0] * np.log(2 * math.pi) / 2)

def dLdT(alpha, iKxx, dKdT):
    """
    calculate the partial derivative of Marginal likeliihood
    :parameter
    dKdT: the derivatives of covariance function
    """

    return 0.5 * np.trace(np.dot((alpha @ alpha.T - iKxx), dKdT))

def dKdsf(x1, x2, var_f, l):
    """
    :param x1: input variable
    :param x2: input variable
    :param var_f: hyperparameter, signal variance
    :param l: hyperparameter, length scale
    :return: Gradient of SE kernel function wrt the signal variance s_f
    """
    return 2 * np.sqrt(var_f) * np.exp(-cdist(x1 , x2) ** 2) / (2 * l ** 2)

def dKdl(x1, x2, var_f, l):
    """
    :param x1: input variable
    :param x2: input variable
    :param var_f: hyperparameter, signal variance
    :param l: hyperparameter, length scale
    :return: Gradient of SE kernel function wrt the length scale
    """
    return var_f ** 2 * np.exp(- cdist(x1, x2) ** 2) / (2 * l ** 2) * (cdist(x1, x2) / l ** 3)

def dKdsn(x1, x2, var_noise):
    """
    :param x1: input variable
    :param x2: input variable
    :param var_f: hyperparameter, signal variance
    :param l: hyperparameter, length scale
    :return: Gradient of SE kernel function wrt the length scale noise variance s_n
    """
    return 2 * np.sqrt(np.diag(var_noise))

def dfx(x, *args):
    """
    :param x: input variable
    :param args: other variable
    :return: Gradient of the objective with respect to length scale, signal variance, noise variance
    """
    X, y, kernel, x_star = args
    var_f, l, var_noise = x

    Kxx = kernel(X, X, var_f, l)
    L = np.linalg.cholesky(Kxx + var_noise * np.eye(X.shape[0]))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

    iKxx = np.linalg.inv(Kxx + var_noise * np.eye(X.shpae[0]))

    J = np.empty([3, ])
    J[0] = dLdT(alpha, iKxx, dKdsf(X, X, var_f, l))
    J[1] = dLdT(alpha, iKxx, dKdl(X, X, var_f, l))
    J[2] = dLdT(alpha, iKxx, dKdsn(X, X, var_noise))

    return J

def optimize(x_train, y_train, kernel, x_star):
    bounds = [[0, np.inf], [0, np.inf], [0.0001, np.inf]]
    args = (x_train, y_train, kernel, x_star)
    x0 = [1, 1, 0.1]
    x0 = np.array(x0)
    res = opt.fmin_l_bfgs_b(optimize_f, x0, args = args, approx_grad = True, bounds = bounds)
    var_f = res[0][0]
    l = res[0][1]
    var_noise = res[0][2]

    print("Signal variance: %.6f\nLengthscale: %.6f\nNoise variance: %.6f" % (var_f, l, var_noise))

    return var_f, l, var_noise