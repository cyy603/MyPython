import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import cdist
import RW

def kernel(x1, x2, sigma_f, l):
    """
    Using squared exponential function as kernel

    :parameter
        x1: value of one point
        x2: value of one point
        sigma_f: a constant, use as a control variable to limit the range of variance of GPR
        l: a positive constant, use to control the length scale of |x1 - x2|
    :return: the value of kernel function
    """

    return sigma_f * np.exp(-cdist(x1, x2) / (2 * l ** 2))

def sample(mu, var, jitter, N):
    """Generate N samples from a multivariate Gaussian N(mu, var)"""

    L = np.linalg.cholesky(var + jitter * np.eye(var.shape[0]))# cholesky decomposition (square root) of covariance matrix
    f_post = mu + L @ np.random.normal(size = (n, N))

    return f_post

f = lambda x : np.sin(x) + 0.1 * np.cos(2 * x)




if __name__ == '__main__':
    n = 100 #number of prediction points
    x_min = -5
    x_max = 5
    x_star = np.linspace(x_min * 1.4, x_max * 1.4, n).reshape(-1, 1) #prediction points
    jitter = 1e-10
    l = 1 #hyper-perameter length-scale
    N = 12 # number of training points
    sigma_f = 1

    Kss = kernel(x_star, x_star, sigma_f, l)#prior convariance

    Ns = 10#number of samples
    f_prior = sample(0, Kss, jitter, Ns)

    #compute standard divation of sample

    std = np.sqrt(np.diag(Kss))

    x = np.random.uniform(x_min, x_max, size = (N, 1))# training inputs

    y_train = f(x)

    mu, var = RW.gp_regression(x, y_train, kernel, x_star, sigma_f, l)


"""
    #plot the training data
    plt.figure()

    plt.plot(x_star, f(x_star), color = "r", label = "underlying function")
    plt.scatter(x, y_train, color = "b", label = "Training data points")
    plt.legend()
    plt.show()


    fig = plt.figure(figsize = (12, 12))
    plt.subplots_adjust(wspace = 0.3, hspace = 0.5)

    plt.subplot(2, 2, 1)
    plt.plot(x_star, f_prior)
    plt.title("%i samples from the GP prior"% Ns)
    plt.fill_between(x_star.flatten(), 0-2 * std, 0+2 * std, label = '$\pm$2 standard deviations of posterior', color="#dddddd")

    #visualize the covariance function
    plt.subplot(2, 2, 2)
    plt.title("Prior covariance $K(X_*, X_*)$")
    plt.contourf(Kss)
    plt.show()
"""
