import numpy as np
import matplotlib.pyplot as plt
import RW
from scipy.spatial.distance import cdist

f = lambda x : np.sin(x) + 0.1 * np.cos(2 * x)
y = lambda x, s : f(x) + np.random.normal(0, np.sqrt(s), x.shape) # generate noisy observation

def plot_training_data(x_train, y_train, x_star):
    plt.figure(figsize = (12, 12))
    plt.plot(x_star, f(x_star), color = 'r', label = 'underlying function')
    plt.scatter(x_train, y_train, color = 'b', label = 'training data')
    plt.legend()
    plt.show()

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

def plot_posterior(x_train, y_train, x_star, mu, var):
    samples = 30

    std = np.sqrt(np.diag(var))
    f_post = sample(mu, var, jitter, samples)

    plt.figure(figsize = (12, 12))
    plt.subplots_adjust(wspace = 0.3, hspace = 0.5)

    #plot underlying function, training data, posterior means, and +-/2 standard deviation
    plt.subplot(2, 2, 1)
    plt.title('GP posterior')
    plt.fill_between(x_star.flatten(), mu.flatten() - 2 * std, mu.flatten() + 2 * std, label = 'standard deviation of posterior', color = '#dddddd')
    plt.plot(x_star, f(x_star), 'b-', label = "underlying function")
    plt.plot(x_train, y_train, 'kx', label = 'training data')
    plt.plot(x_star, mu, 'r-', label = 'mean of posterior')
    plt.legend()

    #plot samples from posterior
    plt.subplot(2, 2, 2)
    plt.plot(x_star, f_post)
    plt.title('%i samples from GP posterior' % samples)
    plt.plot(x_train, y_train, 'kx', ms = 8, label = 'training data')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    n = 100  # number of prediction points
    x_min = -5
    x_max = 5
    x_star = np.linspace(x_min * 1.4, x_max * 1.4, n).reshape(-1, 1)  # prediction points
    jitter = 1e-10
    l = 1  # hyperparameter length-scale
    N = 12  # number of training points
    sigma_f = 1
    x = np.random.uniform(x_min, x_max, size=(N, 1))  # training inputs
    var_noise_ran = 0.01

    y_train = y(x, var_noise_ran)

    #plot_training_data(x, y_train, x_star)

    var_f = 1
    var_noise = 0.1

    mu, var = RW.gp_regression_noisy(x, y_train, kernel, x_star, var_noise, var_f, l)

    plot_posterior(x, y_train, x_star, mu, var)

