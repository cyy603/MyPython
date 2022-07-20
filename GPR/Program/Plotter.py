import numpy as np
import matplotlib.pyplot as plt

def sample(mu, var, jitter, N):
    """Generate N samples from a multivariate Gaussian N(mu, var)"""
    n = 100

    L = np.linalg.cholesky(var + jitter * np.eye(var.shape[0]))# cholesky decomposition (square root) of covariance matrix
    f_post = mu + L @ np.random.normal(size = (n, N))

    return f_post

f = lambda x : np.sin(x) + 0.1 * np.cos(2 * x)

def plot_posterior(x_train, y_train, x_star, mu, var):
    samples = 30
    jitter = 1e-10

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