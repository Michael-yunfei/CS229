# Mathematical Visualisation
# @ Michael

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns

sns.set_style('darkgrid')


x = np.linspace(-1, 2, 100)
y = np.linspace(-2, 3, 100)
X, Y = np.meshgrid(x, y)
Z1 = X**2 + Y**2


fig, ax = plt.subplots(figsize=(6, 5.6))
CS = ax.contour(X, Y, Z1, np.arange(-1, 8, 1))
ax.clabel(CS, inline=1, fontsize=10)
ax.plot(x, 2-x)
ax.set_title('Contour Plot With Constraint')
# plt.savefig('/Users/Michael/Library/Mobile
# Documents/com~apple~CloudDocs/ComputerScience/
# Optimization/Notes/figures/contour1.png', bbox_inches='tight', dpi=360)


# an optimization example

def f(x):
    return -(x[0]*x[1] + x[1]*x[2])


cons = ({'type': 'eq',
         'fun': lambda x: np.array([x[0] + 2*x[1] - 6, x[0] - 3*x[2]])})

x0 = np.array([2, 2, 0.67])
res = minimize(f, x0, constraints=cons)
res.fun


# multivariate normial distribution

# Univariate normal distribution
def univariate_normal(x, mean, variance):
    """pdf of the univariate normal distribution"""
    return ((1. / np.sqrt(2 * np.pi * variance))*np.exp(
        -(x-mean)**2 / (2 * variance)))


x = np.linspace(-3, 5, num=150)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(x, univariate_normal(x, mean=0, variance=1),
        label='$\mathcal{N}(0, 1)$')
ax.plot(x, univariate_normal(x, mean=2, variance=3),
        label="$\mathcal{N}(2, 3)$")
ax.plot(x, univariate_normal(x, mean=0, variance=0.2),
        label="$\mathcal{N}(0, 0.2)$")
plt.xlabel('$x$', fontsize=13)
plt.ylabel('density: $f(x)$', fontsize=13)
plt.title('Univariate normal distributions')
plt.ylim([0, 1])
plt.xlim([-3, 5])
plt.legend(loc=1)
fig.subplots_adjust(bottom=0.15)
# fig.show()


# multivariate normal
def multivariate_normal(x, d, mean, covariance):
    """pdf of the multivariate normal distribution."""
    x_m = x - mean
    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(
        covariance))) * np.exp(-(np.linalg.solve(
            covariance, x_m).T.dot(x_m)) / 2))


# Plot bivariate distribution
def generate_surface(mean, covariance, d):
    """Helper function to generate density surface."""
    nb_of_x = 100  # grid size
    x1s = np.linspace(-5, 5, num=nb_of_x)
    x2s = np.linspace(-5, 5, num=nb_of_x)
    x1, x2 = np.meshgrid(x1s, x2s)  # Generate grid
    pdf = np.zeros((nb_of_x, nb_of_x))
    # Fill the cost matrix for each combination of weights
    for i in range(nb_of_x):
        for j in range(nb_of_x):
            pdf[i, j] = multivariate_normal(
                np.matrix([[x1[i, j]], [x2[i, j]]]),
                d, mean, covariance)
    return x1, x2, pdf  # x1, x2, pdf(x1,x2)


# Plot of independent Normals
bivariate_mean = np.matrix([[0.], [0.]])  # Mean
bivariate_covariance = np.matrix([
    [1., 0.],
    [0., 1.]])  # Covariance
d = 2  # dimension
x1, x2, p = generate_surface(bivariate_mean, bivariate_covariance, d)
