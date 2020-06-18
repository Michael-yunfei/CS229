# SVM
# @ Michael
# Reference: Nagi El Hachem


import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix, solvers  # convex optimization tool


def gene_gaus_data(N, means, std_dev, dim=2):
    """ A function that generates data according to a Gaussian distribution
    Input
    ------
    N : size of sample for two categories
    means : fixed mean
    std_dev : standard devaition of gaussian distribution
    dim : dimension of dataset

    Output
    ------
    X: 2d array containing the x and y coordinates of each sample
    Y: 1d array containing the labels of each sample (label = {1, -1})

    datatype
    -------
    int, float
    """

    mean_len = len(means)
    if len(std_dev) != mean_len:
        print('means and std_deviation must have the same length')
        return

    X = np.empty([dim, N])
    Y = np.empty([N])
    sample_num = N // mean_len  # calculate the sample size for each category

    for i in range(mean_len):
        M = means[i]
        S = std_dev[i] * np.eye(dim)
        i1 = i * sample_num
        i2 = i1 + sample_num
        X[:, i1:i2] = np.random.multivariate_normal(M, S, sample_num).T
        Y[i1: i2] = (i+1) * np.ones([sample_num])

    if i2 != N:
        X[:, i2:] = np.random.multivariate_normal(M, S, (N-i2)).T
        Y[i2:] = (i+1) * np.ones((N-i2))

    return X, Y


# plot function with labels
def plot_data(X, Y):
    """Plot data with labels"""

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'grey']
    unique = np.unique(Y)
    if len(X.shape) == 1:
        X = np.array([X]).T

    for i, c in enumerate(unique):
        idx = np.where(Y == c)[0]
        x_sub = X[:, idx]
        if X.shape[1] == 1:
            y = [1] * x_sub.shape[1]
        else:
            y = x_sub[1, :]
        y = x_sub[1, :]
        plt.scatter(x_sub[0, :], y, c=colors[i])
    plt.show()


X, Y = gene_gaus_data(61, [[2, 2], [5, 5]], [1, 1])
plot_data(X, Y)


# Kernels
def linear_kernel(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.dot(x, y)


def poly_kernel(x, y, d=4):
    x = np.array(x)
    y = np.array(y)
    return (np.dot(x, y) + 1)**d


def rbf_kernel(x, y, sigma=1):
    x = np.array(x)
    y = np.array(y)
    norm = np.linalg.norm(x-y)
    return np.exp(-(norm**2) / (sigma**2))


# SVM Class !
class SVM:
    """
    Attributes:
        weights : w
        bias : b (in the formula b = y - estimatiton)
        alphas: lagrange multipliers
        C : slack penalty
        gram_matrix : gram matrix,
                    contains the computed inner product (xi.T).xj * yi *yj

    """

    def __init__(self):
        self.kernel = linear_kernel
        self.kernel_args = []
        self.colors = ['red', 'blue', 'green', 'purple', 'orange', 'grey']
        self.cmaps = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges', 'Grey']

    def compute_bias(self, EPSILON=1e-2):
        self.lagrange_multipliers[self.lagrange_multipliers < EPSILON] = 0
        self.support_vectors_idx = np.where(self.lagrange_multipliers > 0)[0]
        if self.support_vectors_idx.shape[0] == 0:
            return 0
        bias = 0
        for i in self.support_vectors_idx:
            kernels = np.array([self.kernel(self.data[:, j],
                                            self.data[:, i],
                                            *self.kernel_args)
                                for j in range(self.data.shape[1])])
            bias += self.labels[i] - np.sum(self.lagrange_multipliers
                                            * self.labels * kernels)
        return bias / self.support_vectors_idx.shape[0]

    def set_kernel(self, kernel, *args):
        self.kernel = kernel
        self.kernel_args = args

    def __compute_gram_matrix(self):
        """computes the gram coefficients for each pair
        where gram_coef = yi * yj * (xi.T). xj
        """

        gram_matrix = np.zeros([self.labels.shape[0], self.labels.shape[0]])
        size = self.labels.shape[0]
        for i in range(size):
            for j in range(size):
                gram_matrix[i, j] = (self.labels[i] * self.labels[j]
                                     * self.kernel(self.data[:, i],
                                                   self.data[:, j],
                                                   *self.kernel_args))
        return gram_matrix

    def __compute_lagrange_multipliers(self):
        # set up solver inputs
        data_size = self.labels.shape[0]
        P = matrix(self.gram_matrix, tc='d')  # d means floats
        q = matrix(np.full(self.labels.shape, -1, dtype=float), tc='d')
        G = matrix(-np.identity(data_size), tc='d') if self.C is None \
            else matrix(np.concatenate((-np.identity(data_size),
                                        np.identity(data_size))), tc='d')
        b = matrix(np.zeros(1), tc='d')
        A = matrix(self.labels, tc='d').T
        h = matrix(np.zeros(data_size), tc='d') if self.C is None \
            else matrix(np.concatenate((np.zeros(data_size), self.C
                                        * np.ones(data_size))), tc='d')
        solvers.options['show_progress'] = self.show_progress
        solution = solvers.qp(P, q, G, h, A, b)['x']  # get coptimal values
        return np.asarray(solution).reshape((data_size, ))  # make it array

    def __train(self, data, labels):
        """ find the separator coordiantes
        """

        self.data, self.labels = data, labels
        self.gram_matrix = self.__compute_gram_matrix()
        self.lagrange_multipliers = self.__compute_lagrange_multipliers()
        self.bias = self.compute_bias()

    def train(self, data, labels, C=None, show_progress=False):
        self.C = C
        self.show_progress = show_progress
        classes = np.unique(labels)
        Y = np.empty(shape=labels.shape)
        N = classes.shape[0]
        self.multi_lagrange_multipliers = [None] * N
        self.multi_bias = [None] * N
        self.multi_labels = [None] * N
        self.labels_all = labels
        for i, c in enumerate(classes):
            # if we have more than 2 classes
            Y[:] = -1
            idx = np.where(labels == c)[0]
            Y[idx] = 1
            self.__train(data, Y)
            temp_short_line = svm.lagrange_multipliers.copy()
            self.multi_lagrange_multipliers[i] = temp_short_line
            self.multi_labels[i] = Y.copy()
            self.multi_bias[i] = self.bias

    # now we are ready for classifying
    def decision(self, X):
        kernels = np.array([self.kernel(self.data[:, i], X, *self.kernel_args)
                            for i in range(self.data.shape[1])])
        desc = np.sum(self.lagrange_multipliers*self.labels*kernels)+self.bias
        return desc

    def process(self, data):
        """
        returns a list of labels of data
        """
        labels = self.decision(data)
        labels[labels <= 0] = -1
        labels[labels > 0] = 1
        return labels

    def print_2Ddecision(self, nb_samples=50, print_sv=True, print_non_sv=True,
                         levels=[0., 1.], color='seismic'):
        eps = 0.2
        xmin, ymin = np.min(self.data, axis=1) - eps
        xmax, ymax = np.max(self.data, axis=1) + eps

        # generate nb_samples floats between ximin, xmax
        x = np.linspace(xmin, xmax, nb_samples)
        y = np.linspace(ymin, ymax, nb_samples)
        x, y = np.meshgrid(x, y)  # generates a grid with x, y values

        num_categories = len(self.multi_lagrange_multipliers)
        c_z = np.empty(shape=(num_categories, ) + x.shape)
        alpha = 1

        for c in range(num_categories):
            svm.lagrange_multipliers = self.multi_lagrange_multipliers[c]
            svm.labels = self.multi_labels[c]
            svm.bias = self.multi_bias[c]

            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    c_z[c, i, j] = svm.decision(np.array((x[i, j], y[i, j])))

            if num_categories != 2:
                plt.pcolor(x, y, c_z[c, :, :],
                           cmap=self.cmaps[c],
                           vmin=-1, vmax=1, alpha=alpha)
            elif c == 0:
                plt.pcolor(x, y, c_z[c, :, :], cmap=color,
                           vmin=-1, vmax=1)

            if 0. in levels:
                plt.contour(x, y, c_z[c, :, :], levels=[0.], colors='white',
                            alpha=0.5, linewidths=3)
            if 1. in levels:
                plt.contour(x, y, c_z[c, :, :], levels=[1.],
                            colors=self.colors[c], linestyles='dashed',
                            linewidths=1.5)
            if -1. in levels:
                plt.contour(x, y, c_z[c, :, :], levels=[-1.],
                            colors=self.colors[c],
                            linestyles='dashed', linewidths=0.5)

            alpha /= 2

        plt.axis([x.min(), x.max(), y.min(), y.max()])
        self.__plot_data(print_sv, print_non_sv)

    def __plot_data(self, print_sv, print_non_sv):
        classes = np.unique(self.labels_all)
        for i, c in enumerate(classes):
            idx = None
            # filter support vector
            lagrange_multipliers = self.multi_lagrange_multipliers[i]
            if print_sv and not print_non_sv:
                idx = np.where(lagrange_multipliers != 0)[0]
            elif not print_sv and print_non_sv:
                idx = np.where(lagrange_multipliers == 0)[0]
            elif not print_sv and not print_non_sv:
                idx = [-1]
            # get class data
            tmp = np.where(self.labels_all == c)[0]
            idx = tmp if idx is None else np.intersect1d(idx, tmp)
            x_sub = self.data[:, idx]
            # scatter filtered data
            plt.scatter(x_sub[0, :], x_sub[1, :], c=self.colors[i])


# Test SVM
X = np.array([[1., 2., 3.], [1., 1., 1.]])
Y = np.array([-1., 1., 1.])
plot_data(X, Y)
data, labels = X, Y
svm = SVM()
svm.train(data, labels)
svm.print_2Ddecision()


# example 2
X = np.array([[0., 0., 1., 1.], [0., 1., 0., 1.]])
Y = np.array([-1., 1., 1., 1.])
plot_data(X, Y)

data, labels = X, Y
svm = SVM()
svm.train(data, labels)
svm.print_2Ddecision()

# nonlinear case
X = np.array([[0., 0., 1., 1.], [0., 1., 0., 1.]])
Y = np.array([-1., 1., 1., -1.])
plot_data(X, Y)

data, labels = X, Y
svm = SVM()
svm.set_kernel(poly_kernel, 2)  # set polynomial kernels
svm.train(data, labels)
svm.print_2Ddecision()

# generate gaussian data
X, Y = gene_gaus_data(100, [[1, 1], [4, 4]], [15, 2])
plot_data(X, Y)
data, labels = X, Y
svm = SVM()
svm.set_kernel(poly_kernel, 2)  # set polynomial kernels
svm.train(data, labels, C=10)
svm.print_2Ddecision(print_non_sv=False, color='coolwarm')

# multiple classes: linear
X, Y = gene_gaus_data(100, [[0, 0], [0, 6], [6, 0], [6, 6]],
                      [0.5, 0.5, 0.5, 0.5])
plot_data(X, Y)
svm = SVM()
svm.train(X, Y)
svm.print_2Ddecision()

# multiple classes: nonlinear
X, Y = gene_gaus_data(100, [[0, 0], [0, 6], [6, 0], [6, 6], [3, 3]],
                      [1, 1, 1, 1, 2.5])
plot_data(X, Y)
svm = SVM()
svm.set_kernel(poly_kernel, 2)
svm.train(X, Y, C=3)
svm.print_2Ddecision()

# set different kernel
svm.set_kernel(rbf_kernel, 10)
svm.train(X, Y, C=3)
svm.print_2Ddecision(levels=[0., 1.])
#
