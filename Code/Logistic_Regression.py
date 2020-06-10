# Logistic Regression and Generalized Linear Model
# @ Michael
# Reference

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp


# Read the dataset
lgx = pd.read_csv('Dataset/Logistic_x.txt', sep=" +", names=['x1', 'x2'],
                  header=None, engine='python')
lgy = pd.read_csv('Dataset/Logistic_y.txt', sep=" +", names=['y'],
                  header=None, engine='python')
lgx.head()
lgy.head()  # the value is float
lgy.astype(int)  # convert all values to int

# construct matrix
lg1_xm = np.hstack([np.ones([lgx.shape[0], 1]), np.asmatrix(lgx)])
lg1_xm.shape
lg1_ym = np.asmatrix(lgy)
lg1_ym.shape


# Define sigmoid function
def sigmoid(x):
    return 1/(1 + np.exp(-x))


def gradient_L(x, y, theta):
    '''
    Input
    ------
    x : input matrix, n by m
    y : input matrix, n by 1
    theta : initial parameters, m by 1

    Output
    -------
    gradient : set it to negative value so hessian will become positive
               gradient is calculated based on the average empirical loss
               or sign function,
    '''
    z = np.multiply(y, (x @ theta))  # element wise multiplication
    gradient = -np.mean(np.multiply((1 - sigmoid(z)),
                                    np.multiply(y, x)), axis=0)
    return gradient


def hessian_L(x, y, theta):
    '''
    Input
    ------
    x : input matrix, n by m
    y : input matrix, n by 1
    theta : initial parameters, m by 1

    Output
    -------
    hessian : Hessian Matrix, m by m
    '''
    hessian = np.zeros([x.shape[1], x.shape[1]])
    z = np.multiply(y, (x @ theta))
    for i in range(hessian.shape[0]):
        for j in range(hessian.shape[1]):
            if i <= j:
                wise_multp1 = np.multiply(sigmoid(z), (1 - sigmoid(z)))
                wise_multp2 = np.multiply(x[:, i], x[:, j])
                hessian[i][j] = np.mean(np.multiply(wise_multp1,
                                                    wise_multp2))
                if i != j:
                    hessian[j][i] = hessian[i][j]  # hessian is symmetric
    return hessian


# Test functions first
sigmoid(lg1_xm).shape  # (99, 3)
theta_inital = np.zeros([lg1_xm.shape[1], 1])
gradient_L(lg1_xm, lg1_ym, theta_inital)  # (1, 3)
hessian_L(lg1_xm, lg1_ym, theta_inital)
# array([[ 0.25      ,  0.98082384, -0.08742426],
#        [ 0.98082384,  4.76984544, -0.18171064],
#        [-0.08742426, -0.18171064,  0.80740309]])


# Newton method for this case
def newton(x, y, theta_0, G, H, epsilon, maxiter):
    '''
    A netwon method for estimating prameters

    Input
    ------
    x : input matrix, n by m
    y : input matrix, n by 1
    theta_inital : initial parameters, m by 1
    G : gradient function
    H : hessian function
    epsilon : convergence rate
    maxiter : maximum iteration

    Output
    -------
    theta : estimated parameters
    '''

    theta = theta_0.copy()  # it's important to use copy()!!
    delta = 1
    it = 0
    while delta > epsilon and it < maxiter:
        theta_old = theta.copy()  # you need copy value, otherwise values
        # will change for all variables
        theta -= np.linalg.inv(H(x, y, theta)) @ G(x, y, theta).transpose()
        delta = np.linalg.norm(theta - theta_old, ord=1)
        it += 1
    return theta


# Test function
theta_estimated = newton(lg1_xm, lg1_ym, theta_inital,
                         gradient_L, hessian_L, 1e-6, 100)

















#
