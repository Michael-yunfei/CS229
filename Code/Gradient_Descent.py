# Gradient Descent
# @ Michael
# Reference

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import random

# Newton's method


# Example 2.2 implemenation : solve square root function
def Square_root(value):
    """A function that solve the square root equations,
    input is the value, and supposed to be postive"""
    if value < 0:
        raise ValueError('The input value has to be positive')
    else:
        x0 = 1  # set initial guess to be 1
        while abs(x0**2 - value) >= 0.0001:
            fxn = x0**2 - value
            fxn_der = 2*x0
            x1 = x0 - fxn/fxn_der
            x0 = x1
    return x0


# Test function
Square_root(-3)
a3 = Square_root(3)
print(a3)   # # 1.7320508100147276
print(math.sqrt(3))  # 1.7320508075688772
# You can change the convergence rate from 0.0001 to 0.001


# General implemenation of Newton Method with lambda
def newton(f, Df, x0, epsilon, max_iter):
    '''Approximate solution of f(x)=0 by Newton's method.

    Parameters
    ----------
    f : function
        Function for which are searching for a solution f(x) = 0
    Df : function
        Derivative of f(x)
    x0 : number
        Initial guess for a solution f(x) = 0
    epsilon: number
        Stopping criteria is abs(f(x)) < epsilon
    max_iter : integer
        Maximum number of iteration of Newton's method

    Returns
    -------
    xn: number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/Df(xn)
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.'''

    xn = x0
    for n in range(0, max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            print('Found solution after', n, 'iterations.')
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None


# Test function
def p(x):
    return x**3 - x**2 - 1


def Dp(x):
    return 3*x**2 - 2*x


approx = newton(p, Dp, 1, 1e-10, 10)
print(approx)
# Found solution after 6 iterations.
# 1.4655712318767877


# divergent case
def L(x):
    return x**(1/3)


def Dl(x):
    return (1/3)*x**(-2/3)


approx = newton(L, Dl, 0.1, 1e-2, 100)
# Newton's method diverges in certain cases. For example,
# if the tangent line at the root is vertical


# Case Study

# Define the cost function
def square_loss(x, y, theta):
    '''
    A function that calculates the square loss

    Input
    -------
    x : dataset of dependent variables, an n by m matrix
    y : a vector of independent variable, an n by 1 vector
    theta : parameters (or estimations), m by 1 vetor

    Output
    -------
    loss : a number
    '''

    n = x.shape[0]
    yhat = np.dot(x, theta)
    loss = 1/2 * 1/n * np.sum(np.square(yhat - y))
    return loss


# Define gradient descent Function
def gradient_descent(x, y, theta, convergence,
                     learn_rate=0.1, iterations=100):
    '''
    Implementation of Gradient Descent Algorithm

    Input
    ------
    x : dataset of dependent variables, an n by m matrix
    y : a vector of independent variable, an n by 1 vector
    theta : parameters (or estimations), m by 1 vetor
    convergence : convergence rate measures when loop should stop
    learning rate : default value = 0.1
    interations : the maximum of interation

    Output
    -------
    theta: the convergent parameters
    cost_history : cost vector
    theta_history : trace of theta updating
    '''

    n = x.shape[0]
    m = x.shape[1]
    cost_history = np.zeros([iterations, 1])
    theta_history = np.zeros([iterations, m])
    current_theta = theta
    it = 0
    initial_converg = 10
    while initial_converg > convergence and it <= iterations:
        yhat = x @ current_theta
        theta_update = current_theta-(1/n)*learn_rate*x.transpose() @ (yhat-y)
        theta_history[it, :] = theta_update.transpose()
        cost_history[it, :] = square_loss(x, y, theta)
        it += 1
        initial_converg = np.max(np.abs(theta_update - current_theta))
        current_theta = theta_update

    return current_theta, cost_history, theta_history


# Test function
# read the dataset and visualize it
hp_data = pd.read_csv('Dataset/GradientEx1.csv',
                      names=['Population', 'Profit'])
hp_data.head()
hp_data.shape

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(hp_data.Population, hp_data.Profit)
ax.set(xlabel='Population', ylabel='Profit',
       title='Scatter Plot of X and Y')
fig.show()


# transfer dataframe into the matrix
hp_x = np.asmatrix(hp_data.Population).transpose()
hp_y = np.asmatrix(hp_data.Profit).transpose()
matrix_x = np.hstack([np.ones([hp_x.shape[0], 1]), hp_x])
matrix_x.shape

hp_theta = np.zeros([2, 1])

hp_loss = square_loss(matrix_x, hp_y, hp_theta)

hp_parameters, hp_cost_history, hp_theta_hisotry = gradient_descent(
    matrix_x, hp_y, hp_theta, 0.00001, 0.01, 50000
)

matrix([[-3.89024352],
        [ 1.19247736]])

# Now we plot the regression line
reg_x = np.linspace(5, 23, 1000).reshape([1000, 1])
reg_y = reg_x * hp_parameters[1] + hp_parameters[0]

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(hp_data.Population, hp_data.Profit)
ax.plot(reg_x, reg_y, 'r')
ax.set(xlabel='Population', ylabel='Profit',
       title='Scatter Plot of X and Y')
fig.show()


# Example: Prostate Cancer

prostate = pd.read_csv('Dataset/GradientEx2.csv')
prostate.head()
prostate.shape
prostate.corr()

prostate_y = np.asmatrix(prostate.PSA).reshape([prostate.shape[0], 1])
prostate_x = np.asmatrix(prostate.drop(['PSA', 'ID'], axis=1))
prostate_x.shape

prostate_theta = np.zeros([7, 1])
prostate_loss = square_loss(prostate_x, prostate_y, prostate_theta)

prostate_parameters, prostate_cost_history, prostate_para_hist = gradient_descent(
    prostate_x, prostate_y, prostate_theta, 0.001, 0.1, 1000
)

# It's important to normalize your dataset!

prostate_parameters, prostate_cost_history, prostate_para_hist = gradient_descent(
    preprocessing.scale(prostate_x), preprocessing.scale(prostate_y),
    prostate_theta, 0.001, 0.1, 1000
)


# Stochastic Graident Descent
def Stcst_gd(x, y, theta, samplesize, learning=0.1, iterations=100):
    '''
    Implementing the algorithm to do stochastic gradient descent
    Input
    ------
    x : dataset of dependent variables, an n by m matrix
    y : a vector of independent variable, an n by 1 vector
    theta : parameters (or estimations), m by 1 vetor
    samplesize : sample size of reshuffle, sample size <= x.shape[0]
    learning : learning rate, default value = 0.1
    interations : the maximum of interation

    Output
    -------
    theta: the convergent parameters
    cost_history : cost vector
    theta_history : trace of theta updating
    '''

    if samplesize > x.shape[0]:
        raise ValueError('Sample size must be less than population size')

    n = x.shape[0]
    m = x.shape[1]
    current_theta = theta
    alpha = learning
    cost_history = np.zeros([iterations, 1])
    theta_history = np.zeros([iterations, m])
    for it in range(iterations):
        randomIndex = random.sample(range(samplesize), samplesize)
        xtrain = x[randomIndex]
        ytrain = y[randomIndex]
        for i in range(samplesize):
            x_i = xtrain[i].reshape(1, -1)
            y_i = ytrain[i].reshape(-1, 1)
            fx = x_i @ current_theta
            update_theta = (current_theta
                             - alpha * 2
                             * x_i.transpose() @ (fx - y_i))
            current_theta = update_theta
        theta_history[it, :] = update_theta.transpose()
        cost_history[it, :] = square_loss(x, y, update_theta)

    return current_theta, cost_history, theta_history


hp_stc, stc_cost_history, stc_theta_hisotry = Stcst_gd(
    matrix_x, hp_y, hp_theta, 50, 0.01, 150
)

# matrix([[-3.4037112 ],
#         [ 1.06040721]])

hp_stc, stc_cost_history, stc_theta_hisotry = Stcst_gd(
    matrix_x, hp_y, hp_theta, 120, 0.01, 150
)
#
