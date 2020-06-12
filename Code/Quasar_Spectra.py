# Case Study: denoising quasar spectra
# @ Michael
# Case Study from Stanford CS229

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib  as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# read the dataset
quasar_train = pd.read_csv('Dataset/quasar_train.csv')
quasar_train.head()
cols_train = quasar_train.columns.values.astype(float).astype(int)
quasar_train.columns = cols_train
quasar_train.head()
quasar_train.shape  # 200, 450

quasar_test = pd.read_csv('Dataset/quasar_test.csv')
quasar_test.head()
cols_test = quasar_test.columns.values.astype(float).astype(int)
quasar_test.columns = cols_test
quasar_test.head()
quasar_test.shape  # 50, 450


# closed form solution (or normal_eqaution)
def normal_equation(X, Y, weight=None):
    '''
    A function that calculate the coefficients based on the
    closed form solution: (X'X)^{-1}X'Y

    Input
    ------
    X : n by m matrix
    Y : n by 1 matrix
    weight : weith matrix, n by n matrix diagonal matrix

    Output
    -------
    coefficients : estimated coefficients
    '''

    if weight is None:
        coefficients = np.linalg.inv(X.T @ X) @ X.T @ Y
    else:
        coefficients = np.linalg.inv(X.T @ weight @ X) @ X.T @ weight @ Y

    return coefficients


# weight function
def build_weights(X, X_i, Tau=5):
    '''
    A weight function that assign the weight to each X
    What weight measures is the distance between different wavebadth

    Input
    ------
    X : the features from test dataset, n by m matix, includes the constant
        that's why we only take the second column[:, 1]
    X_i: a specific row in X, 1 by m vector
    Tau : the weight parameter

    Output:
    -------
    weights : a n by n diagnoal matrix
    '''

    weights = np.diag(np.exp(-((X - X_i)[:, 1]**2)/(2*Tau**2)))

    return weights


y = quasar_train.head(1).values.T  # the fulx we measured
x = np.vstack([np.ones(cols_train.shape), cols_train]).T  # wavebadth+constant
x.shape
theta = normal_equation(x, y)
theta.shape

ax = sns.regplot(x=x[:, 1], y=y, fit_reg=False)
plt.plot(x[:, 1], x@theta, linewidth=5)
ax.set(xlabel='Wavelength', ylabel='Flux')


# tain the regression
pred = []
for k, x_j in enumerate(x):
    w = build_weights(x, x_j, 5)
    theta = normal_equation(x, y, w)
    pred.append((theta.T @ x_j[:, np.newaxis]).ravel()[0])

# np.newaxis is to increase dimension
# ravel() is to reduce dimension
# unlike the very commmon weighted least square model,
# here we construct a systematic weights, which weighted the
# whole dataset in terms of each feature
# they estimated coefficients for each weight
# rather than getting a set of coefficients applicalbe to all
# we get a series of coeeficients !

ax = sns.regplot(x=x[:, 1], y=y, fit_reg=False)
plt.plot(x[:, 1], pred, linewidth=3)
ax.set(xlabel='Wavelength', ylabel='Flux')

# See how the value of tau will affet the predction
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.ravel()
colors = sns.color_palette('muted')
for k, tau in enumerate(np.logspace(0, 3, 4).astype(int)):
    pred = []
    ax = axes[k]
    for x_j in x:
        w = build_weights(x, x_j, tau)
        theta = normal_equation(x, y, w)
        pred.append((theta.T @ x_j[:, np.newaxis]).ravel()[0])
    sns.regplot(x=x[:, 1], y=y, fit_reg=False, ax=ax, color=colors[0])
    ax.plot(x[:, 1], pred, linewidth=3, color=colors[k+1])
    ax.set(xlabel="wavebadth", ylabel='Flux', title='tau = {}'.format(tau))




































#
