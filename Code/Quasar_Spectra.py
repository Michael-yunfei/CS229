# Case Study: denoising quasar spectra
# @ Michael
# Case Study from Stanford CS229

import numpy as np
import pandas as pd
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
for k, x_j in enumerate(x):  # automatically enumerate each row
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


# Smooth the data
y_train = quasar_train.values.T
y_train.shape
y_test = quasar_test.values.T


def smoother(x, y_in, tau):
    '''
    A function that repeats prediction for all observations
    Input
    ------
    x : m by 2, matrix, one column is constant, one column is wavebadth
    y_in : obsevations we got, or noise
    tau : the weight parameter
    Output
    ------
    Pred : a m by n matrix, which includes all the predications
    '''
    pred = np.zeros(y_in.shape)
    for i in range(y_in.shape[1]):
        y = y_in[:, i]  # get each column, or each observation
        for j, x_j in enumerate(x):
            w = build_weights(x, x_j, tau)  # construct weights
            theta = normal_equation(x, y, w)
            pred[j, i] = (theta.T @ x_j[:, np.newaxis]).ravel()[0]
    return pred


y_train_smooth = smoother(x, y_train, 5)
y_train_smooth.shape  # (450, 200)
y_test_smooth = smoother(x, y_test, 5)
y_test_smooth.shape  # (450, 50)


# understand np.argsort
np.argsort(np.array([1, 3, 4, 2]))
# the postion for 1 is zero,
# the position for 3 is 3
# the position for 4 is 1
# sorted_array(1, 2, 3, 4) = position_array([0, 3, 1, 2])

# I didnot finsih all questions for this case study
#
# k = 3
# estimation = []
# errors = []
# train_left, train_right = np.split(y_train_smooth,[150],axis=1)
# for i in range(train_right.shape[0]):
#     row = train_right[i]
#     dist = ((train_right-row)**2).sum(axis=1)
#     h = dist.max()
#
#     neighbors = dist.argsort()[:k]
#
#     bottom = 0
#     for j in range(len(neighbors)):
#         bottom += ker(dist[neighbors[j]]/h)
#     left_hat = np.zeros(len(train_left[0]))
#     for j in range(len(neighbors)):
#         left_hat += ker(dist[neighbors[j]]/h)*train_left[neighbors[j]]
#     left_hat /= bottom
#     estimation.append(left_hat)
#     error = np.sum((train_left[i] - left_hat) ** 2)
#     errors.append(error)
# errors = np.array(errors)
# estimation = np.array(estimation)
# print('The average error is %.6f'%errors.mean())
#
