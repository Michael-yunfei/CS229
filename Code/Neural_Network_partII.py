# Neural Network II
# @ Michael
# we will build up an neural network with hidden layers
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model


# first, we creat some dataset
def load_planar_dataset():
    np.random.seed(1)
    m = 400
    N = int(m/2)
    D = 2
    X = np.zeros([m, D])
    Y = np.zeros([m, 1], dtype='uint8')
    a = 4

    for j in range(2):
        ix = range(N*j, N*(j+1))
        t = np.linspace(j*3.12, (j+1)*3.12, N) + np.random.randn(N)*0.2
        r = a*np.sin(4*t) + np.random.randn(N)*0.2
        X[ix] = np.column_stack((r*np.sin(t), r*np.cos(t)))
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y


# test function
X, Y = load_planar_dataset()
X[0, :].shape
Y.shape
# be careful on shapes
plt.scatter(X[0, :], X[1, :], c=Y.ravel(), s=40, cmap=plt.cm.Spectral)


# define the decision boundary
def plot_decision_boundary(model, X, Y):
    x1_min, x1_max = X[0, :].min()-1, X[0, :].max()+1
    x2_min, x2_max = X[1, :].min()-1, X[1, :].max()+1
    h = 0.01
    x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                         np.arange(x2_min, x2_max, h))
    Z = model(np.column_stack((x1.ravel(), x2.ravel())))
    Z = Z.reshape(x1.shape)
    plt.contourf(x1, x2, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=Y.ravel(), cmap=plt.cm.Spectral)


# use the logistic regression model from sklearn.learn model to
sklg = sklearn.linear_model.LogisticRegressionCV()
sklg.fit(X.T, Y.T)

# plot the results
LR_predictions = sklg.predict(X.T)
# accuracy of logistic regression
print(float((np.dot(Y,
                    LR_predictions)+np.dot(1-Y,
                                           1-LR_predictions))/float(Y.size)))
# 47%

plot_decision_boundary(lambda x: sklg.predict(x), X, Y)


# Neural Network with single hidden layer

# Main structure:
def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    """
    A neural network model for doing classification

    Input:
    X -- feature matrix m by n (feature size, sample size)
    Y -- label vector 1 by n (1, sample size)
    n_h -- size of the hidden layer
    num_interations = 10000
    print_cost -- if True, print the cost every 1000 iterations

    Output:
    parameters -- coefficients estimated by the model
    """

    np.random.seed(3)
    n_x = X.shape[0]
    n_y = Y.shape[0]

    # Intialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y)

    # gradient descent
    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_progagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)

    if print_cost and i % 1000 == 0:
        print("Cost after interation %i : %f" % (i, cost))

    return parameters


# Initialize parameters
def initialize_parameters(n_x, n_h, n_y):
    """
    Initialize parameters for one hidden layers with size n_h

    Input:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Output:
    parameters -- pyhon dictionary includes:
                W1 - weight matrix of shape (n_h, n_x)
                b1 - bias vector of shape (n_h, 1)
                W2 - weight matrix of shape (n_y, n_h)
                b2 - bias vector of shape(n_y, 1)
    """
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros([n_h, 1])
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros([n_y, 1])

    parameters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    return parameters


# sigmoid function
def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s


# Forward progagation
def forward_propagation(X, parameters):
    """
    A function of doing forward progagation in neural network model
    Input:
    X -- size(n_x, smaple size)
    parameters -- python dictionary containing all parameters

    Output:
    A2 -- the sigmood output of the second activation
    cahce -- a dictionary containing "Z1", "A1", "Z2", "A2"
    """

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # Implement forward progagation
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {'Z1': Z1, "A1": A1, 'Z2': Z2, 'A2': A2}

    return A2, cache


# Compute cost
def compute_cost(A2, Y, parameters):
    """
    Compute the cross-entropy cost
    """

    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1-A2), 1-Y)
    cost = - np.sum(logprobs)/m

    cost = np.squeeze(cost)  # squeeze dimension into scale

    return cost


# backward progagation
def backward_progagation(parameters, cache, X, Y):
    """
    Implement the backward progagation using derivative rules

    Input: parameters, cache, X, Y

    Output: grads -- python dictionary containing gradients for
            each parameter
    """

    m = X.shape[1]

    W2 = parameters['W2']
    A1 = cache['A1']
    A2 = cache['A2']

    # backward progagation
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis=1, keepdims=True)/m
    dZ1 = np.multiply(np.dot(W2.T, dZ2), (1-np.power(A1, 2)))
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis=1, keepdims=True)/m

    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    return grads


# update_parameters
def update_parameters(parameters, grads, learning_rate=1.2):
    """
    A function that updates the parameters
    """

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1-learning_rate*dW1
    b1 = b1-learning_rate*db1
    W2 = W2-learning_rate*dW2
    b2 = b2-learning_rate*db2

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return parameters


# prediction function
def predict(parameters, X):
    """
    A function that predicts the classification
    """

    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)

    return predictions


# Test the model

# Build a model with a n_h-dimensional hidden layer
parameters = nn_model(X, Y, n_h=4,
                      num_iterations=10000, print_cost=True)

plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)


# see how hidden layer size will change accuray

plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations=5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,
                             predictions.T)
                      + np.dot(1-Y, 1-predictions.T))/float(Y.size)*100)
    print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
