# Neural Network I
# @ Michael

# This code includes a class that fits logistic regression and a class
# fits with the neural network.

import numpy as np
import pandas as pd
import random
import sys


# Class fits with logistic regression model for doing binary classification
# one should initialze the class with Matrix X and Y
class logistic_reg(object):
    """
    A logistic model for binary classification
    Initialze the class with input:
        X - m by n matrix, m = number of features, n = sample size
        Y - 1 by n vectory, n = sample size
        percential - percentage for splitting sample to train and test
        random=false, if it is true, dataset is splitted randomly
    """

    def __init__(self, X, Y, percential, random=False):
        self.X = X
        self.Y = Y
        self.percentile = percential
        try:
            self.X = np.asmatrix(self.X)
            self.Y = np.asmatrix(self.Y)
            if (self.X.shape[1] != self.Y.shape[1]):
                print('Input Y and X \
                      have different sample size')
        except Exception:
            print('There is an error with the input data.\
                  Make sure your input X and Y are either matrix or dataframe')
            sys.exit(0)
        self.xtrain, self.xtest, self.ytrain, self.ytest = (
            logistic_reg.splitSample(self.X, self.Y, percential, random))
        self.w = np.zeros([self.xtrain.shape[0], 1])
        self.b = 0

    @staticmethod
    def splitSample(sampleX, sampleY, trainSize, permute=False):
        '''
        static function to split the sample
        X = m by n, n = sample size
        '''
        sample_length = int(sampleX.shape[1] * trainSize)
        if permute is True:
            random_index = random.sample(range(sampleX.shape[1]),
                                         sample_length)
            trainSampleX = sampleX[:, random_index]
            trainSampleY = sampleY[:, random_index]
            testSampleX = np.delete(sampleX, random_index, 1)
            testSampleY = np.delete(sampleY, random_index, 1)

            return(trainSampleX, testSampleX, trainSampleY, testSampleY)
        else:
            percentile_index = list(range(sample_length))
            trainSampleX = sampleX[:, percentile_index]
            trainSampleY = sampleY[:, percentile_index]
            testSampleX = np.delete(sampleX, percentile_index, 1)
            testSampleY = np.delete(sampleY, percentile_index, 1)

            return(trainSampleX, testSampleX, trainSampleY, testSampleY)

    @staticmethod
    def sigmoid(z):
        """
        Compute the sigmoid of z

        Arguments:
        z -- A scalar or numpy array of any size.

        Return:
        s -- sigmoid(z)
        """
        s = 1/(1+np.exp(-z))

        return s

    @staticmethod
    def propagate(w, b, X, Y):
        """
        Implement the cost and propogation
        """
        m = X.shape[1]
        # FORWARD PROPAGATION (FROM X TO COST)
        A = logistic_reg.sigmoid(np.dot(w.T, X)+b)
        cost = -1/m * np.sum(np.multiply(Y, np.log(A))
                             + np.multiply((1 - Y), np.log(1-A)))

        # BACKWARD PROPAGATION (TO FIND GRAD)
        dw = 1/m * np.dot(X, (A-Y).T)
        db = 1/m * np.sum(A-Y)

        assert(dw.shape == w.shape)
        assert(db.dtype == float)
        cost = np.squeeze(cost)
        assert(cost.shape == ())
        # save it as a dictionary
        grads = {"dw": dw, "db": db}

        return grads, cost

    @staticmethod
    def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
        """
        estimate the parameters by doing gradient descent
        """

        costs = []

        for i in range(num_iterations):
            # Cost and gradient calculation
            grads, cost = logistic_reg.propagate(w, b, X, Y)
            # Retrieve derivatives from grads
            dw = grads["dw"]
            db = grads["db"]
            # update the gradient
            w = w - learning_rate * dw
            b = b - learning_rate * db
            # Record the costs
            if i % 100 == 0:
                costs.append(cost)
            # Print the cost every 100 training iterations
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        params = {"w": w,
                  "b": b}

        grads = {"dw": dw,
                 "db": db}

        return params, grads, costs

    @staticmethod
    def predict(w, b, X):
        '''
        predict the results
        '''
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        w = w.reshape(X.shape[0], 1)
        # Compute vector "A" predicting the probabilities of a cat
        # being present in the picture
        A = logistic_reg.sigmoid(np.dot(w.T, X)+b)

        for i in range(A.shape[1]):
            # Convert probabilities A[0,i] to actual predictions p[0,i]
            A_value = A[0, i]
            if A_value > 0.5:
                Y_prediction[0, i] = 1
            else:
                Y_prediction[0, i] = 0

        assert(Y_prediction.shape == (1, m))

        return Y_prediction

    def fit_model(self, num_iterations=2000, learning_rate=0.5,
                  print_cost=False):
        """
        finally we can fit the model
        """
        parameters, grads, costs = logistic_reg.optimize(self.w, self.b,
                                                         self.xtrain,
                                                         self.ytrain,
                                                         num_iterations,
                                                         learning_rate,
                                                         print_cost)
        w = parameters["w"]
        b = parameters["b"]
        Y_prediction_test = logistic_reg.predict(w, b, self.xtest)
        Y_prediction_train = logistic_reg.predict(w, b, self.xtrain)

        # Print train/test Errors
        print("train accuracy: {} %".format(
            100 - np.mean(np.abs(Y_prediction_train - self.ytrain)) * 100))
        print("test accuracy: {} %".format(
            100 - np.mean(np.abs(Y_prediction_test - self.ytest)) * 100))

        d = {"costs": costs,
             "Y_prediction_test": Y_prediction_test,
             "Y_prediction_train": Y_prediction_train,
             "w": w,
             "b": b,
             "learning_rate": learning_rate,
             "num_iterations": num_iterations}

        return d


# test the code
X = pd.read_csv('Dataset/digits_data.csv', header=None)
y = pd.read_csv('Dataset/digits_labels.csv', header=None)

X.shape  # (5000, 400), each row contains 400 values, 5000 totally
y.shape  # (5000, 1), labels for each row
np.unique(y)  # digit 0 is indicated by number 10

# transfer data into the binary case!
y = (y == 1).astype(int)

# reshape X and y into feature by sampel size
X = X.T
X.shape
y = y.T

digit_logrist_fit = logistic_reg(X, y, 0.9, random=False)
digit_results = digit_logrist_fit.fit_model(num_iterations=2000,
                                            learning_rate=0.5,
                                            print_cost=False)

# train accuracy: 99.28888888888889 %
# test accuracy: 99.4 %

# I have to say that prediction rate is quite high!
#
