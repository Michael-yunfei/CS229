# Classification: LDA, QDA, Navie Bayes
# @ Michael
# We will do this part with a hands-on approach
# For people who want to understand math and algorithm, please read my notes
# We will follow chapter 3 of the book by Aurelien Geron


from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


mnist = fetch_openml('mnist_784', version=1)
mnist.keys()

X, Y = mnist['data'], mnist['target']
X.shape  # (70000, 784)
Y.shape  # (70000, )

# take a look at picture
plt.imshow(X[0, :].reshape(28, 28), cmap=mpl.cm.binary,
           interpolation="nearest")
Y[0]  # '5'
Y = Y.astype(int)  # convert string to int

# Split the dataset
X_train, X_test, Y_train, Y_test = X[:60000], X[60000:], Y[:60000], Y[60000:]

# Binary Classficiation
y_train_5 = (Y_train == 5)
y_test_5 = (Y_test == 5)

# Stochastic Gradient Descent
sgd_clf = SGDClassifier(random_state=42)  # initialize the class
sgd_clf.fit(X_train, y_train_5)
np.sum(sgd_clf.predict(X_test) == y_test_5)/len(Y_test)  # 96.46% accuracy

# K-fold cross-validaton
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# array([0.9532 , 0.95125, 0.9625 ])

# WARNING: This is simply because only about 10% of the images are 5s,
# so if you always guess that an image is not a 5,
# you will be right about 90% of the time

# Confusion Matrix
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
confusion_matrix(y_train_5, y_train_pred)
# array([[52992,  1587],
#        [ 1074,  4347]])

precision_score(y_train_5, y_train_pred)  # 0.732558, accuracy
recall_score(y_train_5, y_train_pred)  # 0.80188, detecing
# it is correct only 73.26% of the time.
# Morover, it only detects 80.2% of the 5s.
# There is  precision/recall tradeoff !


# Inside the function, the score!
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")


plot_precision_recall_vs_threshold(precisions, recalls, thresholds)





























#
