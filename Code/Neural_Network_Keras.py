# Neural Network with Keras
# @ Michael
# Chapter 10 - 12 from Hands-on Machine Learning


# Building a simple image classifer
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd


# dataset: Fashion MNIST
# 70,000 grayscale images of 28 x 28 pixels each, with 10 classes
# load the dataset
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X_train_full.shape  # 60,000 images with 28 by 28
X_train_full.dtype

# create the validate dataset, and normalize the images
X_valid, X_train = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# check the class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class_names[y_train[0]]

# check some pictures
X_train[0].shape
plt.imshow(X_train[0], cmap='Greys')
plt.imshow(X_train[11], cmap='Greys')

# Creating the Model Using the Sequential API
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))  # convert data into 1-D
model.add(keras.layers.Dense(300, activation='relu'))  # 300 neurons/units
model.add(keras.layers.Dense(100, activation='relu'))  # 100 neurons
model.add(keras.layers.Dense(10, activation='softmax'))  # output layer

# model summary
model.summary()
model.layers

# compilig the model (specify the model structure)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# Training the model
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))


# plot the cost curve
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # set the vertical range to [0, 1]
plt.show()

# Predict the model
model.predict_classes(X_test[:3])  # array([9, 2, 1])
#
