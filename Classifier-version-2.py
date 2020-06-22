import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import sys


# A small function to make the program happy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# First we want to get the MNIST data
mnist = keras.datasets.mnist

# And then divide the data into training and testing groups
# X represents the images, and y represents the labels
# Also see that we reshape and regularize the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Now let's use a more in-depth neural network than version 1
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3),
                              activation='relu',
                              input_shape=(28, 28, 1)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation='softmax'))

# And then we compile it
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# In version 1, I did't want users to see all of the hundreds of lines
# of output from fitting the model, but this version only has eight lines
# and I find the waiting to go by quicker while watching the accuracy increase.
model.fit(X_train, y_train, epochs=4)


# Lastly, we evaluate the test data.
loss, accuracy = model.evaluate(X_test, y_test)
print('Accuracy of the training:', accuracy)
