import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import sys

# A small function to make the program happy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# The class HiddenPrints, taken from Alexander Chzhen, will stop unwanted print
#    statements called during training. Credits:
#    https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# First we want to get the MNIST data
mnist = keras.datasets.mnist

# And then divide the data into training and testing groups
# X represents the images, and y represents the labels
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Then we can train with a specified batch-size and epochs
batch_size = np.array([32, 64, 128, 256, 512, 1024, 2048])
epochs = np.array([6, 7, 8, 9, 10, 11, 12])
nodes = np.array([10, 15, 18, 20, 30, 100, 300, 1000, 3000])

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))

# After extensive testing of the batch sizes, epochs, and nodes,
#   the best accuracy was found in the following categories:

# 1. 10 nodes, 7 epochs, 2048 batch
# 2. 15 nodes, 7 epochs, 1024 batch
# 3. 20 nodes, 6 epochs, 2048 batch
# 4. 10 nodes, 6 epochs, 2048 batch
# 5. 18 nodes, 7 epochs, 1024 batch

# so we set the variables to maximum efficiency.
number_nodes = 10
number_epochs = 7
number_batch_size = 2048

# Set up the nodes of the single layer
model.add(keras.layers.Dense(number_nodes))

# And then we compile it
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Now we will begin training repeatedly and use HiddenPrints() to reduce text output.
print('Please wait while the model is training...')
with HiddenPrints():
    i = 0
    while i < 100:
        model.fit(X_train, y_train, batch_size=number_batch_size, epochs=number_epochs)
        i += 1

# Lastly, we evaluate the test data.
loss, accuracy = model.evaluate(X_test, y_test)
print(accuracy)
