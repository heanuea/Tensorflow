# Adapted from: https://github.com/salmanahmad4u/keras-iris/blob/master/iris_nn.py

import csv
import numpy as np
import keras as kr

# Load the Iris dataset.
# Data from: https://github.com/mwaskom/seaborn-data/blob/master/iris.csv
iris = list(csv.reader(open('iris.csv')))[1:]

# The inputs are four floats: sepal length, sepal width, petal length, petal width.
inputs  = np.array(iris)[:,:4].astype(np.float)

# Outputs are initially individual strings: setosa, versicolor or virginica.
outputs = np.array(iris)[:,4]
# Convert the output strings to ints.
outputs_vals, outputs_ints = np.unique(outputs, return_inverse=True)
# Encode the category integers as binary categorical vairables.
outputs_cats = kr.utils.to_categorical(outputs_ints)

# Split the input and output data sets into training and test subsets.
inds = np.random.permutation(len(inputs))
train_inds, test_inds = np.array_split(inds, 2)
inputs_train, outputs_train = inputs[train_inds], outputs_cats[train_inds]
inputs_test,  outputs_test  = inputs[test_inds],  outputs_cats[test_inds]

# Create a neural network.
model = kr.models.Sequential()

# Add an initial layer with 4 input nodes, and a hidden layer with 16 nodes.
model.add(kr.layers.Dense(16, input_shape=(4,)))
# Apply the sigmoid activation function to that layer.
model.add(kr.layers.Activation("sigmoid"))
# Add another layer, connected to the layer with 16 nodes, containing three output nodes.
model.add(kr.layers.Dense(3))
# Use the softmax activation function there.
model.add(kr.layers.Activation("softmax"))

# Configure the model for training.
# Uses the adam optimizer and categorical cross entropy as the loss function.
# Add in some extra metrics - accuracy being the only one.
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Fit the model using our training data.
model.fit(inputs_train, outputs_train, epochs=100, batch_size=1, verbose=1)

# Evaluate the model using the test data set.
loss, accuracy = model.evaluate(inputs_test, outputs_test, verbose=1)

# Output the accuracy of the model.
print("\n\nLoss: %6.4f\tAccuracy: %6.4f" % (loss, accuracy))

# Predict the class of a single flower.
prediction = np.around(model.predict(np.expand_dims(inputs_test[0], axis=0))).astype(np.int)[0]
print("Actual: %s\tEstimated: %s" % (outputs_test[0].astype(np.int), prediction))
print("That means it's a %s" % outputs_vals[prediction.astype(np.bool)][0])

# Save the model to a file for later use.
model.save("iris_nn.h5")
# Load the model again with: model = load_model("iris_nn.h5")