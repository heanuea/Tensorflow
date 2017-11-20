#got the train and split from this stack over flow 
#https://stackoverflow.com/questions/41859605/tensorflow-split-inputs-into-training-and-test-sets

import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils

iris = datasets.load_iris()


# Split Train/Test
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33)

# Build a model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=x_train.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(set(iris.target)), activation='softmax'))

model.compile(optimizer='rmsprop',
             loss='categorical_crossentropy',
             metrics=['accuracy'])


