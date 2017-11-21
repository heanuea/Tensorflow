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

#epochs means how many times you go through your training set.
# Set parameters
epoch = 100
batch_size = 10


one_hot_label_y_train = np_utils.to_categorical(y_train)
one_hot_label_y_test = np_utils.to_categorical(y_test)
model.fit(x_train, one_hot_label_y_train, epochs=epoch, batch_size=batch_size)
score = model.evaluate(x_test, one_hot_label_y_test, batch_size=batch_size)
print("\n{}: {:.2f}%".format(model.metrics_names[1], score[1]*100))

predict_data = np.array([4., 3., 4., 1.2])
x = predict_data.reshape(-1,4)
predict = model.predict(x)

for i in range(len(predict)):    
    guess = iris.target_names[np.argmax(predict[i])]
    actual = iris.target_names[y_train[i]]
    print("Predict: {},\nActual: {},\nIs it Correct: {}\n".format(guess, actual, guess==actual))