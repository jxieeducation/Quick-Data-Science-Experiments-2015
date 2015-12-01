from __future__ import absolute_import
# from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.utils import np_utils
from preprocess import *


X, y = getData()
X_test, y_test = getTestData()


batch_size = 8
nb_epoch = 125

model = Sequential()
model.add(Dense(25, input_shape=(13,)))
model.add(Activation('linear'))
model.add(Dropout(0.10))
model.add(Dense(40))
model.add(Activation('relu'))
model.add(Dropout(0.05))
# model.add(Dense(25))
# model.add(Activation('linear'))
# model.add(Dropout(0.05)) # this layer doesn't add much
model.add(Dense(1))
model.add(Activation('relu'))

rms = RMSprop()
# ada = Adagrad()
model.compile(loss='mean_squared_error', optimizer=rms)

model.fit(X, y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, validation_data=(X_test, y_test))

for i in range(X_test.shape[0]):
	xn = X_test[i]
	yn = y_test[i]
	print yn, model._predict([xn])[0][0]

