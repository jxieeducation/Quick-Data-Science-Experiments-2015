from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import pandas as pd
from sklearn.cross_validation import train_test_split

"""
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python test.py
"""

batch_size = 128
nb_classes = 10
nb_epoch = 12

img_rows, img_cols = 28, 28
nb_filters = 32
nb_pool = 2
nb_conv = 3

data = pd.read_csv('train.csv')
data = data.iloc[np.random.permutation(len(data))]
y = data[['label']]
X = data.drop('label', 1)
X = X.as_matrix()
X.shape = (len(X), 28, 28) # resize from 784 to 28x28

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='full',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

##### kaggle submission

inputs = pd.read_csv('test.csv')
submission = open('submit.txt', 'wb')
for i in range(len(inputs)):
	input_sub = inputs.iloc[i].as_matrix()
	# input_sub /= 255
	input_sub = input_sub.reshape(1, 1, 28, 28)
	output = model.predict(input_sub)[0]
	for j in range(len(output)):
		if(output[j] > 0.5):
			submission.write(str(i + 1) + "," + str(j) + '\n')
