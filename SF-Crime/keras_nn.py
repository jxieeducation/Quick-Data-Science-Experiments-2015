from __future__ import absolute_import
from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
from random import sample
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split

batch_size = 1024
nb_classes = 39
nb_epoch = 100

training = pd.read_csv('train.csv', parse_dates = ["Dates"])
crime_OHE = preprocessing.LabelEncoder()
crime_labels = crime_OHE.fit_transform(training.Category)

def OHE_crime(df):
    days = pd.get_dummies(df.DayOfWeek)
    district = pd.get_dummies(df.PdDistrict)
    hour = pd.get_dummies(df.Dates.dt.hour)
    year = pd.get_dummies(df.Dates.dt.year)
    month = pd.get_dummies(df.Dates.dt.month)
    minute = pd.get_dummies(df.Dates.dt.minute)
    X = df.X
    Y = df.Y
    new_df = pd.concat([days, hour, year, month, district, X, Y], axis = 1)
    return new_df

print("preprocessing")

crimes = OHE_crime(training)
crimes = normalize(crimes, axis=0)
# ds.addSample(crimes.iloc[[k]], crime_labels[k])
crimes_train, crimes_test, target_train, target_test = train_test_split(crimes, crime_labels, test_size=0.33, random_state=42)

target_train = np_utils.to_categorical(target_train, nb_classes)
target_test = np_utils.to_categorical(target_test, nb_classes)


print ("making net")
# convert class vectors to binary class matrices
target_train = np_utils.to_categorical(target_train, nb_classes)
target_test = np_utils.to_categorical(target_test, nb_classes)

model = Sequential()
model.add(Dense(120, input_dim=68, init='uniform'))
model.add(Activation('linear'))
model.add(Dropout(0.15))
model.add(Dense(200, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(200, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(39))
model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)
model.fit(crimes_train, target_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, validation_data=(crimes_test, target_test))
