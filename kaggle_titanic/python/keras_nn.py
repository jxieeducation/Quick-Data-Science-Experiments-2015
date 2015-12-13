from __future__ import division
import pandas as pd
import numpy as np
from sklearn import cross_validation
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

df = pd.read_csv('data/train.csv', header=0)

#formatting
def process(df):
	df = df [['Survived', 'PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
	df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(float)
	df['FamilySize'] = df['SibSp'] + df['Parch']
	df['isOld'] = df['Age'].apply(lambda age: 1 if age > 50 else 0)
	df['isYoung'] = df['Age'].apply(lambda age: 1 if age < 18 else 0)
	df["Embarked"] = df["Embarked"].fillna("S")
	df.loc[df["Embarked"] == "S", "Embarked"] = 0
	df.loc[df["Embarked"] == "C", "Embarked"] = 1
	df.loc[df["Embarked"] == "Q", "Embarked"] = 2
	df = df.drop(['Ticket', 'Cabin'], axis=1) 
	return df
df = process(df)
df = df.dropna()

#separating data
X = df[df.columns[1:]]
y = df[df.columns[0]]
X, X_test, y, y_test = cross_validation.train_test_split(X, y, test_size=0.1, random_state=0)
X, X_test, y, y_test = X.values, X_test.values, y.values, y_test.values

nb_classes = np.max(y)+1
y = np_utils.to_categorical(y, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

batch_size = 16
nb_epoch = 200

model = Sequential()
model.add(Dense(18, input_dim=11, init='uniform'))
model.add(Activation('linear'))
model.add(Dropout(0.15))
model.add(Dense(30, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.15))
# model.add(Dense(30, init='uniform'))
# model.add(Activation('sigmoid'))
# model.add(Dropout(0.15))
model.add(Dense(2))
model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss='binary_crossentropy', optimizer=rms)

model.fit(X, y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, validation_data=(X_test, y_test))

count = 0
for i in range(X_test.shape[0]):
	ny = 0 if y_test[i][0] > y_test[i][1] else 1
	ny_test = model._predict([X_test[i]])[0]
	ny_test = 0 if ny_test[0] > ny_test[1] else 1
	if(ny == ny_test):
		count += 1
print count / X_test.shape[0]


predictions = pd.read_csv('data/test.csv', header=0)
predictions['Survived'] = 0
predictions = process(predictions)
predictions = predictions[predictions.columns[1:]]
predictions = predictions.fillna(predictions.mean())
f = open('data/predictions.csv', 'wb')

for index, row in predictions.iterrows():
	f.write(str(int(row['PassengerId'])))
	f.write(',')
	out = model._predict([row.values])[0]
	out = 0 if out[0] > out[1] else 1
	f.write(str(out))
	f.write('\n')
f.close()
