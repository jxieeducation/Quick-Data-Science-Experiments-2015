from __future__ import division
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.ensemble import AdaBoostRegressor

df = pd.read_csv('data/train.csv', header=0)

#formatting
def process(df):
	df = df [['Survived', 'PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
	df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(float)
	df['FamilySize'] = df['SibSp'] + df['Parch']
	df = df.drop(['Ticket', 'Cabin', 'Embarked'], axis=1) 
	return df
df = process(df)
df = df.dropna()

#separating data
X = df[df.columns[1:]]
y = df[df.columns[0]]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1, random_state=0)

rf = AdaBoostRegressor(n_estimators=100, loss='linear', learning_rate=1)
rf.fit(X_train, y_train)

count = 0
for i in range(X_test.values.shape[0]):
	x_test_val = X_test.values[i]
	output = rf.predict(x_test_val)[0]
	output = 1 if output > 0.5 else 0
	if output == y_test.values[i]:
		count += 1
print count / X_test.values.shape[0]
