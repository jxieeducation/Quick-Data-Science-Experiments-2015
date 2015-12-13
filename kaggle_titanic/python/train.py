import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import cross_validation

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

#training
logreg = linear_model.LogisticRegression()
logreg.fit(X_train, y_train)

print logreg.score(X_test, y_test)

predictions = pd.read_csv('data/test.csv', header=0)
predictions['Survived'] = 0
predictions = process(predictions)
predictions = predictions[predictions.columns[1:]]
predictions = predictions.fillna(predictions.mean())
f = open('data/predictions.csv', 'wb')
for index, row in predictions.iterrows():
	f.write(str(int(row['PassengerId'])))
	f.write(',')
	f.write(str(logreg.predict(row)[0]))
	f.write('\n')
f.close()
