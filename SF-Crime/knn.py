import numpy as np
import pandas as pd
from dateutil.parser import parse
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn.preprocessing import normalize

''' 
This algorithm does not work. Since we assign elements of a catagory numbers, the classifier infers distance. But id1 is as far away from id2 and id3.

RIP
'''

crimes = pd.read_csv("train.csv")

Categories = list(set(crimes['Category'].values))
PdDistricts = list(set(crimes['PdDistrict'].values))
Resolutions = list(set(crimes['Resolution'].values))

def process(crimes):
	#get date data
	crimes['date_obj'] = crimes['Dates'].apply(lambda date: parse(date))
	crimes['year'] = crimes['date_obj'].apply(lambda date: date.year)
	crimes['month'] = crimes['date_obj'].apply(lambda date: date.month)
	crimes['day'] = crimes['date_obj'].apply(lambda date: date.day)
	crimes['dayofweek'] = crimes['date_obj'].apply(lambda date: date.weekday())
	crimes['hour'] = crimes['date_obj'].apply(lambda date: date.hour)
	crimes['minute'] = crimes['date_obj'].apply(lambda date: date.minute)

	# turn strings into ids
	crimes['category_ids'] = crimes['Category'].apply(lambda item: Categories.index(item))
	for i in range(len(PdDistricts)):
		crimes['p_' + str(i)] = crimes['PdDistrict'].apply(lambda item: 1 if item == PdDistricts[i] else 0)
	for i in range(len(Resolutions)):
		crimes['r_' + str(i)] = crimes['Resolution'].apply(lambda item: 1 if item == Resolutions[i] else 0)

	# removing not floats 
	crimes = crimes.drop(['Dates', 'DayOfWeek', 'Address', 'date_obj', 'Descript'], axis=1)
	crimes = crimes.drop(['Category', 'PdDistrict', 'Resolution'], axis=1)

	return crimes

print "preprocessing"
crimes = process(crimes)
X = crimes.drop(['category_ids'], axis=1)
X = normalize(X, axis=0)
y = crimes['category_ids']

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.90, random_state=0)
X_test = X_test[:5000]
y_test = y_test[:5000]


print "training"
clf = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', weights='distance')
model = clf.fit(X_train, y_train)

# get accuracy
print "getting accuracy"
print model.score(X_test, y_test)

