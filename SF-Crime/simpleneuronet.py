from numpy import ravel
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
from pybrain.utilities           import percentError
from pybrain.structure import LinearLayer, SigmoidLayer, FeedForwardNetwork, FullConnection
import pandas as pd
from dateutil.parser import parse
from sklearn.preprocessing import normalize
from random import sample

crimes = pd.read_csv("train.csv")
crimes = crimes.loc[sample(crimes.index, 30000)]

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
# X = normalize(X, axis=0)
y = crimes['category_ids']

print "making net"
ds = ClassificationDataSet(35, 1 , nb_classes=39)
for k in xrange(len(X)): 
    ds.addSample(X.iloc[[k]],y.iloc[[k]])
print "cleaning data"
tstdata, trndata = ds.splitWithProportion( 0.5 )
trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

print "training"
hidden_layer = int((trndata.indim + trndata.outdim) / 2)

fnn = FeedForwardNetwork()
inLayer = LinearLayer(trndata.indim)
outLayer = SoftmaxLayer(trndata.outdim)

prev = None

fnn.addInputModule(inLayer)
for i in range(30):
	hiddenLayer = SigmoidLayer(hidden_layer)
	fnn.addModule(hiddenLayer)
	if i == 0:
		fnn.addConnection(FullConnection(inLayer, hiddenLayer))
	else:
		fnn.addConnection(FullConnection(prev, hiddenLayer))
	prev = hiddenLayer
fnn.addOutputModule(outLayer)
fnn.addConnection(FullConnection(prev, outLayer))
fnn.sortModules()
print fnn

trainer = BackpropTrainer(fnn, dataset=trndata, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01) 

print "WIP"
for i in range(20):
	print i
	trainer.trainEpochs (10)
	trnresult = percentError( trainer.testOnClassData(),trndata['class'] )
	tstresult = percentError( trainer.testOnClassData(dataset=tstdata ), tstdata['class'] )
	print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult

