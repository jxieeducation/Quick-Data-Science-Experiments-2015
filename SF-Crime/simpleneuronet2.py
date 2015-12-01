import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
from pybrain.utilities           import percentError
from pybrain.structure import LinearLayer, SigmoidLayer, FeedForwardNetwork, FullConnection
from sklearn.preprocessing import normalize
from random import sample

training = pd.read_csv('train.csv', parse_dates = ["Dates"])
training = training.loc[sample(training.index, 30000)]

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

crimes = OHE_crime(training)

print "making dataset"
ds = ClassificationDataSet(68, 1 , nb_classes=39)
for k in xrange(len(crimes)): 
    print k
    ds.addSample(crimes.iloc[[k]], crime_labels[k])
tstdata, trndata = ds.splitWithProportion( 0.5 )
trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()

print "making net"
hidden_layer = int((trndata.indim + trndata.outdim) / 2)
fnn = buildNetwork(trndata.indim, hidden_layer, trndata.outdim, bias=True, outclass=SoftmaxLayer)
print fnn

trainer = BackpropTrainer(fnn, dataset=trndata, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01) 

print "WIP"
for i in range(100):
    print i
    trainer.trainEpochs (10)
    trnresult = percentError( trainer.testOnClassData(),trndata['class'] )
    tstresult = percentError( trainer.testOnClassData(dataset=tstdata ), tstdata['class'] )
    print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult

