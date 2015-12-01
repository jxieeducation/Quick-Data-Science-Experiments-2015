import csv
import numpy as np
import scipy as sp

from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from pybrain.structure.modules import SoftmaxLayer


def readFile(filename, startCol):
    f = csv.reader(open(filename, 'rb'))
    header = f.next()

    data = []
    for row in f:
        data.append(row[startCol:])
    
    return np.array(data)

def parseRow(row):
    # Cabin class
    cabin_class = row[0]
    # Sex
    sex = 0 if row[2] == 'female' else 2
    # Age
    age = 1000 if row[3] == '' else row[3]
    # Number of siblings/spouse on board
    siblings_spouse = 0 if row[4] == '' else row[4]
    # Number of parents/children on board
    parents_children = 0 if row[5] == '' else row[5]
    # Fare
    fare = 0 if row[7] == '' else row[7]
    # Point of embarcation
    embarcation = 0
    if row[9] == 'S':
        embarcation = 1
    elif row[9] == 'Q':
        embarcation = 2
    elif row[9] == 'C':
        embarcation = 3

    return [cabin_class, sex, age, siblings_spouse, parents_children, fare, embarcation]


def constructDataset(data):
    dataset = ClassificationDataSet(7)
    for row in data:
        dataset.addSample(parseRow(row[1:]), row[0])

    return dataset
    
def main():
    training_data = readFile('data/train.csv', 1)
    test_data = readFile('data/test.csv', 0)

    dataset = constructDataset(training_data)
    tstdata, trndata = dataset.splitWithProportion( 0.25 )
    network = buildNetwork(7, 10, 1, outclass=SoftmaxLayer)
    trainer = BackpropTrainer(network, trndata)
    for i in range(100):
        trainer.trainEpochs(1)
        trnresult = percentError(trainer.testOnClassData(),
                                 trndata['class'])
        tstresult = percentError(trainer.testOnClassData(
                                 dataset=tstdata), tstdata['class'])
        print("epoch: %4d" % trainer.totalepochs,
              "  train error: %5.2f%%" % trnresult,
              "  test error: %5.2f%%" % tstresult)


if __name__ == '__main__':
    main()