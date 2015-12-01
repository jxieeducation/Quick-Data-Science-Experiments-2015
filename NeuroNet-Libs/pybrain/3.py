from pybrain.datasets.classification import ClassificationDataSet
# below line can be replaced with the algorithm of choice e.g.
# from pybrain.optimization.hillclimber import HillClimber
from pybrain.optimization.populationbased.ga import GA
from pybrain.tools.shortcuts import buildNetwork

# create XOR dataset
d = ClassificationDataSet(2)
d.addSample([0., 0.], [0.])
d.addSample([0., 1.], [1.])
d.addSample([1., 0.], [1.])
d.addSample([1., 1.], [0.])
# d.setField('class', [ [0.],[1.],[1.],[0.]])

nn = buildNetwork(2, 3, 5, 9, 5, 3, 1)
# d.evaluateModuleMSE takes nn as its first and only argument
ga = GA(d.evaluateModuleMSE, nn, minimize=True)
for i in range(500):
    nn = ga.learn(0)[0]

print nn.activate([0,0])
print nn.activate([1,0])
print nn.activate([0,1])
print nn.activate([1,1])
