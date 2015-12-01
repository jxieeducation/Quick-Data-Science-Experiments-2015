from pybrain.tools.shortcuts import buildNetwork
net = buildNetwork(2, 3, 1)
print net.activate([2, 1])

print net['in']
print net['hidden0']
print net['out']

###########
from pybrain.structure import TanhLayer
net = buildNetwork(2, 3, 1, hiddenclass=TanhLayer)
print net['hidden0']

###########

from pybrain.structure import SoftmaxLayer
net = buildNetwork(2, 3, 2, hiddenclass=TanhLayer, outclass=SoftmaxLayer)
print net.activate((2, 3))

###########
###########
###########
