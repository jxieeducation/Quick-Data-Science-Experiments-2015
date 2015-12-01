from __future__ import division
import random
import numpy as np
import scipy.stats as stats
import pylab as pl
import sys
import os


''' this is the distribution '''
p1 = 0.8
num1 = 13
p2 = 0.2
num2 = 19
def getNum():
	num = random.random()
	if num >= p1:
		return num2
	return num1

''' this generates distribution'''

def getDistribution(size=2):
	curr_list = []
	curr_sum = 0
	for i in range(500):
		for j in range(size):
			curr_sum += getNum()
		curr_list += [curr_sum / size]
		curr_sum = 0
	# print curr_list
	return curr_list

def plot(curr_list):
	fig = pl.figure()
	h = sorted(curr_list)
	fit = stats.norm.pdf(h, np.mean(h), np.std(h))  
	pl.plot(h,fit,'-o')
	pl.hist(h,bins=20)
	pl.savefig("size_" + str(size) + ".png", dpi=fig.dpi)

size = 2
while size < 3201:
	plot(getDistribution(size))
	size *= 2
	print size
