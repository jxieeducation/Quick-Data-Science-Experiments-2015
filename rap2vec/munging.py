from nltk.corpus import cmudict
dic = dict(cmudict.entries())
dic_vocab = dic.keys()
import sys
from os import listdir
from os.path import isfile, join
import cPickle as pickle

def getListOfRhymes(lines):
	list_of_rhymes = []

	current_list_of_rhymes = []
	last_stresses = set()
	for line in lines:
		word = line.split()
		if len(word) > 1 and word[-1] in dic_vocab:
			word = word[-1]
			syls = dic[word]
			stresses = set([syl for syl in syls if syl[-1] in '0123456789'])
			if not stresses.intersection(last_stresses):
				list_of_rhymes.append(current_list_of_rhymes)
				current_list_of_rhymes = []
			last_stresses = stresses
			current_list_of_rhymes += [word]
		else:
			list_of_rhymes.append(current_list_of_rhymes)
			current_list_of_rhymes = []
			last_stresses = []
	list_of_rhymes.append(current_list_of_rhymes)
	list_of_rhymes = [rhymes for rhymes in list_of_rhymes if rhymes and len(rhymes) > 1]
	return list_of_rhymes

datadir = "data/cleaned-rap-lyrics/"
files = [join(datadir, f) for f in listdir(datadir) if isfile(join(datadir, f)) and ".txt" in f]
training_set = []
count = 0 
for rap_file in files:
	rap_file = open(rap_file)
	lines = rap_file.read().split('\n')
	training_set += getListOfRhymes(lines)
	print str(count) + "/" + str(len(files))
	count += 1

pickle.dump(training_set, open('data/munged_dataset.pickle', 'wb'))
