import numpy as np 
import pickle
import lda

vocab = open("vocab.enron.txt").read().splitlines()
doc_raw = open("docword.enron.txt").read().splitlines()[3:]

kos = []
curr_doc = 1
curr_array = np.zeros(len(vocab), dtype=int)
count = 0

for line in doc_raw:
	fields = line.split(" ")
	doc_num = int(fields[0])
	vocab_num = int(fields[1]) - 1
	freq = int(fields[2])
	if doc_num != curr_doc:
		kos += [curr_array]
		curr_array = np.zeros(len(vocab), dtype=int)
		curr_doc = doc_num
		curr_array[vocab_num] = freq
	else:
		curr_array[vocab_num] = freq
	print count
	count += 1

kos = np.array(kos)

model = lda.LDA(n_topics=5, n_iter=300, random_state=1)
model.fit(kos)
pickle.dump(model, open("enron.model", 'w'))
