from nltk.corpus import cmudict
from gensim.models import Word2Vec

dic = dict(cmudict.entries())
model = Word2Vec.load('data/rap.model')
print "here we go~~~~~~"

'''assume that keyword is in dic'''
def get_rhyme_similiarity(keyword, suggestion):
	if not suggestion in dic.keys():
		return 0
	keyword_syl = dic[keyword]
	suggestion_syl = dic[suggestion]

	overlap = len(set(keyword_syl).intersection(suggestion_syl))
	base = overlap / float(len(keyword_syl))
	pair = overlap / float(len(suggestion_syl))
	return base * pair


def suggest_rhyme(keyword):
	if keyword not in dic.keys():
		return []
	suggestions = model.most_similar(keyword, topn=1000)
	suggestion_list = []
	for word, semantic_score in suggestions:
		rhyme_score = get_rhyme_similiarity(keyword, word)
		combined_score = rhyme_score
		if combined_score != 0:
			suggestion_list += [(word, combined_score)]

	return sorted(suggestion_list, reverse=True)
