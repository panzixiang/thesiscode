from gensim import corpora, models, similarities
from itertools import chain
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer


""" DEMO """
documents = ['We give necessary and sufficient conditions for the (bounded) law of theiterated logarithm for $U$-statistics in Hilbert spaces. As a tool we alsodevelop moment and tail estimates for canonical Hilbert-space valued$U$-statistics of arbitrary order, which are of independent interest.', 'Generalization of the Kac integral and Kac method for paths measure based onthe Levy distribution has been used to derive fractional diffusion equation. Application to nonlinear fractional Ginzburg-Landau equation is discussed.']

#remove common words and tokenize
stoplist = get_stop_words('en')
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]
tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(texts)

# remove words that appear only once
all_tokens = sum(texts, [])
#tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
#texts = [[word for word in text if word not in tokens_once] for text in texts]

# Create Dictionary.
id2word = corpora.Dictionary()
# Creates the Bag of Word corpus.
mm = [id2word.doc2bow(text) for text in texts]

# Trains the LDA models.
lda = models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=3, passes=10)

# Prints the topics.
for top in lda.print_topics():
  print top
print

# Assigns the topics to the documents in corpus
lda_corpus = lda[mm]

# Find the threshold, let's set the threshold to be 1/#clusters,
# To prove that the threshold is sane, we average the sum of all probabilities:
scores = list(chain(*[[score for topic_id,score in topic] \
                      for topic in [doc for doc in lda_corpus]]))
threshold = sum(scores)/len(scores)
print threshold
print

cluster1 = [j for i,j in zip(lda_corpus,documents) if i[0][1] > threshold]
cluster2 = [j for i,j in zip(lda_corpus,documents) if i[1][1] > threshold]
cluster3 = [j for i,j in zip(lda_corpus,documents) if i[2][1] > threshold]

print cluster1
print cluster2
print cluster3
