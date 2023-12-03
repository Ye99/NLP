import nltk
from gensim.models.word2vec import Word2Vec
import string
nltk.download('punkt')
nltk.download('gutenberg')
from nltk.corpus import gutenberg
from gensim.models.phrases import Phraser, Phrases

print(len(gutenberg.fileids()))

gberg_sents = gutenberg.sents()

lower_sents = []
for s in gberg_sents:
    lower_sents.append([w.lower() for w in s if w.lower() not in list(string.punctuation)])

# Train bi-gram
lower_bigram = Phraser(Phrases(lower_sents))

clean_sents = []
for s in lower_sents:
    clean_sents.append(lower_bigram[s]) # apply bi-gram.

model = Word2Vec(sentences=clean_sents, vector_size=64, sg=1, window=10, min_count=5, workers=24)
model.save('raw_gutenberg_model.w2v')

print("Done training model.")

print(f'Similar word to dog is: {model.wv.most_similar("dog")}')

print(
    f'mother father daughter dog, the does not match word is {model.wv.doesnt_match("mother father daughter dog".split())}')

similar = model.wv.most_similar(positive=['husband', 'woman'], negative=['man'])
print(f'husband - man + woman is {similar}')

