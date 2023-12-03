import nltk
from gensim.models.word2vec import Word2Vec

nltk.download('punkt')
nltk.download('gutenberg')
from nltk.corpus import gutenberg

print(len(gutenberg.fileids()))

gberg_sents = gutenberg.sents()
model = Word2Vec(sentences=gberg_sents, vector_size=64, sg=1, window=10, min_count=5, workers=24)
model.save('raw_gutenberg_model.w2v')

print("Done training model.")

print(model.wv.most_similar('dog'))
print(
    f'mother father daughter dog, the does not match word is {model.wv.doesnt_match("mother father daughter dog".split())}')

similar = model.wv.most_similar(positive=['husband', 'woman'], negative=['man'])
print(f'husband - man + woman is {similar}')

