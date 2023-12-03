import pickle
from pprint import pprint
import numpy as np

data = pickle.load(open("mary.pickle", "rb"))
mary_word_list = data['word_list']
mary_word_dict = data['word_dict']
text_words = data['text_words']

print(text_words)


def cooccurence_matrix(word_dict, text_words, window_size=1):
    vocabulary_size = len(word_dict)
    matrix = np.zeros((vocabulary_size, vocabulary_size), dtype='int')

    for i in range(window_size + 1, len(text_words) - window_size):
        word_id = word_dict[text_words[i]]

        for j in range(i - window_size, i + window_size + 1):
            if j == i:
                continue

            context_id = word_dict[text_words[j]]

            matrix[word_id, context_id] += 1

    return matrix

matrix = cooccurence_matrix(mary_word_dict, text_words)
pprint(matrix)
Prob = matrix/matrix.sum(axis=0)
print(Prob[mary_word_dict['mary'], mary_word_dict['had']])

for i in range(len(text_words)):
    if text_words[i] == 'mary':
        if i != 0:
            print(text_words[i-1], text_words[i], text_words[i+1])
        else:
            print(None, text_words[i], text_words[i+1])