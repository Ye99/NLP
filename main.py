# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import string
import gzip
from collections import Counter
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import normalize

from pprint import pprint


# %matplotlib inline

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def most_similar(word, embeddings, dictionary, reverse_dictionary, top_k=8):
    valid_word = dictionary[word]
    similarity = cosine_similarity(embeddings, embeddings[valid_word, :].reshape(1, -1))
    nearest = (-similarity).argsort(axis=0)[1:top_k + 1].flatten()
    return reverse_dictionary[nearest]


def evaluate_analogy(question):
    word1, word2, word3, word4 = question

    if word1 not in word_dict or \
       word2 not in word_dict or \
       word3 not in word_dict or \
       word4 not in word_dict:
        return None

    key1 = word_dict[word1]
    key2 = word_dict[word2]
    key3 = word_dict[word3]
    key4 = word_dict[word4]

    vec1 = embeddings[key1, :]
    vec2 = embeddings[key2, :]
    vec3 = embeddings[key3, :]
    vec4 = embeddings[key4, :]

    predict = vec2-vec1+vec3

    sim = np.matmul(predict, embeddings.T)
    nearest = np.argsort(-sim)[:10]

    return word4 in word_list[nearest]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    # Find related words
    word_list, embeddings = pd.read_pickle('polyglot-en.pkl')
    embeddings = normalize(embeddings)
    word_list = np.array(word_list)
    word_dict = dict(zip(word_list, range(embeddings.shape[0])))
    pprint(embeddings.shape)

    pprint(most_similar("queen", embeddings, word_dict, word_list, top_k=20))

    # Find analogies
    # questions = pd.read_table('questions-words.txt', comment=':', sep=' ', header=None)
    # print(questions.shape)
    # results = [evaluate_analogy(questions.iloc[i]) for i in range(1000)]
    # clean_results = [res for res in results if res is not None]
    # accuracy = np.mean(clean_results)
    # print(accuracy)

    # Visualize
    # plt.figure(figsize=(15, 5))
    # plt.imshow(embeddings.T, aspect=300, cmap=cm.jet)
    # plt.xlabel("vocabulary")
    # plt.ylabel("embeddings dimensions")
    # plt.show()

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_only = 500  # Plot only 500 words
    low_dim_embs = tsne.fit_transform(np.array(embeddings)[:plot_only, :])
    labels = [word_list[i] for i in range(plot_only)]

    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.show()

