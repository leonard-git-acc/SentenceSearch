from gensim.models import Word2Vec, KeyedVectors
from nltk.tokenize import word_tokenize
import numpy as np

def load_word_vectors(path, binary=True):
    word_vectors = KeyedVectors.load_word2vec_format(path, binary=binary)
    return word_vectors

def vectorize_string(vectors, string):
    words = word_tokenize(string)
    words = list(filter(lambda x: x in vectors, words))
    
    sentence = []
    for w in words:
        vec = vectors[w]
        sentence.extend(vec)

    return sentence

def get_vector_shape(vectors):

    return item.shape
