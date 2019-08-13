from gensim.models import Word2Vec, KeyedVectors
from nltk.tokenize import word_tokenize
import numpy as np

def load_word_vectors(path, binary=True):
    word_vectors = KeyedVectors.load_word2vec_format(path, binary=binary, unicode_errors='ignore')
    return word_vectors

def vectorize_string(vectors, string):
    words = word_tokenize(string)
    words = list(filter(lambda x: x in vectors, words))
    
    sentence = np.zeros((len(words), vectors.vector_size))
    for i, w in enumerate(words):
        vec = vectors[w]
        sentence[i][:len(vec)] = vec

    return sentence

#vec = load_word_vectors("./data/german.model")
#out = vec.most_similar(positive=["Koenig", "Frau"], negative=["Mann"])
#print(out)
#print(vectorize_string(vec, "Hallo, mein Name ist").shape)