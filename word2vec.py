from gensim.models import Word2Vec, KeyedVectors
from nltk.tokenize import word_tokenize
import numpy as np

def load_word_vectors(path, binary=True):
    word_vectors = KeyedVectors.load_word2vec_format(path, binary=binary, unicode_errors='ignore')
    return word_vectors

def vectorize_string(vectors, string):
    words = word_tokenize(string.lower())
    words = list(filter(lambda x: x in vectors, words))
    
    sentence = np.zeros((len(words), vectors.vector_size))
    for i, w in enumerate(words):
        vec = vectors[w]
        sentence[i][:len(vec)] = vec

    return sentence

def devectorize_array(vectors, vectorized_string):
    amount = vectorized_string.shape[0] // vectors.vector_size
    result = []
    for i in range(amount):
        low = i * vectors.vector_size
        high = low + vectors.vector_size
        vec = vectorized_string[low:high]

        result.append(vectors.most_similar(positive=[vec], topn=1))
    return result


#vec = load_word_vectors("./data/german.model")
#out = vec.most_similar(positive=["Koenig", "Frau"], negative=["Mann"])
#print(out)
#print(vectorize_string(vec, "Hallo, mein Name ist").shape)