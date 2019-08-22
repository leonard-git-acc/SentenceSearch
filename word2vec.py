from gensim.models import Word2Vec, KeyedVectors
from nltk.tokenize import word_tokenize
import numpy as np

def load_word_vectors(path, binary=True):
    word_vectors = KeyedVectors.load_word2vec_format(path, binary=binary, unicode_errors='ignore')
    return word_vectors

def vectorize_string(vectors, string):
    words = word_tokenize(string)
    words = __filter_words__(vectors, words)
    
    sentence = np.zeros((len(words), vectors.vector_size))
    for i, w in enumerate(words):
        vec = None
        if w in vectors:
            vec = vectors[w]
        else:
            vec = vectors["unknown"]
        sentence[i][:len(vec)] = vec

    return sentence

def devectorize_array(vectors, vectorized_string):
    amount = vectorized_string.shape[0] // vectors.vector_size
    result = []
    for i in range(amount):
        low = i * vectors.vector_size
        high = low + vectors.vector_size
        vec = vectorized_string[low:high]
        word = vectors.most_similar(positive=[vec], topn=1)
        if word[0][1] > 0.0:
            result.append(word[0][0])
    return result

def __filter_words__(vectors, words):
    for i, w in enumerate(words):
        if any(not char.isalpha() for char in w) or w == "and" or w == "a" or w == "of":
            words[i] = ""
        if any(char.isdigit() for char in w):
            words[i] = "number"
        if w == "?":
            words[i] = "question"
        if w == "." or w == "!":
            words[i] = "dot"
        
        words[i] = words[i].lower()

    words = list(filter(lambda x: x != "", words))
    return words



#vec = load_word_vectors("./data/german.model")
#out = vec.most_similar(positive=["Koenig", "Frau"], negative=["Mann"])
#print(out)
#print(vectorize_string(vec, "Hallo, mein Name ist").shape)