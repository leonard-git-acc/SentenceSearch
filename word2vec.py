from gensim.models import Word2Vec, KeyedVectors
from nltk.tokenize import word_tokenize
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

class WordVectors:
    def __init__(self):
        self.embedder = hub.Module("https://tfhub.dev/google/nnlm-de-dim128-with-normalization/1")
        self.vector_size = 128
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.tables_initializer())

    def embed_words(self, words):
        vectors = self.session.run(self.embedder(words))
        return vectors[0]

    def vectorize_string(self, string):
        words = word_tokenize(string)
        words = __filter_words__(self, words)
    
        sentence = np.zeros((len(words), self.vector_size))
        for i, w in enumerate(words):
            vec = None
            if w in self:
                vec = self[w]
            else:
                vec = self["unknown"]
            sentence[i][:len(vec)] = vec

        return sentence


def load_word_vectors(path, binary=True):
    word_vectors = KeyedVectors.load_word2vec_format(path, binary=binary, unicode_errors='ignore')
    return word_vectors





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