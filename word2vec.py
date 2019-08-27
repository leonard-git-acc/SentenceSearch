from gensim.models import Word2Vec, KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

class WordVectors:
    def __init__(self):
        self.embedder = hub.Module("https://tfhub.dev/google/nnlm-de-dim128-with-normalization/1")
        self.stop_words = set(stopwords.words("english"))
        self.vector_size = 128
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.tables_initializer())


    def __call__(self, words):
        """Embedds a list of words as vector array"""
        vectors = self.session.run(self.embedder(words))
        return vectors


    def vectorize_string(self, string):
        """Embedds and preprocesses a string as vector array"""
        words = word_tokenize(string)
        words = self._filter_words(words)
    
        vec = self(words)
        
        return vec

    def vectorize_sentences(self, sentences):
        """Embedds and preprocesses a list of strings as list of vector arrays."""
        words = []
        word_count = []

        for sen in sentences: 
            w = word_tokenize(sen)
            w = self._filter_words(w)

            words.extend(w) #collect all words in list
            word_count.append(len(w)) #save sentence word count after filter

        vec = self(words)
        senVec = []

        for i, val in enumerate(word_count): #reconstruct the sentences
            if i == 0: 
                senVec.append(vec[0:val])
            else:
                senVec.append(vec[word_count[i - 1]:val])

        return senVec


    def _filter_words(self, words):
        for i, w in enumerate(words):

            if w in ".:-_,;#'+*/()[]{}&$%§=":        #filter symbols
                words[i] = ""
            if w in self.stop_words:                #filter stop words
                words[i] = ""
            if any(char.isdigit() for char in w):   #filter numbers
                words[i] = "[NUM]"
            if w == "?":                            #mark question ending
                words[i] = "[QES]"
            if w == "." or w == "!":                #mark sentence ending
                words[i] = "[SEP]"
            
            words[i] = words[i].lower()
        
        words = list(filter(lambda x: x != "", words))
        return words



#vec = load_word_vectors("./data/german.model")
#out = vec.most_similar(positive=["Koenig", "Frau"], negative=["Mann"])
#print(out)
#print(vectorize_string(vec, "Hallo, mein Name ist").shape)