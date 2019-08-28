"""Word/Sentence/Text embeddings as vectors"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

class WordVectors:
    def __init__(self):
        self.module = hub.Module("https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1")
        self.placeholder = tf.placeholder(tf.string, [None])
        self.embedder = self.module(self.placeholder)

        self.stop_words = set(stopwords.words("english"))
        self.vector_size = 128
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.tables_initializer())


    def __call__(self, strings):
        """Embedds a list of strings as vector array"""
        vectors = self.session.run(self.embedder, feed_dict={self.placeholder: strings})
        return vectors


    def vectorize_string(self, string):
        """Embedds and preprocesses a string as vector"""
        filtered = self._filter_string(string)

        vec = self([filtered])[0]
        
        return vec


    def vectorize_strings(self, strings):
        """Embedds and preprocesses a list of strings as list of vectors."""
        filtered = []
        for s in strings:
            filtered.append(self._filter_string(s))

        vecs = self(filtered)

        return vecs


    def _filter_string(self, string):
        words = word_tokenize(string)

        for i, w in enumerate(words):
            if w in ".:-_,;#'+*/()[]{}&$%§´`'=":       #filter symbols
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
        string = self._reconstruct_string(words)

        return string


    def _reconstruct_string(self, words):
        result = ""
        for i, w in enumerate(words):
            if i == 0:
                result = w
            else:
                result = result + " " + w
        return result