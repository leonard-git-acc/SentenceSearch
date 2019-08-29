"""Searching in unknown text files"""

import os
import json
import numpy as np
from word2vec import WordVectors
import tensorflow as tf
from nltk.tokenize import sent_tokenize

TEXT_FILE = "./data/text.txt"
MODEL_PATH = "./output/sentsearch_gru_1567070506.model"


def main():
    word_vec = WordVectors()
    model = tf.keras.models.load_model(MODEL_PATH)

    f = open(TEXT_FILE, "r")
    text = f.read()
    textVec = word_vec.vectorize_string(text)
    sentences = sent_tokenize(text)
    sentVec = word_vec.vectorize_strings(sentences)

    while True:
        print("Question:")
        question = input()
        quesVec = word_vec.vectorize_string(question)

        active = 0
        for i in range(len(sentVec) - 1):
            sample = [[quesVec, textVec, sentVec[active], sentVec[i + 1]]]
            res = model.predict(np.array(sample))[0]
            if res[0] < res[1]:
                active = i + 1
        print(sentences[active])

        active = len(sentVec) - 1
        for i in range(len(sentVec), -1, -1):
            sample = [[quesVec, textVec, sentVec[active], sentVec[i - 1]]]
            res = model.predict(np.array(sample))[0]
            if res[0] < res[1]:
                active = i - 1
        print(sentences[active])




if __name__ == "__main__":
    main()