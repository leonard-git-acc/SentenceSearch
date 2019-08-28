import os
import json
import numpy as np
from word2vec import WordVectors
import tensorflow as tf
from nltk.tokenize import sent_tokenize

TEXT_FILE = "./text.txt"
MODEL_PATH = "./nlp.model"

word_vec = WordVectors()
def main():
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

        results = []
        for i, sent in enumerate(sentVec):
            res = model.predict(np.array([[quesVec, sent, textVec]]))[0]
            results.append(res[1] - res[0])
        results = np.array(results)
        for i in range(3):
            index = np.argmax(results)
            results[index] = -1 
            print(f"{i}: {sentences[index]}")


if __name__ == "__main__":
    main()