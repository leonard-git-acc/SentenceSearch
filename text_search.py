import os
import json
import numpy as np
import word2vec
import tensorflow as tf
from preprocessing import doc_padding, sentence_padding, vectorize_sentences
from nltk.tokenize import sent_tokenize

TEXT_FILE = "./data/text.txt"
KEYEDVECTORS_PATH = "./data/english.bin"
MODEL_PATH = "./output/sentsearch_nn_1565784664.model"


def main():
    word_vec = word2vec.load_word_vectors(KEYEDVECTORS_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)

    f = open(TEXT_FILE, "r")
    text = f.read()
    sentences = sent_tokenize(text)
    sentVec = vectorize_sentences(sentences, word_vec)

    docs = []
    doc = []
    for i, sen in enumerate(sentVec):
        doc.append(sen)
        if len(doc) >= 16 or i >= len(sentVec) - 1:
            padded = doc_padding(np.array(doc), 16, 24, word_vec.vector_size)
            docs.append(padded)
            doc = []

    while True:
        question = input()
        quesVec = word2vec.vectorize_string(word_vec, question).flatten()
        quesVec = sentence_padding(quesVec, 24, word_vec.vector_size)

        for d in docs:
            res = evaluate_doc(quesVec, d, model)
            print(sentences[np.argmax(res)])

def evaluate_doc(quesVec, docVec, model):
    predictions = np.zeros(docVec.shape[0])
    docVecFlat = docVec.flatten()
    for i, senVec in enumerate(docVec):
         sample = np.concatenate([quesVec, senVec, docVecFlat])
         pred = model.predict(np.array([sample]))
         predictions[i] = pred
    return predictions
        


if __name__ == "__main__":
    main()