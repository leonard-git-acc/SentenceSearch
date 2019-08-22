import os
import json
import numpy as np
import word2vec
import tensorflow as tf
from preprocessing import doc_padding, sentence_padding, vectorize_sentences
from nltk.tokenize import sent_tokenize

TEXT_FILE = "./data/text.txt"
KEYEDVECTORS_PATH = "./data/english.bin"
MODEL_PATH = "./output/sentsearch_nn_1566389698.model"

DOC_SIZE = 16
word_vec = word2vec.load_word_vectors(KEYEDVECTORS_PATH)
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
        if len(doc) >= DOC_SIZE or i >= len(sentVec) - 1:
            padded = doc_padding(np.array(doc), 16, 24, word_vec.vector_size)
            docs.append(padded)
            doc = []

    while True:
        question = input()
        quesVec = word2vec.vectorize_string(word_vec, question).flatten()
        quesVec = sentence_padding(quesVec, 24, word_vec.vector_size)

        print(word2vec.devectorize_array(word_vec, quesVec))

        for i, d in enumerate(docs):
            res = evaluate_doc(quesVec, d, model)
            print("---------")
            print(sentences[DOC_SIZE * i + np.argmax(res)])


def evaluate_doc(quesVec, docVec, model):
    predictions = np.zeros(docVec.shape[0])
    docVecFlat = docVec.flatten()
    for i, senVec in enumerate(docVec):
         sample = np.concatenate([quesVec, senVec, docVecFlat])
         print(word2vec.devectorize_array(word_vec, sample))
         sample = np.array([sample])
         pred = model.predict(sample)
         predictions[i] = pred[0]
    return predictions


if __name__ == "__main__":
    main()