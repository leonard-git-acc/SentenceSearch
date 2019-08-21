import os
import json
import numpy as np
import word2vec
import tensorflow as tf
from preprocessing import doc_padding, sentence_padding, vectorize_sentences
from nltk.tokenize import sent_tokenize

TEST_PATH = "./data/test.json"
KEYEDVECTORS_PATH = "./data/english.bin"
MODEL_PATH = "./output/sentsearch_nn_1566389698.model"

DOC_SIZE = 16
SENTENCE_SIZE = 24

def main():
    testJSON = json.load(open(TEST_PATH, "r"))
    word_vec = word2vec.load_word_vectors(KEYEDVECTORS_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)

    while True:
        print("Question: ")
        question = input()

        find_answers(testJSON, word_vec, model, question)
    

def find_answers(testJSON, word_vec, model, question):
    quesVec = word2vec.vectorize_string(word_vec, question).flatten()
    quesVec = sentence_padding(quesVec, SENTENCE_SIZE, word_vec.vector_size)

    docAnswerSentences = [] #index of the sentence with highest value in doc
    docAnswerValues = [] #value of each sentence

    for i, doc in enumerate(testJSON):
        docVec = vectorize_sentences(doc["doc"], word_vec)
        docVec = doc_padding(docVec, DOC_SIZE, SENTENCE_SIZE, word_vec.vector_size)
        res = evaluate_doc(quesVec, docVec, model)
        print("Doc complete: " + str(i))
            
        argmax = np.argmax(res)
        val = res[argmax]

        docAnswerSentences.append(argmax)
        docAnswerValues.append(val)

        if i > 100:
            break

    docAnswerValues = np.array(docAnswerValues)
    for i in range(5):
        result = np.argmax(docAnswerValues)
        docAnswerValues[result] = -1
        sentenceIndex = docAnswerSentences[result]
        print(f"Answer[{i}]: {testJSON[result]['doc'][sentenceIndex]}")


def evaluate_doc(quesVec, docVec, model):
    predictions = np.zeros(docVec.shape[0])
    docVecFlat = docVec.flatten()
    for i, senVec in enumerate(docVec):
         sample = np.concatenate([quesVec, senVec, docVecFlat])
         sample = np.array([sample])
         pred = model.predict(sample)
         predictions[i] = pred[0]
    return predictions

if __name__ == "__main__":
    main()