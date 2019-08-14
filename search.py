import os
import json
import numpy as np
import word2vec
import tensorflow as tf
from preprocessing import sentence_padding
from nltk.tokenize import sent_tokenize

DOC_DIR = "C:/Users/leona/Downloads/dateien/datasearch1/ee_to_trax/storage/content"
KEYEDVECTORS_PATH = "./data/german.model"
MODEL_PATH = ""

DOC_SIZE = 16
SENTENCE_SIZE = 24

def main():
    files = os.listdir(DOC_DIR)
    word_vec = word2vec.load_word_vectors(KEYEDVECTORS_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Question: ")
    question = input()
    quesVec = word2vec.vectorize_string(word_vec, question)
    quesVec = sentence_padding(quesVec, SENTENCE_SIZE, word_vec.vector_size)

    docNames = [] #document names
    docAnswerSentences = [] #index of the sentence with highest value in doc
    docAnswerValues = [] #value of each sentence

    for name in files:
        if ".meta.npy" in name:
            docVec = np.load(os.path.join())
            res = evaluate_doc(quesVec, docVec, model)
            
            argmax = np.argmax(res)
            val = res[argmax]

            docNames.append(name.replace(".meta.npy", ""))
            docAnswerSentences.append(argmax)
            docAnswerValues.append(val)

    docAnswerValues = np.array(docAnswerValues)
    result = np.argmax(docAnswerValues)

    name = docNames[result]
    sentenceIndex = docAnswerSentences[result]

    targetFile = open(os.path.join(DOC_DIR, name), "r")
    targetJSON = json.load(targetFile)
    text = targetJSON["content"]
    sentences = sent_tokenize(text)
    
    print("Answer: " + sentences[sentenceIndex])


def evaluate_doc(quesVec, docVec, model):
    predictions = np.zeros(docVec.shape[0])
    docVecFlat = docVec.flatten()
    for i, senVec in enumerate(docVec):
         sample = np.concatenate([quesVec, senVec, docVecFlat])
         pred = model.predict(sample)
         predictions[i] = pred
    return predictions

if __name__ == "__main__":
    main()