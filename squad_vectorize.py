import os
import json
import pickle
from word2vec import WordVectors

IN_PATH = "./data/train.json"
OUT_PATH = "./data/train.pkl"

word_vec = WordVectors()

f = open(IN_PATH, "r")
train = json.load(f)

data = []

for i, paragraph in enumerate(train):
    doc = paragraph["doc"]
    qas = paragraph["qas"]

    docVec = word_vec.vectorize_sentences(doc)
    qasVec = []

    for qa in qas:
        vec = word_vec.vectorize_string(qa["question"])
        qasVec.append({"question": vec, "answerSentence": qa["answerSentence"]})

    data.append({"doc": docVec, "qas": qasVec})
    print(i)

f = open(OUT_PATH, "wb")

pickle.dump(data, f)

