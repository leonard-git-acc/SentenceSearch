import os
import json
import word2vec
import numpy as np
import compressed_pickle as pickle

KEYEDVECTORS_FILE = "./data/GoogleNews-vectors-negative300-SLIM.bin"
INTERMEDIATE_FILE = "./data/intermediate.json"
OUT_FILE = ""
MAX_SENTENCE_WORDS = 24
MAX_DOCUMENT_SENTENCES = 16

def main():
    interFile = open(INTERMEDIATE_FILE, "r")
    interJSON = json.load(interFile)

    word_vectors = word2vec.load_word_vectors(KEYEDVECTORS_FILE)
    print(type(word_vectors.vocab))
    data = []
    labels = []

    for paragraph in interJSON:
        doc = paragraph["doc"]

        docVec = vectorize_sentences(doc, word_vectors)
        for i in range(len(docVec)):
            docVec[i] = sentence_padding(docVec, MAX_SENTENCE_WORDS, word_vectors.vector_size)

        for qa in paragraph["qas"]:
            question = qa["question"]
            answer = qa["answerSentence"]

            quesVec = word2vec.vectorize_string(word_vectors, question)
            ansVec = docVec[answer]


def sentence_padding(sentence, sentenceSize, vectorSize):
    targetSize = sentenceSize * vectorSize
    target = sentence

    if len(sentence) < targetSize:
        padding = targetSize - len(sentence)
        target = np.pad(target, (0, padding), 'constant')
    elif len(sentence) > targetSize:
        target = target[:targetSize]

    return target


def vectorize_sentences(sentences, vectors):
    vec = []
    for sen in sentences:
        vec.append(word2vec.vectorize_string(vectors, sen))
    return np.array(vec)


if __name__ == "__main__":
    main()
