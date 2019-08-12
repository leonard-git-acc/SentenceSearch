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

    data = []
    labels = []

    for paragraph in interJSON:
        doc = paragraph["doc"]

        docVec = vectorize_sentences(doc, word_vectors)
        docVec = doc_padding(docVec, MAX_DOCUMENT_SENTENCES, MAX_SENTENCE_WORDS, word_vectors.vector_size)
        docFlatVec = docVec.flatten()
        print(docVec.shape)

        for qa in paragraph["qas"]:
            question = qa["question"]
            answer = qa["answerSentence"]

            quesVec = word2vec.vectorize_string(word_vectors, question).flatten()
            quesVec = sentence_padding(quesVec, MAX_SENTENCE_WORDS, word_vectors.vector_size)
            ansVec = docVec[answer]

            for sentence in docVec:
                input = np.concatenate([quesVec, sentence, docFlatVec])
                input = input.astype(np.float32)
                data.append(input)
                print(len(data))
            
            pLabel = np.zeros(MAX_DOCUMENT_SENTENCES, np.integer)
            pLabel[answer] = 1
            labels.extend(list(pLabel))
    
                

def doc_padding(doc, docSize, sentenceSize, vectorSize):
    """Adds padding to a document"""
    target = np.zeros((docSize, sentenceSize * vectorSize))
    for i in range(min(len(doc), docSize)):
        sentence = sentence_padding(doc[i], sentenceSize, vectorSize)
        target[i][:len(sentence)] = sentence

    return target


def sentence_padding(sentence, sentenceSize, vectorSize):
    """Adds padding to a sentence"""
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
        res = word2vec.vectorize_string(vectors, sen).flatten()
        vec.append(res)

    return np.array(vec)


if __name__ == "__main__":
    main()
