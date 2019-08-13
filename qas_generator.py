import os
import json
import word2vec
import numpy as np
import compressed_pickle as pickle

def create_qas_generator(inputPath, keyedVectorsPath, maxDocumentSentences, maxSentenceWords, batchSize=32, mode="train"):
    interFile = open(inputPath, "r")
    interJSON = json.load(interFile)

    word_vectors = word2vec.load_word_vectors(keyedVectorsPath)
    inputSize = get_sample_len(maxDocumentSentences, maxSentenceWords, word_vectors.vector_size)
    
    while True: 
        data = np.zeros((batchSize, inputSize))
        labels = np.zeros(batchSize)
        batchCount = 0
        totalCount = 0
        
        for paragraph in interJSON:
            doc = paragraph["doc"]

            docVec = vectorize_sentences(doc, word_vectors)
            docVec = doc_padding(docVec, maxDocumentSentences, maxSentenceWords, word_vectors.vector_size)
            docFlatVec = docVec.flatten()

            for qa in paragraph["qas"]:
                question = qa["question"]
                answer = qa["answerSentence"]

                quesVec = word2vec.vectorize_string(word_vectors, question).flatten()
                quesVec = sentence_padding(quesVec, maxSentenceWords, word_vectors.vector_size)

                for i, sentence in enumerate(docVec):
                    data[batchCount] = np.concatenate([quesVec, sentence, docFlatVec])
                    labels[batchCount] = int(answer == i)
                    batchCount = batchCount + 1
                    totalCount = totalCount + 1

                    if not (batchCount < batchSize):
                        if mode == "train":
                            shuffle_in_unison(data, labels)
                            yield (data, labels)
                        elif mode == "eval":
                            yield (data, labels)
                        
                        data = np.zeros((batchSize, inputSize))
                        labels = np.zeros(batchSize)
                        batchCount = 0
                                     

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


def get_sample_len(docSize, sentenceSize, vectorSize):
    return sentenceSize * vectorSize * 2 + docSize * sentenceSize * vectorSize

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)