import os
import json
import word2vec
import math
import numpy as np
from preprocessing import doc_padding, sentence_padding, vectorize_sentences, shuffle_in_unison, checksum

def create_qas_generator(inputPath, keyedVectorsPath, maxDocumentSentences, maxSentenceWords, batchSize=32, mode="train"):
    interFile = open(inputPath, "r")
    interJSON = json.load(interFile)

    word_vectors = word2vec.load_word_vectors(keyedVectorsPath)
    word_amount = maxDocumentSentences * maxSentenceWords + maxSentenceWords * 2
    
    while True: 
        data = np.zeros((batchSize, word_amount, word_vectors.vector_size))
        labels = np.zeros((batchSize, 2))

        batchCount = 0
        totalCount = 0
        trainSwitch = False
        
        for paragraph in interJSON:
            doc = paragraph["doc"]

            docVec = vectorize_sentences(doc, word_vectors) #3D array (sentences, words, vector_size)
            docVec = doc_padding(docVec, maxDocumentSentences, maxSentenceWords, word_vectors.vector_size) #3D array padded
            docVecFlat = docVec.reshape(maxDocumentSentences * maxSentenceWords, word_vectors.vector_size)

            for qa in paragraph["qas"]:
                question = qa["question"]
                answer = qa["answerSentence"]

                quesVec = word2vec.vectorize_string(word_vectors, question) #2D array (words, vector_size)
                quesVec = sentence_padding(quesVec, maxSentenceWords, word_vectors.vector_size) #2D array padded

                for i, sentence in enumerate(docVec):
                    #balances trainig data: 50% positives and 50% negatives
                    if mode == "train":
                        if trainSwitch and answer != i:
                            continue

                    trainSwitch = not trainSwitch
                    item = np.concatenate([quesVec, sentence, docVecFlat])
                    
                    data[batchCount] = item
                    labels[batchCount][int(answer == i)] = 1

                    batchCount = batchCount + 1
                    totalCount = totalCount + 1

                    if not (batchCount < batchSize):
                        if mode == "train":
                            shuffle_in_unison(data, labels)
                            #print(checksum(data))
                            yield (data, labels)
                        elif mode == "eval":
                            print("TestBatch: " + str(totalCount))
                            yield (data, labels)
                        
                        data = np.zeros((batchSize, word_amount, word_vectors.vector_size))
                        labels = np.zeros((batchSize, 2))
                        batchCount = 0