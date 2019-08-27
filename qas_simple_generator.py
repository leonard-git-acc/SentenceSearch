import os
import json
import math
import time
import numpy as np
from word2vec import WordVectors
from preprocessing import doc_padding, sentence_padding, vectorize_sentences, shuffle_in_unison, checksum

def create_qas_simple_generator(inputPath, batchSize=32, mode="train"):
    interFile = open(inputPath, "r")
    interJSON = json.load(interFile)

    word_vectors = WordVectors()

    while True: 
        rawBatch = []

        data = np.zeros((batchSize, 3, word_vectors.vector_size))
        labels = np.zeros((batchSize, 2))

        batchCount = 0
        totalCount = 0
        trainSwitch = False
        
        for paragraph in interJSON:
            doc = paragraph["doc"]
            docText = word_vectors._reconstruct_string(doc)

            for qa in paragraph["qas"]:
                question = qa["question"]
                answer = qa["answerSentence"]

                collection = [question, docText]
                collection.extend(doc)

                for i, sentence in enumerate(doc):
                    #balances trainig data: 50% positives and 50% negatives
                    if mode == "train":
                        if trainSwitch and answer != i:
                            continue

                    trainSwitch = not trainSwitch

                    rawBatch.extend([question, sentence, docText]) #collect all samples as strings
                    labels[batchCount][int(answer == i)] = 1

                    batchCount = batchCount + 1
                    totalCount = totalCount + 1

                    if not (batchCount < batchSize):
                        result = word_vectors.vectorize_strings(rawBatch) #embedd all samples at once, to improve performance

                        for i in range(len(result) // 3):
                            if i == 0:
                                data[i] = np.array(result[0:(i + 1) * 3])
                            else:
                                data[i] = np.array(result[i * 3:(i + 1) * 3])
                        
                        if mode == "train":
                            shuffle_in_unison(data, labels)
                            #print(checksum(data))
                            yield (data, labels)
                        elif mode == "eval":
                            print("TestBatch: " + str(totalCount))
                            yield (data, labels)
                        
                        data = np.zeros((batchSize, 3, word_vectors.vector_size))
                        labels = np.zeros((batchSize, 2))
                        rawBatch = []
                        batchCount = 0