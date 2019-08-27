import os
import json
import math
import numpy as np
from word2vec import WordVectors
from preprocessing import doc_padding, sentence_padding, vectorize_sentences, shuffle_in_unison, checksum

def create_qas_simple_generator(inputPath, batchSize=32, mode="train"):
    interFile = open(inputPath, "r")
    interJSON = json.load(interFile)

    word_vectors = WordVectors()

    while True: 
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

                quesVec, docTextVec, *docVecs = word_vectors.vectorize_strings(collection)

                for i, sentence in enumerate(docVecs):
                    #balances trainig data: 50% positives and 50% negatives
                    if mode == "train":
                        if trainSwitch and answer != i:
                            continue

                    trainSwitch = not trainSwitch
                    sample = np.array([quesVec, sentence, docTextVec])
                    
                    data[batchCount] = sample
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
                        
                        data = np.zeros((batchSize, 3, word_vectors.vector_size))
                        labels = np.zeros((batchSize, 2))
                        batchCount = 0