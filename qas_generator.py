import os
import json
import numpy as np
from word2vec import WordVectors
from preprocessing import shuffle_in_unison, checksum

def create_qas_generator(inputPath, batchSize=32, mode="train"):
    interFile = open(inputPath, "r")
    interJSON = json.load(interFile)

    word_vectors = WordVectors()
    sample_size = 4

    while True: 
        rawBatch = []
        data = np.zeros((batchSize, sample_size, word_vectors.vector_size))
        labels = np.zeros((batchSize, 2))

        batchCount = 0
        totalCount = 0
        balanceSwitch = False
        
        for paragraph in interJSON:
            doc = paragraph["doc"]
            if len(doc) <= 1:
                continue
            docText = word_vectors._reconstruct_string(doc)

            for qa in paragraph["qas"]:
                question = qa["question"]
                answer = qa["answerSentence"]
                random = np.random.randint(0, len(doc))

                while random == answer:
                    random = np.random.randint(0, len(doc))
                
                sentences = [doc[answer], doc[random]]

                rawBatch.extend([question, docText, sentences[int(balanceSwitch)], sentences[int(not balanceSwitch)]]) #collect all samples as strings
                labels[batchCount][int(balanceSwitch)] = 1

                balanceSwitch = not balanceSwitch
                batchCount = batchCount + 1
                totalCount = totalCount + 1
                
                if not (batchCount < batchSize):
                    result = word_vectors.vectorize_strings(rawBatch) #embedd all samples at once, to improve performance

                    for i in range(len(result) // sample_size):
                        data[i] = np.array(result[i * sample_size:(i + 1) * sample_size])
                        
                    if mode == "train":
                        shuffle_in_unison(data, labels)
                        yield (data, labels)
                    elif mode == "eval":
                        print("TestBatch: " + str(totalCount))
                        yield (data, labels)
                        
                    data = np.zeros((batchSize, sample_size, word_vectors.vector_size))
                    labels = np.zeros((batchSize, 2))
                    rawBatch = []
                    batchCount = 0