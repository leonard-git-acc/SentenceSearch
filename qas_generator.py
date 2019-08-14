import os
import json
import word2vec
import numpy as np
from preprocessing import doc_padding, sentence_padding, vectorize_sentences, shuffle_in_unison, get_sample_len, checksum

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
                            #print(checksum(data))
                            yield (data, labels)
                        elif mode == "eval":
                            yield (data, labels)
                        
                        data = np.zeros((batchSize, inputSize))
                        labels = np.zeros(batchSize)
                        batchCount = 0