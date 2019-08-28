"""
Parses the SQuAD dataset to an easier to understand format. It aims at finding the correct sentence,
instead of the correct span in the text and therefore removes answer_start and answer completely and
replaces it with answerSentence, which is a index of the correct sentence.
"""

import os
import json
from nltk.tokenize import sent_tokenize
from preprocessing import get_sentence_index

INPUT_PATH = "./train-v1.json" #Path to squad json file
OUT_PATH = "./train.json" #output for parsed json file
DOC_SIZE = 16 #how many sentences per "doc"

def main():
    squadFile = open(INPUT_PATH, "r")
    squadJSON = json.load(squadFile)
    data = []

    total_texts = [] #list of all topics
    total_qas = [] #list of all questions and answers of all topics

    for dataSQUAD in squadJSON["data"]: 
        topic_text = [] #list of all sentences in a topic
        topic_qas = [] #list of all questions and answers in a topic 

        for paragraphSQUAD in dataSQUAD["paragraphs"]:
            context = paragraphSQUAD["context"]
            sentences = sent_tokenize(context)
            for qaSQUAD in paragraphSQUAD["qas"]:
                questionStr = qaSQUAD["question"]
                answerStart = qaSQUAD["answers"][0]["answer_start"]

                answerSentence = get_sentence_index(sentences, answerStart)
                answerSentence = len(topic_text) + answerSentence
                qa = {"question": questionStr, "answerSentence": answerSentence}
                topic_qas.append(qa)

            topic_text.extend(sentences)

        total_texts.append(topic_text)
        total_qas.append(topic_qas)

    #Create docs, that have DOC_SIZE sentences within them
    for topic_text, topic_qas in zip(total_texts, total_qas):
        obj = {}
        for i in range(len(topic_text) // DOC_SIZE + DOC_SIZE):
            low = i * DOC_SIZE
            high = low + DOC_SIZE

            doc = topic_text[low:high]
            qas = []
            if doc:
                for qa in topic_qas:
                    ans = qa["answerSentence"]
                    if ans >= low and ans < high:
                        qa["answerSentence"] = ans - low
                        qas.append(qa)
                obj = {"doc": doc, "qas": qas}
                data.append(obj)

    outFile = open(OUT_PATH, "w")
    json.dump(data, outFile)


if __name__ == "__main__":
    main()
