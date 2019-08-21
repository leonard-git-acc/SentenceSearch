import os
import json
import sys
from nltk.tokenize import sent_tokenize
from preprocessing import get_sentence_index

DOC_SIZE = 16
SENTENCE_SIZE = 24

def main():
    args = sys.argv
    squad_path = args[1]
    out_path = args[2]

    squadFile = open(squad_path, "r")
    squadJSON = json.load(squadFile)
    data = []

    total_texts = []
    total_qas = []

    for dataSQUAD in squadJSON["data"]: 
        topic_text = []
        topic_qas = []
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

    outFile = open(out_path, "w")
    json.dump(data, outFile)


if __name__ == "__main__":
    main()
