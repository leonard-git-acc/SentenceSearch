import os
import json
import sys
from nltk.tokenize import sent_tokenize
from preprocessing import get_sentence_index

def main():
    args = sys.argv
    squad_path = args[1]
    out_path = args[2]

    squadFile = open(squad_path, "r")
    squadJSON = json.load(squadFile)
    data = []

    for dataSQUAD in squadJSON["data"]:
        for paragraphSQUAD in dataSQUAD["paragraphs"]:
            context = paragraphSQUAD["context"]
            sentences = sent_tokenize(context)
            qas = []
            for qaSQUAD in paragraphSQUAD["qas"]:
                questionStr = qaSQUAD["question"]
                answerStart = qaSQUAD["answers"][0]["answer_start"]

                answerSentence = get_sentence_index(sentences, answerStart)

                qa = {}
                qa["question"] = questionStr
                qa["answerSentence"] = answerSentence
                qas.append(qa)

            para = {}
            para["doc"] = sentences
            para["qas"] = qas
            data.append(para)

    outFile = open(out_path, "w")
    json.dump(data, outFile)


if __name__ == "__main__":
    main()
