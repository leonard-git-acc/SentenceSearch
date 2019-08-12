import os
import json
import nltk

SQUAD_FILE = "./data/train-v1.1.json"
OUT_FILE = "./data/intermediate.json"

def main():
    squadFile = open(SQUAD_FILE, "r")
    squadJSON = json.load(squadFile)
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    data = []

    for dataSQUAD in squadJSON["data"]:
        for paragraphSQUAD in dataSQUAD["paragraphs"]:
            context = paragraphSQUAD["context"]
            sentences = tokenizer.tokenize(context)
            qas = []
            for qaSQUAD in paragraphSQUAD["qas"]:
                questionStr = qaSQUAD["question"]
                answerStr = qaSQUAD["answers"][0]["text"]
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

    outFile = open(OUT_FILE, "w")
    json.dump(data, outFile)

def get_sentence_index(sentences, answerStart):
    count = 0
    for i, se in enumerate(sentences):
        count = count + len(se)
        if count > answerStart:
            return i



if __name__ == "__main__":
    main()
