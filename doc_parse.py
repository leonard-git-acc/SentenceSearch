import os
import json
import random
from nltk.tokenize import sent_tokenize

DOC_DIR = "C:/Users/leona/Downloads/dateien/datasearch1/ee_to_trax/storage/content"
OUT_PATH = "./data/documents.json"

def main():
    files = os.listdir(DOC_DIR)

    data = []
    for name in files:
        try:
            f = open(os.path.join(DOC_DIR, name), "r")
            obj = json.load(f)
    
            title = get_title(obj)
            doc, answer = get_sentences(obj)
            qas = [{"question": title, "answerSentence": answer}]
            data.append({"doc": doc, "qas": qas})

        except Exception as e:
            print(e)
            print("JSON parsing failed!")
    
    out = open(OUT_PATH, "w")
    json.dump(data, out)


def get_title(obj):
    fields = obj["fields"]
    for field in fields:
        if field["name"] == "title":
            return field["value"]


def get_sentences(obj):
    text = obj["content"].replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    sentences = sent_tokenize(text)
    rndNum = random.randint(0, len(sentences) - 1)

    buffer = sentences[0]
    sentences[0] = sentences[rndNum]
    sentences[rndNum] = buffer

    return (sentences, rndNum)
    

if __name__ == "__main__":
    main()