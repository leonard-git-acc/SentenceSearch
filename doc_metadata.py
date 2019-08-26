import os
import json
import pickle
import numpy as np
from word2vec import WordVectors
from nltk.tokenize import sent_tokenize
from preprocessing import doc_padding, sentence_padding, vectorize_sentences, shuffle_in_unison

DOC_DIR = "C:/Users/leona/Downloads/dateien/datasearch1/ee_to_trax/storage/content"
KEYEDVECTORS_PATH = "./data/german.model"

DOC_SIZE = 16
SENTENCE_SIZE = 24

def main():
    files = os.listdir(DOC_DIR)
    word_vec = WordVectors

    for name in files:
        try:
            f = open(os.path.join(DOC_DIR, name), "r")
            obj = json.load(f)
            doc = get_vec_doc(obj["content"], word_vec)

            np.save(os.path.join(DOC_DIR, name + ".meta"), doc)

        except Exception as e:
            print(e)
            print("JSON parsing failed!")


def get_vec_doc(text, vectors):
    text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    sentences = sent_tokenize(text)
    sentences = vectors.vectorize_sentences(sentences, vectors)
    doc = doc_padding(sentences, DOC_SIZE, SENTENCE_SIZE, vectors.vector_size)

    return doc

if __name__ == "__main__":
    main()