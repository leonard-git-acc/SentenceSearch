import word2vec
import numpy as np

def doc_padding(doc, docSize, sentenceSize, vectorSize):
    """
    Padds/Truncates sentences of a document and words of its sentences.
    Parameters:
    -----------
    doc : 2D numpy array
        Takes in a array to be transformed.
        This array represents a document, containing sentences,
        which contain words embedded as vectors.
    docSize : integer
        Target size of the first dimension of 'doc'. 
        Represents amount of sentences in a document. 
    sentenceSize : integer
        Part of the target size for second dimension. 
        [targetSize = sentenceSize * vectorSize]
        Represents amount of words in a sentence.
    vectorSize : integer
        Size of the embedded word vectors.
    Returns:
    --------
    out : 2D numpy array
        Returns 'doc' in the correct size.
    """
    target = np.zeros((docSize, sentenceSize * vectorSize))
    for i in range(min(len(doc), docSize)):
        sentence = sentence_padding(doc[i], sentenceSize, vectorSize)
        target[i][:len(sentence)] = sentence

    return target


def sentence_padding(sentence, sentenceSize, vectorSize):
    """
    Padds/Truncates words of a sentence.
    Parameters:
    -----------
    sentence : 1D numpy array
        Takes in a array to be transformed.
        This array represents a sentence,
        with words embedded as vectors.
    sentenceSize : integer
        Part of the target size for the array. 
        [targetSize = sentenceSize * vectorSize]
        Represents amount of words in a sentence.
    vectorSize : integer
        Size of the embedded word vectors
    Returns:
    --------
    out : 1D numpy array
        Returns 'sentence' in the correct size.
    """
    targetSize = sentenceSize * vectorSize
    target = sentence

    if len(sentence) < targetSize:
        padding = targetSize - len(sentence)
        target = np.pad(target, (0, padding), 'constant')
    elif len(sentence) > targetSize:
        target = target[:targetSize]

    return target


def vectorize_sentences(sentences, vectors):
    """
    Vectorizes sentences with word2vec.
    Parameters:
    -----------
    sentences : list of strings
    vectors : Word2VecKeyedVectors
    Returns:
    -------
    out : 2D numpy array
        Contains sentences with word embeddings as vectors
    """
    vec = []
    for sen in sentences:
        res = word2vec.vectorize_string(vectors, sen).flatten()
        vec.append(res)

    return np.array(vec)

def get_sentence_index(sentences, charIndex):
    """Finds the index of a sentence by a character index from the previous text"""
    count = 0
    for i, se in enumerate(sentences):
        count = count + len(se)
        if count > charIndex:
            return i
        elif i >= len(sentences) - 1:
            return i


def get_sample_len(docSize, sentenceSize, vectorSize):
    """Calculates the length of one sample, containing question, possible answer and entire document"""
    return sentenceSize * vectorSize * 2 + docSize * sentenceSize * vectorSize


def shuffle_in_unison(a, b):
    """Shuffles two arrays in unison"""
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def checksum(data):
    sum = 0
    for doc in data:
        docsum = 0
        for sample in doc:
            docsum = docsum + np.sum(sample)
        sum = sum + docsum

    return sum
