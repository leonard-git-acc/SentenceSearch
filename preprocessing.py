import word2vec
import numpy as np

def doc_padding(doc, docSize, sentenceSize, vectorSize):
    """
    Padds/Truncates sentences of a document and words of its sentences.
    Parameters:
    -----------
    doc : 3D numpy array (sentences, words, vector_size)
        Takes in a array to be transformed.
        This array represents a document, containing sentences,
        which contain words embedded as vectors.
    docSize : integer
        Target size of the first dimension of 'doc'. 
        Represents amount of sentences in a document. 
    sentenceSize : integer
        PTarget size for the third dimension. 
        Represents amount of words in a sentence.
    vectorSize : integer
        Size of the embedded word vectors.
    Returns:
    --------
    out : 3D numpy array
        Returns 'doc' in the correct size.
    """
    target = np.zeros((docSize, sentenceSize, vectorSize))
    for sent in range(min(len(doc), docSize)):
        sentence = sentence_padding(doc[sent], sentenceSize, vectorSize)
        for word in range(sentenceSize):
            target[sent][word] = sentence[word]

    return target


def sentence_padding(sentence, sentenceSize, vectorSize):
    """
    Padds/Truncates words of a sentence.
    Parameters:
    -----------
    sentence : 2D numpy array (words, vector_size)
        Takes in a array to be transformed.
        This array represents a sentence,
        with words embedded as vectors.
    sentenceSize : integer
        Target size of the first dim of the array. 
        Represents amount of words in a sentence.
    vectorSize : integer
        Size of the embedded word vectors
    Returns:
    --------
    out : 2D numpy array
        Returns 'sentence' in the correct size.
    """
    target = np.zeros((sentenceSize, vectorSize))

    for i in range(min(len(sentence), sentenceSize)):
        target[i] = sentence[i]

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
        res = word2vec.vectorize_string(vectors, sen)
        vec.append(res)
    vec = np.array(vec)
    
    return vec

def get_sentence_index(sentences, charIndex):
    """Finds the index of a sentence by a character index from the previous text"""
    count = 0
    for i, se in enumerate(sentences):
        count = count + len(se)
        if count > charIndex:
            return i
        elif i >= len(sentences) - 1:
            return i


def get_sample_shape(docSize, sentenceSize, vectorSize):
    """Calculates the length of one sample, containing question, possible answer and entire document"""
    return (docSize * sentenceSize + 2 * sentenceSize, vectorSize)


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
