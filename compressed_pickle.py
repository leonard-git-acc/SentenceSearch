import pickle
import gzip


def save(object, filename, bin = 1):
    file = gzip.GzipFile(filename, 'wb')
    pickle.dump(object, file, bin)
    file.close()



def load(filename):
    file = gzip.GzipFile(filename, 'rb')
    object = pickle.load(file)
    file.close()

    return object
