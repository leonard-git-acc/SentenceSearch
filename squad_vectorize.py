import numpy as np
import tensorflow as tf
from qas_simple_generator import create_qas_simple_generator

INPUT_PATH = "./data/train.json"
OUT_PATH = "./data/train"
BATCH_SIZE = 512
TOTAL_BATCHES = 375

tf.logging.set_verbosity(tf.logging.ERROR)

generator = create_qas_simple_generator(INPUT_PATH, BATCH_SIZE, "train")

data = np.zeros((TOTAL_BATCHES, BATCH_SIZE, 3, 128))
labels = np.zeros((TOTAL_BATCHES, BATCH_SIZE, 2))
print("Started:")
for i, batch in enumerate(generator):
    if i < TOTAL_BATCHES:
        data[i] = batch[0]
        labels[i] = batch[1]
        print(f"Batch {i + 1} of {TOTAL_BATCHES} completed!")
    else:
        break
print("Reshaping:")
data.reshape((BATCH_SIZE * TOTAL_BATCHES, 3, 128))
labels.reshape((BATCH_SIZE * TOTAL_BATCHES, 2))
print("Saving:")
np.save(open(OUT_PATH + "_data.npy", "wb"), data)
np.save(open(OUT_PATH + "_labels.npy", "wb"), labels)
