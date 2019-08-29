"""
Embedds training data with the qas_generator and serializes it with numpy for faster training.
IMPORTANT: Creates large files => memory intensive (~600MB). These have to be loaded into memory while training.

TRAINING:
---------
If you want to use the serialized numpy arrays for training, it is necessary to modify train.py
The call to model.fit_generator needs to be replaced with:

        x_train = np.load("[YOURPATH]/train_data.npy")
        y_train = np.load("[YOURPATH]/train_labels.npy")
        
        model.fit(x_train, y_train, FLAGS.batch_size, FLAGS.epochs)
"""

import numpy as np
import tensorflow as tf
from qas_generator import create_qas_generator

INPUT_PATH = "./data/train.json"
OUT_PATH = "./data/train"
BATCH_SIZE = 64
TOTAL_BATCHES = 1500

def main():
    tf.logging.set_verbosity(tf.logging.ERROR)

    generator = create_qas_generator(INPUT_PATH, BATCH_SIZE, "train")

    data = np.zeros((TOTAL_BATCHES, BATCH_SIZE, 4, 128))
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
    data = data.reshape((BATCH_SIZE * TOTAL_BATCHES, 4, 128))
    labels = labels.reshape((BATCH_SIZE * TOTAL_BATCHES, 2))
    print("Saving:")
    np.save(open(OUT_PATH + "_data.npy", "wb"), data)
    np.save(open(OUT_PATH + "_labels.npy", "wb"), labels)

if __name__ == "__main__":
    main()
