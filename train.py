import time
import tensorflow as tf
import numpy as np
from qas_generator import create_qas_generator, get_sample_len
from create_model import create_model

INTERMEDIATE_FILE = "./data/intermediate.json"
KEYEDVECTORS_FILE = "./data/GoogleNews-vectors-negative300-SLIM.bin"
MAX_SENTENCE_WORDS = 24
MAX_DOCUMENT_SENTENCES = 16

NAME = "textgen_rnn_{}".format(int(time.time()))
STEPS_PER_EPOCH = 100000
EPOCHS = 4

def main():
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(NAME))

    train_gen = create_qas_generator(INTERMEDIATE_FILE, KEYEDVECTORS_FILE, MAX_DOCUMENT_SENTENCES, MAX_SENTENCE_WORDS)
    model = create_model()

    model.fit_generator(
        train_gen,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=train_gen,
        validation_steps=STEPS_PER_EPOCH,
        epochs=EPOCHS)


if __name__ == "__main__":
    main()