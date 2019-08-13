import time
import os
import numpy as np
import tensorflow as tf
from qas_generator import create_qas_generator, get_sample_len
from create_model import create_model

TRAIN_FILE = "./data/train.json"
TEST_FILE = "./data/test.json"
KEYEDVECTORS_FILE = "./data/GoogleNews-vectors-negative300-SLIM.bin"
SAVE_DIR = "./models"
MAX_SENTENCE_WORDS = 24
MAX_DOCUMENT_SENTENCES = 16

NAME = "sentsearch_nn_{}".format(int(time.time()))
STEPS_PER_EPOCH = 100000
EPOCHS = 4
BATCH_SIZE = 32

def main():
    log_dir = os.path.join("logs", "train", NAME)

    tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                batch_size=BATCH_SIZE)

    train_gen = create_qas_generator(
            TRAIN_FILE, 
            KEYEDVECTORS_FILE, 
            MAX_DOCUMENT_SENTENCES, 
            MAX_SENTENCE_WORDS,
            batchSize=BATCH_SIZE,
            mode="train")

    test_gen = create_qas_generator(
            TEST_FILE, 
            KEYEDVECTORS_FILE, 
            MAX_DOCUMENT_SENTENCES, 
            MAX_SENTENCE_WORDS, 
            mode="eval")

    model = create_model()

    model.fit_generator(
        train_gen,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=train_gen,
        validation_steps=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        callbacks=[tensorboard])

    model.save(os.path.join(SAVE_DIR, NAME + ".model"))

    val_loss, val_acc = model.evaluate_generator(
            test_gen,
            steps=STEPS_PER_EPOCH,
            callbacks=[tensorboard])

    print("Loss: " + val_loss)
    print("Accuracy: " + val_acc)

if __name__ == "__main__":
    main()