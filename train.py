import time
import os
import numpy as np
import tensorflow as tf
from preprocessing import get_sample_shape
from qas_generator import create_qas_generator
from create_model import create_model_nn, create_model_cnn, create_model_lstm

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("keyedvectors_path", None, "Path to keyed vectors file")
flags.DEFINE_string("train_path", None, "Path to train json file")
flags.DEFINE_string("test_path", None, "Path to test json file")
flags.DEFINE_string("out_dir", None, "Output dir of the model and logs")
flags.DEFINE_string("saved_model_path", None, "Saved path for a model to load")

flags.DEFINE_integer("max_sentence_words", 24, "Maximum words a sentence can have")
flags.DEFINE_integer("max_document_sentences", 16, "Maximum sentences a document can have")

flags.DEFINE_integer("batches_per_epoch", 100000, "Steps per epoch")
flags.DEFINE_integer("epochs", 1, "Amount of total epochs")
flags.DEFINE_integer("batch_size", 32, "Amount of samples the generator will create per batch")

flags.DEFINE_boolean("do_train", True, "True, if model should be trained")
flags.DEFINE_boolean("do_test", True, "True, if model should be tested")

NAME = "sentsearch_nn_{}".format(int(time.time()))

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    model = None
    if FLAGS.saved_model_path != None:
        model = tf.keras.models.load_model(FLAGS.saved_model_path)
        model.summary()
    else:
        model = create_model_lstm(get_sample_shape(FLAGS.max_document_sentences, FLAGS.max_sentence_words, 300))

    if not os.path.isdir(FLAGS.out_dir):
        tf.io.gfile.mkdir(FLAGS.out_dir)
    if FLAGS.do_train:
        train_gen = create_qas_generator(
                FLAGS.train_path, 
                FLAGS.keyedvectors_path, 
                FLAGS.max_document_sentences, 
                FLAGS.max_sentence_words, 
                batchSize=FLAGS.batch_size,
                mode="train")

        model.fit_generator(
                train_gen,
                steps_per_epoch=FLAGS.batches_per_epoch,
                epochs=FLAGS.epochs)
                
        model.save(os.path.join(FLAGS.out_dir, NAME + ".model"))
        
    if FLAGS.do_test:
        print("Test started!")
        test_gen = create_qas_generator(
                FLAGS.test_path, 
                FLAGS.keyedvectors_path, 
                FLAGS.max_document_sentences, 
                FLAGS.max_sentence_words, 
                mode="eval")

        val_loss, val_acc = model.evaluate_generator(
                test_gen,
                steps=FLAGS.batches_per_epoch)

        print("Loss: " + str(val_loss))
        print("Accuracy: " + str(val_acc))


if __name__ == "__main__":
    flags.mark_flag_as_required("keyedvectors_path")
    flags.mark_flag_as_required("out_dir")
    tf.app.run(main=main)