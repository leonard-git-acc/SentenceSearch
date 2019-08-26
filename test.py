import tensorflow as tf
import tensorflow_hub as hub




#  = hub.Module("https://tfhub.dev/google/elmo/2")

def init_graph():
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

def embed(words):
  vectors=sess.run(embedder([words]))
  return vectors[0]


def is_number(s):            #isint isnum
  return isinstance(s,int) or isinstance(s,float) or isinstance(s,str) and s.isdigit() # is number isnumeric
  

print(embed("HELLO").shape)