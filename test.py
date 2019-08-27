from word2vec import WordVectors

word_vec = WordVectors()

string = "This is a very interesting program!"

res = word_vec([string])
print(res)
print(res.shape)