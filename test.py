import word2vec

vec = word2vec.load_word_vectors("./data/english.bin")
string = "Marchioness had been hired for the evening for a birthday party and had about 130 people on board, four of whom were crew and bar staff."
res = word2vec.vectorize_string(vec, string).flatten()
print(word2vec.devectorize_array(vec, res))
