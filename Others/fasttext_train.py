from preprocess import preprocess
from gensim.models import FastText
from gensim.models import KeyedVectors

filename = './data.csv'
save_file = './fasttext2word2vec.txt'
sentences, index, ids, label = preprocess(filename)
print(sentences[0])
model = FastText(sentences, size=100, iter=10, min_n = 3, max_n = 6, word_ngrams = 0)
model.wv.save_word2vec_format(save_file, binary=False)