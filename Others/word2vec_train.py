import logging
from gensim.models import word2vec
import pandas as pd
import re
from nltk.corpus import stopwords
from preprocess import preprocess

if __name__ == "__main__":
    text_list_n, index, ids, label = preprocess('./data.csv')

    model = word2vec.Word2Vec(text_list_n, size=100, min_count=2)     # 训练
    model.save('./word2vecmodel')      # 模型存储