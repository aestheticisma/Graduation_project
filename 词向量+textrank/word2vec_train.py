import logging
from gensim.models import word2vec
import pandas as pd
import re
from nltk.corpus import stopwords

def preprocess(filepath):
    text_list = []
    text_list_n = []
    sen_new = []
    stop_words = stopwords.words('english')
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    test = pd.read_csv(filepath, dtype={'id':str})
    sentences = test['sentence'].values
    index = test['index'].values
    ids = test['id'].values
    label = test['label'].values
    # print(ids)
    # print(len(sentences))
    for sentence in sentences:
        sentence = re.split(r'[\[\]\"\'\,]', sentence)
        sentence = list(filter(lambda s: s and s.strip(), sentence))
        text_list.append(sentence)
    for i in text_list:
        for j in i:
            if j not in stop_words:
                sen_new.append(j)
        text_list_n.append(sen_new)
        sen_new = []
    # print(text_list_n)
    return text_list_n, index, ids, label
# sentences = word2vec.Text8Corpus(wikipath)  # 加载语料
if __name__ == "__main__":
    text_list_n, index, ids, label = preprocess('./test.csv')

    model = word2vec.Word2Vec(text_list_n, size=100, min_count=2)     # 训练
    model.save('./model')      # 模型存储
    y2 = model.wv.similarity('problem', 'bug')
    print(y2)
    for i in model.wv.most_similar('rev'):
        print (i[0],i[1])
    print(text_list_n[-1])
    print(len(index))