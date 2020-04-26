from nltk.corpus import stopwords
import logging
import pandas as pd
import re

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