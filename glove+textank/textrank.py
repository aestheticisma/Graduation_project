from gensim.models import word2vec
import pandas as pd
from word2vec_train import preprocess
from itertools import product, count
import numpy as np
import math
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
sentences = []

def Model(modelpath):
    glove_file = './glove.6B.100d.txt'
    tmp_file = "./test_word2vec.txt"
    glove2word2vec(glove_file, tmp_file)
    # model = word2vec.Word2Vec.load(modelpath)  # 加载训练好的模型
    model = KeyedVectors.load_word2vec_format(tmp_file)
    return model

# 传入句子列表  返回句子之间相似度的图
def create_graph(model, word_sent):
    num = len(word_sent)
    board = [[0.0 for _ in range(num)] for _ in range(num)]
    for i, j in product(range(num), repeat=2):      # range(num)未0-15的整数
        if i != j:
            board[i][j] = compute_similarity_by_avg(model, word_sent[i], word_sent[j])
    return board

# 对两个句子求平均词向量
def compute_similarity_by_avg(model, sents_1, sents_2):
    if len(sents_1) == 0 or len(sents_2) == 0:
        return 0.0
    vec1 = model[sents_1[0]]
    for word1 in sents_1[1:]:
        vec1 = vec1 + model[word1]
    vec2 = model[sents_2[0]]
    for word2 in sents_2[1:]:
        vec2 = vec2 + model[word2]
    similarity = cosine_similarity(vec1 / len(sents_1), vec2 / len(sents_2))
    return similarity

# 计算两个向量之间的余弦相似度
def cosine_similarity(vec1, vec2):
    tx = np.array(vec1)
    ty = np.array(vec2)
    cos1 = np.sum(tx * ty)
    cos21 = np.sqrt(sum(tx ** 2))
    cos22 = np.sqrt(sum(ty ** 2))
    cosine_value = cos1 / float(cos21 * cos22)
    return cosine_value

def filter_model(model, sents):
    _sents = []
    for sentence in sents:
        for word in sentence:
            if word not in model.wv.vocab:
                # print('remove' + word)
                sentence.remove(word)       # 剔除没有在训练模型中的字
        if sentence:
            _sents.append(sentence)     # 剔除后加入_sent并返回
    return _sents

# 判断前后分数有无变化
def different(scores, old_scores):

    flag = False
    for i in range(len(scores)):
        if math.fabs(scores[i] - old_scores[i]) >= 0.0001:
            flag = True
            break
    return flag

# 计算句子在图中的分数
def calculate_score(weight_graph, scores, i):
    length = len(weight_graph)
    d = 0.85
    added_score = 0.0
    for j in range(length):
        fraction = 0.0
        denominator = 0.0
        # 计算分子
        fraction = weight_graph[j][i] * scores[j]
        # 计算分母
        for k in range(length):
            denominator += weight_graph[j][k]
            if denominator == 0:
                denominator = 1
        added_score += fraction / denominator
    # 算出最终的分数
    weighted_score = (1 - d) + d * added_score
    return weighted_score

# 输入相似度的图（矩阵),返回各个句子的分数
def weight_sentences_rank(weight_graph):
    # 初始分数设置为0.5
    scores = [0.5 for _ in range(len(weight_graph))]
    old_scores = [0.0 for _ in range(len(weight_graph))]
    # 开始迭代
    while different(scores, old_scores):
        for i in range(len(weight_graph)):
            old_scores[i] = scores[i]
        for i in range(len(weight_graph)):
            scores[i] = calculate_score(weight_graph, scores, i)
    return scores

if __name__ == "__main__":
    text, index, ids, label = preprocess('./test.csv')
    num_list = [len(i) for i in text]
    scores = []
    count = [0 for i in range(37)]
    print(type(index[0]))
    for i in index:
        count[i] += 1
    print('count',count)
    print(sum(count))

    model = Model('./model')
    sents = filter_model(model, text)
    sents = filter_model(model, text)
    sents = filter_model(model, text)
    for i in range(36):
        graph = create_graph(model, text[sum(count[:i]):count[i+1]+sum(count[:i])])     # 传入句子链表  返回句子之间相似度的图
        score = weight_sentences_rank(graph)
        for j in score:
            scores.append(j)

    data_dict = {'index':index, 'id':ids, 'score':scores, 'len':num_list, 'label':label}
    data_df = pd.DataFrame(data_dict)
    print(data_df)
    data_df.to_csv('./result.csv')