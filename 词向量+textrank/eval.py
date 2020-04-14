import numpy as np
import tensorflow as tf
import pandas as pd
from word2vec_train import preprocess

# def evaluate(y,pred):
#     print('y', y)
#     print('pred', pred)
#     acclist = []
#     prlist = []
#     relist = []
#     f1list = []
#     recall = tf.keras.metrics.Recall()
#     precision = tf.keras.metrics.Precision()
#     accuracy = tf.keras.metrics.Accuracy()
#     for per_y,per_pred in zip(y,pred):
#         #计算recall
#         recall.update_state(per_y, per_pred)
#         re = recall.result()
#         # print('recall',re)
#         #计算precision
#         precision.update_state(per_y, per_pred)
#         pr = precision.result()
#         # print('precision',pr)
#         #计算f1
#         if pr+re == 0.:
#             f1 = 0
#         else:
#             f1 = 2.0*pr*re/(pr+re)
#         # print('f1',f1)
#         #计算accuracy
#         accuracy.update_state(per_y, per_pred)
#         acc = accuracy.result()
#         # print('acc',acc)
#         prlist.append(pr)
#         relist.append(re)
#         acclist.append(acc)
#         f1list.append(f1)

#     mean_pr = np.mean(prlist)
#     mean_re = np.mean(relist)
#     mean_acc = np.mean(acclist)
#     mean_f1 = np.mean(f1list)
#     return mean_acc, mean_pr, mean_re, mean_f1

def bugsum(id_socre_length):
    id_socre_length = sorted(id_socre_length, key=lambda x: x[1], reverse=True)
    extra_word = sum([i[2] for i in id_socre_length])*0.32
    num = 0
    extra_num = 0
    for data in id_socre_length:
        if num+data[2] < extra_word:
            num += data[2]
            extra_num = extra_num + 1
        else:
            break
    for i in range(len(id_socre_length)):
        if i < extra_num:
            id_socre_length[i].append(1)
        else:
            id_socre_length[i].append(0)
    return id_socre_length

if __name__ == "__main__":
    report = {}
    ids_list = []
    score_list = []
    label_list = []
    id_score = []
    id_socre_length = []
    
    report_len = []
    report_result = []

    count = [0 for i in range(37)]
    acc_list = []
    pr_list = []
    re_list = []
    f1_list = []

    sentence_list = []
    num_list = []
    results = pd.read_csv('./result.csv', dtype={'index':int, 'id':str, 'score':float})

    index = list(map(int, results['index'].values))
    ids = list(map(str, results['id'].values))
    score = list(map(float, results['score'].values))
    label = list(map(int, results['label'].values))
    sentence_len = list(map(int, results['len'].values))

    # 计算数量
    for i in index:
        count[i] += 1
    for i in range(36):
        ids_list.append(ids[sum(count[:i+1]):count[i+1]+sum(count[:i+1])])
        score_list.append(score[sum(count[:i+1]):count[i+1]+sum(count[:i+1])])
        label_list.append(label[sum(count[:i+1]):count[i+1]+sum(count[:i+1])])
        report_len.append(sentence_len[sum(count[:i+1]):count[i+1]+sum(count[:i+1])])
        # sentence_list.append(sentences[sum(count[:i+1]):count[i+1]+sum(count[:i+1])])
    for id, score, len_sentence, label in zip(ids_list, score_list, report_len, label_list):
        id_score += [[i, j, k, t] for i, j, k, t in zip(id, score, len_sentence, label)]
    for i in range(36):
        id_socre_length.append(id_score[sum(count[:i+1]):count[i+1]+sum(count[:i+1])])
    # print(id_socre_list[1])
    for i in range(len(id_socre_length)):
        report_result.append(bugsum(id_socre_length[i]))
        num_acc = 0
        tp = 0
        num_pr = 0
        num_re = 0
        for sent_result in report_result[i]:
            if sent_result[3] == sent_result[4]:
                num_acc += 1
            if sent_result[3]==1 and sent_result[4]==1:
                tp += 1
            if sent_result[3]==1:
                num_re += 1
            if sent_result[4]==1:
                num_pr += 1
        acc_list.append(num_acc/len(report_result[i]))
        pr_list.append(tp/num_pr)
        re_list.append(tp/num_re)
        f1_list.append((2*pr_list[i]*re_list[i])/(pr_list[i]+re_list[i]))
    print('Acc: %f' % (sum(acc_list)/len(acc_list)))
    print('Pr: %f' % (sum(pr_list)/len(pr_list)))
    print('Re: %f' % (sum(re_list)/len(re_list)))
    print('F1: %f' % (sum(f1_list)/len(f1_list)))
    # print(report_result[0])
    ### 计算acc



