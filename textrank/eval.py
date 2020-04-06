import numpy as np
import tensorflow as tf

def index2pred(index,ids):
    index = np.array(index)
    idlist = [np.zeros_like(id,dtype=np.int32) for id in ids]
    for i, id in enumerate(idlist):
        id[index[i]] = 1
    idlist = [list(id) for id in idlist]
    return idlist

def label2y(label,ids):
    y = ids.copy()
    for index,(id,la) in enumerate(zip(y,label)):
        #ids[index][0] = len(id)
        for i,value in enumerate(id):
            if value in la:
                y[index][i] = 1
            else:
                y[index][i] = 0
    return y

def evaluate(y,pred):
    acclist = []
    prlist = []
    relist = []
    f1list = []
    recall = tf.keras.metrics.Recall()
    precision = tf.keras.metrics.Precision()
    accuracy = tf.keras.metrics.Accuracy()
    for per_y,per_pred in zip(y,pred):
        #计算recall
        recall.update_state(per_y, per_pred)
        re = recall.result()
        # print('recall',re)
        #计算precision
        precision.update_state(per_y, per_pred)
        pr = precision.result()
        # print('precision',pr)
        #计算f1
        if pr+re == 0.:
            f1 = 0
        else:
            f1 = 2.0*pr*re/(pr+re)
        # print('f1',f1)
        #计算accuracy
        accuracy.update_state(per_y, per_pred)
        acc = accuracy.result()
        # print('acc',acc)
        prlist.append(pr)
        relist.append(re)
        acclist.append(acc)
        f1list.append(f1)

    mean_pr = np.mean(prlist)
    mean_re = np.mean(relist)
    mean_acc = np.mean(acclist)
    mean_f1 = np.mean(f1list)
    return mean_acc, mean_pr, mean_re, mean_f1