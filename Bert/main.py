from utils import read_label, get_ids
from bert_serving.client import BertClient

from keras import Model
from keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout, Input, Bidirectional, LSTM
from keras.callbacks import LearningRateScheduler, EarlyStopping
import numpy as np
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical
from Attention_define import AttentionLayer

# 修改学习率
def scheduler(epoch):
    # 每隔10个epoch，学习率减小为原来的1/10
    if epoch == 0:
        lr = lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 5)
        print("lr changed to {}".format(lr * 5))
    if epoch % 10 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.5)
        print("lr changed to {}".format(lr * 0.5))
    return K.get_value(model.optimizer.lr)

def create_classify_dense(EMBEDDING_DIM):
    # 输入层 (None, 768)
    inputs = Input(shape=(EMBEDDING_DIM,))
    outputs = Dense(2, activation='sigmoid')(inputs)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()  # 输出模型结构和参数数量
    return model

def create_classify_textcnn(EMBEDDING_DIM):
    kernel_sizes=[3, 4, 5]
    convs = []
    # 输入层 (None, 12, 64)
    inputs = Input(shape=(12, int(EMBEDDING_DIM/12)))
    for kernel_size in kernel_sizes:
        c = Conv1D(128, kernel_size, activation='relu')(inputs)
        c = GlobalMaxPooling1D()(c)
        convs.append(c)
    c = Concatenate()(convs)
    outputs = Dense(2, activation='sigmoid')(c)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()  # 输出模型结构和参数数量
    return model

def create_classify_lstm(EMBEDDING_DIM, HIDDEN_SIZE):
    # 输入层 (None, 2, 384)
    inputs = Input(shape=(2, int(EMBEDDING_DIM/2)))
    x = Bidirectional(LSTM(HIDDEN_SIZE, dropout=0.2, return_sequences=True))(inputs)
    outputs = Dense(2, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()  # 输出模型结构和参数数量
    return model

def create_classify_lstm_att(EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
    # 输入层 (None, 2, 384)
    inputs = Input(shape=(2, int(EMBEDDING_DIM/2)))
    x = Bidirectional(LSTM(HIDDEN_SIZE, dropout=0.2))(inputs)
    # Attention层
    x = AttentionLayer(attention_size=ATTENTION_SIZE)(x)
    outputs = Dense(2, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()  # 输出模型结构和参数数量
    return model

def get_data(sentences, labels, i, step):
    train_X1, train_Y1 = [], []
    test_X = sentences[i:i+step]
    test_Y = labels[i:i+step]
    train_X = sentences[0:i] + sentences[i+step:]
    train_Y = labels[0:i] + labels[i+step:]
    for num in range(len(train_X)):
        for sent_num in range(len(train_X[num])):
            train_X1.append(train_X[num][sent_num])
            train_Y1.append(train_Y[num][sent_num])
    return train_X1, train_Y1, test_X, test_Y

def eval(test_X, test_Y, model, bc, EMBEDDING_DIM):
    acc_list, pre_list, re_list, f1_list = [], [], [], []
    for num in range(6):
        num_acc, tp, num_pr, num_re = 0, 0, 0, 0
        predict_temp = test_X[num]
        predict_temp = bc.encode(predict_temp)
        predict_temp = predict_temp.reshape(-1, int(768/EMBEDDING_DIM), EMBEDDING_DIM)
        predict_temp = model.predict(predict_temp)
        predict_temp = np.argmax(predict_temp, axis=1)
        Y_temp = test_Y[num]
        Y_temp = np.array(Y_temp)
        for index in range(len(predict_temp)):
            if predict_temp[index]==Y_temp[index]:
                num_acc += 1
            if predict_temp[index]==1 and Y_temp[index]==1:
                tp += 1
            if predict_temp[index]==1:
                num_pr += 1
            if Y_temp[index] == 1:
                num_re += 1
        acc = num_acc/len(Y_temp)
        if num_pr == 0:
            precision = 0
        else:
            precision = tp/num_pr
        recall = tp/num_re
        acc_list.append(acc)
        pre_list.append(precision)
        re_list.append(recall)
        if precision == 0 and recall == 0:
            f1_list.append(0)
        else:
            f1_list.append((2*precision*recall)/(precision+recall))
    acc = sum(acc_list)/len(acc_list)
    precision = sum(pre_list)/len(pre_list)
    recall = sum(re_list)/len(re_list)
    f1score = sum(f1_list)/len(f1_list)
    return acc, precision, recall, f1score

def clear_data(sentences, labels):
    sentences_all, labels_all = [], []
    for i, sentence_list in enumerate(sentences):
        sentences_new, labels_new = [], []
        for j, sentence in enumerate(sentence_list):
            if len(sentence.strip()) != 0:
                sentences_new.append(sentence.strip())
                labels_new.append(labels[i][j])
        sentences_all.append(sentences_new)
        labels_all.append(labels_new)
    return sentences_all, labels_all

if __name__ == "__main__":
    EMBEDDING_DIM = 768
    HIDDEN_SIZE = 256
    ATTENTION_SIZE = 256
    EPOCHS = 50
    batch_size = 64
    # 存储结果列表
    acc_list_all = []
    pre_list_all = []
    re_list_all = []
    f1_list_all = []
    filepath = './bugreports_sds/'
    step = 6
    bc = BertClient()
    sentences = []
    vectors = []
    for i in range(36):
        report_sent = []
        with open(filepath + str(i+1)+ '.txt', "r", encoding='utf-8') as f:
            for line in f.readlines():
                report_sent.append(line.strip('\n'))
        sentences.append(report_sent)
    labels_ids = read_label('./data/goldset_sds.txt')
    ids = get_ids()
    labels = []
    for index, id_list in enumerate(ids):
        label = []
        for id in id_list:
            if id in labels_ids[index]:
                label.append(1)
            else:
                label.append(0)
        labels.append(label)
    sentences, labels = clear_data(sentences, labels)
    for i in range(0, 36, step):
        # model = create_classify_dense(EMBEDDING_DIM)
        model = create_classify_lstm_att(EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        # model = create_classify_textcnn(EMBEDDING_DIM)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        train_X, train_Y, test_X, test_Y = get_data(sentences, labels, i, step)
        train_X = bc.encode(train_X)
        # 不同模型修改不同的shape: Dense: 不修改; TextCNN: (-1, 12, 64); LSTM/LSTM+Attention: (-1, 2, 384)
        train_X = train_X.reshape(-1, 2, int(EMBEDDING_DIM/2))
        # 随不同的模型修改dim
        dim = int(EMBEDDING_DIM/2)
        train_Y = np.array(train_Y)
        # 生成label
        lb = LabelBinarizer()
        train_Y = lb.fit_transform(train_Y)
        train_Y = to_categorical(train_Y)
        # early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max')
        reduce_lr = LearningRateScheduler(scheduler)
        model.fit(train_X, train_Y, epochs=EPOCHS, batch_size=batch_size, callbacks=[reduce_lr])
        temp_list = []
        acc, precision, recall, f1score = eval(test_X, test_Y, model, bc, dim)
        print(acc, precision, recall, f1score)
        acc_list_all.append(acc)
        pre_list_all.append(precision)
        re_list_all.append(recall)
        f1_list_all.append(f1score)
    acc = sum(acc_list_all)/len(acc_list_all)
    precision = sum(pre_list_all)/len(pre_list_all)
    recall = sum(re_list_all)/len(re_list_all)
    f1score = sum(f1_list_all)/len(f1_list_all)
    print('Acc:%f, Precision: %f, Recall: %f, F1-score: %f' % (acc, precision, recall, f1score))


