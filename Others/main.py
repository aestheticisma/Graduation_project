from Attention_define import create_classify_model
from preprocess import preprocess
from keras.preprocessing.sequence import pad_sequences
import gensim
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical
from keras import backend as K
from keras.callbacks import LearningRateScheduler, EarlyStopping
from text_cnn import create_textcnn

from bert_serving.client import BertClient

# 序号化 文本，tokenizer句子，并返回每个句子所对应的词语索引
def tokenizer(sentences, word_index, MAX_SEQUENCE_LENGTH):
    data = []
    for sentence in sentences:
        new_txt = []
        for word in sentence:
            try:
                new_txt.append(word_index[word])  # 把句子中的 词语转化为index
            except:
                new_txt.append(0)
        data.append(new_txt)
    # 使用kears的内置函数padding对齐句子,好处是输出numpy数组，不用自己转化了
    texts = pad_sequences(data, maxlen=MAX_SEQUENCE_LENGTH)
    return texts

def load_word2vec(sengtences, wvmodel_Path, options=1):
    if options == 1:
        Word2VecModel = gensim.models.Word2Vec.load(wvmodel_Path)
    elif options == 2:
        tmp_file = "./test_word2vec.txt"
        glove2word2vec(wvmodel_Path, tmp_file)
        Word2VecModel = KeyedVectors.load_word2vec_format(tmp_file)
    elif options == 3 or  options == 4:
        Word2VecModel = KeyedVectors.load_word2vec_format(wvmodel_Path, binary=False)
    # sentences, index, ids, label = preprocess('./data.csv')
    vocab_list = []
    for sentence in sentences:
        for word in sentence:
            if word not in vocab_list:
                vocab_list.append(word)
    vocab_list2 = [word for word, Vocab in Word2VecModel.wv.vocab.items()]
    vocab_list3 = []
    for word in vocab_list:
        if word in vocab_list2:
            vocab_list3.append(word)
    word_index = {" ": 0}# 初始化 [word : token]
    word_vector = {} # 初始化`[word : vector]`字典
    embeddings_matrix = np.zeros((len(vocab_list3) + 1, Word2VecModel.vector_size))
    ## 3 填充 上述 的字典 和 大矩阵
    for i in range(len(vocab_list3)):
        # print(i)
        word = vocab_list3[i]  # 每个词语
        word_index[word] = i + 1 # 词语：序号
        word_vector[word] = Word2VecModel.wv[word] # 词语：词向量
        embeddings_matrix[i + 1] = Word2VecModel.wv[word]
    return word_index, embeddings_matrix, word_vector

def load_bert_vectors(sengtences):
    bc = BertClient()
    vocab_list = []
    for sentence in sentences:
        for word in sentence:
            if word not in vocab_list:
                vocab_list.append(word)
    word_index = {" ": 0}# 初始化 [word : token]
    word_vector = {} # 初始化`[word : vector]`字典
    embeddings_matrix = np.zeros((len(vocab_list) + 1, 768))
    for i in range(len(vocab_list)):
        word = vocab_list[i]
        word_index[word] = i + 1
        word_vector[word] = bc.encode([word])
        embeddings_matrix[i+1] = word_vector[word]
    return word_index, embeddings_matrix, word_vector
# 计算每个报告的句子数量 count[0] = 0 报告计数下标从1开始
def cal_count(index):
    count = [0 for i in range(37)]
    for i in index:
        count[i] += 1
    return count

def count_word(sentences, count):
    word_num = []
    for i in range(len(count)-1):
        word_count = []
        report = sentences[sum(count[:i+1]) : sum(count[:i+2])]
        sentence_length = [len(sentence) for sentence in report]
        word_length = sum(sentence_length)
        word_count.append(word_length)
        word_count += sentence_length
        word_num.append(word_count)
    return word_num


# 6个报告作为测试集、30个报告作为训练集   i为6个报告一组的第一个下标，step=6
def get_data(X, label, count, i, step):
    test_X = X[sum(count[:i+1]) : sum(count[:i+step+1])]
    test_Y = label[sum(count[:i+1]) : sum(count[:i+step+1])]
    train_X = np.vstack((X[ : sum(count[:i+1])], X[sum(count[:i+step+1]) : ]))
    train_Y = np.vstack((label[ : sum(count[:i+1])], label[sum(count[:i+step+1]) : ]))
    return train_X, train_Y, test_X, test_Y

def get_summary_label(predict_y, word_num_temp):
    report_length = word_num_temp[0]
    summary_length = 0.25*report_length
    word_count = 0
    predict_fin_y = []
    predict_label = []
    for id in predict_y:
        if word_count + word_num_temp[id[1]+1] < summary_length:
            predict_fin_y.append(id[1])
            word_count += word_num_temp[id[1]+1]
        else:
            break
    for i in range(len(predict_y)):
        if i in predict_fin_y:
            predict_label.append(1)
        else:
            predict_label.append(0)
    return predict_label

def eval(test_X, test_Y, temp_list, model, word_num_temp):
    # predict_test = model.predict(test_X)
    # predict_test = np.argmax(predict_test, axis=1)
    # Y = np.argmax(test_Y, axis=1)
    acc_list, pre_list, re_list, f1_list = [], [], [], []
    for num in range(6):
        num_acc, tp, num_pr, num_re = 0, 0, 0, 0

        test_X_temp = test_X[sum(temp_list[:num+1]) : sum(temp_list[:num+2])]
        test_Y_temp = test_Y[sum(temp_list[:num+1]) : sum(temp_list[:num+2])]
        predict_temp = model.predict(test_X_temp)
        # predict_temp = np.argmax(predict_temp, axis=1)
        predict_y = []
        for index, result in enumerate(predict_temp):
            predict_y.append((result[1], index))
        predict_y = sorted(predict_y, key=lambda x: x[0], reverse=True)
        predict_label = get_summary_label(predict_y, word_num_temp[num])

        Y_temp = np.argmax(test_Y_temp, axis=1)

        for index in range(len(predict_label)):
            if predict_label[index]==Y_temp[index]:
                num_acc += 1
            if predict_label[index]==1 and Y_temp[index]==1:
                tp += 1
            if predict_label[index]==1:
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

if __name__ == "__main__":
    word2vec_path = ['./word2vecmodel', '../glove+LSTM/glove.6B.100d.txt', 'enwiki_20180420_100d.txt.bz2', './fasttext2word2vec.txt']
    options = 1
    sentences, index, ids, label = preprocess('./data.csv')
    word_index, embeddings_matrix, word_vector = load_word2vec(sentences, word2vec_path[options-1], options)
    # word_index, embeddings_matrix, word_vector = load_bert_vectors(sentences)
    # 定义参数
    MAX_SEQUENCE_LENGTH = 50
    MAX_NB_WORDS = len(embeddings_matrix)
    EMBEDDING_DIM = 100
    EPOCHS = 50
    batch_size = 64
    HIDDEN_SIZE = 128
    ATTENTION_SIZE = 128
    trainable = False
    step = 6

    # 存储结果列表
    acc_list_all = []
    pre_list_all = []
    re_list_all = []
    f1_list_all = []

    count = cal_count(index)
    word_num = count_word(sentences, count)
    # padding sentence为相同长度 返回index array
    X = tokenizer(sentences, word_index, MAX_SEQUENCE_LENGTH)
    # 生成label
    lb = LabelBinarizer()
    label = lb.fit_transform(label)
    label = to_categorical(label)
    num = 0
    for i in range(0, 36, step):
        word_num_temp = word_num[i:i+step]
        num += 1
        #划分数据集
        train_X, train_Y, test_X, test_Y = get_data(X, label, count, i, step)
        # model = create_classify_model(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, embeddings_matrix, HIDDEN_SIZE, ATTENTION_SIZE, trainable)
        model = create_textcnn(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, embeddings_matrix, trainable)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        reduce_lr = LearningRateScheduler(scheduler)
        # early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max')
        model.fit(train_X, train_Y, epochs=EPOCHS, batch_size=batch_size, callbacks=[reduce_lr])
        # predict_test = model.predict(test_X)
        # predict_test = np.argmax(predict_test, axis=1)
        # Y = np.argmax(test_Y, axis=1)
        temp_list = []
        for j in range(0,7):
            if j == 0:
                temp_list.append(0)
            else:
                temp_list.append(count[i+j])
        acc, precision, recall, f1score = eval(test_X, test_Y, temp_list, model, word_num_temp)
        print(num, acc, precision, recall, f1score)
        acc_list_all.append(acc)
        pre_list_all.append(precision)
        re_list_all.append(recall)
        f1_list_all.append(f1score)

    acc = sum(acc_list_all)/len(acc_list_all)
    precision = sum(pre_list_all)/len(pre_list_all)
    recall = sum(re_list_all)/len(re_list_all)
    f1score = sum(f1_list_all)/len(f1_list_all)
    print('Acc:%f, Precision: %f, Recall: %f, F1-score: %f' % (acc, precision, recall, f1score))