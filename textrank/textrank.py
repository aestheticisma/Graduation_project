import codecs
from textrank4zh import TextRank4Keyword, TextRank4Sentence
import os

# 使用textrank4zh进行textrank对sentence计算得分，摘要单词数不超过总单词数*0.32
def textrank(text,sum_word,len_sentence):
    tr4s = TextRank4Sentence(delimiters='\n')
    tr4s.analyze(text=text, lower=True, source='all_filters')
    print('摘要：')
    sentences = []
    weight = []
    index = []
    for item in tr4s.get_key_sentences(num=1000):
        print(item.index, item.weight, item.sentence)  # index是语句在文本中位置，weight是权重
        sentences.append(item.sentence)
        weight.append(item.weight)
        index.append(item.index)
    extra_word = int(sum_word*0.32)
    num = 0
    extra_num = 0
    for data in len_sentence:
        if num+data<extra_word:
            num = num+data
            extra_num = extra_num + 1
        else:
            break
    result = index[:extra_num]
    return result
# 读取./bug_reports目录下的txt并调用textrank()
def bugsum(path,sum_word,len_sentence):
    result_samples = []
    files = os.listdir(path)
    files.sort(key=lambda x: int(x[:-4])) #去掉文件扩展名进行排序，即去掉'.txt'
    # print(files)
    for i,file in enumerate(files):
        file_path = os.path.join(path, file)
        text = codecs.open(file_path, 'r', 'utf-8').read()
        result = textrank(text,sum_word[i],len_sentence[i])
        result_samples.append(result)
    return result_samples

