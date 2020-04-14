from xml.dom import minidom
import os 
from nltk.corpus import stopwords
import re

# 读取"bugreports.xml"数据
def read_xml(filename):
    xml=minidom.parse(filename)
    root=xml.documentElement
    bugslist = []
    bugreports = root.getElementsByTagName('BugReport')
    for bugreport in bugreports:
            bugdict = {'BugreportID':0,'Title':'','Content':[]}
            #判断是否有id属性
            if bugreport.hasAttribute('ID'):
                #不加上面的判断也可以，若找不到属性，则返回空
                bugreportId = int(bugreport.getAttribute('ID'))
                bugdict['BugreportID'] = bugreportId
            title = bugreport.getElementsByTagName('Title')[0].firstChild.data
            bugdict['Title'] = title[1:-1]
            turns = bugreport.getElementsByTagName('Turn')
            turnslist = []
            for turn in turns:
                turndict = {'Date':'','Person':'','Text':[]}
                date = turn.getElementsByTagName('Date')[0].firstChild.data
                person = turn.getElementsByTagName('From')[0].firstChild.data
                textNode = turn.getElementsByTagName('Sentence')
                turndict['Date'] = date[1:-1]
                turndict['Person'] = person[1:-1]
                textlist = []
                for i, sentenceNode in enumerate(textNode):
                    sentence_id = sentenceNode.getAttribute('ID')
                    sentence_context = sentenceNode.firstChild.data
                    sentence_tuple = (sentence_id, sentence_context)
                    textlist.append(sentence_tuple)
                turndict['Text'] = textlist
                turnslist.append(turndict)
            bugdict['Content'] = turnslist
            bugslist.append(bugdict)
    return bugslist
# 读取goldset.txt
def read_label(filename):
    label = []
    with open(filename, 'r') as file:
        for line in file:  # 设置文件对象并读取每一行文件
            line = line.strip().split(',')
            label.append(line)
    label_list = []
    for i in range(int(label[-1][0])):
        id_list = []
        for index,id in label:
            if int(index)==i+1:
                id_list.append(id)
        label_list.append(id_list)
    return label_list
# 获取samples and ids 清除噪声
def get_content(bugslist):
    samples = []
    ids = []
    # stop_words = stopwords.words('english')
    for bugreport in bugslist:
        titles = []
        id = []
        sentences = []
        titles.append(bugreport['Title'])
        for turn in bugreport['Content']:
            for id_sentence, sentence in turn['Text']:
                # sentence = ' '.join(sentence.replace("[^a-zA-Z]", " ").strip().lower().split())
                # sen_new = ' '.join(i for i in sentence.split() if i not in stop_words)
                sentence = sentence.strip().lower()
                sentence = sentence.replace('&gt', '')
                sentence = re.split(r'[\s\-\.\(\),\*:"\d\\_\[\];\?=#\+/&\$@!%~}{|]', sentence)
                sentences.append(list(filter(lambda s: s and s.strip(), sentence)))
                id.append(id_sentence)
        ids.append(id)
        samples.append(sentences)
    return samples,ids
# 计算每个sentence有多少个单词、每个bug_reports有多少个单词
def count_word(samples):
    num_word_list = []
    for sample in samples:
        num = []
        for sentence in sample:
            num.append(len(sentence.split(' ')))
        num_word_list.append(num)
    numword = [sum(i) for i in num_word_list]
    return num_word_list,numword
# 将每个Bug_report保存成单个文件
def savefile(samples):
    for i,sample in enumerate(samples):
        string = ''
        for sentence in sample:
            string = string + sentence + '\n'
        with open("bug_reports/" + str(i+1)+".txt", "w", encoding='utf-8') as f:
            f.write(string)

# filename = "goldset.txt"
# label_list = read_label(filename)