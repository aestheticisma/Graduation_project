import utils
import os
import pandas as pd

data_file = 'bugreports.xml'
label_file = 'goldset.txt'
data_path = './origin_data'

if __name__ == "__main__":
    test = []
    buglist = utils.read_xml(os.path.join(data_path, data_file))
    labels = utils.read_label(os.path.join(data_path, label_file))
    data = pd.DataFrame(buglist)
    # print(data['Title'][0])
    samples, ids = utils.get_content(buglist)
    count, flag = 0, 0
    for sentences, id_list in zip(samples, ids):
        count += 1
        for sentence, id in zip(sentences, id_list):
            if id in labels[count-1]:
                test.append({'index': count, 'title': data['Title'][count-1], 'sentence': sentence, 'id': id, 'label': 1})
            else:
                test.append({'index': count, 'title': data['Title'][count-1], 'sentence': sentence, 'id': id, 'label': 0})

    test = pd.DataFrame(test)
    # print(test['sentence'].values)
    count = 0
    # test = test.drop(index=1)

    print(test['sentence'])
    for index, i in enumerate(test['sentence'].values):
        if len(i) < 3:
            count += 1
            test = test.drop(index=index)
            # test = test.drop(index=(test.loc[(test['sentence']==i)].index))
    # test.drop(index=(test.loc[(len(test['sentence'])<3)].index))
            
    print(test)
    print(count)
    test.to_csv('test.csv', index = None)
    # print(len(samples[-1]))
    # print(len(ids[-1]))