import utils
import textrank
import eval


def main():
    filename = './origin_data/bugreports.xml'
    path = './bug_reports'
    bugslist = utils.read_xml(filename)
    # print(bugslist)
    label = utils.read_label('./origin_data/goldset.txt')
    # print(label)
    samples, ids = utils.get_content(bugslist)
    # print(samples)
    num_word_list,numword = utils.count_word(samples)
    # print(len(num_word_list))

    # for i in num_word_list:
    #     num_sentence.append(len(i))
    utils.savefile(samples)
    # print(num_sentence)
    results = textrank.bugsum(path,numword,num_word_list)
    print(len(i) for i in results)
    # extra_ids = index2id(results,ids)
    # print(len(extra_ids))
    pred = eval.index2pred(results,ids)
    y = eval.label2y(label,ids)
    mean_acc, mean_pr, mean_re, mean_f1 = eval.evaluate(y,pred)
    print('mean_acc, mean_pr, mean_re, mean_f1',mean_acc, mean_pr, mean_re, mean_f1)

if __name__ == "__main__":
    main()