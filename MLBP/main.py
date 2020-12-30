# -*- coding: utf-8 -*-
# @Author  : twd
# @FileName: demo.py
# @Software: PyCharm


import os
import time
import numpy as np
from pathlib import Path
dir = 'BiGRU_base'
Path(dir).mkdir(exist_ok=True)
t = time.localtime(time.time())
with open(os.path.join(dir, 'time.txt'), 'w') as f:
    f.write('start time: {}m {}d {}h {}m {}s'.format(t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec))
    f.write('\n')
from sklearn.model_selection import train_test_split



def GetSourceData(root, dir, lb):
    seqs = []
    print('\n')
    print('now is ', dir)
    file = '{}CD.txt'.format(dir)
    file_path = os.path.join(root, dir, file)

    with open(file_path) as f:
        for each in f:
            if each == '\n' or each[0] == '>':
                continue
            else:
                seqs.append(each.rstrip())

    # data and label
    label = len(seqs) * [lb]
    seqs_train, seqs_test, label_train, label_test = train_test_split(seqs, label, test_size=0.2, random_state=0)
    print('train data:', len(seqs_train))
    print('test data:', len(seqs_test))
    print('train label:', len(label_train))
    print('test_label:', len(label_test))
    print('total numbel:', len(seqs_train)+len(seqs_test))

    return seqs_train, seqs_test, label_train, label_test



def DataClean(data):
    max_len = 0
    for i in range(len(data)):
        st = data[i]
        # get the maximum length of all the sequences
        if(len(st) > max_len): max_len = len(st)

    return data, max_len



def PadEncode(data, max_len):

    # encoding
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    data_e = []
    for i in range(len(data)):
        length = len(data[i])
        elemt, st = [], data[i]
        for j in st:
            index = amino_acids.index(j)
            elemt.append(index)
        if length < max_len:
            elemt += [0]*(max_len-length)
        data_e.append(elemt)

    return data_e



def GetSequenceData(dirs, root):
    # getting training data and test data
    count, max_length = 0, 0
    tr_data, te_data, tr_label, te_label = [], [], [], []
    for dir in dirs:
        # 1.getting data from file
        tr_x, te_x, tr_y, te_y = GetSourceData(root, dir, count)
        count += 1

        # 2.getting the maximum length of all sequences
        tr_x, len_tr = DataClean(tr_x)
        te_x, len_te = DataClean(te_x)
        if len_tr > max_length: max_length = len_tr
        if len_te > max_length: max_length = len_te

        # 3.dataset
        tr_data += tr_x
        te_data += te_x
        tr_label += tr_y
        te_label += te_y


    # data coding and padding vector to the filling length
    traindata = PadEncode(tr_data, max_length)
    testdata = PadEncode(te_data, max_length)

    # data type conversion
    train_data = np.array(traindata)
    test_data = np.array(testdata)
    train_label = np.array(tr_label)
    test_label = np.array(te_label)

    return [train_data, test_data, train_label, test_label]



def GetData(path):
    dirs = ['AMP', 'ACP', 'ADP', 'AHP', 'AIP'] # functional peptides

    # get sequence data
    sequence_data = GetSequenceData(dirs, path)

    return sequence_data



def TrainAndTest(tr_data, tr_label, te_data, te_label):

    from train import train_main # load my training function

    train = [tr_data, tr_label]
    test = [te_data, te_label]

    threshold = 0.5
    model_num = 9  # model number
    test.append(threshold)
    train_main(train, test, model_num, dir)

    ttt = time.localtime(time.time())
    with open(os.path.join(dir, 'time.txt'), 'a+') as f:
        f.write('finish time: {}m {}d {}h {}m {}s'.format(ttt.tm_mon, ttt.tm_mday, ttt.tm_hour, ttt.tm_min, ttt.tm_sec))



def main():
    # I.get sequence data
    path = 'data' # data path
    sequence_data = GetData(path)


    # sequence data partitioning
    tr_seq_data,te_seq_data,tr_seq_label,te_seq_label = \
        sequence_data[0],sequence_data[1],sequence_data[2],sequence_data[3]


    # II.training and testing
    TrainAndTest(tr_seq_data, tr_seq_label, te_seq_data, te_seq_label)



if __name__ == '__main__':
    # executing the main function
    main()