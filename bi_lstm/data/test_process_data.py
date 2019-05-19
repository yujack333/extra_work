# -*- coding: utf-8 -*-
# @Time    : 2019/5/13 10:47
# @Author  : YuYi


import numpy as np
import sys
sys.path.append('../')
import parameter as pa

a = open(pa.train_raw_data_path, 'r').readlines()
x = []
y = []


def trans_label_fn(y_i):
    trans_dict = {}
    trans_y = []
    keys = 0
    for i in y_i:
        if i in trans_dict.keys():
            trans_y.append(trans_dict[i])
        else:
            trans_dict[i] = keys
            trans_y.append(keys)
            keys += 1

    trans_y_i = np.array(trans_y)
    return trans_y_i

for a_i in a[:100]:

    split_i = a_i.strip('\n').split('output')
    x_i = np.array([np.float32(num_i) for num_i in split_i[0].split(' ')[:-1]])
    x_i = np.reshape(x_i, [20, 2])
    x.append(x_i)

    y_i = np.array([np.int32(num_i) for num_i in split_i[1].split(' ')[1:]])
    y_i = np.reshape(y_i, [20])
    y.append(y_i)

x = np.array(x)
y = np.array(y)
print(x.shape, y.shape)