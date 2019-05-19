# -*- coding: utf-8 -*-
# @Time    : 2019/5/13 14:20
# @Author  : YuYi

import numpy as np
import parameter as pa
import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def check_feasibility(x_i, y_i):
    '''
    check if this is a feasibility solution
    :param x_i: [20, 2] float
    :param y_i: [20] int
    :return: True or false
    '''
    num_boxes = np.max(y_i) + 1
    boxes = np.zeros((num_boxes, 2))
    for x, y in zip(x_i, y_i):
        boxes[y, :] += x

    return np.max(boxes) <= 2 and np.min(boxes) >= 0


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


def convert_txt_to_tfrecord(is_train):
    if is_train:
        writer = tf.python_io.TFRecordWriter(pa.train_tfrecord_path)
        raw_data = open(pa.train_raw_data_path, 'r').readlines()
    else:
        writer = tf.python_io.TFRecordWriter(pa.test_tfrecord_path)
        raw_data = open(pa.test_raw_data_path, 'r').readlines()
    max_num_boxes = -1
    for i, data_i in enumerate(raw_data):

        # read and process one line
        split_i = data_i.strip('\n').split('output')
        x_i = np.array([np.float32(num_i) for num_i in split_i[0].split(' ')[:-1]])
        x_i = np.reshape(x_i, [20, 2])

        y_i = np.array([np.int32(num_i) for num_i in split_i[1].split(' ')[1:]])
        y_i = np.reshape(y_i, [20])
        y_i = trans_label_fn(y_i)
        max_num_boxes = max(np.max(y_i), max_num_boxes)

        assert check_feasibility(x_i, y_i)

        features = tf.train.Features(feature={
            'X': _bytes_feature(x_i.tostring()),
            'Y': _bytes_feature(y_i.tostring())
        })

        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
        if i % 1000 == 0:
            print(i)
        if i % 10000 == 0:
            print(y_i)
    writer.close()
    print('convertion is done!!!', '\n', i)
    print('max num boxes', max_num_boxes)


if __name__ == '__main__':
    # train
    convert_txt_to_tfrecord(is_train=True)
    # test
    convert_txt_to_tfrecord(is_train=False)
