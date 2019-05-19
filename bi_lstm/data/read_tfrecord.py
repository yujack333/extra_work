# -*- coding: utf-8 -*-
# @Time    : 2019/5/13 15:25
# @Author  : YuYi

import tensorflow as tf
import sys
sys.path.append('../')
import parameter as cfg
import numpy as np
from data.convert_txt_to_tfrecord import check_feasibility as check


def read_and_decode_single_example(example_proto):
    features = {
        'X': tf.FixedLenFeature([], tf.string),
        'Y': tf.FixedLenFeature([], tf.string)
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    x = tf.decode_raw(parsed_features['X'], tf.float32)
    x = tf.reshape(x, shape=[20, 2])
    y = tf.decode_raw(parsed_features['Y'], tf.int32)
    y = tf.reshape(y, shape=[20, 1])
    y = tf.cast(y, dtype=tf.float32)

    concat_tensor = tf.concat((x, y), axis=1)
    concat_tensor = tf.random_shuffle(concat_tensor)
    x, y = tf.split(concat_tensor, [2, 1], axis=1)
    y = tf.reshape(tf.cast(y, dtype=tf.int32), shape=[20])

    return x, y


def train_input_fn(batch_size, epoch):
    dataset = tf.data.TFRecordDataset(cfg.train_tfrecord_path)
    dataset = dataset.map(read_and_decode_single_example)
    dataset = dataset.shuffle(200).repeat(epoch).batch(batch_size)

    return dataset


def test_input_fn(batch_size=1):
    dataset = tf.data.TFRecordDataset(cfg.test_tfrecord_path)
    dataset = dataset.map(read_and_decode_single_example)
    dataset = dataset.batch(batch_size)

    return dataset


def predict_input_fn():
    files = open(cfg.predict_raw_data_path, 'r').readlines()
    features = []
    for file in files:
        x_i = np.array([np.float32(num_i) for num_i in file.split(' ')])
        features.append(np.reshape(x_i, (20, 2)))
    features = np.array(features)
    label_size = features.shape[:2]
    labels = np.zeros(label_size)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    return dataset.batch(label_size[0])


if __name__ == '__main__':

    predict_input_fn()
    # data_i = train_input_fn(20, 1)
    # sess = tf.Session()
    # a, b = sess.run(data_i)
    # for i, j in zip(a, b):
    #     c = check(i, j)
    #     print(c)
    # print(a.shape, b.shape)
