# -*- coding: utf-8 -*-
# @Time    : 2019/5/13 11:36
# @Author  : YuYi


import os
import tensorflow as tf

# data path
root_path = 'G:\\tensorflow\\bi_lstm'
data_path = os.path.join(root_path, 'data')
train_raw_data_path = os.path.join(data_path, 'raw_data', 'train.txt')
test_raw_data_path = os.path.join(data_path, 'raw_data', 'test.txt')
predict_raw_data_path = os.path.join(data_path, 'raw_data', 'predict.txt')

train_tfrecord_path = os.path.join(data_path, 'tfrecord', 'train.tfrecord')
test_tfrecord_path = os.path.join(data_path, 'tfrecord', 'test.tfrecord')


# model parameter
seq_len = 20
batch_size = 32
model_dir = os.path.join(root_path, 'train_output_1')
epoch = None
step = 60000
params = {'boundaries': [10000, 30000],
          'values': [0.05, 0.005, 0.0001],
          'lstm_hidden_dim': 10,
          'max_number_boxes': 10,
          'seq_len': seq_len
          }

