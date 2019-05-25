# -*- coding: utf-8 -*-
# @Time    : 2019/5/24 22:33
# @Author  : YuYi
import os
import math


# data related
root_path = 'G:/extra_work/Multi_label_Image_Classification/'
train_txt_path = os.path.join(root_path, 'data', 'train2014.txt')
num_total_data = len(open(train_txt_path).readlines())
train_image_path = os.path.join(root_path, 'data', 'image_data', 'train2014')
train_dataset_files = os.path.join(root_path, 'data', 'train.txt')
trainval_dataset_files = os.path.join(root_path, 'data', 'trainval.txt')
trainval_data_ratio = 0.1
num_trainval_img = math.ceil(num_total_data * trainval_data_ratio)
num_train_img = num_total_data - num_trainval_img

# tf.data parameters
prefetech_buffer = 5  # Prefetech_buffer used in tf.data pipeline.

# train parameters
num_threads = 10
image_w = 224
image_h = 224
batch_size = 16
epoch = 10
train_num_step_per_epoch = math.ceil(float(num_train_img)/batch_size)
trainval_num_step_per_epoch = math.ceil(float(num_trainval_img)/batch_size)
save_model_dir = os.path.join(root_path, 'model_ckp/')
log_dir = os.path.join(root_path, 'model_summary')

# Learning rate and optimizer
optimizer_name = 'adam'  # Chosen from [sgd, momentum, adam, rmsprop]
save_optimizer = True  # Whether to save the optimizer parameters into the checkpoint file.
learning_rate_init = 1e-3
lr_type = 'piecewise'  # Chosen from [fixed, exponential, cosine_decay_restart, piecewise]
# piecewise params
pw_boundaries = [6000, 20000]  # epoch based boundaries
pw_values = [learning_rate_init, learning_rate_init/5.0, learning_rate_init/10.0]