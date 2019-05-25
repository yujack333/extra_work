# -*- coding: utf-8 -*-
# @Time    : 2019/5/24 22:33
# @Author  : YuYi
import os
root_path = 'G:/extra_work/Multi_label_Image_Classification/'
train_txt_path = os.path.join(root_path, 'data', 'train2014.txt')
train_image_path = os.path.join(root_path, 'data', 'image_data', 'train2014')

# tf.data parameters
num_train_img = 30000
prefetech_buffer = 5  # Prefetech_buffer used in tf.data pipeline.

num_threads = 10
image_w = 224
image_h = 224
batch_size = 1


# Learning rate and optimizer
optimizer_name = 'adam'  # Chosen from [sgd, momentum, adam, rmsprop]
save_optimizer = True  # Whether to save the optimizer parameters into the checkpoint file.
learning_rate_init = 1e-3
lr_type = 'piecewise'  # Chosen from [fixed, exponential, cosine_decay_restart, piecewise]
# piecewise params
pw_boundaries = [6000, 20000]  # epoch based boundaries
pw_values = [learning_rate_init, learning_rate_init/5.0, learning_rate_init/10.0]