# -*- coding: utf-8 -*-
# @Time    : 2019/5/24 14:58
# @Author  : YuYi

import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
from utils.layers_utils import conv2d


class Model:
    def __init__(self, num_class=20, weight_decay=1e-4):
        self.num_class = num_class
        self.weight_decay = weight_decay

    def forward(self, inputs, is_training=False):
        # set batch norm params
        batch_norm_params = {
            'decay': 0.999,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            biases_initializer=None,
                            activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                            weights_regularizer=slim.l2_regularizer(self.weight_decay)):
            with tf.variable_scope('darknet_body'):
                net = darknet53_body(inputs)
                net = slim.conv2d(net, self.num_class, 3,
                                  stride=1, normalizer_fn=None,
                                  activation_fn=None,
                                  biases_initializer=tf.zeros_initializer())
        with tf.name_scope('global_avg_pool'):
            logits = tf.reduce_mean(net, axis=[1, 2])
        return logits

    def probability(self, logits):
        return tf.nn.sigmoid(logits)

    def prediction(self, probability, threshold):
        '''
        if probability greater threshold than it a object
        :param probability: a np.array which shape is [batch_size, 20]
        :param threshold:  np.float32
        :return:
        prediction_list:[batch_size], prediction_list[i] is a set
        '''
        prediction_list = []
        for i in range(probability.shape[0]):
            prediction_list.append(set(np.where(probability[i] > threshold)[0]))
        return prediction_list

    def correct_sample(self, prediction_list, label_list):
        '''
        :param prediction_list: [batch_size], prediction_list[i] is a set
        :param label_list: a list which shape is [batch_size],label_list[i] is a set
        :return:
        '''
        assert len(prediction_list) == len(label_list)
        number_correct = 0
        correct_sample = []
        for i in range(len(prediction_list)):
            if prediction_list[i].issubset(label_list[i]):
                number_correct += 1
                correct_sample.append(1)
            else:
                correct_sample.append(0)
        return number_correct, correct_sample

    def compute_loss(self, logits, labels):
        with tf.name_scope('loss'):
            return tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)


def darknet53_body(inputs):
    def res_block(inputs, filters):
        shortcut = inputs
        net = conv2d(inputs, filters * 1, 1)
        net = conv2d(net, filters * 2, 3)

        net = net + shortcut

        return net

    # first two conv2d layers
    net = conv2d(inputs, 32, 3, strides=1)
    net = conv2d(net, 64, 3, strides=2)

    # res_block * 1
    net = res_block(net, 32)

    net = conv2d(net, 128, 3, strides=2)

    # res_block * 2
    for i in range(2):
        net = res_block(net, 64)

    net = conv2d(net, 256, 3, strides=2)

    # res_block * 8
    for i in range(2):
        net = res_block(net, 128)

    # route_1 = net
    net = conv2d(net, 512, 3, strides=2)

    # res_block * 8
    for i in range(2):
        net = res_block(net, 256)

    # route_2 = net
    net = conv2d(net, 1024, 3, strides=2)

    # res_block * 4
    for i in range(2):
        net = res_block(net, 512)
    out = net

    return out

