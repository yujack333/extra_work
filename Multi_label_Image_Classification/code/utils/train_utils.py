# -*- coding: utf-8 -*-
# @Time    : 2019/5/25 0:28
# @Author  : YuYi
import tensorflow as tf
import sys
sys.path.append('../')
import args


def config_learning_rate(global_step):
    if args.lr_type == 'fixed':
        return tf.convert_to_tensor(args.learning_rate_init, name='fixed_learning_rate')
    elif args.lr_type == 'piecewise':
        return tf.train.piecewise_constant(global_step, boundaries=args.pw_boundaries, values=args.pw_values,
                                           name='piecewise_learning_rate')
    else:
        raise ValueError('Unsupported learning rate type!')


def config_optimizer(optimizer_name, learning_rate, decay=0.9, momentum=0.9):
    if optimizer_name == 'momentum':
        return tf.train.MomentumOptimizer(learning_rate, momentum=momentum)
    elif optimizer_name == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=momentum)
    elif optimizer_name == 'adam':
        return tf.train.AdamOptimizer(learning_rate)
    elif optimizer_name == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Unsupported optimizer type!')
