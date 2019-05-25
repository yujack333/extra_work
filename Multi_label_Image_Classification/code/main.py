# -*- coding: utf-8 -*-
# @Time    : 2019/5/24 16:57
# @Author  : YuYi

import tensorflow as tf
import numpy as np
import args
import sys
from utils.data_utils import parse_line
from utils.train_utils import config_learning_rate, config_optimizer
from model import  Model
import time


def main():

    ####################
    # tf.data pipeline #
    ####################
    train_dataset = tf.data.TextLineDataset(args.train_txt_path)

    train_dataset = train_dataset.shuffle(args.num_train_img)
    train_dataset = train_dataset.map(
        lambda x: tf.py_func(parse_line,
                             inp=[x],
                             Tout=[tf.float32, tf.float32]),
        num_parallel_calls=args.num_threads
    )
    train_dataset = train_dataset.batch(args.batch_size)
    train_dataset = train_dataset.prefetch(args.prefetech_buffer)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)

    # get batch data from dataset
    images, labels = iterator.get_next()
    images.set_shape([None, None, None, 3])
    labels.set_shape([None, None])

    ###############
    # build model #
    ###############
    model = Model()
    logits = model.forward(images, is_training=True)
    loss = model.compute_loss(logits=logits, labels=labels)
    l2_loss = tf.losses.get_regularization_loss()
    global_step = tf.train.get_or_create_global_step()
    lr = config_learning_rate(global_step)
    optimizer = config_optimizer(args.optimizer_name, lr)

    # set dependencies for BN ops
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss+l2_loss, global_step=global_step)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        # initial train data \ model weights
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        for epoch_i in range(args.epoch):
            sess.run(train_init_op)
            for step in range(args.num_step_per_epoch):
                s_time = time.time()
                a, _ = sess.run((loss, train_op))
                e_time = time.time()

                if step % 1 == 0:
                    print(e_time-s_time, step, a)

                if step % 20 == 0 and step > 0:
                    # summary
                    break
                    pass

            if epoch_i % 2 == 0 and epoch_i > 0:
                # save model
                pass
            if epoch_i %2 == 0 and epoch_i > 0:
                # eval model
                pass


if __name__ == '__main__':
    main()

