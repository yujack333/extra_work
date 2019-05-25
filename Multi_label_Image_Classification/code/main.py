# -*- coding: utf-8 -*-
# @Time    : 2019/5/24 16:57
# @Author  : YuYi

import tensorflow as tf
import numpy as np
import args
import sys
from utils.data_utils import parse_line
from utils.train_utils import config_learning_rate, config_optimizer
from utils.layers_utils import make_summary
from model import  Model
import time



def main():

    ####################
    # tf.data pipeline #
    ####################
    # train dataset
    train_dataset = tf.data.TextLineDataset(args.train_dataset_files)

    train_dataset = train_dataset.shuffle(args.num_train_img)
    train_dataset = train_dataset.map(
        lambda x: tf.py_func(parse_line,
                             inp=[x],
                             Tout=[tf.float32, tf.float32]),
        num_parallel_calls=args.num_threads
    )
    train_dataset = train_dataset.batch(args.batch_size)
    train_dataset = train_dataset.prefetch(args.prefetech_buffer)

    # trainval dataset
    trainval_dataset = tf.data.TextLineDataset(args.trainval_dataset_files)

    trainval_dataset = trainval_dataset.shuffle(args.num_trainval_img)
    trainval_dataset = trainval_dataset.map(
        lambda x: tf.py_func(parse_line,
                             inp=[x],
                             Tout=[tf.float32, tf.float32]),
        num_parallel_calls=args.num_threads
    )
    trainval_dataset = trainval_dataset.batch(args.batch_size)
    trainval_dataset = trainval_dataset.prefetch(args.prefetech_buffer)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    trainval_init_op = iterator.make_initializer(trainval_dataset)

    # get batch data from dataset
    images, labels = iterator.get_next()
    images.set_shape([None, None, None, 3])
    labels.set_shape([None, None])

    ###############
    # build model #
    ###############
    model = Model()
    logits = model.forward(images, is_training=True)
    prob = model.probability(logits)
    loss = model.compute_loss(logits=logits, labels=labels)
    l2_loss = tf.losses.get_regularization_loss()
    global_step = tf.train.get_or_create_global_step()
    lr = config_learning_rate(global_step)
    optimizer = config_optimizer(args.optimizer_name, lr)

    # summary
    tf.summary.scalar(name='learning_rate', tensor=lr)
    tf.summary.scalar(name='loss', tensor=loss)
    tf.summary.scalar(name='l2loss', tensor=l2_loss)
    tf.summary.histogram(name='probability', values=prob)

    # set dependencies for BN ops
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss+l2_loss, global_step=global_step)

    # model saver
    saver = tf.train.Saver()
    best_saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        # initial model weights
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        # summary
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(args.log_dir, sess.graph)

        for epoch_i in range(args.epoch):
            sess.run(train_init_op)
            for train_step in range(args.train_num_step_per_epoch):
                s_time = time.time()
                _, summary, step_loss, __global_step = sess.run((train_op, merged, loss, global_step))
                e_time = time.time()

                if train_step % 20 == 0:
                    print('batch_time:%.4f' % (e_time-s_time),
                          'epoch:%d--step:%d' % (epoch_i, train_step),
                          'loss:%.4f' % step_loss)

                if train_step % 1 == 0 and train_step > -1:
                    # summary
                    writer.add_summary(summary, global_step=__global_step)

            if epoch_i % 2 == 0 and epoch_i > 0:
                # save model
                saver.save(sess,
                           args.save_model_dir,
                           global_step=global_step)
            if epoch_i % 1 == 0 and epoch_i > 0:
                # eval model
                sess.run(trainval_init_op)
                best_acc = -1
                number_correct = 0
                correct_sample = []
                for trainval_step in range(args.trainval_num_step_per_epoch):
                    prob_i, labels_i = sess.run((prob, labels))
                    prediction_list = model.prediction(prob_i, 0.5)
                    label_list = model.prediction(labels_i, 0.5)
                    number_correct_i, correct_sample_i = model.correct_sample(prediction_list,
                                                                              label_list)
                    number_correct += number_correct_i
                    correct_sample.append(correct_sample_i)
                acc = float(number_correct)/args.num_trainval_img
                if acc > best_acc:
                    best_acc = acc
                    best_saver.save(sess,
                                    args.save_model_dir+'model-epoch_%d_acc_%.4f' % (epoch_i, best_acc))
                print('----epoch:%d' % epoch_i, '--trainval_acc:%.4f------' % acc)


if __name__ == '__main__':
    main()

