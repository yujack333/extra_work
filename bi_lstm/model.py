# -*- coding: utf-8 -*-
# @Time    : 2019/5/13 16:30
# @Author  : YuYi

import parameter as cfg
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMBlockCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode


def Bi_LSTM_CRF_mdoel(features, labels, mode, params):
    '''
    :param features: [batch_size, 20, 2] float32(input_fn first output)
    :param labels: [batch_size,20] int32(input_fn second output)
    :param model: train, eval or test
    :param params: dict model hyperparameter
    :return:  EstimatorSpec class
    '''
    batch_size = tf.shape(features)[0]
    with tf.variable_scope('bi_lstm'):
        cell_fw = LSTMBlockCell(params['lstm_hidden_dim'])
        cell_bw = LSTMBlockCell(params['lstm_hidden_dim'])
        (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                            cell_bw=cell_bw,
                                                                            inputs=features,
                                                                            dtype=tf.float32)
        output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
    with tf.variable_scope('project'):
        output = tf.reshape(output,
                            [-1, params['lstm_hidden_dim']*2])
        logits = tf.layers.dense(output,
                                 params['max_number_boxes'],
                                 activation=None)
        logits = tf.reshape(logits, [-1, params['seq_len'], params['max_number_boxes']])

    seq_len = tf.ones(batch_size, dtype=tf.int32) * tf.constant(cfg.seq_len, dtype=tf.int32)

    if mode == tf.estimator.ModeKeys.PREDICT:
        transition_params = tf.get_variable(name='CRF/transitions', shape=[10, 10])
        predict_id, _ = tf.contrib.crf.crf_decode(logits, transition_params, seq_len)
        prediction = {'predict_id': predict_id}
        return tf.estimator.EstimatorSpec(mode, predictions=prediction)

    with tf.variable_scope('CRF'):

        log_likelihood, transition_params = crf_log_likelihood(inputs=logits,
                                                               tag_indices=labels,
                                                               sequence_lengths=seq_len)
    loss = -tf.reduce_mean(log_likelihood)

    if mode == tf.estimator.ModeKeys.EVAL:
        predict_id, _ = tf.contrib.crf.crf_decode(logits, transition_params, seq_len)
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predict_id)
        eval_metric = {'acc': accuracy}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric)

    assert mode == tf.estimator.ModeKeys.TRAIN

    global_step = tf.train.get_global_step()
    learning_rate = tf.train.piecewise_constant(global_step, params['boundaries'], params['values'])
    tf.summary.scalar(name='learning_rate', tensor=learning_rate)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)



