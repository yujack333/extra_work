# -*- coding: utf-8 -*-
# @Time    : 2019/5/14 10:47
# @Author  : YuYi
from model import Bi_LSTM_CRF_mdoel as model_fn
import tensorflow as tf
import parameter as cfg
from data.read_tfrecord import train_input_fn, test_input_fn, predict_input_fn


def main(argv):

    model = tf.estimator.Estimator(model_fn=model_fn,
                                   model_dir=cfg.model_dir,
                                   params=cfg.params)

    # train the model
    # model.train(input_fn=lambda: train_input_fn(batch_size=cfg.batch_size, epoch=cfg.epoch), steps=cfg.step)
    #
    # # test the model
    # test_result = model.evaluate(input_fn=lambda: test_input_fn(batch_size=32))
    # print(test_result)

    # predict by the model
    predictions = model.predict(input_fn=lambda: predict_input_fn())
    for i in predictions:
        print(i)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)



