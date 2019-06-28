#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zhangqifan
# @Date  : 2019/6/1
import time
import tensorflow as tf
from lib.tf_dataset import input_fn, FileListGenerator
from lib.read_conf import Config
from lib.build_estimator import build_estimator


def main():

    CONFIG = Config()
    model_conf = CONFIG.read_model_conf()['model_conf']
    traindata_list = FileListGenerator(model_conf['data_dir_train']).generate()
    testdata_list = FileListGenerator(model_conf['data_dir_pred']).generate()

    model = build_estimator()

    traindata = next(traindata_list)
    testdata = next(testdata_list)

    t0 = time.time()
    tf.logging.info('Start training {}'.format(traindata))

    model.train(
            input_fn=lambda: input_fn(traindata),
            hooks=None,
            steps=None,
            max_steps=None,
            saving_listeners=None)
    t1 = time.time()
    tf.logging.info('Finish training {}, take {} mins'.format(traindata, float((t1 - t0) / 60)))

    tf.logging.info('Start evaluating {}'.format(testdata))
    t2 = time.time()


    results = model.evaluate(
            input_fn=lambda: input_fn(testdata),
            steps=None,  # Number of steps for which to evaluate model.
            hooks=None,
            checkpoint_path=None,  # latest checkpoint in model_dir is used.
            name=None)
    t3 = time.time()
    tf.logging.info('Finish evaluation {}, take {} mins'.format(testdata, float((t3 - t2) / 60)))

    # Display evaluation metrics
    for key in sorted(results):
        print('{}: {}'.format(key, results[key]))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
