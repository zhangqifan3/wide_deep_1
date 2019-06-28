#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zhangqifan
# @Date  : 2019/6/18
# ==============================================================================
import tensorflow as tf
import os
import sys

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)
from os.path import dirname, abspath
from lib.build_feature_columns import build_model_columns
from lib.read_conf import Config

MODEL_DIR = os.path.join(dirname(dirname(dirname(abspath(__file__)))), 'model')
print(MODEL_DIR)
CONF = Config()

def build_estimator():
  """Build an estimator using official tf.estimator API.
      Args:
          model_dir: model save base directory
          model_type: one of {`wide`, `deep`, `wide_deep`}
      Returns:
          model instance of tf.estimator.Estimator class
      """
  wide_columns, deep_columns = build_model_columns()  # 确定模型的特征输入


  return tf.estimator.DNNLinearCombinedClassifier(
                              model_dir=MODEL_DIR,
                              linear_feature_columns=wide_columns,
                              linear_optimizer=tf.train.FtrlOptimizer(
                                  learning_rate=0.1,
                                  l1_regularization_strength=0.5,
                                  l2_regularization_strength=1),
                              dnn_feature_columns=deep_columns,
                              dnn_optimizer=tf.train.ProximalAdagradOptimizer(
                                  learning_rate=0.05,
                                  l1_regularization_strength=0.01,
                                  l2_regularization_strength=0.01),
                              dnn_hidden_units=[32, 16, 2],
                              dnn_activation_fn=tf.nn.relu,
                              #    dnn_dropout= ,
                              n_classes=2,
                              #    weight_column=weight_column,
                              label_vocabulary=None,
                              input_layer_partitioner=None)

if __name__=="__main__":
    pass
    build_estimator()