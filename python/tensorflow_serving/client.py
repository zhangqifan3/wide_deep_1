from __future__ import print_function

from os.path import dirname, abspath, join
import sys
import threading

import numpy as np
import tensorflow as tf

from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from collections import OrderedDict
PACKAGE_DIR = dirname(dirname(abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)

from lib.read_conf import Config
from lib.tf_dataset import TF_Data

PRED_FILE = '/home/zhangqifan/data/part_0.csv'


def _read_test_input():
    for line in open(PRED_FILE):
        line = line.strip('\n').split(' ')
        print(line)
        line.pop(0)
        print(line)
        return line

# Example Features for a movie recommendation application:
#    feature {
#      key: "age"
#      value { float_list {
#        value: 29.0
#      }}
#    }
#    feature {
#      key: "movie"
#      value { bytes_list {
#        value: "The Shawshank Redemption"
#        value: "Fight Club"
#      }}
#    }


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(value, encoding='utf-8')]))


def pred_input_fn(csv_data):
    """Prediction input fn for a single data, used for serving client"""
    conf = Config()
  #  feature = conf.read_schema_conf().values()
  #  feature_unused = conf.get_feature_name('unused')
    feature_conf = conf.read_feature_conf()[1]
    csv_default = TF_Data('/home/zhangqifan/data/part_0.csv')._column_to_csv_defaults()
    csv_default.pop('label')
    print(csv_default)

    feature_dict = {}
    for idx, f in enumerate(csv_default.keys()):
        print(f)
        print(type(csv_default[f]))
        if f in feature_conf:

            if csv_default[f] == ['']:
                print('yes')
                feature_dict[f] = _bytes_feature(csv_data[idx])
            else:
                feature_dict[f] = _float_feature(float(csv_data[idx]))
    return feature_dict


def main(_):

    channel = implementations.insecure_channel('139.9.45.232', int(8500))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'test'
    request.model_spec.signature_name = 'serving_default'
    # feature_dict = {'age': _float_feature(value=25),
    #               'capital_gain': _float_feature(value=0),
    #               'capital_loss': _float_feature(value=0),
    #               'education': _bytes_feature(value='11th'.encode()),
    #               'education_num': _float_feature(value=7),
    #               'gender': _bytes_feature(value='Male'.encode()),
    #               'hours_per_week': _float_feature(value=40),
    #               'native_country': _bytes_feature(value='United-States'.encode()),
    #               'occupation': _bytes_feature(value='Machine-op-inspct'.encode()),
    #               'relationship': _bytes_feature(value='Own-child'.encode()),
    #               'workclass': _bytes_feature(value='Private'.encode())}
    # label = 0
    data = _read_test_input()
    print(data)
    feature_dict = pred_input_fn(data)

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    serialized = example.SerializeToString()

    request.inputs['inputs'].CopyFrom(
        tf.contrib.util.make_tensor_proto(serialized, shape=[1]))

    result_future = stub.Predict.future(request, 5.0)
    prediction = result_future.result().outputs['scores']

    # print('True label: ' + str(label))
    print('Prediction: ' + str(np.argmax(prediction.float_val)))

if __name__ == '__main__':
    tf.app.run()