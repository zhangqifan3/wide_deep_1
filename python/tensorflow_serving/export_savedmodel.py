import os
import sys
import tensorflow as tf

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)
from os.path import dirname, abspath
from lib.build_estimator import build_model_columns
from lib.build_estimator import build_estimator

EXPORT_MODEL = os.path.join(dirname(dirname(dirname(abspath(__file__)))), 'export_model')

def main():
    wide_columns, deep_columns = build_model_columns()
    feature_columns = wide_columns + deep_columns
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    print(feature_spec)
    serving_input_receiver_fn = \
        tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    export_dir = EXPORT_MODEL
    model = build_estimator()
    model.export_savedmodel(export_dir,
                            serving_input_receiver_fn,
                            as_text= False)


if __name__ == '__main__':
    main()