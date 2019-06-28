import tensorflow as tf
import numpy as np
import sys
import os
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)

from lib.read_conf import Config


def normalizer_fn_builder(scaler, normalization_params):
    """normalizer_fn builder"""
    if scaler == 'min_max':
        return lambda x: (x - normalization_params[0]) / (
            max(normalization_params[1] - normalization_params[0], 0.001))
    elif scaler == 'standard':
        return lambda x: (x - normalization_params[0]) / normalization_params[1]
    else:
        return lambda x: tf.log(x)

def build_model_columns():
    def embedding_dim(dim):
        """empirical embedding dim"""
        return int(np.power(2, np.ceil(np.log(dim ** 0.5))))

    wide_columns = []
    wide_dim = 0
    deep_columns = []
    deep_dim = 0
    normalizer_scaler = 'min_max'
    _feature_conf_dic = Config().read_feature_conf()[0]
    for feature, conf in _feature_conf_dic.items():
        f_type, f_tran, f_param, is_deep = conf["type"], conf["transform"], conf["parameter"], conf["is_deep"]
        if feature == 'tag' or feature == 'main_actor':
            col = tf.feature_column.categorical_column_with_vocabulary_file(feature,
                                                                            vocabulary_file=f_param)
            wide_columns.append(col)
            wide_dim += int(conf["dim"])
            if is_deep:
                embed_dim = 20
                deep_columns.append(tf.feature_column.embedding_column(col,
                                                                       dimension=embed_dim,
                                                                       combiner='mean',
                                                                       initializer=None,
                                                                       ckpt_to_load_from=None,
                                                                       tensor_name_in_ckpt=None,
                                                                       max_norm=None,
                                                                       trainable=True))
                deep_dim += embed_dim

        else:
            if f_type == 'category':
                if f_tran == 'hash_bucket':
                    hash_bucket_size = int(f_param)
                    col = tf.feature_column.categorical_column_with_hash_bucket(feature,
                                                                                hash_bucket_size=hash_bucket_size,
                                                                                dtype=tf.string)
                    wide_columns.append(col)
                    wide_dim += hash_bucket_size
                    if is_deep:
                        embed_dim = embedding_dim(hash_bucket_size)
                        deep_columns.append(tf.feature_column.embedding_column(col,
                                                                               dimension=embed_dim,
                                                                               combiner='mean',
                                                                               initializer=None,
                                                                               ckpt_to_load_from=None,
                                                                               tensor_name_in_ckpt=None,
                                                                               max_norm=None,
                                                                               trainable=True))
                        deep_dim += embed_dim
                elif f_tran == 'vocab':
                    col = tf.feature_column.categorical_column_with_vocabulary_list(feature,
                                                                                    vocabulary_list=list(
                                                                                        map(str, f_param)),
                                                                                    dtype=None,
                                                                                    default_value=-1,
                                                                                    num_oov_buckets=0)
                    wide_columns.append(col)
                    wide_dim += len(f_param)
                    if is_deep:
                        deep_columns.append(tf.feature_column.indicator_column(col))
                        deep_dim += len(f_param)
                elif f_tran == 'identity':
                    num_buckets = f_param
                    col = tf.feature_column.categorical_column_with_identity(feature,
                                                                             num_buckets=num_buckets,
                                                                             default_value=0)
                    wide_columns.append(col)
                    wide_dim += num_buckets
                    if is_deep:
                        deep_columns.append(tf.feature_column.indicator_column(col))
                        deep_dim += num_buckets
            else:
                normalization_params = []
                normalization_params.append(int(f_param[0]))
                normalization_params.append(int(f_param[2]))
                normalizer_fn = normalizer_fn_builder(normalizer_scaler, tuple(normalization_params))
                col = tf.feature_column.numeric_column(feature,
                                                       shape=(1,),
                                                       default_value=0,
                                                       dtype=tf.float32,
                                                       normalizer_fn=normalizer_fn)
                wide_columns.append(col)
                wide_dim += 1
                if is_deep:
                    deep_columns.append(col)
                    deep_dim += 1

    # for cross_features, hash_bucket_size, is_deep in cross_feature_list:
    #     cf_list = []
    #     for f in cross_features:
    #
    #         f_type = feature_conf_dic[f]["type"]
    #         f_tran = feature_conf_dic[f]["transform"]
    #         f_param = feature_conf_dic[f]["parameter"]
    #         if f_tran == 'identity':
    #             cf_list.append(tf.feature_column.categorical_column_with_identity(f, num_buckets=f_param,
    #                                                                               default_value=0))
    #         else:
    #             cf_list.append(f)
    #     col = tf.feature_column.crossed_column(cf_list, int(hash_bucket_size))
    #     wide_columns.append(col)
    #     wide_dim += int(hash_bucket_size)
    #     if is_deep:
    #         deep_columns.append(tf.feature_column.embedding_column(col, dimension=embedding_dim(int(hash_bucket_size))))
    #         deep_dim += embedding_dim(int(hash_bucket_size))

    tf.logging.info('Build total {} wide columns'.format(len(wide_columns)))
    for col in wide_columns:
        tf.logging.debug('Wide columns: {}'.format(col))
    tf.logging.info('Wide input dimension is: {}'.format(wide_dim))

    tf.logging.info('Build total {} deep columns'.format(len(deep_columns)))
    for col in deep_columns:
        tf.logging.debug('Deep columns: {}'.format(col))
    tf.logging.info('Deep input dimension is: {}'.format(deep_dim))

    return wide_columns, deep_columns