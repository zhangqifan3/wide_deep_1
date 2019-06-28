# 项目运行说明
使用机器：192.168.1.88  

运行环境：/home/guanyue/anaconda3/envs/tf_cpu36/bin/python3.6  

模型测试：

- `cd conf`
修改配置文件：model_conf.ini，测试数据改为自己的目录
- `cd python`
- `python train.py`

# 模型导出及特征解析规范
说明：该项目使用tf.estimator中官方wide&deep模型，使用tf.dataset读取数据。特征解析规范主要在导出和服务阶段使用，是一个字典`{name:output}`格式来描述输出签名

模型导出：

- 模型训练好以后，会保存在./wide_deep/model目录下，确认训练模型已保存。
- `cd python`
- `cd tensorflow_serving`
- `python export_savedmodel.py`
- 运行成功的话导出模型会保存在./wide_deep/export_model目录下，该模型用作线上部署

特征解析规范（各特征使用数据类型说明）：
```'did': VarLenFeature(dtype=tf.string)
'coocaa_v_id': VarLenFeature(dtype=tf.string)
'is_vip': VarLenFeature(dtype=tf.string)
'source': VarLenFeature(dtype=tf.string)
'tag': VarLenFeature(dtype=tf.string)
'director': VarLenFeature(dtype=tf.string)
'area': VarLenFeature(dtype=tf.string)
'year': VarLenFeature(dtype=tf.string)
'score': FixedLenFeature(shape=(1,),
dtype=tf.float32, default_value=(0,))
```

    
