# -*- coding: utf-8 -*-
"""
   File Name：     model
   Description : 模型层
   Author :       mick.yi
   date：          2019/4/1
"""
import keras
from keras import layers, Input, Model
import tensorflow as tf
from east.layers.base_net import resnet50
from east.layers.losses import balanced_cross_entropy, iou_loss, angle_loss
from east.layers.rbox import dist_to_box


def merge_block(f_pre, f_cur, out_channels, index):
    """
    east网络特征合并块
    :param f_pre:
    :param f_cur:
    :param out_channels:输出通道数
    :param index:block index
    :return:
    """
    # 上采样
    up_sample = layers.UpSampling2D(size=2, name="east_up_sample_f{}".format(index - 1))(f_pre)
    # 合并
    merge = layers.Concatenate(name='east_merge_{}'.format(index))([up_sample, f_cur])
    # 1*1 降维
    x = layers.Conv2D(out_channels, (1, 1), padding='same', name='east_reduce_channel_conv_{}'.format(index))(merge)
    x = layers.BatchNormalization(name='east_reduce_channel_bn_{}'.format(index))(x)
    x = layers.Activation(activation='relu', name='east_reduce_channel_relu_{}'.format(index))(x)
    # 3*3 提取特征
    x = layers.Conv2D(out_channels, (3, 3), padding='same', name='east_extract_feature_conv_{}'.format(index))(x)
    x = layers.BatchNormalization(name='east_extract_feature_bn_{}'.format(index))(x)
    x = layers.Activation(activation='relu', name='east_extract_feature_relu_{}'.format(index))(x)
    return x


def east(features):
    """
    east网络头
    :param features: 特征列表： f1, f2, f3, f4分别代表32,16,8,4倍下采样的特征
    :return:
    """
    f1, f2, f3, f4 = features
    # 特征合并分支
    h2 = merge_block(f1, f2, 128, 2)
    h3 = merge_block(h2, f3, 64, 3)
    h4 = merge_block(h3, f4, 32, 4)
    # 提取g4特征
    x = layers.Conv2D(32, (3, 3), padding='same', name='east_g4_conv')(h4)
    x = layers.BatchNormalization(name='east_g4_bn')(x)
    x = layers.Activation(activation='relu', name='east_g4_relu')(x)

    # 预测得分
    predict_score = layers.Conv2D(1, (1, 1), name='predict_score_map')(x)
    # 预测距离
    predict_geo_dist = layers.Conv2D(4, (1, 1), name='predict_geo_dist')(x)
    # 预测角度
    predict_geo_angle = layers.Conv2D(1, (1, 1), name='predict_geo_angle')(x)

    return predict_score, predict_geo_dist, predict_geo_angle


def east_net(config, stage='train'):
    # 输入
    h, w = list(config.IMAGE_SHAPE)[:2]
    h, w = h / 4, w / 4
    input_image = Input(shape=config.IMAGE_SHAPE, name='input_image')
    input_score_map = Input(shape=(h, w), name='input_score')
    input_geo_dist = Input(shape=(h, w, 4), name='input_geo_dist')  # rbox 4个边距离
    input_geo_angle = Input(shape=(h, w), name='input_geo_angle')  # rbox 角度
    input_mask = Input(shape=(h, w), name='input_mask')

    # 网络
    features = resnet50(input_image)
    predict_score, predict_geo_dist, predict_geo_angle = east(features)

    if stage == 'train':
        # 增加损失函数层
        score_loss = layers.Lambda(lambda x: balanced_cross_entropy(*x), name='score_loss')(
            [input_score_map, predict_score, input_mask])
        geo_dist_loss = layers.Lambda(lambda x: iou_loss(*x), name='dist_loss')(
            [input_geo_dist, predict_geo_dist, input_score_map, input_mask])
        geo_angle_loss = layers.Lambda(lambda x: angle_loss(*x), name='angle_loss')(
            [input_geo_angle, predict_geo_angle, input_score_map, input_mask])

        return Model(inputs=[input_image, input_score_map, input_geo_dist, input_geo_angle, input_mask],
                     outputs=[score_loss, geo_dist_loss, geo_angle_loss])
    else:
        # 距离和角度转为顶点坐标
        vertex = layers.Lambda(lambda x: dist_to_box(*x))([predict_geo_dist, predict_geo_angle])
        return Model(inputs=input_image, outputs=[predict_score, vertex])


def compile(keras_model, config, loss_names=[]):
    """
    编译模型，增加损失函数，L2正则化以
    :param keras_model:
    :param config:
    :param loss_names: 损失函数列表
    :return:
    """
    # 优化目标
    optimizer = keras.optimizers.SGD(
        lr=config.LEARNING_RATE, momentum=config.LEARNING_MOMENTUM,
        clipnorm=config.GRADIENT_CLIP_NORM)
    # 增加损失函数，首先清除之前的，防止重复
    keras_model._losses = []
    keras_model._per_input_losses = {}

    for name in loss_names:
        layer = keras_model.get_layer(name)
        if layer is None or layer.output in keras_model.losses:
            continue
        loss = (tf.reduce_mean(layer.output, keepdims=True)
                * config.LOSS_WEIGHTS.get(name, 1.))
        keras_model.add_loss(loss)

    # 增加L2正则化
    # 跳过批标准化层的 gamma 和 beta 权重
    reg_losses = [
        keras.regularizers.l2(config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
        for w in keras_model.trainable_weights
        if 'gamma' not in w.name and 'beta' not in w.name]
    keras_model.add_loss(tf.add_n(reg_losses))

    # 编译
    keras_model.compile(
        optimizer=optimizer,
        loss=[None] * len(keras_model.outputs))  # 使用虚拟损失

    # 为每个损失函数增加度量
    for name in loss_names:
        if name in keras_model.metrics_names:
            continue
        layer = keras_model.get_layer(name)
        if layer is None:
            continue
        keras_model.metrics_names.append(name)
        loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * config.LOSS_WEIGHTS.get(name, 1.))
        keras_model.metrics_tensors.append(loss)


def add_metrics(keras_model, metric_name_list, metric_tensor_list):
    """
    增加度量
    :param keras_model: 模型
    :param metric_name_list: 度量名称列表
    :param metric_tensor_list: 度量张量列表
    :return: 无
    """
    for name, tensor in zip(metric_name_list, metric_tensor_list):
        keras_model.metrics_names.append(name)
        keras_model.metrics_tensors.append(tf.reduce_mean(tensor, keepdims=True))
