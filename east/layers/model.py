# -*- coding: utf-8 -*-
"""
   File Name：     model
   Description : 模型层
   Author :       mick.yi
   date：          2019/4/1
"""
from keras import layers


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
    predict_geo_angle = layers.Conv2D(1, (1, 1), name='predict_geo_dist')(x)

    return [predict_score, predict_geo_dist, predict_geo_angle]
