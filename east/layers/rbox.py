# -*- coding: utf-8 -*-
"""
Created on 2019/3/31 下午4:51

生成rbox的四个坐标

@author: mick.yi

"""
import tensorflow as tf
import numpy as np


def dist_to_box(distances, angles):
    """
    将四边距离和角度转为顶点坐标
    x′=xcosθ−ysinθ
    y′=xsinθ+ycosθ
    :param distances: [batch_size,H,W,(dist_top,dist_right,dist_bottom,dist_left)]
    :param angles: [batch_size,H,W,1]
    :return:
    """
    # lt,rt,rb,lb
    x = tf.stack([distances[..., 3] * -1.,  # lt
                  distances[..., 1],  # rt
                  distances[..., 1],  # rb
                  distances[..., 3] * -1.], axis=-1)  # lb; [batch_size,H,W,4]
    # lt,rt,rb,lb
    y = tf.stack([distances[..., 0] * -1,
                  distances[..., 0] * -1,
                  distances[..., 2],
                  distances[..., 2]], axis=-1)  # [batch_size,H,W,4]
    # 角度约束到0~90
    angles = angles[..., 0]  # 去除最后一维
    angles = tf.where(tf.less(angles, 0.), tf.zeros_like(angles), angles)
    angles = tf.where(tf.greater(angles, np.pi / 2.),
                      tf.ones_like(angles) * (np.pi / 2.), angles)

    # 处理角度,大于45,顺时针，小于45逆时针
    angles = tf.where(tf.greater(angles, np.pi / 4.),
                      np.pi / 2. - angles,  # 90-angle
                      -angles)
    # 扩维[batch_size,H,W,1]
    angles = tf.expand_dims(angles, axis=-1)
    # 旋转后的新坐标
    cos = tf.cos(angles)
    sin = tf.sin(angles)
    new_x = x * cos - y * sin  # [batch_size,H,W,4]
    new_y = x * sin + y * cos  # [batch_size,H,W,4]

    rbox = tf.stack([new_x, new_y], axis=-1)  # [batch_size,H,W,4,2]
    return rbox
