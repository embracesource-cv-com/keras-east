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
    :return:rbox: [batch_size,H,W,4,(x,y)]
    """
    # 计算中心点偏移(旋转在矩形中心点进行)
    half_h = (distances[..., 0] + distances[..., 2]) / 2.
    half_w = (distances[..., 1] + distances[..., 3]) / 2.
    # 当前点相对于矩形框中心点的偏移
    shift_y = distances[..., 0] - half_h
    shift_x = distances[..., 3] - half_w  # 左边距离

    # lt,rt,rb,lb
    x = tf.stack([half_w * -1.,  # lt
                  half_w,  # rt
                  half_w,  # rb
                  half_w * -1.], axis=-1)  # lb; [batch_size,H,W,4]
    # lt,rt,rb,lb
    y = tf.stack([half_h * -1,
                  half_h * -1,
                  half_h,
                  half_h], axis=-1)  # [batch_size,H,W,4]
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
    # 旋转后的顶点新坐标
    cos = tf.cos(angles)
    sin = tf.sin(angles)
    new_x = x * cos - y * sin  # [batch_size,H,W,4]
    new_y = x * sin + y * cos  # [batch_size,H,W,4]

    # 旋转后当前点坐标
    shift_x = tf.expand_dims(shift_x, axis=-1)
    shift_y = tf.expand_dims(shift_y, axis=-1)
    new_shift_x = shift_x * cos - shift_y * sin
    new_shift_y = shift_x * sin + shift_y * cos

    # 处理偏移
    new_x -= new_shift_x
    new_y -= new_shift_y

    # 相对坐标转绝对坐标
    shape = tf.shape(distances)  # (batch_size,H,W,4)
    H, W = shape[1], shape[2]

    ys = tf.range(H, dtype=tf.float32)  # [H]
    ys = tf.expand_dims(tf.expand_dims(ys, axis=-1), axis=-1)  # [H,1,1]
    xs = tf.range(W, dtype=tf.float32)  # [W]
    xs = tf.expand_dims(xs, axis=-1)  # [W,1]

    new_x += xs
    new_y += ys

    rbox = tf.stack([new_x, new_y], axis=-1)  # [batch_size,H,W,4,2]
    return rbox
