# -*- coding: utf-8 -*-
"""
Created on 2019/3/31 上午9:45

损失函数

@author: mick.yi

"""
import tensorflow as tf


def iou_loss(y_true, y_pred, y_score, mask):
    """
    iou损失函数   -log(iou)
    :param y_true: [batch_size,H,W,(dist_top,dist_right,dist_bottom,dist_left)]
    :param y_pred: [batch_size,H,W,(dist_top,dist_right,dist_bottom,dist_left)]
    :param y_score: [batch_size,H,W] 正负样本标志，1-文本区域，0-背景
    :param mask: [batch_size,H,W] 是否参与训练，1-参与，0-不参与
    :return:
    """
    # 计算交集
    h = tf.minimum(y_true[..., 0], y_pred[..., 0]) + tf.minimum(y_true[..., 2], y_pred[..., 2])
    w = tf.minimum(y_true[..., 1], y_pred[..., 1]) + tf.minimum(y_true[..., 3], y_pred[..., 3])
    overlap = h * w  # [batch_size,H,W]
    # 计算R1,R2面积
    area_true = tf.reduce_sum(y_true[..., ::2], axis=-1) * tf.reduce_sum(y_true[..., 1::2], axis=-1)
    area_pred = tf.reduce_sum(y_pred[..., ::2], axis=-1) * tf.reduce_sum(y_pred[..., 1::2], axis=-1)
    # iou
    union = area_true + area_pred - overlap
    iou = overlap / union
    # 只处理参与训练的部分
    mask = y_score * mask  # 正样本计算损失
    iou = tf.boolean_mask(iou, tf.cast(mask, tf.bool))
    return -tf.log(iou)


def angle_loss(y_true, y_pred, y_score, mask):
    """
    角度损失函数 1-cosine(y_pred-y_true)
    :param y_true: [batch_size,H,W]
    :param y_pred: [batch_size,H,W,1] 这里多了一维
    :param y_score: [batch_size,H,W] 正负样本标志，1-文本区域，0-背景
    :param mask: [batch_size,H,W] 是否参与训练，1-参与，0-不参与
    :return:
    """
    loss = 1. - tf.cos(y_pred[..., 0] - y_true)
    mask = y_score * mask  # 正样本计算损失
    return tf.boolean_mask(loss, tf.cast(mask, tf.bool))


def balanced_cross_entropy(y_true, logits, mask):
    """
    平衡交叉熵
    :param y_true: [batch_size,H,W] 1-文本，0-非文本
    :param logits: [batch_size,H,W,1] 预测文本得分,预测的多了一维
    :param mask: [batch_size,H,W] 是否参与训练，1-参与，0-不参与
    :return:
    """
    mask = tf.cast(mask, tf.bool)
    y_true = tf.boolean_mask(y_true, mask)
    logits = tf.boolean_mask(logits[..., 0], mask)

    # 统计正负样本数
    pos_num = tf.minimum(tf.reduce_sum(y_true), 1.)  # 平滑
    neg_num = tf.minimum(tf.reduce_sum(1. - y_true), 1.)
    total = pos_num + neg_num
    # 正负样本权重
    weights = tf.where(tf.equal(y_true, 1.),
                       tf.ones_like(y_true) * neg_num / total,
                       tf.ones_like(y_true) * pos_num / total)
    # 计算损失函数
    loss = tf.nn.sigmoid_cross_entropy_with_logits(y_true, logits)
    loss = loss * weights

    return loss
