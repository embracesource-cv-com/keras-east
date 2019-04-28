# -*- coding: utf-8 -*-
"""
Created on 2019/4/9 上午6:18

@author: mick.yi

工具类

"""
import numpy as np
from shapely.geometry import Polygon


def poly_iou(poly_a, poly_b):
    """
    四边形iou
    :param poly_a: numpy数组[4,(x,y)]
    :param poly_b: numpy数组[4,(x,y)]
    :return:
    """
    a = Polygon(poly_a)
    b = Polygon(poly_b)
    if not a.is_valid or not b.is_valid:
        return 0
    inter = Polygon(a).intersection(Polygon(b)).area
    union = a.area + b.area - inter
    if union == 0:
        return 0
    else:
        return inter / union


def nms(polygons, scores, iou_threshold=0.3):
    """

    :param polygons: 多边形[n,4,(x,y)]
    :param scores: 得分 [n]
    :param iou_threshold:
    :return:
    """
    # 倒序排列
    order_indices = np.argsort(scores)[::-1]
    # 保留的索引号
    keep_indices = []

    while order_indices.size > 0:
        keep_indices.append(order_indices[0])  # 保留当前分值最大的polygon
        # 计算当前polygon与其他所有polygon的iou值
        ious = [poly_iou(polygons[order_indices[0]], polygons[i]) for i in order_indices[1:]]
        keep_indices = keep_indices[1:][np.array(ious) >= iou_threshold]
    # 返回保留的索引号
    return polygons[keep_indices], scores[keep_indices]


def poly_weighted_merge(poly_a, poly_b, score_a, score_b):
    """
    多边形加权合并
    :param poly_a:[n,4,(x,y)]
    :param poly_b:[n,4,(x,y)]
    :param score_a:标量
    :param score_b:标量
    :return:
    """
    poly = (poly_a * score_a + poly_b * score_b) / (score_a + score_b)
    score = score_a + score_b
    return poly, score


def locale_aware_nms(polygons, scores, threshold=0.3):
    """
    locale aware nms
    :param polygons: [n,4,(x,y)]
    :param scores: [n]
    :param threshold: iou阈值,大于阈值就合并多边形
    :return:
    """
    keep_polygons = []
    keep_scores = []
    p = None
    p_score = None
    for g, g_score in zip(polygons, scores):
        if p is not None and poly_iou(g, p) > threshold:
            p, p_score = poly_weighted_merge(g, p, g_score, p_score)
        else:
            if p is not None:
                keep_polygons.append(p)
                keep_scores.append(p_score)
            p = g
            p_score = g_score
    # 处理结尾
    if p is not None:
        keep_polygons.append(p)
        keep_scores.append(p_score)
    # 转为numpy
    keep_polygons = np.array(keep_polygons)
    keep_scores = np.array(keep_scores)

    return keep_polygons, keep_scores if keep_scores.size == 0 \
        else nms(keep_polygons, keep_scores, threshold)


def relative_to_absolute(polygons):
    """
    将预测的相对坐标改为决定坐标
    :param polygons: [batch,H,W,4,(x,y)]
    :return: polygons
    """
    shape = np.shape(polygons[0])  # (H,W,4,2)
    # 高度
    heights = np.arange(shape[0])
    polygons[..., 1] += heights[np.newaxis, :, np.newaxis, np.newaxis]
    # 宽度
    weights = np.arange(shape[1])
    polygons[..., 0] += weights[np.newaxis, np.newaxis, :, np.newaxis]

    return polygons
