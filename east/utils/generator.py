# -*- coding: utf-8 -*-
"""
Created on 2019/3/31 上午11:24

生成器

@author: mick.yi

"""
import numpy as np
import cv2
from east.utils import geo_utils


def shrink_poly_edge(poly, rs, edge_index):
    """
    内缩四边形的某一条边
    :param poly: [4,(xi,yi)] i<4,第一个点为左上，顺时针排列
    :param rs: 每个顶点的R距离，R距离为顶点的短边距离
    :param edge_index: 需要内缩的边索引号
    :return:
    """
    # 边对应的两个顶点索引号
    idx1, idx2 = edge_index, (edge_index + 1) % 4
    # 边对应的两个顶点
    p1, p2 = poly[idx1], poly[idx2]
    # 顶点对应的r
    r1, r2 = rs[idx1], rs[idx2]
    # 两边内缩
    poly[idx1] = geo_utils.point_shift_on_line(p1, p2, r1 * 0.3)
    poly[idx2] = geo_utils.point_shift_on_line(p2, p1, r2 * 0.3)


def shrink_polygon(poly):
    """
    内缩四边形
    :param poly: [4,(xi,yi)] i<4,第一个点为左上，顺时针排列
    :return:
    """
    # top,right,bottom,left四边的距离
    dist_edges = [np.linalg.norm(poly[i] - poly[(i + 1) % 4]) for i in range(4)]
    # lt,rt,rb,lb四个顶点的ri值，ri为顶点相邻两边的短边长度
    rs = [min(dist_edges[i], dist_edges[(i + 3) % 4]) for i in range(4)]

    # 先缩长边再缩短边;对边确定长边
    if dist_edges[0] + dist_edges[2] > dist_edges[1] + dist_edges[3]:
        # 长边
        shrink_poly_edge(poly, rs, 0)
        shrink_poly_edge(poly, rs, 2)
        # 短边
        shrink_poly_edge(poly, rs, 1)
        shrink_poly_edge(poly, rs, 3)
    else:
        # 长边
        shrink_poly_edge(poly, rs, 1)
        shrink_poly_edge(poly, rs, 3)
        # 短边
        shrink_poly_edge(poly, rs, 0)
        shrink_poly_edge(poly, rs, 2)
    return poly


def gen_gt(h, w, polygons, min_text_size):
    """
    生成单个GT
    :param h: 输入高度
    :param w: 输入宽度
    :param polygons: 文本区域多边形[n,4]
    :param min_text_size: 文本区域最小边长
    :return: score_map, geo_map, mask
    """
    poly_mask = np.zeros((h, w), dtype=np.uint8)  # 第几个多边形区域
    score_map = np.zeros((h, w), dtype=np.uint8)  # 是否为文本区域
    geo_map = np.zeros((h, w, 5), dtype=np.float32)  # rbox 4边距离和角度
    mask = np.ones((h, w), dtype=np.uint8)  # 是否参与训练
    for index, polygon in enumerate(polygons):
        # 最小外接矩形和矩形角度
        rect, angle = geo_utils.min_area_rect_and_angle(polygon)
        # 向内收缩多边形
        shrinked_polygon = shrink_polygon(polygon.copy()).astype(np.int32)  # 坐标转为整型
        cv2.fillPoly(poly_mask, shrinked_polygon, index + 1)  # 第index+1个多边形区域
        cv2.fillPoly(score_map, shrinked_polygon, 1)  # 正样本
        # 当前多边形的坐标
        xs, ys = np.where(poly_mask == (index + 1))
        for x, y in zip(xs, ys):
            distances = geo_utils.dist_point_to_rect(np.array([x, y], np.float32), rect)  # [4]
            geo_map[x, y, :4] = distances
            geo_map[x, y, 4] = angle

        # 过滤太小的文本区域
        dist_edges = [np.linalg.norm(polygon[i] - polygon[(i + 1) % 4]) for i in range(4)]
        if min(dist_edges) < min_text_size:
            cv2.fillPoly(mask, shrinked_polygon, 0)

    return score_map, geo_map, mask


class Generator(object):
    def __init__(self, input_shape, annotation_list, batch_size, min_text_size, **kwargs):
        """

        :param h:
        :param w:
        :param annotation_list:
        :param kwargs:
        """
        self.input_shape = input_shape
        self.annotation_list = annotation_list
        self.batch_size = batch_size
        self.min_text_size = min_text_size
        self.size = len(annotation_list)
        super(Generator, self).__init__(**kwargs)

    def gen(self):
        h, w = list(self.input_shape)[:2]
        while True:
            score_map = np.zeros((self.batch_size, h, w), dtype=np.uint8)  # 是否为文本区域
            geo_map = np.zeros((self.batch_size, h, w, 5), dtype=np.float32)  # rbox 4边距离和角度
            mask = np.ones((self.batch_size, h, w), dtype=np.uint8)  # 是否参与训练
            # 随机选择
            indices = np.random.choice(self.size, self.batch_size, replace=False)
            for i, index in enumerate(indices):
                score_map[i], geo_map[i], mask[i] = gen_gt(h,
                                                           w,
                                                           self.annotation_list[i]['polygons'],
                                                           self.min_text_size)

            yield {"input_image": None,
                   "input_score": score_map,
                   "input_geo": geo_map,
                   "input_mask": mask}
