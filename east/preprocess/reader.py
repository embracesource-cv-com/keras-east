# -*- coding: utf-8 -*-
"""
   File Name：     reader
   Description :   读取icdar数据
   Author :       mick.yi
   date：          2019/3/14
"""
import numpy as np
import os
import glob


def load_annotation(annotation_path, image_dir):
    """
    加载标注信息
    :param annotation_path:
    :param image_dir:
    :return:
    """
    image_annotation = {}
    # 文件名称，路径
    base_name = os.path.basename(annotation_path)
    image_name = base_name[3:-3] + '*'  # 通配符 gt_img_3.txt,img_3.jpg or png
    image_annotation["annotation_path"] = annotation_path
    image_annotation["image_path"] = glob.glob(os.path.join(image_dir, image_name))[0]
    image_annotation["file_name"] = os.path.basename(image_annotation["image_path"])  # 图像文件名
    # 读取边框标注
    polygons = []  # 四边形

    with open(annotation_path, "r", encoding='utf-8') as f:
        lines = f.read().encode('utf-8').decode('utf-8-sig').splitlines()
        # lines = f.readlines()
        # print(lines)
    for line in lines:
        line = line.strip().split(",")
        # 左上、右上、右下、左下 四个坐标 如：377,117,463,117,465,130,378,130
        lt_x, lt_y, rt_x, rt_y, rb_x, rb_y, lb_x, lb_y = map(float, line[:8])
        polygons.append(np.array([lt_x, lt_y, rt_x, rt_y, rb_x, rb_y, lb_x, lb_y]).reshape((4, 2)))

    image_annotation["polygons"] = np.asarray(polygons)
    return image_annotation
