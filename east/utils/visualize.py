# -*- coding: utf-8 -*-
"""
Created on 2019/4/14 下午7:55

@author: mick.yi

可视化工具类

"""
import matplotlib.pyplot as plt
from matplotlib import patches
import colorsys
import random
import numpy as np


def random_colors(N, bright=True):
    """
    生成随机RGB颜色
    :param N: 颜色数量
    :param bright:
    :return:
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_polygons(image, polygons, scores=None, figsize=(16, 16), ax=None, colors=None):
    """
    可视化多边形
    :param image: [H,W,3]
    :param polygons: [n,4,(x,y)]
    :param scores:
    :param figsize:
    :param ax:
    :param colors:
    :return:
    """
    auto_show = False
    if ax is None:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True
    if colors is None:
        colors = random_colors(len(polygons))

    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')

    for i, polygon in enumerate(polygons):
        color = colors[i]
        patch = patches.Polygon(polygon, facecolor=None, fill=False, color=color)
        ax.add_patch(patch)
        # 多边形得分
        x1, y1 = polygon[0][:]
        ax.text(x1, y1 - 1, scores[i] if scores is not None else '',
                color='w', size=11, backgroundcolor="none")
    ax.imshow(image.astype(np.uint8))
    if auto_show:
        plt.show()
