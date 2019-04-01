# -*- coding: utf-8 -*-
"""
Created on 2019/3/30 下午9:07

几何图形工具类

@author: mick.yi

"""

import numpy as np
import cv2


def dist_point_to_line(p1, p2, p3):
    """
    点p3到直线(p1,p2)的距离
    :param p1: [x,y]
    :param p2: [x,y]
    :param p3: [x,y]
    :return:
    """
    # 以(p1,p3),(p2,p3)为边的平行四边形面积
    area = np.cross(p1 - p3, p2 - p3)
    # 点p3到直线(p1,p2)的距离=area除以边(p1,p2)的距离
    dist_p1_p2 = np.linalg.norm(p1 - p2)
    return area / dist_p1_p2


def elem_cycle_shift(elements, shift):
    """
    将元素位移指定长度;
    如：elements=[1,2,3,4] shift=1 则返回[2,3,4,1]
                       shift=2 则返回[3,4,1,2]
                       shift=-1 则返回[4,1,2,3]
                       shift > 0 左移，反之，右移
    :param elements: 元素列表list of elem
    :param shift: 位移长度
    :return:
    """
    length = len(elements)
    return [elements[(i + shift) % length] for i in range(length)]


def min_area_rect_and_angle(polygon):
    """
    包围多边形的最小矩形框
    :param polygon:[n,(x,y)]
    :return:rect,angle
    """
    rect = cv2.minAreaRect(polygon)  # ((ctx,cty),(w,h),angle) angle为负
    box = cv2.boxPoints(rect)  # 边框四个坐标[4,2];顺时针排列,第一个坐标是y值最大的那个
    # 角度
    angle = abs(rect[2])
    if angle == 0:
        if box[0][0] == np.max(box[:, 0]):  # box[0]是右下角
            box = elem_cycle_shift(list(box), 2)
        else:  # box[0]是左下角
            box = elem_cycle_shift(list(box), 1)
    elif angle <= 45:  # box[0]是左下角
        box = elem_cycle_shift(list(box), 1)
    else:  # box[0]是右下角
        box = elem_cycle_shift(list(box), 2)

    # angle转为pi表示
    return box, angle * np.pi / 180


def dist_point_to_rect(point, box):
    """
    点到矩形框4条边的距离
    :param point: numpy数组[(x,y)]
    :param box: 矩形框的顶点，numpy数组[4,(xi,yi)];  (x0,y0)代表左上顶点，顺时针方向排列
    :return: 返回点到矩形框上边，右边，下边，左边的距离
    """
    dist_top, dist_right, dist_bottom, dist_left = [dist_point_to_line(
        box[i], box[(i + 1) % 4], point) for i in range(4)]
    return np.array([dist_top, dist_right, dist_bottom, dist_left])


def point_shift_on_line(p1, p2, dist_shift):
    """
    p1向p2移动指定距离的点
    :param p1: 直线的第一个端点
    :param p2: 直线的第二个端点
    :param dist_shift: 相对于第一个点的位移距离
    :return: 目标点
    """
    # 直线距离
    dist_line = np.linalg.norm(p1 - p2)
    # 按比例相加，就是目标点
    point = p1 + (p2 - p1) * dist_shift / dist_line
    return point


def fit_line(p1, p2):
    """
    拟合直线
    :param p1: [x,y]
    :param p2: [x,y]
    :return: 拟合方程系数
    """
    x = [p1[0], p2[0]]
    y = [p1[1], p2[1]]
    # 拟合直线 ax+by+c = 0
    if x[0] == x[1]:
        return [1., 0., -x[0]]
    else:
        [a, c] = np.polyfit(x, y, deg=1)  # b=-1
        return [a, -1., c]


def vertical_line(p1, p2, p3):
    """
    过点p3,与直线p1,p2垂直的线
    互相垂直的线，斜率互为互倒数
    :param p1: [x,y]
    :param p2: [x,y]
    :param p3: [x,y]
    :return: 新方程的系数[na,nb,nc]
    """
    line = fit_line(p1, p2)
    a, b, c = line  # ax+by+c=0;一般b为-1
    # 以下获取垂线的系数na,nb,nc
    if a == 0.:  # 原方程为y=c ;新方程为x=-nc
        na = 1.
        nb = 0.
    elif b == 0.:  # 原方程为x=-c;新方程为y=nc
        na = 0.
        nb = -1.
    else:  # 斜率互为互倒数 a*na=-1;
        na = -1. / a
        nb = -1.
    # 根据ax+by+c=0求解系数c
    nc = -(na * p3[0] + nb * p3[1])
    return [na, nb, nc]


def cross_point(line1, line2):
    """
    两条直线的交点
    y=(c2*a1-c1*a2)/(b1*a2-b2*a1)
    x=(c2*b1-c1*b2)/(a1*b2-a2*b1)
    :param line1: [a1,b1,c1]
    :param line2: [a2,b2,c2]
    :return:
    """
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    x = (c2 * b1 - c1 * b2) / (a1 * b2 - a2 * b1)
    y = (c2 * a1 - c1 * a2) / (b1 * a2 - b2 * a1)
    return np.array([x, y])


def rotate(point, angle):
    """
    饶原点旋转
    x′=xcosθ−ysinθ
    y′=xsinθ+ycosθ
    :param point:
    :param angle: 旋转角度
    :return: 旋转后的坐标
    """
    x, y = point
    new_x = x * np.cos(angle) - y * np.sin(angle)
    new_y = x * np.sin(angle) + y * np.cos(angle)
    return np.array([new_x, new_y])


def main():
    print(elem_cycle_shift([1, 2, 3, 4], 1))
    print(elem_cycle_shift([1, 2, 3, 4], 2))
    print(elem_cycle_shift([1, 2, 3, 4], -1))


if __name__ == '__main__':
    main()
