# -*- coding: utf-8 -*-
"""
   File Name：     image
   Description :  图像处理工具类
   Author :       mick.yi
   date：          2019/2/18
"""
import skimage
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import random


def load_image(image_path):
    """
    加载图像
    :param image_path: 图像路径
    :return: [h,w,3] numpy数组
    """
    image = plt.imread(image_path)
    # 灰度图转为RGB
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
        image = np.tile(image, (1, 1, 3))
    elif image.shape[-1] == 1:
        image = skimage.color.gray2rgb(image)  # io.imread 报ValueError: Input image expected to be RGB, RGBA or gray
    # 标准化为0~255之间
    if image.dtype == np.float32:
        image *= 255
        image = image.astype(np.uint8)
    # 删除alpha通道
    return image[..., :3]


def resize_image_and_gt(image, output_size, gt_polygons=None):
    """
    按照输入大小缩放图像
    :param image:
    :param output_size:
    :param gt_polygons:
    :return:
    image: (H,W,3)
    image_meta: 元数据信息，详见compose_image_meta
    gt_boxes：图像缩放及padding后对于的GT 边框坐标 [N,(y1,x1,y2,x2)]
    """
    original_shape = image.shape
    # resize图像，并获取相关元数据信息
    h, w, window, scale, padding = resize_meta(original_shape[0], original_shape[1], output_size)
    image = resize_image(image, h, w, padding)

    # 组合元数据信息
    image_meta = compose_image_meta(np.random.randint(10000), original_shape, image.shape,
                                    window, scale)
    # 根据缩放及padding调整GT边框
    if gt_polygons is not None and gt_polygons.shape[0] > 0:
        gt_polygons = adjust_polygons(gt_polygons, padding, scale)

    return image, image_meta, gt_polygons


def random_crop_image(image, gt_window):
    """
    随机裁剪图像
    :param image: [H,W,C]
    :param gt_window: 标注区域 (y1,x1,y2,x2)
    :return: 裁剪后的图像和裁剪窗口
    """
    h, w = list(image.shape)[:2]
    y1, x1, y2, x2 = gt_window
    # 每边最多裁剪1/20
    crop_ratio = 0.05
    wy1 = np.random.randint(min(y1 + 1, h * crop_ratio))
    wx1 = np.random.randint(min(x1 + 1, w * crop_ratio))
    wy2 = h - np.random.randint(min(h - y2 + 1, h * crop_ratio))
    wx2 = w - np.random.randint(min(w - x2 + 1, w * crop_ratio))
    return image[wy1:wy2, wx1:wx2], [wy1, wx1, wy2, wx2]


def resize_image(image, h, w, padding):
    """
    缩放图像为正方形，指定长边大小，短边padding;
    :param image: numpy 数组(H,W,3)
    :param h: 缩放后的高度
    :param w: 缩放后的宽度
    :param padding:缩放后增加的padding
    :return: 缩放后的图像,元素图像的宽口位置，缩放尺寸，padding
    """
    image_dtype = image.dtype
    image = transform.resize(image, (h, w), order=1, mode='constant',
                             cval=0, clip=True, preserve_range=True)

    image = np.pad(image, padding, mode='constant', constant_values=0)
    return image.astype(image_dtype)


def resize_meta(h, w, max_dim):
    """
    计算resize的元数据信息
    :param h: 图像原始高度
    :param w: 图像原始宽度
    :param max_dim: 缩放后的边长
    :return:
    """
    scale = max_dim / max(h, w)  # 缩放尺寸
    # 新的高度和宽度
    h, w = round(h * scale), round(w * scale)

    # 计算padding
    top_pad = (max_dim - h) // 2
    bottom_pad = max_dim - h - top_pad
    left_pad = (max_dim - w) // 2
    right_pad = max_dim - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    # 计算窗口
    window = (top_pad, left_pad, h + top_pad, w + left_pad)  #
    return h, w, window, scale, padding


def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale):
    """
    组合图像元数据信息，返回numpy数据
    :param image_id:
    :param original_image_shape: 原始图像形状，tuple(H,W,3)
    :param image_shape: 缩放后图像形状tuple(H,W,3)
    :param window: 原始图像在缩放图像上的窗口位置（y1,x1,y2,x2)
    :param scale: 缩放因子
    :return:
    """
    meta = np.array(
        [image_id] +  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +  # size=3
        list(window) +  # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale]  # size=1
    )
    return meta


def parse_image_meta(meta):
    """
    解析图像元数据信息,注意输入是元数据信息数组
    :param meta: [12]
    :return:
    """
    image_id = meta[0]
    original_image_shape = meta[1:4]
    image_shape = meta[4:7]
    window = meta[7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[11]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32)
    }


def batch_parse_image_meta(meta):
    """
    解析图像元数据信息,注意输入是元数据信息数组
    :param meta: [batch,12]
    :return:
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32)
    }


def adjust_box(boxes, padding, scale):
    """
    根据填充和缩放因子，调整boxes的值
    :param boxes: numpy 数组; GT boxes [N,(y1,x1,y2,x2)]
    :param padding: [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    :param scale: 缩放因子
    :return:
    """
    boxes = boxes * scale
    boxes[:, 0::2] += padding[0][0]  # 高度padding
    boxes[:, 1::2] += padding[1][0]  # 宽度padding
    return boxes


def adjust_polygons(polygons, padding, scale):
    """
    根据填充和缩放因子，调整四边形的值
    :param polygons: numpy 数组; GT polygons[N,4,(x,y)]
    :param padding: [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    :param scale: 缩放因子
    :return:
    """
    polygons = polygons * scale
    polygons[:, :, 1] += padding[0][0]  # 高度padding
    polygons[:, :, 0] += padding[1][0]  # 宽度padding
    return polygons


def recover_detect_boxes(boxes, window, scale):
    """
    将检测边框映射到原始图像上，去除padding和缩放
    :param boxes: numpy数组，[n,(y1,x1,y2,x2)]
    :param window: [(y1,x1,y2,x2)]
    :param scale: 标量
    :return:
    """
    # 去除padding
    boxes[:, 0::2] -= window[0]
    boxes[:, 1::2] -= window[1]
    # 还原缩放
    boxes /= scale
    return boxes


def recover_detect_polygons(polygons, window, scale):
    """
    将检测四边形映射到原始图像上，去除padding和缩放
    :param polygons: numpy数组，[n,4,(x,y)]
    :param window: [(y1,x1,y2,x2)]
    :param scale: 标量
    :return:
    """
    # 去除padding
    polygons[:, 1::2] -= window[0]  # 高度
    polygons[:, 0::2] -= window[1]  # 宽度
    # 还原缩放
    polygons /= scale
    return polygons
