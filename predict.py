# -*- coding: utf-8 -*-
"""
Created on 2019/5/3 上午7:54

@author: mick.yi

预测入口

"""
import os
import sys
import numpy as np
import argparse
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from east.utils import image_utils, visualize, common_utils
from east.config import cur_config as config
from east.layers import models


def main(args):
    # 覆盖参数
    if args.weight_path is not None:
        config.WEIGHT_PATH = args.weight_path
    config.IMAGES_PER_GPU = 1
    # 加载图片
    image = image_utils.load_image(args.image_path)
    image, image_meta, _ = image_utils.resize_image_and_gt(image,
                                                           config.input_shape[0])

    # 加载模型
    m = models.ctpn_net(config, 'test')
    m.load_weights(config.WEIGHT_PATH, by_name=True)
    # m.summary()

    # 模型预测
    predict_scores, predict_vertex, image_metas = m.predict([np.array([image]), np.array([image_meta])])
    # 后处理
    polygons = common_utils.relative_to_absolute(predict_vertex)[0]  # 相对坐标转绝对坐标
    polygons *= 4  # 转为网络输入的大小
    scores = predict_scores[0]
    # 过滤低分值多边形
    ix = np.where(scores >= 0.5)
    polygons = polygons[ix]
    scores = scores[ix]
    polygons, scores = common_utils.locale_aware_nms(polygons, scores, 0.3)

    # 可视化保存图像
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(1, 1, 1)
    visualize.display_polygons(image, polygons.reshape((-1, 8)), scores,
                               ax=ax)
    image_name = os.path.basename(args.image_path)
    fig.savefig('{}.{}.jpg'.format(os.path.splitext(image_name)[0], int(config.USE_SIDE_REFINE)))


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--image_path", type=str, help="image path")
    parse.add_argument("--weight_path", type=str, default=None, help="weight path")
    argments = parse.parse_args(sys.argv[1:])
    main(argments)
