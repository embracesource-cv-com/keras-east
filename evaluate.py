# -*- coding: utf-8 -*-
"""
Created on 2019/4/14 下午8:00

@author: mick.yi

评估入口

"""

import sys
import os
import numpy as np
import argparse
from east.utils import image_utils, file_utils, common_utils
from east.utils.generator import EvaluateGenerator
from east.config import cur_config as config
from east.layers import models
import datetime
import lanms


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # 覆盖参数
    if args.weight_path is not None:
        config.WEIGHT_PATH = args.weight_path
    config.IMAGES_PER_GPU = 1
    # config.IMAGE_SHAPE = (1024, 1024, 3)
    # 图像路径
    image_path_list = file_utils.get_sub_files(args.image_dir)

    # 加载模型
    m = models.east_net(config, 'test')
    m.load_weights(config.WEIGHT_PATH, by_name=True)

    # 预测
    start_time = datetime.datetime.now()
    gen = EvaluateGenerator(config.IMAGE_SHAPE, image_path_list)
    predict_scores, predict_vertex, image_metas = m.predict_generator(generator=gen.gen(),
                                                                      steps=len(image_path_list),
                                                                      use_multiprocessing=True)
    end_time = datetime.datetime.now()
    print("======完成{}张图像评估，耗时:{} 秒".format(len(image_path_list), end_time - start_time))

    # 后处理
    image_metas = image_utils.batch_parse_image_meta(image_metas)

    size = len(image_path_list)
    polygons_list = []
    scores_list = []
    # 高度和宽度打平,逐个样本处理
    for p, s, win, scale in zip(np.reshape(predict_vertex, (size, -1, 4, 2)),
                                np.reshape(predict_scores, (size, -1)),
                                image_metas['window'],
                                image_metas['scale']):
        # 过滤低分值多边形
        ix = np.where(s >= 0.5)
        p = p[ix]
        s = s[ix]
        polygons, scores = common_utils.locale_aware_nms(p, s, 0.3)  # nms'
        # polys = lanms.merge_quadrangle_n9(np.concatenate([np.reshape(p, (-1, 8)),
        #                                                  s[:, np.newaxis]], axis=1), 0.3)
        # polygons = np.reshape(polys[:, :8], (-1, 4, 2))
        # scores = polys[:, 8]
        polygons *= 4  # 转为网络输入的大小
        # 还原检测边框到原图
        polygons = image_utils.recover_detect_polygons(polygons, win, scale)

        polygons_list.append(polygons)
        scores_list.append(scores)

    # 写入文档中
    for image_path, polygons in zip(image_path_list, polygons_list):
        output_filename = os.path.splitext('res_' + os.path.basename(image_path))[0] + '.txt'
        with open(os.path.join(args.output_dir, output_filename), mode='w') as f:
            for poly in polygons.astype(np.int32):
                f.write("{},{},{},{},{},{},{},{}\r\n".format(poly[0][0],
                                                             poly[0][1],
                                                             poly[1][0],
                                                             poly[1][1],
                                                             poly[2][0],
                                                             poly[2][1],
                                                             poly[3][0],
                                                             poly[3][1]))
    print("======总耗时:{} 秒".format(datetime.datetime.now() - start_time))


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--image_dir", type=str, help="image dir")
    parse.add_argument("--output_dir", type=str, help="output dir")
    parse.add_argument("--weight_path", type=str, default=None, help="weight path")
    argments = parse.parse_args(sys.argv[1:])
    main(argments)
