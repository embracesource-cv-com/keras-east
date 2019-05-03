# -*- coding: utf-8 -*-
"""
   File Name：     config
   Description :  配置类
   Author :       mick.yi
   date：          2019/4/1
"""


class Config(object):
    IMAGES_PER_GPU = 4
    IMAGE_SHAPE = (640, 640, 3)
    TEXT_MIN_SIZE = 10

    # 训练超参数
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    # 权重衰减
    WEIGHT_DECAY = 0.0005,
    GRADIENT_CLIP_NORM = 1.0

    LOSS_WEIGHTS = {
        "score_loss": 1.,
        "dist_loss": 1.,
        "angle_loss": 10.
    }

    # 预训练模型
    PRE_TRAINED_WEIGHT = '/opt/pretrained_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    WEIGHT_PATH = '/tmp/east.h5'

    # 数据集路径
    IMAGE_DIR = '/opt/dataset/OCR/ICDAR_2015/train_images'
    IMAGE_GT_DIR = '/opt/dataset/OCR/ICDAR_2015/train_gt'


cur_config = Config()
