# -*- coding: utf-8 -*-
"""
   File Name：     base_net
   Description :  基网络
   Author :       mick.yi
   date：          2019/4/1
"""
import keras
from keras import layers, Model
from keras_applications.resnet50 import conv_block, identity_block


def resnet50(image_input):
    bn_axis = 3

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(image_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    # block 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    f4 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    # # 确定精调层
    no_train_model = Model(inputs=image_input, outputs=x)
    for l in no_train_model.layers:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = True
        else:
            l.trainable = False
    # block 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    f3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    # block 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    f2 = x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    # block 5
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    f1 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    return [f1, f2, f3, f4]
