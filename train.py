# -*- coding: utf-8 -*-
"""
   File Name：     train
   Description :  训练
   Author :       mick.yi
   date：          2019/4/2
"""

import os
import sys
import tensorflow as tf
import keras
import argparse
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from east.layers import models
from east.config import cur_config as config
from east.utils import file_utils
from east.utils.generator import Generator
from east.preprocess import reader


def set_gpu_growth():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    session = tf.Session(config=cfg)
    keras.backend.set_session(session)


def get_call_back():
    """
    定义call back
    :return:
    """
    checkpoint = ModelCheckpoint(filepath='/tmp/east.{epoch:03d}.h5',
                                 monitor='acc',
                                 verbose=1,
                                 save_best_only=False,
                                 period=5)

    # 验证误差没有提升
    lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   cooldown=0,
                                   patience=10,
                                   min_lr=1e-5)
    log = TensorBoard(log_dir='log')
    return [lr_reducer, checkpoint, log]


def main(args):
    set_gpu_growth()
    # 加载标注
    annotation_files = file_utils.get_sub_files(config.IMAGE_GT_DIR)
    image_annotations = [reader.load_annotation(file,
                                                config.IMAGE_DIR) for file in annotation_files]
    # 过滤不存在的图像，ICDAR2017中部分图像找不到
    image_annotations = [ann for ann in image_annotations if os.path.exists(ann['image_path'])]
    # 加载模型
    m = models.east_net(config, 'train')
    models.compile(m, config, loss_names=['score_loss', 'dist_loss', 'angle_loss'])
    if args.init_epochs > 0:
        m.load_weights('/tmp/east.{:03d}.h5'.format(args.init_epochs), by_name=True)
    else:
        m.load_weights(config.PRE_TRAINED_WEIGHT, by_name=True)
    m.summary()
    # 生成器
    generator = Generator(config.IMAGE_SHAPE, image_annotations[-100:],
                          config.IMAGES_PER_GPU, config.TEXT_MIN_SIZE)
    val_gen = Generator(config.IMAGE_SHAPE, image_annotations[:-100],
                        config.IMAGES_PER_GPU, config.TEXT_MIN_SIZE)
    # 训练
    m.fit_generator(generator.gen(),
                    steps_per_epoch=generator.size // config.IMAGES_PER_GPU,
                    epochs=args.epochs,
                    initial_epoch=args.init_epochs,
                    verbose=True,
                    callbacks=get_call_back(),
                    validation_data=val_gen.gen(),
                    validation_steps=val_gen.size,
                    workers=2,
                    use_multiprocessing=True)

    # 保存模型
    m.save(config.WEIGHT_PATH)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--epochs", type=int, default=100, help="epochs")
    parse.add_argument("--init_epochs", type=int, default=0, help="epochs")
    argments = parse.parse_args(sys.argv[1:])
    main(argments)
