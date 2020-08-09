#!/usr/bin/env python
# _*_ encoding: utf-8 _*_
"""
@File       :   demo_classification.py
@Time       :   2020/8/4 15:10
@Author     :   lst
@Version    :   1.0
@Contact    :   liushitong@360humi.com
@License    :   (C)Copyright 2020-2021, humi
@Desc       :   None
"""
from __future__ import division, absolute_import, print_function
import os
# import sys
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)


import tensorflow as tf
import time
import src.args.parser as parser
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from src.models import cnn
from src.util.loader import get_generator


def train(flags):
    # timestamp
    tst = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))

    # load training data
    train_generator, valid_generator, _ = get_generator(flags)

    # tensorboard callback
    tbcb = tf.keras.callbacks.TensorBoard(log_dir=flags.log_path + '/' + tst)

    # create model, compile and fit
    model_name = 'cnn3_'
    model = cnn.cnn3(
        num_classes=train_generator.num_classes,
        image_shape=train_generator.image_shape
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=flags.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.fit(
        x=train_generator,
        validation_data=valid_generator,
        epochs=flags.maxepoch,
        callbacks=[tbcb]
    )

    # save model
    model.save(flags.save_path + '/' + model_name + tst)
    print('model:"' + model_name + tst + '" saved in ' + flags.save_path)


def test(flags):
    # load test data
    _, _, test_generator = get_generator(flags)

    # load the latest model
    md_lists = os.listdir(flags.save_path)
    md_lists.sort(key=lambda x: os.path.getmtime(flags.save_path + '/'))
    md_path = os.path.join(flags.save_path, md_lists[-1])
    model = tf.keras.models.load_model(md_path)

    # evaluate
    _, acc = model.evaluate(test_generator)
    print('The accuracy on test set is: {:6.3f}%'.format(acc * 100))


def predict(flags):
    # load the latest model
    md_lists = os.listdir(flags.save_path)
    md_lists.sort(key=lambda x: os.path.getmtime(flags.save_path + '/'))
    md_path = os.path.join(flags.save_path, md_lists[-1])
    model = tf.keras.models.load_model(md_path)
    model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    # load class names
    _, _, generator = get_generator(flags)
    class_names = {value: key for key, value in generator.class_indices.items()}

    # image path
    paths = listdir(flags.pred_path)
    for path in paths:
        img_raw = tf.io.read_file(flags.pred_path + '/' + path)
        img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
        img_tensor = tf.image.resize(img_tensor, flags.target_size)
        img = img_tensor.numpy() / 255
        prob = model(np.expand_dims(img, 0))
        ans = class_names[np.argmax(prob)]
        plt.figure(ans)
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    print('start')

    # parse arguments
    FLAGS = parser.get_args()

    if FLAGS.train:
        train(FLAGS)
    if FLAGS.test:
        test(FLAGS)
    if FLAGS.predict:
        predict(FLAGS)

    print('done.')
