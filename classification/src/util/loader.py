#!/usr/bin/env python
# _*_ encoding: utf-8 _*_
"""
@File       :   loader.py
@Time       :   2020/8/7 15:45
@Author     :   lst
@Version    :   1.0
@Contact    :   liushitong@360humi.com
@License    :   (C)Copyright 2020-2021, humi
@Desc       :   None
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_generator(flags):
    """ get training valid and test generator

    :param flags: arguments
    :return: generators
    """
    train_datagen = ImageDataGenerator(
        rescale=flags.rescale,
        shear_range=flags.shear_range,
        zoom_range=flags.zoom_range,
        rotation_range=flags.rotation_range,
        horizontal_flip=flags.horizontal_flip
    )
    train_generator = train_datagen.flow_from_directory(
        directory=flags.train_dir,
        target_size=flags.target_size,
        batch_size=flags.bsize,
        class_mode=flags.class_mode
    )

    valid_datagen = ImageDataGenerator(
        rescale=flags.rescale,
        shear_range=flags.shear_range,
        zoom_range=flags.zoom_range,
        rotation_range=flags.rotation_range,
        horizontal_flip=flags.horizontal_flip
    )
    valid_generator = valid_datagen.flow_from_directory(
        directory=flags.valid_dir,
        target_size=flags.target_size,
        batch_size=flags.bsize,
        class_mode=flags.class_mode
    )

    test_datagen = ImageDataGenerator(
        rescale=flags.rescale,
        shear_range=flags.shear_range,
        zoom_range=flags.zoom_range,
        rotation_range=flags.rotation_range,
        horizontal_flip=flags.horizontal_flip
    )
    test_generator = test_datagen.flow_from_directory(
        directory=flags.test_dir,
        target_size=flags.target_size,
        batch_size=flags.bsize,
        class_mode=flags.class_mode
    )

    return train_generator, valid_generator, test_generator

