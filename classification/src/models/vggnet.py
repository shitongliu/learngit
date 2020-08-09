#!/usr/bin/env python
# _*_ encoding: utf-8 _*_
"""
@File       :   classification.py
@Time       :   2020/8/4 15:11
@Author     :   lst
@Version    :   1.0
@Contact    :   liushitong@360humi.com
@License    :   (C)Copyright 2020-2021, humi
@Desc       :   None
"""
from __future__ import division, absolute_import, print_function
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def vgg16(num_classes, image_shape=(224, 224, 3)):
    model = Sequential([
        # 1
        Conv2D(64, (3, 3), padding='same', activation='relu',
               input_shape=image_shape),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(),

        # 2
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(),

        # 3
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(),

        # 4
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(),

        # 5
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(),

        # last
        Flatten(),
        Dense(units=4096, activation='relu'),
        Dense(units=4096, activation='relu'),
        Dense(units=num_classes)
    ])

    return model


def vgg19(num_classes, image_shape=(224, 224, 3)):
    model = Sequential([
        # 1
        Conv2D(
            64, (3, 3), padding='same', activation='relu',
            input_shape=image_shape
        ),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(),

        # 2
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(),

        # 3
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(),

        # 4
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(),

        # 5
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(),

        # last
        Flatten(),
        Dense(units=4096, activation='relu'),
        Dense(units=4096, activation='relu'),
        # Dense(units=num_classes, activation=softmax)
        Dense(units=num_classes)
    ])

    return model

