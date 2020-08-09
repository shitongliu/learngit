#!/usr/bin/env python
# _*_ encoding: utf-8 _*_
"""
@File       :   cnn.py
@Time       :   2020/8/5 14:59
@Author     :   lst
@Version    :   1.0
@Contact    :   liushitong@360humi.com
@License    :   (C)Copyright 2020-2021, humi
@Desc       :   None
"""
from __future__ import division, absolute_import, print_function
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import he_normal


def cnn3(num_classes, image_shape):
    model = Sequential([
        Conv2D(
            16, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001),
            kernel_initializer=he_normal(), input_shape=image_shape
        ),
        Dropout(0.2),
        MaxPooling2D(),
        Conv2D(
            32, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001),
            kernel_initializer=he_normal()
        ),
        Dropout(0.2),
        MaxPooling2D(),
        Conv2D(
            64, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001),
            kernel_initializer=he_normal()
        ),
        Dropout(0.2),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(
            64, activation='relu', kernel_regularizer=regularizers.l2(0.001),
            kernel_initializer=he_normal()
        ),
        Dense(num_classes)
    ])

    return model
