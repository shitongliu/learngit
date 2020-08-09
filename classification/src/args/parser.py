#!/usr/bin/env python
# _*_ encoding: utf-8 _*_
"""
@File       :   parser.py
@Time       :   2020/8/4 15:17
@Author     :   lst
@Version    :   1.0
@Contact    :   liushitong@360humi.com
@License    :   (C)Copyright 2020-2021, humi
@Desc       :   None
"""
from __future__ import division, absolute_import, print_function
import argparse


def get_args():
    """ parse argument

    :return: parse result
    """
    parser = argparse.ArgumentParser()

    # opinion
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--test', action='store_true',
                        help='Evaluate the model')
    parser.add_argument('--predict', action='store_true',
                        help='Get prediction result')

    # pre-processing parameters
    parser.add_argument('--rescale', type=float, default=1/255,
                        help='Rescale factor in pre-processing')
    parser.add_argument('--shear_range', type=float, default=0.2,
                        help='Shear range in pre-processing')
    parser.add_argument('--zoom_range', type=float, default=0.2,
                        help='Zoom in/out range in pre-processing')
    parser.add_argument('--rotation_range', type=int, default=20,
                        help='Rotation range in pre-processing')
    parser.add_argument('--horizontal_flip', type=bool, default=True,
                        help='Horizontal flip in pre-processing')
    parser.add_argument('--target_size', type=tuple, default=(256, 256),
                        help='Image size in pre-processing')
    parser.add_argument('--class_mode', type=str, default='sparse',
                        help='Class mode of samples, One of "categorical", "binary", "sparse","input", or None')

    # training parameters
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--bsize', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--maxepoch', type=int, default=30,
                        help='Max number of epochs for training')

    # directory
    parser.add_argument('--train_dir', type=str,
                        default=r'..\data\train',
                        help='Directory of training data')
    parser.add_argument('--valid_dir', type=str,
                        default=r'..\data\valid',
                        help='Directory of valid data')
    parser.add_argument('--test_dir', type=str,
                        default=r'..\data\test',
                        help='Directory of test data')
    parser.add_argument('--pred_path', type=str,
                        default=r'..\data\predict',
                        help='Path of predict data')
    parser.add_argument('--log_path', type=str,
                        default=r'..\output\log',
                        help='Path of training log')
    parser.add_argument('--save_path', type=str,
                        default=r'..\output\saved_model',
                        help='Path of saved model')

    return parser.parse_args()


