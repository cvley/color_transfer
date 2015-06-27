#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © shwley <finalley@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""
import numpy
import sys
from PIL import Image

ConvertMatrix1 = numpy.array([
    [0.3811, 0.5783, 0.0402],
    [0.1967, 0.7244, 0.0782],
    [0.0241, 0.1288, 0.8444]])
ConvertMatrix2 = numpy.array([
    [4.4679, -3.5873, 0.1193],
    [-1.2186, 2.3809, -0.1624],
    [0.0497, -0.2439, 1.2045]])

MTX1 = numpy.array([
    [1/numpy.sqrt(3), 0, 0],
    [0, 1/numpy.sqrt(6), 0],
    [0, 0, 1/numpy.sqrt(2)]])
MTX2 = numpy.array([
    [1, 1, 1],
    [1, 1, -2],
    [1, -1, 0]])
MTX = MTX1.dot(MTX2)


def rgb2lab(arr):
    if arr.ndim != 3:
        raise ValueError("input array must have 3 dimensions")

    # RGB => LMS
    # L     |0.3811  0.5783  0.0402| |R|
    # M  =  |0.1967  0.7244  0.0782| |G|
    # S     |0.0241  0.1288  0.8444| |B|
    lms = numpy.dot(arr, ConvertMatrix1)

    LMS = numpy.log10(lms+1)

    # LMS =>  lab
    # l   |1/sqrt(3)  0  0| | 1  1  1| |L|
    # a = | 0 1/sqrt(6)  0| | 1  1 -2| |M|
    # b   | 0  0 1/sqrt(2)| | 1 -1  0| |S|
    lab = numpy.dot(LMS, MTX)

    return lab


def lab2rgb(lab):
    lms = numpy.dot(lab, MTX.T)
    LMS = pow(10, lms) - 1
    rgb = numpy.dot(LMS, ConvertMatrix2)
    return rgb


def colorTransfer(target, origin):
    t_lab = rgb2lab(target)
    t_lmean = numpy.mean(t_lab[:, :, 0])
    t_amean = numpy.mean(t_lab[:, :, 1])
    t_bmean = numpy.mean(t_lab[:, :, 2])
    t_std = numpy.std(t_lab)

    o_lab = rgb2lab(origin)
    o_lmean = numpy.mean(o_lab[:, :, 0])
    o_amean = numpy.mean(o_lab[:, :, 1])
    o_bmean = numpy.mean(o_lab[:, :, 2])
    o_std = numpy.std(o_lab)

    """ subtract the mean from data"""
    scale = t_std / o_std
    ori = numpy.zeros(origin.shape)
    ori[:, :, 0] = (o_lab[:, :, 0] - o_lmean) * scale + t_lmean
    ori[:, :, 1] = (o_lab[:, :, 1] - o_amean) * scale + t_amean
    ori[:, :, 2] = (o_lab[:, :, 2] - o_bmean) * scale + t_bmean
    return lab2rgb(ori)

if __name__ == "__main__":
    source_im = Image.open(sys.argv[1])
    target_im = Image.open(sys.argv[2])
    source_arr = numpy.asarray(source_im)
    target_arr = numpy.asarray(target_im)

    trans_arr = colorTransfer(target_arr, source_arr)
    img = Image.fromarray(trans_arr.astype(numpy.uint8))
    img.save(sys.argv[3])
