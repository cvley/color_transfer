#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2014 ley <finalley@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""
import numpy
from PIL import Image

def rgb2lab(arr):
    arr = arr + 1
    if arr.ndim != 3:
        raise ValueError("input array must have 3 dimensions")

    # RGB => LMS
    # L     |0.3811  0.5783  0.0402| |R|
    # M  =  |0.1967  0.7244  0.0782| |G|
    # S     |0.0241  0.1288  0.8444| |B|

    convert_matrix = numpy.array([
        [0.3811, 0.5783, 0.0402],
        [0.1967, 0.7244, 0.0782],
        [0.0241, 0.1288, 0.8444]])

    lms = arr.dot(convert_matrix)
    
    LMS = numpy.log(lms)
    
    # LMS =>  lab
    # l   |1/sqrt(3)  0  0| | 1  1  1| |L|
    # a = | 0 1/sqrt(6)  0| | 1  1 -2| |M|
    # b   | 0  0 1/sqrt(2)| | 1 -1  0| |S|
    mtx1 = numpy.array([
        [1/numpy.sqrt(3), 0, 0],
        [0, 1/numpy.sqrt(6), 0],
        [0, 0, 1/numpy.sqrt(2)]])
    mtx2 = numpy.array([
        [1, 1, 1],
        [1, 1, -2],
        [1, -1, 0]])
    mtx = mtx1.dot(mtx2)

    lab = LMS.dot(mtx)

    return lab
