#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

# MIT License
#
# Copyright (c) 2025 Synthetic Ocean AI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


DEFAULT_KNN_NUMBER_NEIGHBORS = 5
DEFAULT_KNN_WEIGHTS = "uniform"
DEFAULT_KNN_ALGORITHM = "auto"
DEFAULT_KNN_LEAF_SIZE = 30
DEFAULT_KNN_METRIC = "minkowski"


def add_argument_knn(parser):

    parser.add_argument('--knn_number_neighbors', type=int, default=DEFAULT_KNN_NUMBER_NEIGHBORS,
                        help='Number of neighbors to use by default for KNN.')

    parser.add_argument('--knn_weights', type=str, default=DEFAULT_KNN_WEIGHTS,
                        help='Weight function used in prediction for KNN.')

    parser.add_argument('--knn_algorithm', type=str, default=DEFAULT_KNN_ALGORITHM,
                        help='Algorithm used to compute nearest neighbors.')

    parser.add_argument('--knn_leaf_size', type=int, default=DEFAULT_KNN_LEAF_SIZE,
                        help='Leaf size passed to BallTree or KDTree in KNN.')

    parser.add_argument('--knn_metric', type=str, default=DEFAULT_KNN_METRIC,
                        help='The distance metric to use for KNN.')

    return parser
