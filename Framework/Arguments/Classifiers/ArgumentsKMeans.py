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


DEFAULT_KMEANS_N_CLUSTERS = 2
DEFAULT_KMEANS_INIT = 'k-means++'
DEFAULT_KMEANS_MAX_ITERATIONS = 300
DEFAULT_KMEANS_TOLERANCE = 1e-4
DEFAULT_KMEANS_RANDOM_STATE = None

def add_argument_k_means(parser):

    parser.add_argument('--k_means_number_clusters', type=int, default=DEFAULT_KMEANS_N_CLUSTERS,
                        help='Number of clusters to form.')

    parser.add_argument('--k_means_init', type=str, default=DEFAULT_KMEANS_INIT,
                        help='Method for initialization of centroids.')

    parser.add_argument('--k_means_max_iterations', type=int, default=DEFAULT_KMEANS_MAX_ITERATIONS,
                        help='Maximum number of iterations for the K-Means algorithm.')

    parser.add_argument('--k_means_tolerance', type=float, default=DEFAULT_KMEANS_TOLERANCE,
                        help='Convergence tolerance for the K-Means algorithm.')

    parser.add_argument('--k_means_random_state', type=int, nargs='?', default=DEFAULT_KMEANS_RANDOM_STATE,
                        help='Seed for the random number generator.')

    return parser