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


DEFAULT_SPECTRAL_N_CLUSTERS = 8
DEFAULT_SPECTRAL_EIGEN_SOLVER = None
DEFAULT_SPECTRAL_AFFINITY = 'rbf'
DEFAULT_SPECTRAL_ASSIGN_LABELS = 'kmeans'
DEFAULT_SPECTRAL_RANDOM_STATE = None

def add_argument_spectral_clustering(parser):

    parser.add_argument('--spectral_number_clusters', type=int, default=DEFAULT_SPECTRAL_N_CLUSTERS,
                        help='Number of clusters to form for spectral clustering.')

    parser.add_argument('--spectral_eigen_solver', type=str, default=DEFAULT_SPECTRAL_EIGEN_SOLVER,
                        help='The eigenvalue decomposition method to use. If None, the default solver is chosen.')

    parser.add_argument('--spectral_affinity', type=str, default=DEFAULT_SPECTRAL_AFFINITY,
                        help="How to construct the affinity matrix. Options: 'nearest_neighbors', 'rbf', etc.")

    parser.add_argument('--spectral_assign_labels', type=str, default=DEFAULT_SPECTRAL_ASSIGN_LABELS,
                        help="The strategy to use for assigning labels in the embedding space. Options: 'kmeans', 'discretize'.")

    parser.add_argument('--spectral_random_state', type=int, nargs='?', default=DEFAULT_SPECTRAL_RANDOM_STATE,
                        help='Seed for the random number generator.')

    return parser

