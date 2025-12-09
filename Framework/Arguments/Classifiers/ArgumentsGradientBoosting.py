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


DEFAULT_GRADIENT_BOOSTING_LOSS = 'log_loss'
DEFAULT_GRADIENT_BOOSTING_LEARNING_RATE = 0.1
DEFAULT_GRADIENT_BOOSTING_NUMBER_ESTIMATORS = 100
DEFAULT_GRADIENT_BOOSTING_SUBSAMPLE = 1.0
DEFAULT_GRADIENT_BOOSTING_CRITERION = 'friedman_mse'

def list_of_floats(arg):
    return list(map(float, arg.split(',')))

def add_argument_gradient_boosting(parser):

    parser.add_argument('--gradient_boosting_loss', type=str, default=DEFAULT_GRADIENT_BOOSTING_LOSS,
                        help='Loss function to be used by the gradient boosting model.')

    parser.add_argument('--gradient_boosting_learning_rate', type=float,default=DEFAULT_GRADIENT_BOOSTING_LEARNING_RATE,
                        help='Learning rate for the gradient boosting model.')

    parser.add_argument('--gradient_boosting_number_estimators', type=int,
                        default=DEFAULT_GRADIENT_BOOSTING_NUMBER_ESTIMATORS,
                        help='Number of estimators (trees) in the gradient boosting model.')

    parser.add_argument('--gradient_boosting_subsample', type=float, default=DEFAULT_GRADIENT_BOOSTING_SUBSAMPLE,
                        help='Fraction of samples used for fitting the individual base learners.')

    parser.add_argument('--gradient_boosting_criterion', type=str, default=DEFAULT_GRADIENT_BOOSTING_CRITERION,
                        help='Criterion to measure the quality of a split.')

    return parser