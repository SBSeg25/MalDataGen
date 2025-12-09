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


DEFAULT_LINEAR_REGRESSION_FIT_INTERCEPT = True
DEFAULT_LINEAR_REGRESSION_NORMALIZE = False
DEFAULT_LINEAR_REGRESSION_COPY_X = True
DEFAULT_LINEAR_REGRESSION_N_JOBS = None

def add_argument_linear_regression(parser):

    parser.add_argument('--linear_regression_fit_intercept', type=bool, default=DEFAULT_LINEAR_REGRESSION_FIT_INTERCEPT,
                        help='Whether to calculate the intercept for the model.')

    parser.add_argument('--linear_regression_normalize', type=bool, default=DEFAULT_LINEAR_REGRESSION_NORMALIZE,
                        help='This parameter is ignored when `fit_intercept=False`. If True, the regressors X will be normalized.')

    parser.add_argument('--linear_regression_copy_X', type=bool, default=DEFAULT_LINEAR_REGRESSION_COPY_X,
                        help='If True, X will be copied; else, it may be overwritten.')

    parser.add_argument('--linear_regression_number_jobs', type=int, nargs='?', default=DEFAULT_LINEAR_REGRESSION_N_JOBS,
                        help='The number of jobs to use for the computation. This will only provide speedup in cases of large datasets.')

    return parser
