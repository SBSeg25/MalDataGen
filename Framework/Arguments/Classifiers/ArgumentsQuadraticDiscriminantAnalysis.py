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


DEFAULT_QUADRATIC_DISCRIMINANT_ANALYSIS_PRIORS = None
DEFAULT_QUADRATIC_DISCRIMINANT_ANALYSIS_REGULARIZATION = 0.0
DEFAULT_QUADRATIC_DISCRIMINANT_THRESHOLD = 0.0001

def add_argument_quadratic_discriminant_analysis(parser):

    parser.add_argument('--quadratic_discriminant_analysis_priors', type=str,
                        default=DEFAULT_QUADRATIC_DISCRIMINANT_ANALYSIS_PRIORS,
                        help='Prior probabilities of the classes in QDA.')

    parser.add_argument('--quadratic_discriminant_analysis_regularization', type=float,
                        default=DEFAULT_QUADRATIC_DISCRIMINANT_ANALYSIS_REGULARIZATION,
                        help='Regularization parameter in QDA.')

    parser.add_argument('--quadratic_discriminant_analysis_threshold', type=float,
                        default=DEFAULT_QUADRATIC_DISCRIMINANT_THRESHOLD,
                        help='Threshold value for QDA.')

    return parser