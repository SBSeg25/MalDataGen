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


DEFAULT_SUPPORT_VECTOR_MACHINE_REGULARIZATION = 1.0
DEFAULT_SUPPORT_VECTOR_MACHINE_KERNEL = "rbf"
DEFAULT_SUPPORT_VECTOR_MACHINE_KERNEL_DEGREE = 3
DEFAULT_SUPPORT_VECTOR_MACHINE_GAMMA = "scale"


def add_argument_support_vector_machine(parser):

    parser.add_argument('--support_vector_machine_regularization', type=float,
                        default=DEFAULT_SUPPORT_VECTOR_MACHINE_REGULARIZATION,
                        help='Regularization parameter for SVM.')

    parser.add_argument('--support_vector_machine_kernel', type=str,
                        default=DEFAULT_SUPPORT_VECTOR_MACHINE_KERNEL,
                        help='Kernel type for SVM.')

    parser.add_argument('--support_vector_machine_kernel_degree', type=int,
                        default=DEFAULT_SUPPORT_VECTOR_MACHINE_KERNEL_DEGREE,
                        help='Degree for polynomial kernel function (SVM).')

    parser.add_argument('--support_vector_machine_gamma', type=str,
                        default=DEFAULT_SUPPORT_VECTOR_MACHINE_GAMMA,
                        help='Kernel coefficient for SVM.')

    return parser