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


DEFAULT_STOCHASTIC_GRADIENT_DESCENT_LOSS = 'hinge'
DEFAULT_STOCHASTIC_GRADIENT_DESCENT_PENALTY = 'l2'
DEFAULT_STOCHASTIC_GRADIENT_DESCENT_ALPHA = 0.0001
DEFAULT_STOCHASTIC_GRADIENT_DESCENT_MAX_ITERATIONS = 1000
DEFAULT_STOCHASTIC_GRADIENT_DESCENT_TOLERANCE = 1e-3

def add_argument_stochastic_gradient_descent(parser):

    parser.add_argument('--stochastic_gradient_descent_loss', type=str,
                        default=DEFAULT_STOCHASTIC_GRADIENT_DESCENT_LOSS,
                        help='Loss function to be used by the SGD algorithm.')

    parser.add_argument('--stochastic_gradient_descent_penalty', type=str,
                        default=DEFAULT_STOCHASTIC_GRADIENT_DESCENT_PENALTY,
                        help='Penalty term for regularization.')

    parser.add_argument('--stochastic_gradient_descent_alpha', type=float,
                        default=DEFAULT_STOCHASTIC_GRADIENT_DESCENT_ALPHA,
                        help='Constant that multiplies the regularization term.')

    parser.add_argument('--stochastic_gradient_descent_max_iterations', type=int,
                        default=DEFAULT_STOCHASTIC_GRADIENT_DESCENT_MAX_ITERATIONS,
                        help='Maximum number of iterations for the SGD algorithm.')

    parser.add_argument('--stochastic_gradient_descent_tolerance', type=float,
                        default=DEFAULT_STOCHASTIC_GRADIENT_DESCENT_TOLERANCE,
                        help='Tolerance for stopping criteria.')

    return parser