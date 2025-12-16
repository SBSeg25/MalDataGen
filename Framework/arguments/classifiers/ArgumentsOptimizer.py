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


DEFAULT_ADAM_LEARNING_RATE = 0.001
DEFAULT_ADAM_BETA_1 = 0.9
DEFAULT_ADAM_BETA_2 = 0.999
DEFAULT_ADAM_EPSILON = 1e-7

DEFAULT_SGD_LEARNING_RATE = 0.01
DEFAULT_SGD_MOMENTUM = 0.0
DEFAULT_SGD_NESTEROV = False
DEFAULT_SGD_DECAY = 0.0

DEFAULT_RMSPROP_LEARNING_RATE = 0.001
DEFAULT_RMSPROP_MOMENTUM = 0.0
DEFAULT_RMSPROP_EPSILON = 1e-7
DEFAULT_RMSPROP_DECAY = 0.0
DEFAULT_RMSPROP_CENTERED = False

DEFAULT_ADAGRAD_LEARNING_RATE = 0.01
DEFAULT_ADAGRAD_INITIAL_ACCUMULATOR_VALUE = 0.1
DEFAULT_ADAGRAD_EPSILON = 1e-7

DEFAULT_ADAMAX_LEARNING_RATE = 0.002
DEFAULT_ADAMAX_BETA_1 = 0.9
DEFAULT_ADAMAX_BETA_2 = 0.999
DEFAULT_ADAMAX_EPSILON = 1e-7

DEFAULT_NADAM_LEARNING_RATE = 0.002
DEFAULT_NADAM_BETA_1 = 0.9
DEFAULT_NADAM_BETA_2 = 0.999
DEFAULT_NADAM_EPSILON = 1e-7

DEFAULT_ADADELTA_LEARNING_RATE = 1.0
DEFAULT_ADADELTA_RHO = 0.95
DEFAULT_ADADELTA_EPSILON = 1e-7

DEFAULT_FTRL_LEARNING_RATE = 0.001
DEFAULT_FTRL_L1_REGULARIZATION_STRENGTH = 0.0
DEFAULT_FTRL_L2_REGULARIZATION_STRENGTH = 0.0
DEFAULT_FTRL_L2_SHRINKAGE_REGULARIZATION_STRENGTH = 0.0



def create_argument_parser_optimizer(parser):

    parser.add_argument('--optimizer', type=str,
                        default='Adam',
                        help='Optimizer to use for training.',
                        choices=['Adam', 'SGD', 'RMSprop', 'Adagrad', 'Adamax', 'Nadam', 'Adadelta', 'FTRL'])

    parser.add_argument('--adam_learning_rate', type=float, default=DEFAULT_ADAM_LEARNING_RATE,
                        help='Learning rate for Adam optimizer.')

    parser.add_argument('--adam_beta_1', type=float, default=DEFAULT_ADAM_BETA_1,
                        help='Beta 1 parameter for Adam optimizer.')

    parser.add_argument('--adam_beta_2', type=float, default=DEFAULT_ADAM_BETA_2,
                        help='Beta 2 parameter for Adam optimizer.')

    parser.add_argument('--adam_epsilon', type=float, default=DEFAULT_ADAM_EPSILON,
                        help='Epsilon parameter for Adam optimizer.')

    parser.add_argument('--sgd_learning_rate', type=float, default=DEFAULT_SGD_LEARNING_RATE,
                        help='Learning rate for SGD optimizer.')

    parser.add_argument('--sgd_momentum', type=float, default=DEFAULT_SGD_MOMENTUM,
                        help='Momentum parameter for SGD optimizer.')

    parser.add_argument('--sgd_nesterov', type=bool, default=DEFAULT_SGD_NESTEROV,
                        help='Whether to use Nesterov momentum with SGD optimizer.')

    parser.add_argument('--sgd_decay', type=float, default=DEFAULT_SGD_DECAY,
                        help='Decay parameter for SGD optimizer.')

    parser.add_argument('--rmsprop_learning_rate', type=float, default=DEFAULT_RMSPROP_LEARNING_RATE,
                        help='Learning rate for RMSprop optimizer.')

    parser.add_argument('--rmsprop_momentum', type=float, default=DEFAULT_RMSPROP_MOMENTUM,
                        help='Momentum parameter for RMSprop optimizer.')

    parser.add_argument('--rmsprop_epsilon', type=float, default=DEFAULT_RMSPROP_EPSILON,
                        help='Epsilon parameter for RMSprop optimizer.')

    parser.add_argument('--rmsprop_decay', type=float, default=DEFAULT_RMSPROP_DECAY,
                        help='Decay parameter for RMSprop optimizer.')

    parser.add_argument('--rmsprop_centered', type=bool, default=DEFAULT_RMSPROP_CENTERED,
                        help='Whether to use centered RMSprop.')

    parser.add_argument('--adagrad_learning_rate', type=float, default=DEFAULT_ADAGRAD_LEARNING_RATE,
                        help='Learning rate for Adagrad optimizer.')

    parser.add_argument('--adagrad_initial_accumulator_value', type=float,
                        default=DEFAULT_ADAGRAD_INITIAL_ACCUMULATOR_VALUE,
                        help='Initial accumulator value for Adagrad optimizer.')

    parser.add_argument('--adagrad_epsilon', type=float, default=DEFAULT_ADAGRAD_EPSILON,
                        help='Epsilon parameter for Adagrad optimizer.')

    parser.add_argument('--adamax_learning_rate', type=float, default=DEFAULT_ADAMAX_LEARNING_RATE,
                        help='Learning rate for Adamax optimizer.')

    parser.add_argument('--adamax_beta_1', type=float, default=DEFAULT_ADAMAX_BETA_1,
                        help='Beta 1 parameter for Adamax optimizer.')

    parser.add_argument('--adamax_beta_2', type=float, default=DEFAULT_ADAMAX_BETA_2,
                        help='Beta 2 parameter for Adamax optimizer.')

    parser.add_argument('--adamax_epsilon', type=float, default=DEFAULT_ADAMAX_EPSILON,
                        help='Epsilon parameter for Adamax optimizer.')

    parser.add_argument('--nadam_learning_rate', type=float, default=DEFAULT_NADAM_LEARNING_RATE,
                        help='Learning rate for Nadam optimizer.')

    parser.add_argument('--nadam_beta_1', type=float, default=DEFAULT_NADAM_BETA_1,
                        help='Beta 1 parameter for Nadam optimizer.')

    parser.add_argument('--nadam_beta_2', type=float, default=DEFAULT_NADAM_BETA_2,
                        help='Beta 2 parameter for Nadam optimizer.')

    parser.add_argument('--nadam_epsilon', type=float, default=DEFAULT_NADAM_EPSILON,
                        help='Epsilon parameter for Nadam optimizer.')

    parser.add_argument('--adadelta_learning_rate', type=float, default=DEFAULT_ADADELTA_LEARNING_RATE,
                        help='Learning rate for Adadelta optimizer.')

    parser.add_argument('--adadelta_rho', type=float, default=DEFAULT_ADADELTA_RHO,
                        help='Rho parameter for Adadelta optimizer.')

    parser.add_argument('--adadelta_epsilon', type=float, default=DEFAULT_ADADELTA_EPSILON,
                        help='Epsilon parameter for Adadelta optimizer.')

    parser.add_argument('--ftrl_learning_rate', type=float, default=DEFAULT_FTRL_LEARNING_RATE,
                        help='Learning rate for FTRL optimizer.')

    parser.add_argument('--ftrl_l1_regularization_strength', type=float,
                        default=DEFAULT_FTRL_L1_REGULARIZATION_STRENGTH,
                        help='L1 regularization strength for FTRL optimizer.')

    parser.add_argument('--ftrl_l2_regularization_strength', type=float,
                        default=DEFAULT_FTRL_L2_REGULARIZATION_STRENGTH,
                        help='L2 regularization strength for FTRL optimizer.')

    parser.add_argument('--ftrl_l2_shrinkage_regularization_strength', type=float,
                        default=DEFAULT_FTRL_L2_SHRINKAGE_REGULARIZATION_STRENGTH,
                        help='L2 shrinkage regularization strength for FTRL optimizer.')

    return parser