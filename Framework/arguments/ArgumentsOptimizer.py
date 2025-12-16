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


# Default values for Adam optimizer
DEFAULT_ADAM_LEARNING_RATE = 0.001
DEFAULT_ADAM_BETA_1 = 0.9
DEFAULT_ADAM_BETA_2 = 0.999
DEFAULT_ADAM_EPSILON = 1e-7
DEFAULT_ADAM_AMSGRAD = False

# Default values for AdaDelta optimizer
DEFAULT_ADADELTA_LEARNING_RATE = 0.001
DEFAULT_ADADELTA_RHO = 0.95
DEFAULT_ADADELTA_EPSILON = 1e-7
DEFAULT_ADADELTA_USE_EMA = False
DEFAULT_ADADELTA_EMA_MOMENTUM = 0.99

# Default values for Nadam optimizer
DEFAULT_NADAM_LEARNING_RATE = 0.001
DEFAULT_NADAM_BETA_1 = 0.9
DEFAULT_NADAM_BETA_2 = 0.999
DEFAULT_NADAM_EPSILON = 1e-7
DEFAULT_NADAM_USE_EMA = False
DEFAULT_NADAM_EMA_MOMENTUM = 0.99

# Default values for RMSprop optimizer
DEFAULT_RMSPROP_LEARNING_RATE = 0.001
DEFAULT_RMSPROP_RHO = 0.95
DEFAULT_RMSPROP_MOMENTUM = 0.0
DEFAULT_RMSPROP_EPSILON = 1e-7
DEFAULT_RMSPROP_USE_EMA = False
DEFAULT_RMSPROP_EMA_MOMENTUM = 0.99

# Default values for SGD optimizer
DEFAULT_SGD_LEARNING_RATE = 0.01
DEFAULT_SGD_MOMENTUM = 0.0
DEFAULT_SGD_NESTEROV = False
DEFAULT_SGD_USE_EMA = False
DEFAULT_SGD_EMA_MOMENTUM = 0.99

# Default values for FTRL optimizer
DEFAULT_FTRL_LEARNING_RATE = 0.001
DEFAULT_FTRL_LEARNING_RATE_POWER = -0.5
DEFAULT_FTRL_INITIAL_ACCUMULATOR_VALUE = 0.1
DEFAULT_FTRL_L1_REG_STRENGTH = 0.0
DEFAULT_FTRL_L2_REG_STRENGTH = 0.0
DEFAULT_FTRL_L2_SHRINKAGE_REG_STRENGTH = 0.0
DEFAULT_FTRL_BETA = 0.0
DEFAULT_FTRL_USE_EMA = False
DEFAULT_FTRL_EMA_MOMENTUM = 0.99


def add_argument_optimizers(parser):

    # Adam optimizer arguments

    parser.add_argument('--optimizer', choices=['adadelta', 'adam', 'ftrl', 'nadam', 'rsmprop', 'sgd'],
                        default='adam', help='Select the optimizer')

    parser.add_argument("--adam_optimizer_learning_rate", type=float,
                        default=DEFAULT_ADAM_LEARNING_RATE,
                        help="Learning rate for Adam optimizer. Default is 0.001.")

    parser.add_argument("--adam_optimizer_beta_1", type=float, default=DEFAULT_ADAM_BETA_1,
                        help="The exponential decay rate for the 1st moment estimates. Default is 0.9.")

    parser.add_argument("--adam_optimizer_beta_2", type=float, default=DEFAULT_ADAM_BETA_2,
                        help="The exponential decay rate for the 2nd moment estimates. Default is 0.999.")

    parser.add_argument("--adam_optimizer_epsilon", type=float, default=DEFAULT_ADAM_EPSILON,
                        help="A small constant for numerical stability. Default is 1e-7.")

    parser.add_argument("--adam_optimizer_amsgrad", type=bool, default=DEFAULT_ADAM_AMSGRAD,
                        help="Whether to apply AMSGrad variant of Adam. Default is False.")


    # AdaDelta optimizer arguments
    parser.add_argument("--ada_delta_optimizer_learning_rate", type=float,
                        default=DEFAULT_ADADELTA_LEARNING_RATE,
                        help="Learning rate for AdaDelta optimizer. Default is 0.001.")

    parser.add_argument("--ada_delta_optimizer_rho", type=float, default=DEFAULT_ADADELTA_RHO,
                        help="Decay rate for AdaDelta optimizer. Default is 0.95.")

    parser.add_argument("--ada_delta_optimizer_epsilon", type=float, default=DEFAULT_ADADELTA_EPSILON,
                        help="A small constant for numerical stability. Default is 1e-7.")

    parser.add_argument("--ada_delta_optimizer_use_ema", type=bool, default=DEFAULT_ADADELTA_USE_EMA,
                        help="Whether to use Exponential Moving Average for AdaDelta. Default is False.")

    parser.add_argument("--ada_delta_optimizer_ema_momentum", type=float, default=DEFAULT_ADADELTA_EMA_MOMENTUM,
                        help="Momentum for EMA in AdaDelta optimizer. Default is 0.99.")


    # Nadam optimizer arguments
    parser.add_argument("--nadam_optimizer_learning_rate", type=float, default=DEFAULT_NADAM_LEARNING_RATE,
                        help="Learning rate for Nadam optimizer. Default is 0.001.")

    parser.add_argument("--nadam_optimizer_beta_1", type=float, default=DEFAULT_NADAM_BETA_1,
                        help="The exponential decay rate for the 1st moment estimates. Default is 0.9.")

    parser.add_argument("--nadam_optimizer_beta_2", type=float, default=DEFAULT_NADAM_BETA_2,
                        help="The exponential decay rate for the 2nd moment estimates. Default is 0.999.")

    parser.add_argument("--nadam_optimizer_epsilon", type=float, default=DEFAULT_NADAM_EPSILON,
                        help="A small constant for numerical stability. Default is 1e-7.")

    parser.add_argument("--nadam_optimizer_use_ema", type=bool, default=DEFAULT_NADAM_USE_EMA,
                        help="Whether to use Exponential Moving Average for Nadam. Default is False.")

    parser.add_argument("--nadam_optimizer_ema_momentum", type=float, default=DEFAULT_NADAM_EMA_MOMENTUM,
                        help="Momentum for EMA in Nadam optimizer. Default is 0.99.")


    # RMSprop optimizer arguments
    parser.add_argument("--rsmprop_optimizer_learning_rate", type=float,
                        default=DEFAULT_RMSPROP_LEARNING_RATE,
                        help="Learning rate for RMSprop optimizer. Default is 0.001.")

    parser.add_argument("--rsmprop_optimizer_rho", type=float, default=DEFAULT_RMSPROP_RHO,
                        help="Discounting factor for RMSprop. Default is 0.95.")

    parser.add_argument("--rsmprop_optimizer_momentum", type=float, default=DEFAULT_RMSPROP_MOMENTUM,
                        help="Momentum for RMSprop optimizer. Default is 0.0.")

    parser.add_argument("--rsmprop_optimizer_epsilon", type=float, default=DEFAULT_RMSPROP_EPSILON,
                        help="A small constant for numerical stability. Default is 1e-7.")

    parser.add_argument("--rsmprop_optimizer_use_ema", type=bool, default=DEFAULT_RMSPROP_USE_EMA,
                        help="Whether to use Exponential Moving Average for RMSprop. Default is False.")

    parser.add_argument("--rsmprop_optimizer_ema_momentum", type=float,
                        default=DEFAULT_RMSPROP_EMA_MOMENTUM,
                        help="Momentum for EMA in RMSprop optimizer. Default is 0.99.")


    # SGD optimizer arguments
    parser.add_argument("--sgd_optimizer_learning_rate", type=float,
                        default=DEFAULT_SGD_LEARNING_RATE,
                        help="Learning rate for SGD optimizer. Default is 0.01.")

    parser.add_argument("--sgd_optimizer_momentum", type=float, default=DEFAULT_SGD_MOMENTUM,
                        help="Momentum for SGD optimizer. Default is 0.0.")

    parser.add_argument("--sgd_optimizer_nesterov", type=bool, default=DEFAULT_SGD_NESTEROV,
                        help="Whether to apply Nesterov momentum for SGD. Default is False.")

    parser.add_argument("--sgd_optimizer_use_ema", type=bool, default=DEFAULT_SGD_USE_EMA,
                        help="Whether to use Exponential Moving Average for SGD. Default is False.")

    parser.add_argument("--sgd_optimizer_ema_momentum", type=float,
                        default=DEFAULT_SGD_EMA_MOMENTUM,
                        help="Momentum for EMA in SGD optimizer. Default is 0.99.")


    # FTRL optimizer arguments
    parser.add_argument("--ftrl_optimizer_learning_rate", type=float,
                        default=DEFAULT_FTRL_LEARNING_RATE,
                        help="Learning rate for FTRL optimizer. Default is 0.001.")

    parser.add_argument("--ftrl_optimizer_learning_rate_power", type=float,
                        default=DEFAULT_FTRL_LEARNING_RATE_POWER,
                        help="The power value for learning rate in FTRL. Default is -0.5.")

    parser.add_argument("--ftrl_optimizer_initial_accumulator_value", type=float,
                        default=DEFAULT_FTRL_INITIAL_ACCUMULATOR_VALUE,
                        help="Initial accumulator value for FTRL. Default is 0.1.")

    parser.add_argument("--ftrl_optimizer_l1_regularization_strength", type=float,
                        default=DEFAULT_FTRL_L1_REG_STRENGTH,
                        help="L1 regularization strength for FTRL. Default is 0.0.")

    parser.add_argument("--ftrl_optimizer_l2_regularization_strength", type=float,
                        default=DEFAULT_FTRL_L2_REG_STRENGTH,
                        help="L2 regularization strength for FTRL. Default is 0.0.")

    parser.add_argument("--ftrl_optimizer_l2_shrinkage_regularization_strength", type=float,
                        default=DEFAULT_FTRL_L2_SHRINKAGE_REG_STRENGTH,
                        help="L2 shrinkage regularization strength for FTRL. Default is 0.0.")

    parser.add_argument("--ftrl_optimizer_beta", type=float, default=DEFAULT_FTRL_BETA,
                        help="Beta parameter for FTRL optimizer. Default is 0.0.")

    parser.add_argument("--ftrl_optimizer_use_ema", type=bool, default=DEFAULT_FTRL_USE_EMA,
                        help="Whether to use Exponential Moving Average for FTRL. Default is False.")

    parser.add_argument("--ftrl_optimizer_ema_momentum", type=float, default=DEFAULT_FTRL_EMA_MOMENTUM,
                        help="Momentum for EMA in FTRL optimizer. Default is 0.99.")


    return parser