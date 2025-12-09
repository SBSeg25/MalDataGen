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


DEFAULT_PERCEPTRON_TRAINING_ALGORITHM = "Adam"
DEFAULT_PERCEPTRON_LOSS = "binary_crossentropy"
DEFAULT_PERCEPTRON_LAYERS_SETTINGS = [512, 256, 256]
DEFAULT_PERCEPTRON_DROPOUT_DECAY_RATE = 0.2
DEFAULT_PERCEPTRON_METRIC = ["accuracy"]
DEFAULT_PERCEPTRON_LAYER_ACTIVATION = 'relu'
DEFAULT_PERCEPTRON_LAST_LAYER_ACTIVATION = "sigmoid"
DEFAULT_PERCEPTRON_NUMBER_EPOCHS = 1



def add_argument_perceptron(parser):

    parser.add_argument('--perceptron_training_algorithm', type=str, default=DEFAULT_PERCEPTRON_TRAINING_ALGORITHM,
                        help='Training algorithm for Perceptron.')

    parser.add_argument('--perceptron_training_loss', type=str, default=DEFAULT_PERCEPTRON_LOSS,
                        help='Loss function for Perceptron.')

    parser.add_argument('--perceptron_layers_settings', nargs='+', type=int, default=DEFAULT_PERCEPTRON_LAYERS_SETTINGS,
                        help='Layer settings for Perceptron.')

    parser.add_argument('--perceptron_dropout_decay_rate', type=float, default=DEFAULT_PERCEPTRON_DROPOUT_DECAY_RATE,
                        help='Dropout decay rate in Perceptron.')

    parser.add_argument('--perceptron_training_metric', nargs='+', type=str, default=DEFAULT_PERCEPTRON_METRIC,
                        help='Evaluation metrics for Perceptron.')

    parser.add_argument('--perceptron_layer_activation', type=str, default=DEFAULT_PERCEPTRON_LAYER_ACTIVATION,
                        help='Activation function for layers in Perceptron.')

    parser.add_argument('--perceptron_last_layer_activation', type=str,
                        default=DEFAULT_PERCEPTRON_LAST_LAYER_ACTIVATION,
                        help='Activation function for last layer in Perceptron.')

    parser.add_argument('--perceptron_number_epochs', type=int, default=DEFAULT_PERCEPTRON_NUMBER_EPOCHS,
                        help='Number of epochs for Perceptron.')

    return parser