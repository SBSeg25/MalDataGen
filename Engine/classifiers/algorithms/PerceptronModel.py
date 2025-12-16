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

try:
    import sys
    import numpy

    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import Input
    from keras.models import Model
    from tensorflow import keras

except ImportError as error:
    print(error)
    sys.exit(-1)

class PerceptronMultilayer:
    def __init__(self,
                 layers_settings,
                 training_metric,
                 training_loss,
                 training_algorithm,
                 data_type,
                 layer_activation,
                 last_layer_activation,
                 dropout_decay_rate):

        self._training_algorithm = training_algorithm
        self._training_loss = training_loss
        self._data_type = data_type
        self._layer_activation = layer_activation
        self._last_layer_activation = last_layer_activation
        self._dropout_decay_rate = dropout_decay_rate
        self._training_metric = training_metric
        self._layers_settings = layers_settings

    def get_model(self, input_shape):

        input_layer = Input(shape=(input_shape, ), dtype=self._data_type)

        dense_layer = Dense(self._layers_settings[0], self._layer_activation)(input_layer)

        for number_neurons in self._layers_settings[1:]:
            dense_layer = Dense(number_neurons, self._layer_activation)(dense_layer)
            dense_layer = Dropout(self._dropout_decay_rate)(dense_layer)

        dense_layer = Dense(1, self._last_layer_activation)(dense_layer)
        neural_network_model = Model(input_layer, dense_layer)
        neural_network_model.compile(loss="binary_crossentropy")

        return neural_network_model

    def set_training_algorithm(self, training_algorithm):

        self._training_algorithm = training_algorithm

    def set_training_loss(self, training_loss):

        self._training_loss = training_loss

    def set_data_type(self, data_type):

        self._data_type = data_type

    def set_layer_activation(self, layer_activation):

        self._layer_activation = layer_activation

    def set_last_layer_activation(self, last_layer_activation):

        self._last_layer_activation = last_layer_activation

    def set_dropout_decay_rate(self, dropout_decay_rate):

        self._dropout_decay_rate = dropout_decay_rate
