#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Kayuã Oleques']


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

    from Engine.activations.GLU import GLU
    from Engine.activations.ELU import ELU

    from Engine.activations.ReLU import ReLU
    from Engine.activations.SELU import SELU
    from Engine.activations.CeLU import CeLU
    from Engine.activations.Tanh import Tanh

    from Engine.activations.Swish import Swish
    from Engine.activations.PReLU import PReLU

    from Engine.activations.Linear import Linear

    from Engine.activations.Sigmoid import Sigmoid
    from Engine.activations.Softmax import Softmax

    from Engine.activations.Softplus import Softplus
    from Engine.activations.SoftSign import SoftSign

    from Engine.activations.LeakyRelu import LeakyReLU
    from Engine.activations.LogSigmoid import LogSigmoid

    from Engine.activations.HardSigmoid import HardSigmoid
    from Engine.activations.Exponential import Exponential



except ImportError as error:
    print(error)
    sys.exit(-1)


class Activations:
    """
    A utility class for managing the addition of various activation functions
    to neural network layers.

    This class provides a static method to add an activation layer to a neural
    network model. The supported activation functions include commonly used
    activation types such as ReLU, Sigmoid, Tanh, and more. The method ensures
    that the specified activation function is valid and supported.
    """

    def __init__(self):
        """
        Initializes the activations class.

        This constructor is currently empty, as no instance attributes are needed.
        The class is designed to work with static methods, and thus does not require
        instance initialization.
        """
        pass

    @staticmethod
    def _add_activation_layer(neural_nodel, activation):
        """
        Adds the specified activation function to a given neural network layer.

        This method maps the provided activation function name to a corresponding
        Keras or TensorFlow activation function. If the provided activation function
        is supported, it returns the corresponding activation layer. If the activation
        function is not recognized, it raises a ValueError.

        Args:
            neural_nodel (tensorflow.keras.layers.Layer): The neural network layer to
                                                          which the activation function
                                                          will be applied.
            activation (str): The name of the activation function to add. This should be
                              one of the supported activation names, e.g., 'relu', 'sigmoid'.

        Returns:
            tensorflow.keras.layers.Layer: The neural network layer with the specified
                                           activation function applied.

        Raises:
            ValueError: If the specified activation function is not supported.

        Supported Activation Functions:
            - 'leakyrelu': LeakyReLU activation.
            - 'relu': ReLU activation.
            - 'prelu': PReLU activation.
            - 'sigmoid': Sigmoid activation.
            - 'tanh': Tanh activation.
            - 'elu': ELU activation.
            - 'softmax': Softmax activation.
            - 'swish': Swish activation.
            - 'softplus': Softplus activation.
            - 'hardsigmoid': HardSigmoid activation.
            - 'selu': SELU activation.
            - 'exponential': Exponential activation.
            - 'linear': Linear activation.

        Example:
            model.add(activations._add_activation_layer(layer, 'relu'))
        """

        # Dictionary mapping activation names to their corresponding activation functions
        activations = {
            'leakyrelu': LeakyReLU(),
            'relu': ReLU(),
            'prelu': PReLU(),
            'sigmoid': Sigmoid(),
            'tanh': Tanh(),
            'elu': ELU(),
            'glu': GLU(),
            'logsigmoid': LogSigmoid(),
            'celu': CeLU(),
            'softsign': SoftSign(),
            'softmax': Softmax(),
            'swish': Swish(),
            'softplus': Softplus(),
            'hardsigmoid': HardSigmoid(),
            'selu': SELU(),
            'exponential': Exponential(),
            'linear': Linear(),
        }

        # Convert the activation function name to lowercase to handle case insensitivity
        activation_lower = activation.lower()

        # Check if the activation is supported and return the corresponding layer
        if activation_lower in activations:
            return activations[activation_lower](neural_nodel)

        else:
            # Raise an error if the activation function is unsupported
            print(f"Unsupported activation function: '{activation}'. Please choose from: {list(activations.keys())}")
            raise ValueError(
                f"Unsupported activation function: '{activation}'. Please choose from: {list(activations.keys())}")