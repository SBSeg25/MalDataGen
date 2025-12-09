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
    import torch.nn as nn

except ImportError as error:
    print(error)
    sys.exit(-1)


class Activations:
    """
    A utility class that provides access to various activation functions for PyTorch neural networks.

    This class serves as a mixin or utility that can be inherited by other classes (like neural network
    models) to provide easy access to activation function modules. It supports common activation functions
    used in deep learning, including ReLU, LeakyReLU, Tanh, Sigmoid, Softmax, ELU, SELU, GELU, SiLU (Swish),
    and linear (identity) activations.

    The class normalizes activation function names by converting them to lowercase and removing underscores,
    making it flexible for different naming conventions (e.g., 'leaky_relu', 'LeakyReLU', 'leakyrelu').

    Methods:
        _get_activation(activation_name, alpha=0.2):
            Returns the appropriate activation function module based on the given name.

        _add_activation_layer(activation_name):
            Static method that returns an activation function module. Similar to _get_activation
            but without support for parameterized activations.

    Supported Activation Functions:
        - relu: Rectified Linear Unit
        - leakyrelu: Leaky Rectified Linear Unit (default alpha=0.2)
        - tanh: Hyperbolic Tangent
        - sigmoid: Sigmoid function
        - softmax: Softmax function (applied on dim=1)
        - elu: Exponential Linear Unit
        - selu: Scaled Exponential Linear Unit
        - gelu: Gaussian Error Linear Unit
        - swish / silu: Sigmoid Linear Unit (Swish activation)
        - linear: Identity function (no activation)

    Example:
        >>> class MyModel(nn.Module, Activations):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.activation = self._get_activation('relu')
        ...
        ...     def forward(self, x):
        ...         return self.activation(x)
        >>>
        >>> model = MyModel()
        >>> # Or use the static method directly
        >>> activation = Activations._add_activation_layer('leakyrelu')
    """

    @staticmethod
    def _get_activation(activation_name: str, alpha: float = 0.2) -> nn.Module:
        """
        Returns the appropriate activation function module with support for parameterized activations.

        This method normalizes the activation name and returns the corresponding PyTorch activation
        module. It supports parameterized activations like LeakyReLU where the alpha parameter can
        be customized.

        Args:
            activation_name (str): Name of the activation function (case-insensitive, underscores optional).
            alpha (float, optional): Alpha parameter for LeakyReLU. Defaults to 0.2.

        Returns:
            nn.Module: PyTorch activation module ready to be used in forward passes.

        Raises:
            ValueError: If the activation function name is not supported.

        Example:
            >>> activation = Activations._get_activation('leaky_relu', alpha=0.3)
            >>> output = activation(input_tensor)
        """
        activation_name = activation_name.lower().replace('_', '')

        activation_map = {
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU(alpha),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax(dim=1),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'silu': nn.SiLU(),
            'linear': nn.Identity(),
        }

        if activation_name in activation_map:
            return activation_map[activation_name]
        else:
            raise ValueError(
                f"Unsupported activation function: {activation_name}. "
                f"Supported activations: {', '.join(activation_map.keys())}"
            )

    @staticmethod
    def _add_activation_layer(activation_name: str) -> nn.Module:
        """
        Returns the appropriate activation function module.

        This is a simplified version of _get_activation that uses default parameters
        for all activation functions. Useful when you don't need to customize activation
        parameters.

        Args:
            activation_name (str): Name of the activation function (case-insensitive, underscores optional).

        Returns:
            nn.Module: PyTorch activation module.

        Raises:
            ValueError: If the activation function name is not supported.

        Example:
            >>> activation = Activations._add_activation_layer('relu')
            >>> output = activation(input_tensor)
        """
        activation_name = activation_name.lower().replace('_', '')

        activation_map = {
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU(0.2),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax(dim=1),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'silu': nn.SiLU(),
            'linear': nn.Identity(),
        }

        if activation_name in activation_map:
            return activation_map[activation_name]
        else:
            raise ValueError(
                f"Unsupported activation function: {activation_name}. "
                f"Supported activations: {', '.join(activation_map.keys())}"
            )