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
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as error:
    print(error)
    sys.exit(-1)


class Activations:
    """
    Helper class for applying activation functions in PyTorch models.
    This class provides a unified interface for various activation functions.
    """

    def __init__(self):
        pass

    @staticmethod
    def _add_activation_layer(x, activation_name):
        """
        Applies the specified activation function to the input tensor.

        Args:
            x (torch.Tensor): Input tensor
            activation_name (str): Name of the activation function

        Returns:
            torch.Tensor: Output after applying activation
        """
        activation_name = activation_name.lower()

        if activation_name == 'relu':
            return F.relu(x)
        elif activation_name == 'leakyrelu':
            return F.leaky_relu(x, negative_slope=0.2)
        elif activation_name == 'sigmoid':
            return torch.sigmoid(x)
        elif activation_name == 'tanh':
            return torch.tanh(x)
        elif activation_name == 'swish' or activation_name == 'silu':
            return F.silu(x)
        elif activation_name == 'elu':
            return F.elu(x)
        elif activation_name == 'softmax':
            return F.softmax(x, dim=-1)
        elif activation_name == 'linear' or activation_name == 'none':
            return x
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")