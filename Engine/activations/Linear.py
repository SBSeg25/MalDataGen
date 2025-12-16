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

    import tensorflow

    from tensorflow.keras.layers import Layer

except ImportError as error:
    print(error)
    sys.exit(-1)


class Linear(Layer):
    """
    Linear Activation Function Layer (Identity Function).

    The Linear activation function is defined as:

        linear(x) = x

    This function is typically used as the output activation for regression tasks,
    where no non-linearity is desired. Optionally, the output can be scaled by a
    multiplicative factor.

    Attributes
    ----------
        scale : float
            Optional multiplicative scaling factor applied to the output (default is 1.0).

    Methods
    -------
        call(neural_network_flow: tensorflow.Tensor) -> tensorflow.Tensor
            Applies the linear activation function to the input tensor and returns the output tensor.

    Example
    -------
    >>> import tensorflow
    >>> x = tensorflow.constant([-10, -5, 0, 5, 10], dtype=tensorflow.float32)
    >>> linear_layer = Linear(scale=2.0)
    >>> output = linear_layer(x)
    >>> print(output.numpy())  # Output: [-20. -10.   0.  10.  20.]
    """

    def __init__(self, scale=1.0, **kwargs):
        """
        Initializes the Linear activation function layer.

        Parameters
        ----------
        scale : float
            Optional multiplicative scaling factor applied to the output (default is 1.0).
        **kwargs
            Additional keyword arguments passed to the base Layer class.
        """
        super(Linear, self).__init__(**kwargs)
        self.scale = scale

    def call(self, neural_network_flow: tensorflow.Tensor) -> tensorflow.Tensor:
        """
        Applies the linear activation function to the input tensor.

        Parameters
        ----------
            neural_network_flow : tensorflow.Tensor
                Input tensor with any shape.

        Returns
        -------
        tensorflow.Tensor
            Output tensor with the same shape as input, scaled if specified.
        """
        return self.scale * neural_network_flow

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape, which remains the same as the input shape.

        Parameters
        ----------
            input_shape : tuple
                Shape of the input tensor.

        Returns
        -------
        tuple
            Output shape, identical to input shape.
        """
        return input_shape