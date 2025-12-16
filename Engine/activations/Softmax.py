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


class Softmax(Layer):
    """
    Softmax Activation Function Layer.

    The Softmax activation function converts a vector of values into a probability distribution.
    The elements of the output vector are in the range `[0, 1]` and sum to 1.

    It is commonly used in the final layer of classification models to output the probability
    distribution of each class.

    The softmax of each vector x is computed as:

        softmax(x_i) = exp(x_i) / sum(exp(x_j))

    where x_i is an element of the input vector, and the sum is taken over all elements of the vector.

    Attributes
    ----------
        axis : int
            Axis along which the softmax is applied. Default is -1, which applies softmax to the last axis.

    Methods
    -------
        call(neural_network_flow: tensorflow.Tensor) -> tensorflow.Tensor
            Applies the Softmax activation function to the input tensor and returns the output tensor.

    Example
    -------
    >>> import tensorflow
    >>> input_tensor = tensorflow.random.uniform((2, 5, 8))  # Example tensor
    >>> softmax_layer = Softmax()
    >>> output_tensor = softmax_layer(input_tensor)
    >>> print(output_tensor.shape)  # Output shape will be (2, 5, 8)
    """

    def __init__(self, axis=-1, **kwargs):
        """
        Initializes the Softmax activation function layer.

        Parameters
        ----------
        axis : int
            Axis along which the softmax is applied (default is -1).
        **kwargs
            Additional keyword arguments passed to the base Layer class.
        """
        super(Softmax, self).__init__(**kwargs)
        self.axis = axis

    def call(self, neural_network_flow: tensorflow.Tensor) -> tensorflow.Tensor:
        """
        Applies the Softmax activation function to the input tensor.

        Parameters
        ----------
            neural_network_flow : tensorflow.Tensor
                Input tensor with any shape.

        Returns
        -------
        tensorflow.Tensor
            Output tensor with the same shape as input, after applying Softmax transformation.
        """
        return tensorflow.nn.softmax(neural_network_flow, axis=self.axis)

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