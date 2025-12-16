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


class Sigmoid(Layer):
    """
    Sigmoid Activation Function Layer.

    The Sigmoid activation function is defined as:

        sigmoid(x) = 1 / (1 + exp(-x))

    For small values (<-5), sigmoid returns a value close to zero,
    and for large values (>5) the result of the function gets close to 1.
    It always returns a value between 0 and 1.

    Attributes
    ----------
    None

    Methods
    -------
    call(neural_network_flow: tensorflow.Tensor) -> tensorflow.Tensor
        Applies the Sigmoid activation function to the input tensor and returns the output tensor.

    Example
    -------
    >>> import tensorflow
    ...    # Example tensor with shape (batch_size, sequence_length, 8)
    ...    input_tensor = tensorflow.random.uniform((2, 5, 8))
    ...    # Instantiate and apply Sigmoid
    ...    sigmoid_layer = Sigmoid()
    ...    output_tensor = sigmoid_layer(input_tensor)
    ...    # Output shape (batch_size, sequence_length, 8)
    ...    print(output_tensor.shape)
    >>>


    """

    def __init__(self, **kwargs):
        """
        Initializes the Sigmoid activation function layer.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the base Layer class.
        """
        super(Sigmoid, self).__init__(**kwargs)

    def call(self, neural_network_flow: tensorflow.Tensor) -> tensorflow.Tensor:
        """
        Applies the Sigmoid activation function to the input tensor.

        Parameters
        ----------
            neural_network_flow : tensorflow.Tensor
                Input tensor with any shape.

        Returns
        -------
        tensorflow.Tensor
            Output tensor with the same shape as input, after applying Sigmoid transformation.
        """
        output = tensorflow.sigmoid(neural_network_flow)

        # Cache the logits to use for crossentropy loss
        try:
            output._keras_logits = neural_network_flow
        except AttributeError:
            # We're dealing with a C-type
            pass

        return output

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