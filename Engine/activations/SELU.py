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


class SELU(Layer):
    """
    Scaled Exponential Linear Unit (SELU) Activation Function Layer.

    The Scaled Exponential Linear Unit (SELU) was introduced by Klambauer et al. (2017)
    in the paper "Self-Normalizing Neural Networks" (https://arxiv.org/abs/1706.02515).
    It is defined as:

        selu(x) = scale * x if x > 0 else scale * alpha * (exp(x) - 1)

    where `alpha` and `scale` are pre-defined constants:
    - `alpha = 1.67326324`
    - `scale = 1.05070098`

    Attributes
    ----------
        alpha : float
            Scaling factor for negative values (fixed at 1.67326324).
        scale : float
            Scaling factor for all values (fixed at 1.05070098).

    Methods
    -------
        call(neural_network_flow: tf.Tensor) -> tf.Tensor
            Applies the SELU activation function to the input tensor and returns the output tensor.


    Example
    -------
    >>> import tensorflow
    ...    # Example tensor with shape (batch_size, sequence_length, 8) — divisible by 2
    ...    input_tensor = tensorflow.random.uniform((2, 5, 8))
    ...    # Instantiate and apply SELU
    ...    selu_layer = SELU()
    ...    output_tensor = selu_layer(input_tensor)
    ...    # Output shape (batch_size, sequence_length, 4)
    ...    print(output_tensor.shape)
    >>>


    """

    def __init__(self, **kwargs):
        """
        Initializes the SELU activation function layer.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the base Layer class.
        """
        super(SELU, self).__init__(**kwargs)
        self.alpha = 1.67326324
        self.scale = 1.05070098

    def call(self, neural_network_flow: tensorflow.Tensor) -> tensorflow.Tensor:
        """
        Applies the SELU activation function to the input tensor.

        Parameters
        ----------
            neural_network_flow : tf.Tensor
                Input tensor with any shape.

        Returns
        -------
        tf.Tensor
            Output tensor with the same shape as input, after applying SELU transformation.
        """
        return self.scale * tensorflow.where(neural_network_flow > 0,
                                     neural_network_flow,
                                     self.alpha * (tensorflow.exp(neural_network_flow) - 1))

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
