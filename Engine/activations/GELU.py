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

    import tensorflow

    #from keras.src import ops
    from keras import ops 

    from tensorflow.keras.layers import Layer

except ImportError as error:
    print(error)
    sys.exit(-1)

class GELU(Layer):
    """
    Gaussian Error Linear Unit (GELU) Layer.

    The Gaussian Error Linear Unit (GELU) is an activation function introduced by Hendrycks and Gimpel (2016)
    in the paper "Gaussian Error Linear Units (GELUs)" (https://arxiv.org/abs/1606.08415). It is widely used
    in deep learning architectures, including Transformer-based models, due to its smooth and adaptive
    non-linearity.

    GELU is a smoother alternative to ReLU and approximates the behavior of dropout by adaptively gating
    the input using a scaled error function.

    Mathematical Definition
    ----------------------
    Given an input tensor `X`, the GELU activation is defined as:


        `gelu(x) = x * P(X <= x)` where `P(X) ~ N(0, 1)`,
        i.e. `gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`.

        GELU weights inputs by their value, rather than gating
        inputs by their sign as in ReLU.

    This formulation provides a close approximation of the true Gaussian CDF-based activation function.

    Attributes
    ----------
        None (inherits attributes from the base Layer class)

    Methods
    -------
        call(neural_network_flow: tf.Tensor) -> tf.Tensor
            Applies the GELU activation function to the input tensor and returns the output tensor.

    Example
    -------
    >>> import tensorflow as tf
    ...    # Example tensor with shape (batch_size, feature_dim)
    ...    input_tensor = tf.random.uniform((2, 5))
    ...    # Instantiate and apply GELU
    ...    gelu_layer = GELU()
    ...    output_tensor = gelu_layer(input_tensor)
    ...    # Output shape (batch_size, feature_dim)
    ...    print(output_tensor.shape)
    >>>
    """

    def __init__(self, **kwargs):
        """
        Initializes the Gaussian Error Linear Unit (GELU) layer.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the base Layer class.
        """
        super(GELU, self).__init__(**kwargs)

    def call(self, neural_network_flow: tensorflow.Tensor) -> tensorflow.Tensor:
        """
        Applies the GELU activation function to the input tensor.

        The GELU activation uses the tanh approximation for efficiency.

        Parameters
        ----------
            neural_network_flow : tensorflow.Tensor
                Input tensor with any shape.

        Returns
        -------
        tensorflow.Tensor
            Output tensor with the same shape as input, after applying GELU transformation.

        Example
        -------
        >>> input_tensor = tensorflow.random.uniform((2, 5))
        ...     gelu = GELU()
        ...     output = gelu(input_tensor)
        ...     print(output.shape)
        >>>     (2, 5)
        """
        return ops.gelu(neural_network_flow)

    def compute_output_shape(self, input_shape):
        # GELU does not alter the shape, so the output shape is the same as the input shape
        return input_shape
