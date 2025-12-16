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

    from tensorflow.keras import initializers
    from tensorflow.keras.layers import Layer

except ImportError as error:
    print(error)
    sys.exit(-1)


class PReLU(Layer):
    """
    Parametric Rectified Linear Unit (PReLU) Activation Function Layer.

    The PReLU activation function is defined as:

        prelu(x) = x       if x > 0
        prelu(x) = alpha * x if x <= 0

    Where `alpha` is a learnable parameter that controls the slope of the function
    for negative input values. This allows the model to adaptively learn how to handle
    negative activations, improving model expressiveness and convergence in some tasks.

    Attributes
    ----------
        alpha_initializer : str or tf.keras.initializers.Initializer
            Initializer for the learnable `alpha` parameter (default is 'zeros').
        shared_axes : tuple of int
            Axes along which to share the learnable parameter `alpha` (useful for CNNs).

    Methods
    -------
        call(neural_network_flow: tf.Tensor) -> tf.Tensor
            Applies the PReLU activation function to the input tensor and returns the output tensor.

    Example
    -------
    >>> import tensorflow as tf
    >>> x = tf.constant([-10, -5, 0, 5, 10], dtype=tf.float32)
    >>> prelu_layer = PReLU()
    >>> output = prelu_layer(x)
    >>> print(output.numpy())
    """

    def __init__(self, alpha_initializer='zeros', shared_axes=None, **kwargs):
        """
        Initializes the PReLU activation layer.

        Parameters
        ----------
        alpha_initializer : str or tf.keras.initializers.Initializer
            Initializer for the learnable slope parameter for negative inputs.
        shared_axes : tuple of int, optional
            Axes along which to share the `alpha` parameters (useful in conv layers).
        **kwargs
            Additional keyword arguments passed to the base Layer class.
        """
        super(PReLU, self).__init__(**kwargs)
        self.alpha_initializer = initializers.get(alpha_initializer)
        self.shared_axes = shared_axes
        self.alpha = None

    def build(self, input_shape):
        """
        Creates the trainable parameter `alpha` with proper broadcasting and shape.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the input tensor.
        """
        param_shape = list(input_shape[1:])
        if self.shared_axes is not None:
            for axis in self.shared_axes:
                param_shape[axis - 1] = 1  # axis starts at 1 (excluding batch)
        self.alpha = self.add_weight(
            name='alpha',
            shape=param_shape,
            initializer=self.alpha_initializer,
            trainable=True
        )
        super(PReLU, self).build(input_shape)

    def call(self, neural_network_flow: tensorflow.Tensor) -> tensorflow.Tensor:
        """
        Applies the PReLU activation function to the input tensor.

        Parameters
        ----------
            neural_network_flow : tf.Tensor
                Input tensor with any shape.

        Returns
        -------
        tf.Tensor
            Output tensor with the same shape as input.
        """
        return tensorflow.maximum(0.0,
                                  neural_network_flow) + self.alpha * tensorflow.minimum(0.0, neural_network_flow)

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape, which is the same as the input shape.

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