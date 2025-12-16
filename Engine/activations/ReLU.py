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


class ReLU(Layer):
    """
    Rectified Linear Unit (ReLU) Activation Function Layer.

    The ReLU activation function is defined as:

        relu(x) = x if x > 0
        relu(x) = 0 if x <= 0

    However, with modifications to the default parameters, it can be customized
    to handle negative values differently, set a maximum value for the output,
    and apply a threshold below which values are damped or set to zero.

    The function can be configured as:

        relu(x) = max(x, 0)  (standard ReLU)
        relu(x) = negative_slope * x if x < threshold (LeakyReLU)
        relu(x) = min(max(x, threshold), max_value)  (saturation and thresholding)

    Attributes
    ----------
        negative_slope : float
            Slope for negative values (default is 0.0).
        max_value : float, optional
            Saturation threshold for the output (default is None).
        threshold : float
            The threshold value below which the activation is damped or set to zero (default is 0.0).

    Methods
    -------
        call(neural_network_flow: tensorflow.Tensor) -> tensorflow.Tensor
            Applies the ReLU activation function to the input tensor and returns the output tensor.

    Example
    -------
    >>> import tensorflow
    >>> x = tensorflow.constant([-10, -5, 0, 5, 10], dtype=tensorflow.float32)
    >>> relu_layer = ReLU(negative_slope=0.5, max_value=5.0)
    >>> output = relu_layer(x)
    >>> print(output.numpy())  # Output will be adjusted based on negative_slope and max_value
    """

    def __init__(self, negative_slope=0.0, max_value=None, threshold=0.0, **kwargs):
        """
        Initializes the ReLU activation function layer.

        Parameters
        ----------
        negative_slope : float
            Slope for negative values (default is 0.0).
        max_value : float, optional
            Maximum value to apply to the output (default is None).
        threshold : float
            Threshold below which values are damped or set to zero (default is 0.0).
        **kwargs
            Additional keyword arguments passed to the base Layer class.
        """
        super(ReLU, self).__init__(**kwargs)
        self.negative_slope = negative_slope
        self.max_value = max_value
        self.threshold = threshold

    def call(self, neural_network_flow: tensorflow.Tensor) -> tensorflow.Tensor:
        """
        Applies the ReLU activation function to the input tensor.

        Parameters
        ----------
            neural_network_flow : tensorflow.Tensor
                Input tensor with any shape.

        Returns
        -------
        tensorflow.Tensor
            Output tensor with the same shape as input, after applying ReLU transformation.
        """
        # Apply threshold and negative slope
        x = tensorflow.where(neural_network_flow > self.threshold,
                             neural_network_flow,
                             self.negative_slope * neural_network_flow)

        # Apply max_value saturation
        if self.max_value is not None:
            x = tensorflow.minimum(x, self.max_value)

        return x

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