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

    from tensorflow.keras.layers import Layer

except ImportError as error:
    print(error)
    sys.exit(-1)


class TransposeLayer(Layer):
    """
    A custom layer that performs a permutation of the input tensor's dimensions.

    This layer allows you to reorder the dimensions of a tensor according to the specified
    permutation. The `perm` argument defines the order in which the dimensions of the tensor
    will be permuted, enabling flexible manipulation of the tensor's shape. This can be useful
    when working with different data formats or performing operations like matrix transpositions.

    Attributes:
        channels_permutation (list of int): A list or tuple of integers defining the permutation
                                             of the tensor dimensions. For example, [0, 2, 1] swaps
                                             the second and third dimensions of the input tensor.

    Example usage:
    ---------------
    # Create an instance of the TransposeLayer
    transpose_layer = TransposeLayer(perm=[0, 2, 1])

    # Example input tensor of shape (batch_size, sequence_length, embedding_dim)
    input_tensor = tf.random.normal([32, 100, 128])  # Batch of 32, sequence length of 100, 128 features

    # Apply the TransposeLayer
    output_tensor = transpose_layer(input_tensor)

    # Print the output tensor's shape
    print("Output tensor shape:", output_tensor.shape)

    Output:
    -------
    Output tensor shape: (32, 128, 100)
    """

    def __init__(self, perm, **kwargs):
        """
        Initializes the TransposeLayer with the specified permutation.

        Args:
            perm (list or tuple of int): The permutation order of the tensor's dimensions.
                                         This is a list of integers where each integer represents
                                         the index of the dimension in the output tensor.
            **kwargs: Additional arguments passed to the base `Layer` class (e.g., name, trainable).
        """

        # Call to the parent class (Layer) constructor to initialize the layer
        super(TransposeLayer, self).__init__(**kwargs)
        self.channels_permutation = perm

    def call(self, inputs):
        """
        Performs the permutation of the input tensor's dimensions based on the `perm` argument.

        Args:
            inputs (Tensor): The input tensor whose dimensions are to be permuted.

        Returns:
            Tensor: The output tensor with dimensions permuted as per the `perm` argument.
        """
        # Perform the permutation of the input tensor's dimensions
        return tensorflow.transpose(inputs, perm=self.channels_permutation)

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the transposed tensor.

        Args:
            input_shape (tuple of int): The shape of the input tensor.

        Returns:
            list of int: The shape of the output tensor after permutation.
        """
        # Compute the output shape by reordering the input dimensions according to `perm`
        return [input_shape[dim] for dim in self.channels_permutation]

    def get_config(self):
        """
        Returns the configuration of the TransposeLayer for serialization.

        Returns:
            dict: A dictionary containing the configuration of the layer, including the `perm` argument.
        """
        # Get the base layer's configuration and add the `perm` attribute
        config = super(TransposeLayer, self).get_config()
        config.update({
            'perm': self.channels_permutation
        })
        return config
