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

    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Layer

except ImportError as error:
    print(error)
    sys.exit(-1)

DEFAULT_GROUP_NORMALIZATION = 1


class AttentionBlock(Layer):
    """
    AttentionBlock

    Implements a scaled dot-product attention mechanism for use in deep learning models.
    The block includes query, key, and value projections, followed by a final projection
    layer. Additionally, it integrates group normalization to normalize the attention outputs.

    This block is inspired by the attention mechanism described in the paper:

    Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. A., Kaiser, Ł., & Polosukhin, I.
    (2017). Attention is all you need. In Advances in neural information processing systems (Vol. 30).

    Attributes:
        @units (int):
            The number of output units for the dense layers in the attention mechanism.
        @groups (int):
            The number of groups for normalization. Defaults to `DEFAULT_GROUP_NORMALIZATION`.

    Methods:
        @call(inputs):
            Computes the forward pass for the attention block. This method calculates the attention
            scores, applies them to the input, and returns the augmented input after normalization.

    Example:
        >>> attention_block = AttentionBlock(units=64, groups=8)
        >>> output = attention_block(inputs)
    """

    def __init__(self, units, groups=DEFAULT_GROUP_NORMALIZATION, **kwargs):
        """
        Initializes the AttentionBlock by defining the layers and configuration.

        Args:
            units (int):
                Number of units for the dense layers in the attention mechanism.
            groups (int, optional):
                Number of groups for normalization. Defaults to `DEFAULT_GROUP_NORMALIZATION`.
            **kwargs:
                Additional keyword arguments for the parent Layer class.
        """
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)

        # Define dense layers for query, key, value, and final projection
        self.query_weights = Dense(units)
        self.key_weights = Dense(units)
        self.value_weights = Dense(units)
        self.projection_weights = Dense(units)

    def call(self, inputs):
        """
        Performs the forward pass of the attention block.

        Args:
            inputs (Tensor): The input tensor of shape (batch_size, height, embedding_dim).

        Returns:
            Tensor: The output tensor after applying the attention mechanism and projection. 
                    The shape is the same as the input tensor.
        """
        batch_size = tensorflow.shape(inputs)[0]
        height = tensorflow.shape(inputs)[1]
        scale = tensorflow.cast(self.units, tensorflow.float32) ** (-0.5)

        # Compute query, key, and value projections
        query = self.query_weights(inputs)
        key = self.key_weights(inputs)
        value = self.value_weights(inputs)

        # Compute attention scores using scaled dot-product attention
        attention_score = tensorflow.einsum("bhc, bHc->bhH", query, key) * scale
        attention_score = tensorflow.reshape(attention_score, [batch_size, height, height])

        # Apply softmax to obtain attention weights
        attention_score = tensorflow.nn.softmax(attention_score, -1)
        attention_score = tensorflow.reshape(attention_score, [batch_size, height, height])

        # Apply attention weights to the value tensor
        projection = tensorflow.einsum("bhH,bHc->bhc", attention_score, value)
        projection = self.projection_weights(projection)

        # Add the original input to the projection to form the output
        return inputs + projection
