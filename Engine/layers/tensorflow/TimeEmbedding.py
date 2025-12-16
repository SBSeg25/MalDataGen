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
    import numpy

    import tensorflow

    from tensorflow.keras.layers import Layer

except ImportError as error:
    print(error)
    sys.exit(-1)


class TimeEmbedding(Layer):
    """
    This class implements a sinusoidal time embedding layer. The purpose of this layer is to generate a
    time-dependent embedding vector, which can be used to encode temporal information. The embedding
    method is inspired by the positional encoding technique used in transformer models.

    Attributes:
    -----------
        dimension : int
            The dimension of the output embedding. It should be an even number because the embedding is
            split between sine and cosine components.
        _half_dimension : int
            Half of the embedding dimension. Used to calculate the sine and cosine components separately.
        _embedding : Tensor
            Pre-computed scaling factors for each embedding dimension. This is based on the formula
            10000^(2i/d), where 'i' is the index and 'd' is the dimension.

    Methods:
    --------
        __init__(self, dimension, **kwargs)
            Initializes the TimeEmbedding layer, setting the embedding dimension and calculating the scaling factors.

        call(self, inputs)
            Computes the time embedding for the given inputs using sine and cosine functions. This method
            takes a tensor of inputs representing time steps and returns the corresponding embedding.

    """

    def __init__(self, dimension, **kwargs):
        """
        Initializes the TimeEmbedding layer.

        Parameters:
        -----------
        dimension : int
            The dimension of the output embedding vector. Must be an even number for the split
            between sine and cosine components.
        **kwargs : dict
            Additional keyword arguments for the base Layer class.
        """
        super().__init__(**kwargs)
        self._dimension = dimension
        self._half_dimension = dimension // 2

        # Compute the scaling factors for each dimension in the embedding.
        # The formula is log(10000) / (half_dimension - 1).
        self._embedding = numpy.log(10000) / (self._half_dimension - 1)

        # Apply an exponential function to generate the scaling factors.
        # The result is a Tensor of size (_half_dimension,) for scaling each dimension of the time input.
        self._embedding = tensorflow.exp(tensorflow.range(self._half_dimension,
                                                          dtype=tensorflow.float32) * -self._embedding)

    def call(self, inputs):
        """
        Computes the time embedding for a batch of inputs using sine and cosine functions.

        Parameters:
        -----------
        inputs : Tensor
            A 1D or 2D tensor containing time steps, with shape (batch_size,) or (batch_size, 1).

        Returns:
        --------
        Tensor
            A 2D tensor with shape (batch_size, dimension) containing the computed time embeddings.
            The embedding consists of alternating sine and cosine values for each time input.
        """
        # Cast the input to a float32 tensor.
        inputs = tensorflow.cast(inputs, dtype=tensorflow.float32)

        # Scale the input by the pre-computed embedding factors.
        time_embedding = inputs[:, None] * self._embedding[None, :]

        # Concatenate the sine and cosine values along the last axis to form the final embedding.
        time_embedding = tensorflow.concat([tensorflow.sin(time_embedding),
                                            tensorflow.cos(time_embedding)], axis=-1)
        return time_embedding
