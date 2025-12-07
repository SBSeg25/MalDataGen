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
    from tensorflow.keras.layers import Dense

    from tensorflow.keras.layers import Dropout

    from tensorflow.keras.layers import MultiHeadAttention
    from tensorflow.keras.layers import LayerNormalization

except ImportError as error:
    print(error)
    sys.exit(-1)


class PositionalEncoding(Layer):
    """
    Custom TensorFlow layer that adds positional encoding to input embeddings.
    Positional encoding is crucial for transformer models as it injects information
    about the position of each token in a sequence. This layer adds this information
    to the input embeddings, allowing the model to differentiate between tokens
    in different positions within the sequence.

    The positional encoding is based on sine and cosine functions with different frequencies,
    a method first introduced by Vaswani et al. (2017) in the "Attention is All You Need" paper.

    Reference:
        Vaswani et al., "Attention is All You Need" (2017). This paper introduced the Transformer model
        and positional encoding, which allows the model to handle sequences of arbitrary length.

    Args:
        @max_sequence_length (int): The maximum length of the input sequences.
        @embedding_dimension (int): The dimensionality of the input embeddings.

    Attributes:
        @positional_encodings (tf.Tensor): A tensor containing the precomputed positional encodings.
        @max_sequence_length (int): The maximum sequence length for which positional encodings are computed.
        @embedding_dimension (int): The dimensionality of the embeddings.

    Example:
        >>> #Initialize layer with a maximum sequence length of 100 and an embedding dimension of 512.
        ...     positional_encoding_layer = PositionalEncoding(max_sequence_length=100, embedding_dimension=512)
        ...     # Sample input tensor (batch_size=2, sequence_length=10, embedding_dimension=512)
        ...     input_tensor = tf.random.normal((2, 10, 512))
        ...     # Apply positional encoding to the input tensor
        ...     output_tensor = positional_encoding_layer(input_tensor)
        >>>     print(output_tensor.shape)  # Output shape will be (2, 10, 512)
    """

    def __init__(self, max_sequence_length, embedding_dimension):
        """
        Initializes the PositionalEncoding layer with the specified maximum sequence length
        and embedding dimension.

        Parameters
        ----------
        max_sequence_length : int
            The maximum length of input sequences. The positional encodings will be precomputed for
            sequences up to this length.
        embedding_dimension : int
            The dimensionality of input embeddings. This will determine the size of the positional encoding
            for each token.
        """
        super(PositionalEncoding, self).__init__()
        self.positional_encodings = None
        self.max_sequence_length = max_sequence_length
        self.embedding_dimension = embedding_dimension

    def build(self, input_shape):
        """
        Builds the PositionalEncoding layer by computing the positional encodings.
        This method is called automatically before the layer is used for the first time.

        Args:
            input_shape (tuple): The shape of the input data.
        """
        # Precompute the positional encodings for the maximum sequence length and embedding dimension
        self.positional_encodings = self._get_positional_encodings(self.max_sequence_length, self.embedding_dimension)

    @staticmethod
    def _get_positional_encodings(max_seq_length, embedding_dimension):
        """
        Computes the positional encodings for the given sequence length and embedding dimension.
        The encodings are based on sine and cosine functions of different frequencies, as described in
        the "Attention is All You Need" paper by Vaswani et al. (2017).

        Parameters
        ----------
        max_seq_length : int
            The maximum length of input sequences. This determines the number of position encodings.
        embedding_dimension : int
            The dimensionality of the embeddings. This defines the length of the positional encoding vector.

        Returns
        -------
        tf.Tensor
            A tensor of shape (1, max_seq_length, embedding_dimension) containing the positional encodings.
        """
        # Create a range of positions for the sequence
        positional_array = tensorflow.range(max_seq_length, dtype=tensorflow.float32)[:, tensorflow.newaxis]

        # Create an index array for the embedding dimensions
        index = tensorflow.range(embedding_dimension, dtype=tensorflow.float32)[tensorflow.newaxis, :]

        # Compute the angles for the sine and cosine functions
        angles = positional_array / tensorflow.pow(10000.0, (2 * (index // 2)) / tensorflow.cast(embedding_dimension,
                                                                                                 tensorflow.float32))

        # Apply sine to even indices in the embedding dimensions
        angles_sin = tensorflow.math.sin(angles[:, 0::2])

        # Apply cosine to odd indices in the embedding dimensions
        angles_cos = tensorflow.math.cos(angles[:, 1::2])

        # Concatenate the sine and cosine encodings along the last axis
        positional_encodings = tensorflow.concat([angles_sin, angles_cos], axis=-1)

        # Add a batch dimension to the positional encodings
        return positional_encodings[tensorflow.newaxis, ...]

    def call(self, x):
        """
        Adds the precomputed positional encodings to the input embeddings.

        Parameters
        ----------
        x : tensorflow.Tensor
            The input tensor of shape (batch_size, sequence_length, embedding_dimension).

        Returns
        -------
        tensorflow.Tensor
            A tensor of the same shape as the input, with positional encodings added
            to the embeddings.
        """
        # Get the sequence length of the input tensor
        sequence_length = tensorflow.shape(x)[1]

        # Retrieve the positional encodings corresponding to the current sequence length
        positional_encodings = self.positional_encodings[:, :sequence_length, :]

        # Add the positional encodings to the input embeddings
        return x + positional_encodings
