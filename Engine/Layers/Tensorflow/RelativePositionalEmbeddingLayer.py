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

DEFAULT_MAX_LENGTH = 128
DEFAULT_EMBEDDING_DIMENSION = 64


class RelativePositionalEmbedding(Layer):
    """
    A custom TensorFlow layer that adds relative positional embeddings to the input tensor.

    This layer generates positional embeddings for input sequences to encode information about
    the relative position of each element in the sequence. The embeddings are added to the input tensor
    to allow the model to take into account the order of tokens in sequence-based tasks, such as NLP.

    **Positional Embeddings** are critical in models like Transformers, where they allow the model
    to encode the position of each token in the input sequence. This implementation generates
    relative positional embeddings, which can improve performance for tasks where relative position matters
    (e.g., machine translation, text generation).

    Reference:
        Vaswani et al., "Attention is All You Need" (2017), where positional embeddings were introduced
        in Transformer-based models. This work laid the foundation for using positional embeddings in self-attention.

    Attributes
    ----------
        @max_length : int
            The maximum length of the sequence for which positional embeddings are created.
            This defines the upper limit on the length of input sequences that the model can process.
        @embedding_dimension : int
            The dimension of the positional embeddings. This is the size of the vector that represents each position.
            positional_embeddings : tf.Variable
            A matrix containing the positional embeddings for each position (from 0 to max_length-1),
            initialized with a uniform distribution.

    Example
    -------
        >>> # Create a RelativePositionalEmbedding layer with maximum sequence length of 50 and embedding dimension of 128
        ...     pos_embed_layer = RelativePositionalEmbedding(max_length=50, embedding_dimension=128)
        ...     # Sample input tensor (batch_size=2, sequence_length=10, embedding_dim=128)
        ...     inputs = tf.random.normal((2, 10, 128))
        ...     # Build the layer (this must be done before using it)
        ...     pos_embed_layer.build(inputs.shape)
        ...     # Apply positional embedding to input tensor
        ...     output = pos_embed_layer(inputs)
        >>>     print(output.shape)  # Output tensor with positional embeddings added
    """

    def __init__(self,
                 max_length=DEFAULT_MAX_LENGTH,
                 embedding_dimension=DEFAULT_EMBEDDING_DIMENSION,
                 **kwargs):
        """
        Initializes the RelativePositionalEmbedding layer with the specified parameters.

        Parameters
        ----------
        max_length : int
            The maximum length of the sequence for which positional embeddings are created.
            Default is `DEFAULT_MAX_LENGTH`.
        embedding_dimension : int
            The dimension of the positional embeddings.
            Default is `DEFAULT_EMBEDDING_DIMENSION`.
        **kwargs : Additional keyword arguments.
        """
        super(RelativePositionalEmbedding, self).__init__(**kwargs)
        self.positional_embeddings = None
        self.max_length = max_length
        self.embedding_dimension = embedding_dimension

    def build(self, input_shape):
        """
        Builds the layer by creating the positional embeddings matrix. The matrix is initialized
        using a uniform distribution.

        Parameters
        ----------
        input_shape : tuple
            The shape of the input tensor (batch_size, sequence_length, embedding_dim).
            The sequence_length is used to determine how much of the positional embeddings are needed.

        Notes
        -----
        This method initializes the positional embeddings with a shape of (max_length, embedding_dimension).
        The embeddings are not trainable and are added directly to the input during the call.
        """
        self.positional_embeddings = self.add_weight(
            name="pos_embed",
            shape=(self.max_length, self.embedding_dimension),
            initializer='uniform'
        )

    def call(self, inputs):
        """
        Adds the relative positional embeddings to the input tensor. For each input sequence,
        the corresponding positional embeddings are added to the input features.

        The positional embeddings are sliced to match the sequence length of the input.

        Parameters
        ----------
        inputs : tf.Tensor
            The input tensor of shape (batch_size, sequence_length, embedding_dim).
            Each element represents a token embedding in the sequence.

        Returns
        -------
        tf.Tensor
            The input tensor with the positional embeddings added, resulting in a tensor
            of the same shape (batch_size, sequence_length, embedding_dim).
        """
        # Create a range of indices representing the positions (from 0 to max_length - 1)
        positional_index = tensorflow.range(start=0, limit=self.max_length, delta=1)

        # Look up the positional embeddings corresponding to each position
        positional_embedding = tensorflow.nn.embedding_lookup(self.positional_embeddings, positional_index)

        # Add positional embeddings to the input tensor. Slice positional_embeddings to match the input sequence length.
        return inputs + positional_embedding[:tensorflow.shape(inputs)[1], :]  # Add positional embeddings for the current sequence length
