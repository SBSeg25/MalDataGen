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


class TransformerEncoder(Layer):
    """
    Custom TensorFlow layer implementing a Transformer encoder block.

    The Transformer encoder block is a fundamental component of the Transformer architecture.
    It consists of a multi-head self-attention mechanism followed by a feedforward neural network.
    The encoder layer normalizes and applies dropout to the results of both the attention mechanism
    and the feedforward network. This block is typically used as part of the encoder in sequence-to-sequence models.

    The encoder takes an input sequence and learns contextual relationships between tokens in the sequence.
    It computes a representation of the input, which is later used by the decoder in sequence-to-sequence tasks.

    Reference:
        Vaswani et al., "Attention is All You Need" (2017). This work introduced the Transformer architecture
        and the multi-head self-attention mechanism.

    Args:
        @embedding_dimension (int): The dimensionality of the input embeddings.
        @number_heads (int): The number of attention heads in the multi-head attention mechanism.
        @feedforward_dimension (int): The dimensionality of the feedforward network's hidden layer.
        @dropout_rate (float): The dropout rate applied after attention and feedforward layers. Default is 0.1.

    Attributes:
        @embedding_dimension (int): The dimensionality of the input embeddings.
        @number_heads (int): The number of attention heads in the multi-head attention mechanism.
        @feedforward_dimension (int): The dimensionality of the feedforward network's hidden layer.
        @dropout_rate (float): The dropout rate applied after attention and feedforward layers.
        @mult_head_attention (MultiHeadAttention): The multi-head self-attention layer.
        @feedforward_layer (Sequential): A sequential model consisting of two Dense layers.
        @first_layer_normalization (LayerNormalization): The layer normalization applied after attention.
        @second_layer_normalization (LayerNormalization): The layer normalization applied after the feedforward network.
        @first_dropout (Dropout): The dropout layer applied after the attention output.
        @second_dropout (Dropout): The dropout layer applied after the feedforward output.

    Example:
        >>> # Create a TransformerEncoder layer with embedding dimension of 128, 8 attention heads,
        ...     # feedforward layer dimension of 512, and a dropout rate of 0.1
        ...     encoder_layer = TransformerEncoder(embedding_dimension=128, number_heads=8, feedforward_dimension=512)
        ...     # Sample input tensor (batch_size=2, sequence_length=10, embedding_dim=128)
        ...     encoder_input = tf.random.normal((2, 10, 128))  # Encoder input
        ...     # Apply the Transformer encoder layer
        ...     output = encoder_layer(encoder_input, training=True)
        >>>     print(output.shape)  # Output tensor with shape (batch_size, sequence_length, embedding_dim)
    """

    def __init__(self, embedding_dimension, number_heads, feedforward_dimension, dropout_rate=0.1):
        """
        Initializes the TransformerEncoder with specified parameters for embeddings, attention heads,
        and feedforward network dimensions, along with the dropout rate.

        Parameters
        ----------
        embedding_dimension : int
            The dimensionality of the input embeddings. This is the size of the vector that represents each token.
        number_heads : int
            The number of attention heads in the multi-head attention mechanism.
        feedforward_dimension : int
            The dimensionality of the feedforward network's hidden layer.
        dropout_rate : float
            The dropout rate applied after attention and feedforward layers. Default is 0.1.
        """
        super(TransformerEncoder, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.number_heads = number_heads
        self.feedforward_dimension = feedforward_dimension
        self.dropout_rate = dropout_rate

        # Multi-head self-attention layer
        self.mult_head_attention = MultiHeadAttention(num_heads=number_heads, key_dim=embedding_dimension)

        # Feedforward network consisting of two Dense layers
        self.feedforward_layer = tensorflow.keras.Sequential([
            Dense(feedforward_dimension, activation='relu'),  # First layer with ReLU activation
            Dense(embedding_dimension)  # Output layer projecting back to embedding dimension
        ])

        # First layer normalization, applied after the attention layer
        self.first_layer_normalization = LayerNormalization(epsilon=1e-6)

        # Second layer normalization, applied after the feedforward network
        self.second_layer_normalization = LayerNormalization(epsilon=1e-6)

        # Dropout layers applied after attention and feedforward network
        self.first_dropout = Dropout(dropout_rate)
        self.second_dropout = Dropout(dropout_rate)

    def call(self, x, training):
        """
        Performs the forward pass of the Transformer encoder. Applies multi-head self-attention,
        followed by layer normalization. Then, a feedforward network is applied followed by another
        layer normalization.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor of shape (batch_size, sequence_length, embedding_dimension).
        training : bool
            Whether the layer is in training mode (applies dropout) or inference mode.

        Returns
        -------
        tf.Tensor
            The output tensor of the same shape as the input, after processing by the Transformer encoder block.
        """
        # Apply multi-head self-attention to the input
        attention_output = self.mult_head_attention(x, x)

        # Apply dropout to the attention output if in training mode
        attention_output = self.first_dropout(attention_output, training=training)

        # Add the attention output to the input and apply the first layer normalization
        output_normalization = self.first_layer_normalization(x + attention_output)

        # Pass the normalized output through the feedforward network
        feedforward_output = self.feedforward_layer(output_normalization)

        # Apply dropout to the feedforward output if in training mode
        feedforward_output = self.second_dropout(feedforward_output, training=training)

        # Add the feedforward output to the normalized output and apply the second layer normalization
        output_second_normalization = self.second_layer_normalization(output_normalization + feedforward_output)

        return output_second_normalization
