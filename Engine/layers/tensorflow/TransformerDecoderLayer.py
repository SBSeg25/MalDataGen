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
    from tensorflow.keras.layers import Dense

    from tensorflow.keras.layers import Dropout

    from tensorflow.keras.layers import MultiHeadAttention
    from tensorflow.keras.layers import LayerNormalization

except ImportError as error:
    print(error)
    sys.exit(-1)


class TransformerDecoder(Layer):
    """
    Custom TensorFlow layer implementing a Transformer decoder block.

    The Transformer decoder block is a key component of the Transformer architecture,
    commonly used in sequence-to-sequence tasks such as machine translation, text generation,
    and summarization. It consists of two multi-head attention layers followed by a feedforward
    neural network. Layer normalization and dropout are applied after each step to stabilize
    training and regularize the model.

    The decoder takes both the current input sequence (decoder input) and the output from the encoder
    as inputs. It performs two types of attention:
        1. **Self-attention**: Attention over the decoder's own input sequence.
        2. **Encoder-decoder attention**: Attention over the encoder's output, allowing the decoder
             to focus on relevant parts of the encoder's context.

    Reference:
        Vaswani et al., "Attention is All You Need" (2017). The Transformer architecture introduced
        multi-head attention mechanisms and positional encodings to better capture sequential dependencies.

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
        @first_mult_head_attention (MultiHeadAttention): The first multi-head self-attention layer.
        @second_mult_head_attention (MultiHeadAttention): The second multi-head attention layer
         for attending to encoder output.
        @feedforward (Sequential): A sequential model consisting of two Dense layers.
        @first_layer_normalization (LayerNormalization): The first layer normalization applied
        after the first attention layer.
        @second_layer_normalization (LayerNormalization): The second layer normalization applied
         after the second attention layer.
        @third_layer_normalization (LayerNormalization): The third layer normalization applied
         after the feedforward layer.
        @first_dropout (Dropout): The dropout layer applied after the first attention output.
        @second_dropout (Dropout): The dropout layer applied after the second attention output.
        @third_dropout (Dropout): The dropout layer applied after the feedforward output.

    Example:
        >>> # Create a TransformerDecoder layer with embedding dimension of 128, 8 attention heads,
        ...     # feedforward layer dimension of 512, and a dropout rate of 0.1
        ...     decoder_layer = TransformerDecoder(embedding_dimension=128, number_heads=8, feedforward_dimension=512)
        ...     # Sample input tensors (batch_size=2, sequence_length=10, embedding_dim=128)
        ...     decoder_input = tf.random.normal((2, 10, 128))  # Decoder input
        ...     encoder_output = tf.random.normal((2, 10, 128))  # Encoder output
        ...     # Apply the Transformer decoder layer
        ...     output = decoder_layer(decoder_input, encoder_output, training=True)
        >>>     print(output.shape)  # Output tensor with shape (batch_size, sequence_length, embedding_dim)
    """

    def __init__(self, embedding_dimension, number_heads, feedforward_dimension, dropout_rate=0.1):
        """
        Initializes the TransformerDecoder with the specified parameters for embeddings, attention heads,
        and feedforward network dimensions, along with the dropout rate.

        Parameters
        ----------
        embedding_dimension : int
            The dimensionality of the input embeddings. This is the size of the vector that represents each token.
        number_heads : int
            The number of attention heads in the multi-head attention mechanism.
        feedforward_dimension : int
            The dimensionality of the feedforward network hidden layer.
        dropout_rate : float
            The dropout rate applied after attention and feedforward layers. Default is 0.1.
        """
        super(TransformerDecoder, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.number_heads = number_heads
        self.feedforward_dimension = feedforward_dimension
        self.dropout_rate = dropout_rate

        # First multi-head self-attention layer for the decoder input
        self.first_mult_head_attention = MultiHeadAttention(num_heads=number_heads, key_dim=embedding_dimension)

        # Second multi-head attention layer for attending to the encoder output
        self.second_mult_head_attention = MultiHeadAttention(num_heads=number_heads, key_dim=embedding_dimension)

        # Feedforward network consisting of two Dense layers
        self.feedforward = tensorflow.keras.Sequential([
            Dense(feedforward_dimension, activation='relu'),  # First layer with ReLU activation
            Dense(embedding_dimension)  # Output layer projecting back to embedding dimension
        ])

        # Layer normalization applied after each step
        self.first_layer_normalization = LayerNormalization(epsilon=1e-6)
        self.second_layer_normalization = LayerNormalization(epsilon=1e-6)
        self.third_layer_normalization = LayerNormalization(epsilon=1e-6)

        # Dropout layers applied after each step for regularization
        self.first_dropout = Dropout(dropout_rate)
        self.second_dropout = Dropout(dropout_rate)
        self.third_dropout = Dropout(dropout_rate)

    def call(self, x, encoder_output, training):
        """
        Performs the forward pass of the Transformer decoder. It applies the first multi-head self-attention
        on the decoder input, then the second multi-head attention on the encoder output, followed by a feedforward network.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor of shape (batch_size, sequence_length, embedding_dimension) to the decoder.
        encoder_output : tf.Tensor
            The output tensor from the encoder, used as context in the decoder's attention mechanism.
        training : bool
            A flag indicating whether the layer is in training mode (applies dropout) or inference mode.

        Returns
        -------
        tf.Tensor
            The output tensor of shape (batch_size, sequence_length, embedding_dimension),
            processed by the Transformer decoder block.
        """
        # Apply the first multi-head self-attention to the input
        first_attention_output = self.first_mult_head_attention(x, x)

        # Apply dropout to the first attention output if in training mode
        first_attention_output = self.first_dropout(first_attention_output, training=training)

        # Add the first attention output to the input and apply the first layer normalization
        output_normalization = self.first_layer_normalization(x + first_attention_output)

        # Apply the second multi-head attention using the encoder output as context
        second_attention_output = self.second_mult_head_attention(output_normalization, encoder_output)

        # Apply dropout to the second attention output if in training mode
        second_attention_output = self.second_dropout(second_attention_output, training=training)

        # Add the second attention output to the normalized output and apply the second layer normalization
        second_output_normalization = self.second_layer_normalization(output_normalization + second_attention_output)

        # Pass the normalized output through the feedforward network
        feedforward_output = self.feedforward(second_output_normalization)

        # Apply dropout to the feedforward output if in training mode
        feedforward_output = self.third_dropout(feedforward_output, training=training)

        # Add the feedforward output to the normalized output and apply the third layer normalization
        third_output_normalization = self.third_layer_normalization(second_output_normalization + feedforward_output)

        return third_output_normalization