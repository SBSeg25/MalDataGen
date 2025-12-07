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

    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Layer

except ImportError as error:
    print(error)
    sys.exit(-1)


class CrossAttentionBlock(Layer):
    """
    A custom Keras layer that implements a Cross-Attention mechanism.

    This layer computes cross-attention between two sets of input sequences:
    queries and key-value pairs. The cross-attention mechanism computes attention
    scores between the queries and keys and uses the resulting attention weights
    to perform weighted aggregation of the values. The output of the attention is
    projected  into a desired number of units. It also incorporates a residual
    connection between the input and the attention output.

    The CrossAttentionBlock layer is inspired by the self-attention mechanism described
    in the paper "Attention is All You Need" (Vaswani et al., 2017), but this version
    operates with queries and key-value pairs from separate inputs, making it suitable
    for tasks such as cross-modal learning or multi-view attention.

    References:
        - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. A., Kaiser,
        Ł., & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information
        Processing Systems, 30. URL: https://arxiv.org/abs/1706.03762

    Mathematical Definition:
        Let Q represent the query matrix of shape (batch_size, seq_len, num_channels),
        K represent the key matrix, and V represent the value matrix, both of shape (batch_size, seq_len).

        The attention mechanism can be described as:

        Attention Scores = (Q ⋅ Kᵀ) / √d_k
        where d_k is the dimension of the key vectors (i.e., the number of units in this case).

        Then, the attention weights are computed as:

        Attention Weights = softmax(Attention Scores)

        The output of the attention mechanism is:

        Attention Output = Attention Weights ⋅ V

        Finally, the output is projected by applying a linear transformation and adding the input values as a residual:

        Final Output = Input + Projection(Attention Output)

    Attributes:
        @units: Integer representing the number of output units for each attention head.
        @query_weights: Dense layer for the linear transformation of the query inputs.
        @key_weights: Dense layer for the linear transformation of the key inputs.
        @value_weights: Dense layer for the linear transformation of the value inputs.
        @projection_weights: Dense layer for projecting the attention output.

    Methods:
        call(input_values): Computes the attention output and applies the residual connection.

    Example:
    >>>     python3
    ...     import tensorflow as tf
    ...     # Define input tensors (batch_size=2, seq_len=5, num_channels=3)
    ...     query_inputs = tf.random.normal((2, 5, 3))
    ...     key_value_inputs = tf.random.normal((2, 5))
    ...     # Instantiate the CrossAttentionBlock
    ...     cross_attention_block = CrossAttentionBlock(units=4)
    ...     # Call the layer with input values
    ...     output = cross_attention_block([query_inputs, key_value_inputs])
    >>>     print(output.shape)  # Expected output shape: (2, 5, 4)

    """

    def __init__(self, units, **kwargs):
        """
        Initializes the CrossAttentionBlock layer.

        Args:
            units (int): The number of output units for the attention block.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        super().__init__(**kwargs)
        self.units = units

        # Initialize the weight matrices for query, key, value, and output projection
        self.query_weights = Dense(units)
        self.key_weights = Dense(units)
        self.value_weights = Dense(units)
        self.projection_weights = Dense(units)

    def call(self, input_values):
        """
        Performs the forward pass of the CrossAttentionBlock layer.

        Args:
            input_values (list): A list containing two tensors:
                - query_inputs (Tensor): The query tensor of shape (batch_size, seq_len, num_channels).
                - key_value_inputs (Tensor): The key-value tensor of shape (batch_size, seq_len).

        Returns:
            Tensor: The resulting tensor of shape (batch_size, seq_len, units),
                    which is the sum of the original query inputs and the attention output.
        """
        # Extract query inputs and key-value inputs from the input values
        query_inputs, key_value_inputs = input_values  # (batch_size, seq_len, num_channels), (batch_size, seq_len)

        # Get the dimensions for batch size, sequence length, and number of channels
        number_channels = tensorflow.shape(query_inputs)[2]

        # Expand key_value_inputs to match the shape (batch_size, seq_len, num_channels)
        key_value_inputs = tensorflow.tile(tensorflow.expand_dims(key_value_inputs, axis=-1), [1, 1, number_channels])

        # Calculate scaling factor for attention scores
        scale = tensorflow.cast(self.units, tensorflow.float32) ** -0.5

        # Apply linear projections to queries, keys, and values
        query = self.query_weights(query_inputs)  # (batch_size, seq_len, units)
        key = self.key_weights(key_value_inputs)  # (batch_size, seq_len, units)
        value = self.value_weights(key_value_inputs)  # (batch_size, seq_len, units)

        # Compute attention scores
        attention_scores = tensorflow.matmul(query, key, transpose_b=True) * scale  # (batch_size, seq_len, seq_len)

        # Apply softmax to obtain attention weights
        attention_weights = tensorflow.nn.softmax(attention_scores, axis=-1)  # (batch_size, seq_len, seq_len)

        # Compute attention output by applying attention weights to values
        attention_output = tensorflow.matmul(attention_weights, value)  # (batch_size, seq_len, units)

        # Project the attention output to the desired dimensionality
        attention_output = self.projection_weights(attention_output)  # (batch_size, seq_len, units)

        # Apply residual connection by adding the input query values to the attention output
        return query_inputs + attention_output
