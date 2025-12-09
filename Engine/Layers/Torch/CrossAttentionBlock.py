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
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

except ImportError as error:
    print(error)
    sys.exit(-1)


class CrossAttentionBlock(nn.Module):
    """
    A custom PyTorch module that implements a Cross-Attention mechanism.

    This layer computes cross-attention between two sets of input sequences:
    queries and key-value pairs. The cross-attention mechanism computes attention
    scores between the queries and keys and uses the resulting attention weights
    to perform weighted aggregation of the values. The output of the attention is
    projected into a desired number of units. It also incorporates a residual
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
        units (int): The number of output units for each attention head.
        query_weights (nn.Linear): Linear layer for the transformation of the query inputs.
        key_weights (nn.Linear): Linear layer for the transformation of the key inputs.
        value_weights (nn.Linear): Linear layer for the transformation of the value inputs.
        projection_weights (nn.Linear): Linear layer for projecting the attention output.

    Methods:
        forward(query_inputs, key_value_inputs): Computes the attention output and applies the residual connection.

    Example:
    >>>     import torch
    ...     # Define input tensors (batch_size=2, seq_len=5, num_channels=3)
    ...     query_inputs = torch.randn(2, 5, 3)
    ...     key_value_inputs = torch.randn(2, 5)
    ...     # Instantiate the CrossAttentionBlock
    ...     cross_attention_block = CrossAttentionBlock(units=4)
    ...     # Call the layer with input values
    ...     output = cross_attention_block(query_inputs, key_value_inputs)
    >>>     print(output.shape)  # Expected output shape: torch.Size([2, 5, 4])

    """

    def __init__(self, units):
        """
        Initializes the CrossAttentionBlock layer.

        Args:
            units (int): The number of output units for the attention block.
        """
        super(CrossAttentionBlock, self).__init__()
        self.units = units

        # Initialize the weight matrices for query, key, value, and output projection
        # Note: PyTorch Linear layers will be created dynamically in forward pass
        # to handle variable input dimensions
        self.query_weights = None
        self.key_weights = None
        self.value_weights = None
        self.projection_weights = nn.Linear(units, units)

    def forward(self, query_inputs, key_value_inputs):
        """
        Performs the forward pass of the CrossAttentionBlock layer.

        Args:
            query_inputs (torch.Tensor): The query tensor of shape (batch_size, seq_len, num_channels).
            key_value_inputs (torch.Tensor): The key-value tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: The resulting tensor of shape (batch_size, seq_len, units),
                         which is the sum of the original query inputs and the attention output.
        """
        # Get the dimensions for batch size, sequence length, and number of channels
        batch_size, seq_len, num_channels = query_inputs.shape

        # Expand key_value_inputs to match the shape (batch_size, seq_len, num_channels)
        key_value_inputs = key_value_inputs.unsqueeze(-1).expand(batch_size, seq_len, num_channels)

        # Initialize linear layers if not already created
        if self.query_weights is None:
            self.query_weights = nn.Linear(num_channels, self.units).to(query_inputs.device)
            self.key_weights = nn.Linear(num_channels, self.units).to(query_inputs.device)
            self.value_weights = nn.Linear(num_channels, self.units).to(query_inputs.device)

        # Calculate scaling factor for attention scores
        scale = self.units ** -0.5

        # Apply linear projections to queries, keys, and values
        query = self.query_weights(query_inputs)  # (batch_size, seq_len, units)
        key = self.key_weights(key_value_inputs)  # (batch_size, seq_len, units)
        value = self.value_weights(key_value_inputs)  # (batch_size, seq_len, units)

        # Compute attention scores
        # query: (batch_size, seq_len, units)
        # key.transpose(-2, -1): (batch_size, units, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale  # (batch_size, seq_len, seq_len)

        # Apply softmax to obtain attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, seq_len, seq_len)

        # Compute attention output by applying attention weights to values
        attention_output = torch.matmul(attention_weights, value)  # (batch_size, seq_len, units)

        # Project the attention output to the desired dimensionality
        attention_output = self.projection_weights(attention_output)  # (batch_size, seq_len, units)

        # Apply residual connection by adding the input query values to the attention output
        # Note: If query_inputs has different dimensions than attention_output, we need to project it
        if query_inputs.shape[-1] != self.units:
            residual_proj = nn.Linear(query_inputs.shape[-1], self.units).to(query_inputs.device)
            return residual_proj(query_inputs) + attention_output

        return query_inputs + attention_output