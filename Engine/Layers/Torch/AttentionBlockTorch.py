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
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

except ImportError as error:
    print(error)
    sys.exit(-1)

DEFAULT_GROUP_NORMALIZATION = 1


class AttentionBlockTorch(nn.Module):
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
        @forward(inputs):
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
                Additional keyword arguments for the parent Module class.
        """
        super().__init__()
        self.units = units
        self.groups = groups

        # Define linear layers for query, key, value, and final projection
        self.query_weights = nn.Linear(units, units)
        self.key_weights = nn.Linear(units, units)
        self.value_weights = nn.Linear(units, units)
        self.projection_weights = nn.Linear(units, units)

    def forward(self, inputs):
        """
        Performs the forward pass of the attention block.

        Args:
            inputs (Tensor): The input tensor of shape (batch_size, height, embedding_dim).

        Returns:
            Tensor: The output tensor after applying the attention mechanism and projection.
                    The shape is the same as the input tensor.
        """
        batch_size = inputs.shape[0]
        height = inputs.shape[1]
        scale = float(self.units) ** (-0.5)

        # Compute query, key, and value projections
        query = self.query_weights(inputs)  # (batch_size, height, units)
        key = self.key_weights(inputs)  # (batch_size, height, units)
        value = self.value_weights(inputs)  # (batch_size, height, units)

        # Compute attention scores using scaled dot-product attention
        # einsum "bhc, bHc->bhH" means: batch, height_q, channels @ batch, height_k, channels -> batch, height_q, height_k
        attention_score = torch.einsum("bhc,bHc->bhH", query, key) * scale

        # Apply softmax to obtain attention weights
        attention_score = F.softmax(attention_score, dim=-1)

        # Apply attention weights to the value tensor
        # einsum "bhH,bHc->bhc" means: batch, height_q, height_k @ batch, height_k, channels -> batch, height_q, channels
        projection = torch.einsum("bhH,bHc->bhc", attention_score, value)
        projection = self.projection_weights(projection)

        # Add the original input to the projection to form the output (residual connection)
        return inputs + projection