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


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer for discrete latent representations.

    Args:
        num_embeddings (int): Number of embeddings in the codebook
        embedding_dim (int): Dimensionality of each embedding
        commitment_cost (float): Weight for commitment loss (default: 0.25)
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, name="vector_quantizer"):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.name = name

        # Initialize embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        # Losses storage
        self._vq_loss = None
        self._commitment_loss = None

    def forward(self, x):
        """
        Forward pass through vector quantization.

        Args:
            x: Input tensor of shape (batch_size, embedding_dim)

        Returns:
            Quantized tensor with straight-through estimator
        """
        # Flatten input if needed
        input_shape = x.shape
        flat_input = x.view(-1, self.embedding_dim)

        # Calculate distances to embeddings
        distances = (
                torch.sum(flat_input ** 2, dim=1, keepdim=True)
                + torch.sum(self.embeddings.weight ** 2, dim=1)
                - 2 * torch.matmul(flat_input, self.embeddings.weight.t())
        )

        # Find nearest embeddings
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embeddings(encoding_indices)

        # Calculate VQ loss (codebook loss)
        self._vq_loss = F.mse_loss(quantized.detach(), flat_input)

        # Calculate commitment loss
        self._commitment_loss = self.commitment_cost * F.mse_loss(quantized, flat_input.detach())

        # Straight-through estimator
        quantized = flat_input + (quantized - flat_input).detach()

        # Reshape to original shape
        quantized = quantized.view(input_shape)

        return quantized

    @property
    def losses(self):
        """Returns list of VQ losses"""
        if self._vq_loss is not None and self._commitment_loss is not None:
            return [self._vq_loss + self._commitment_loss]
        return []