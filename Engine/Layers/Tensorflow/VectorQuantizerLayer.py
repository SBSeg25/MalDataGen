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


class VectorQuantizer(Layer):
    """
    Implements the vector quantization layer for VQ-VAE models.

     This layer:
        1. Discretizes continuous latent variables using a learned codebook
        2. Implements the commitment loss and codebook loss
        3. Uses straight-through gradient estimation

     Args:
         number_embeddings (int): Size K of the codebook
         embedding_dimension (int): Dimension D of each embedding vector
         beta (float, optional): Weight for commitment loss. Default 0.25
         **kwargs: Additional base layer arguments

    Mathematical Definition:
    ------------------------

    Given:
        - Input tensor x ∈ ℝ^(B×D), where B is batch size, D is embedding dimension
        - Codebook vectors e ∈ ℝ^(K×D), where K is number of embeddings
        - β: commitment loss weight (default 0.25)

    1. Quantization Process:
       a. Compute pairwise distances between x and all e_i:
          d(x_j, e_i) = ||x_j - e_i||² = ||x_j||² + ||e_i||² - 2⟨x_j, e_i⟩

       b. Find nearest neighbor for each x_j:
          k_j = argmin_i d(x_j, e_i)

       c. Quantize inputs:
          q(x_j) = e_{k_j}

    2. Loss Components:
       a. Codebook Loss:
          L_code = ||sg[x] - q(x)||²
          (Moves codebook vectors toward encoder outputs)

       b. Commitment Loss:
          L_commit = β||x - sg[q(x)]||²
          (Encourages encoder to commit to codebook)

       Where sg[·] is the stop-gradient operator.

    3. Straight-Through Estimator:
       Output = x + sg[q(x) - x]
       (Forward: quantized values, Backward: gradients bypass quantization)

    Reference:
    van den Oord et al., "Neural Discrete Representation Learning", NeurIPS 2017
    arXiv:1711.00937

        Example:
        >>> vq = VectorQuantizer(number_embeddings=512,
        ...                    embedding_dimension=64)
        >>> quantized = vq(encoder_outputs)
    """


    def __init__(self, number_embeddings, embedding_dimension, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        """Initialize the VQ layer with codebook and hyperparameters."""
        self.embedding_dim = embedding_dimension
        self.number_embeddings = number_embeddings
        self.beta = beta
        # Initialize codebook with uniform distribution
        # Shape: (embedding_dim, num_embeddings) for efficient matmul
        weight_init = tensorflow.random_uniform_initializer()
        self.embeddings = self.add_weight(
            name="embeddings_vqvae",
            shape=(self.embedding_dim, self.number_embeddings),
            initializer=weight_init,
            trainable=True
        )

    def call(self, input_flow):
        """Forward pass with quantization and loss computation.

        Args:
            x: Input tensor of shape (..., embedding_dim)

        Returns:
            Quantized tensor with same shape as input
        """
        x, y = input_flow

        # Store original shape for reconstruction
        input_shape = tensorflow.shape(x)

        # Flatten to 2D: [batch*sequence, embedding_dim]
        flattened = tensorflow.reshape(x, [-1, self.embedding_dim])

        # Get indices of nearest codebook vectors
        encoding_indices = self.get_code_indices(flattened)

        # Convert to one-hot encodings
        encodings = tensorflow.one_hot(encoding_indices, self.number_embeddings)

        # Quantize by matrix multiplication with codebook
        # q(x) = e_k where k = argmin||x-e_i||
        quantized = tensorflow.matmul(encodings, self.embeddings, transpose_b=True)


        # Reshape back to original input dimensions
        quantized = tensorflow.reshape(quantized, input_shape)

        # Compute commitment loss: ||sg[q(x)] - x||²
        commitment_loss = tensorflow.reduce_mean((tensorflow.stop_gradient(quantized) - x) ** 2)

        # Compute codebook loss: ||q(x) - sg[x]||²
        codebook_loss = tensorflow.reduce_mean((quantized - tensorflow.stop_gradient(x)) ** 2)

        # Add weighted loss to layer
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator:
        # Forward: quantized values
        # Backward: gradients bypass quantization
        quantized = x + tensorflow.stop_gradient(quantized - x)

        return [quantized, y]

    def get_code_indices(self, flattened_inputs):
        """Compute nearest codebook indices for each input.

        Args:
            flattened_inputs: 2D tensor of shape [N, D]

        Returns:
            1D tensor of indices [N]
        """
        # Compute similarity matrix: ⟨x_j, e_i⟩
        similarity = tensorflow.matmul(flattened_inputs, self.embeddings)

        # Compute squared L2 distances using:
        # ||x - e||² = ||x||² + ||e||² - 2⟨x,e⟩
        distances = (tensorflow.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)+
                     tensorflow.reduce_sum(self.embeddings ** 2, axis=0) - 2 * similarity)

        # Find index of minimum distance for each input
        encoding_indices = tensorflow.argmin(distances, axis=1)

        return encoding_indices
