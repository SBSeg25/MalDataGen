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

import tensorflow
from tensorflow.keras.layers import Layer


class ClusteringLayer(Layer):
    """
    A custom Keras layer that performs clustering by selecting latent vectors
    based on the similarity between input vectors and predefined latent vectors.
    This layer is designed to be used in scenarios where clustering or assigning
    input samples to specific clusters is required during the model's forward pass.

    Attributes:
        @number_vectors (int):
            The number of clusters (vectors) to select from.
        @latent_dimension (int):
            The dimension of each latent vector.
        @number_time_steps (int):
            The number of time steps in the sequence.
        @alpha (float, optional):
            A scaling factor for the similarity calculation. Default is 0.5.
        @latent_vectors (tensorflow.Variable):
            The latent vectors to be optimized during training.

    Methods:
        build(input_shape):
            Initializes the trainable latent vectors.
        call(inputs, training=False):
            Performs the clustering operation by selecting the most similar latent vectors based on input vectors.

    Example:
        >>>     # Create a model with the ClusteringLayer
        ...     model = tensorflow.keras.Sequential([
        ...     tensorflow.keras.layers.InputLayer(input_shape=(None, 64, 1)),  # Input shape (timesteps, feature_size, 1)
        ...     ClusteringLayer(number_vectors=10, latent_dimension=64, number_steps=5)
        ...     ])
        ...
        ...     # Prepare input and index (e.g., [inputs, indices])
        ...     inputs = tensorflow.random.normal(shape=(32, 64, 1))  # Example input data with batch_size=32
        ...     indices = tensorflow.random.uniform(shape=(32,), minval=0, maxval=5, dtype=tf.int32)  # Example indices
        ...
        ...     output = model([inputs, indices])  # Call the model with input and indices
        ...
        >>>     print(output.shape)  # The output will contain selected latent vectors
    """

    def __init__(self, number_vectors, latent_dimension, number_steps, alpha=0.5, **kwargs):
        """
        Initialize the ClusteringLayer.

        Args:
            number_vectors (int): The number of clusters (vectors) to select from.
            latent_dimension (int): The dimension of each latent vector.
            number_steps (int): The number of time steps in the sequence.
            alpha (float, optional): A scaling factor for the similarity calculation. Default is 0.5.
            **kwargs: Additional arguments to be passed to the base class Layer.

        """
        super(ClusteringLayer, self).__init__(**kwargs)
        self.number_vectors = number_vectors
        self.latent_dimension = latent_dimension
        self.number_time_steps = number_steps
        self.alpha = alpha
        self.latent_vectors = None

    def build(self, input_shape):
        """
        Initialize the latent vectors as trainable parameters during the build process.

        Args:
            input_shape (tuple): Shape of the input tensor. This is passed by Keras during the model creation.

        """
        # Create the latent vectors as a trainable weight
        self.latent_vectors = self.add_weight(
            shape=(self.number_time_steps, self.number_vectors, self.latent_dimension),
            initializer="glorot_uniform",
            trainable=True,
            name="latent_vectors"
        )
        super(ClusteringLayer, self).build(input_shape)

    def call(self, inputs, training=False):
        """
        Perform the clustering operation by selecting the most similar latent vectors
        for each input sample.

        Args:
            inputs (tuple): A tuple containing the input tensor and index tensor.
                             The input tensor is expected to be of shape (batch_size, timesteps, feature_size, 1),
                             and the index tensor of shape (batch_size,).
            training (bool, optional): Whether the model is in training mode or not. Default is False.

        Returns:
            tensorflow.Tensor: A tensor of shape (batch_size, latent_dimension), which contains the selected
                               latent vectors based on the input data and the indices.
        """
        inputs, index = inputs  # Unpack inputs and indices

        # Remove singleton dimension at the last axis of the inputs
        inputs_squeezed = tensorflow.squeeze(inputs, axis=-1)

        # Gather the current latent vectors based on the provided indices
        current_latent_vectors = tensorflow.gather(self.latent_vectors, index, axis=0)

        # Normalize the inputs and latent vectors
        inputs_normalised = tensorflow.nn.l2_normalize(inputs_squeezed, axis=-1)
        latent_normalised = tensorflow.nn.l2_normalize(current_latent_vectors, axis=-1)

        # Expand dimensions to align tensors for element-wise multiplication
        inputs_normalised = tensorflow.expand_dims(inputs_normalised, axis=1)

        # Calculate the similarity between inputs and latent vectors using dot product
        similarity = tensorflow.multiply(latent_normalised, inputs_normalised)

        # Sum the similarity scores across the last axis (feature dimension)
        similarity_sum = tensorflow.reduce_sum(similarity, axis=-1)

        # Find the index of the best matching latent vector
        best_match_indices = tensorflow.argmax(similarity_sum, axis=-1)
        best_match_indices = tensorflow.expand_dims(best_match_indices, axis=-1)

        # Select the latent vectors corresponding to the best match indices
        selected_weights = tensorflow.gather(current_latent_vectors, best_match_indices, axis=1, batch_dims=1)

        # Squeeze the selected weights to remove extra dimensions
        selected_weights_squeezed = tensorflow.squeeze(selected_weights, axis=1)

        return selected_weights_squeezed
