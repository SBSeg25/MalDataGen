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
    import numpy

    from typing import Any
    from typing import Dict
    from typing import Optional

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from Engine.Activations.Activations import Activations

except ImportError as error:
    print(error)
    sys.exit(-1)


class CrossAttentionLayer(nn.Module):
    """
    Cross-Attention layer for conditioning on label information.

    The input features act as Query, while the label embedding provides Key and Value.
    """

    def __init__(self, embed_dim, num_heads=4):
        super(CrossAttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Query projection (from input features)
        self.query_dense = nn.Linear(embed_dim, embed_dim)

        # Key and Value projections (from label embedding)
        self.key_dense = nn.Linear(embed_dim, embed_dim)
        self.value_dense = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_dense = nn.Linear(embed_dim, embed_dim)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, head_dim)"""
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_input, key_value_input):
        batch_size = query_input.size(0)

        # Ensure inputs have sequence dimension
        if len(query_input.shape) == 2:
            query_input = query_input.unsqueeze(1)
        if len(key_value_input.shape) == 2:
            key_value_input = key_value_input.unsqueeze(1)

        # Linear projections
        Q = self.query_dense(query_input)
        K = self.key_dense(key_value_input)
        V = self.value_dense(key_value_input)

        # Split heads
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Scaled dot-product attention
        matmul_qk = torch.matmul(Q, K.transpose(-2, -1))
        dk = torch.tensor(self.head_dim, dtype=torch.float32)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)

        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)

        # Concatenate heads
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        concat_attention = attention_output.view(batch_size, -1, self.embed_dim)

        # Final linear projection
        output = self.out_dense(concat_attention)

        # Remove sequence dimension if it was added
        output = output.squeeze(1)

        return output


class VanillaEncoderTorch(Activations, nn.Module):
    """
    VanillaEncoder with Cross-Attention

    A class representing a Vanilla Encoder model with cross-attention mechanism for label conditioning.
    Instead of simple concatenation, the encoder uses cross-attention where the input features query
    the label embedding information. The encoder is designed to process inputs and labels, apply a
    series of dense layers with activations and dropout, and output a latent representation of the
    input data. This model is typically used in applications such as autoencoders, variational
    autoencoders, or other generative models.

    Attributes:
        @encoder_latent_dimension (int):
            The dimensionality of the latent space that the model will output.
        @encoder_output_shape (tuple):
            The desired output shape of the encoder, defining the shape of the encoded representation.
        @encoder_activation_function (str):
            The activation function applied to each layer of the encoder (e.g., 'ReLU', 'LeakyReLU').
        @encoder_last_layer_activation (str):
            The activation function applied to the final output layer.
        @encoder_dropout_decay_rate_encoder (float):
            The rate of dropout applied during encoding to improve generalization (must be between 0 and 1).
        @encoder_number_neurons_encoder (list):
            A list specifying the number of neurons (or units) in each layer of the encoder network.
        @encoder_dataset_type (dtype):
            The data type of the input dataset, default is numpy.float32.
        @encoder_initializer_mean (float):
            The mean for the normal distribution used to initialize the weights.
        @encoder_initializer_deviation (float):
            The standard deviation for the normal distribution used to initialize the weights.
        @encoder_number_samples_per_class (Optional[dict]):
            An optional dictionary containing metadata about the number of samples per class.
        @encoder_attention_embed_dim (int):
            Embedding dimension for cross-attention mechanism.
        @encoder_attention_num_heads (int):
            Number of attention heads in cross-attention layer.

    Raises:
        ValueError:
            Raised when the following invalid arguments are passed during initialization:
            - `latent_dimension` is not a positive integer.
            - `initializer_mean` or `initializer_deviation` is not a number.
            - `dropout_decay_encoder` is outside the valid range [0, 1].
            - `number_neurons_encoder` is not a non-empty list or contains non-positive integers.
            - `number_samples_per_class` is provided but is not a dictionary.
            - `attention_embed_dim` is not divisible by `attention_num_heads`.

    Example:
        >>> encoder = VanillaEncoderTorch(
        ...     latent_dimension=128,
        ...     output_shape=(64, 64, 1),
        ...     activation_function='ReLU',
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_encoder=0.5,
        ...     last_layer_activation='sigmoid',
        ...     number_neurons_encoder=[512, 256, 128],
        ...     dataset_type=numpy.float32,
        ...     number_samples_per_class={"number_classes": 10},
        ...     attention_embed_dim=128,
        ...     attention_num_heads=4
        ... )
    """

    def __init__(self, latent_dimension: int, output_shape: tuple, activation_function: str, initializer_mean: float,
                 initializer_deviation: float, dropout_decay_encoder: float, last_layer_activation: str,
                 number_neurons_encoder: list, dataset_type: Any = numpy.float32,
                 number_samples_per_class: Optional[Dict[str, Any]] = None,
                 attention_embed_dim: int = 128, attention_num_heads: int = 4):
        """
        Initializes the VanillaEncoder with cross-attention and the provided parameters.

        Args:
            latent_dimension (int): The dimension of the latent space.
            output_shape (tuple): The desired output shape of the encoder.
            activation_function (str): The activation function to use for the layers.
            initializer_mean (float): The mean for weight initialization.
            initializer_deviation (float): The standard deviation for weight initialization.
            dropout_decay_encoder (float): The rate of dropout applied during encoding.
            last_layer_activation (str): The activation function for the last layer.
            number_neurons_encoder (list): List specifying the number of neurons in each encoder layer.
            dataset_type (dtype, optional): The data type of the input dataset. Defaults to numpy.float32.
            number_samples_per_class (dict, optional): Specifies the number of samples per class.
            attention_embed_dim (int, optional): Embedding dimension for cross-attention (default: 128).
            attention_num_heads (int, optional): Number of attention heads (default: 4).
        """
        # Initialize Activations first (if it's a class with __init__)
        try:
            Activations.__init__(self)
        except:
            pass

        # Initialize nn.Module
        nn.Module.__init__(self)

        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")

        if not isinstance(initializer_mean, (int, float)):
            raise ValueError("initializer_mean must be a number.")

        if not isinstance(initializer_deviation, (int, float)):
            raise ValueError("initializer_deviation must be a number.")

        if not isinstance(dropout_decay_encoder, (int, float)) or not (0 <= dropout_decay_encoder <= 1):
            raise ValueError("dropout_decay_encoder must be a float between 0 and 1.")

        if not isinstance(number_neurons_encoder, list) or len(number_neurons_encoder) == 0:
            raise ValueError("number_neurons_encoder must be a non-empty list.")

        for neurons in number_neurons_encoder:
            if not isinstance(neurons, int) or neurons <= 0:
                raise ValueError("Each element in number_neurons_encoder must be a positive integer.")

        if number_samples_per_class is not None:
            if not isinstance(number_samples_per_class, dict):
                raise ValueError("number_samples_per_class must be a dictionary.")

        if not isinstance(attention_embed_dim, int) or attention_embed_dim <= 0:
            raise ValueError(f"Invalid value for attention_embed_dim: {attention_embed_dim}. It must be a positive integer.")

        if not isinstance(attention_num_heads, int) or attention_num_heads <= 0:
            raise ValueError(f"Invalid value for attention_num_heads: {attention_num_heads}. It must be a positive integer.")

        if attention_embed_dim % attention_num_heads != 0:
            raise ValueError(f"attention_embed_dim ({attention_embed_dim}) must be divisible by attention_num_heads ({attention_num_heads}).")

        self._encoder_latent_dimension = latent_dimension
        self._encoder_output_shape = output_shape
        self._encoder_activation_function = activation_function
        self._encoder_last_layer_activation = last_layer_activation
        self._encoder_dropout_decay_rate_encoder = dropout_decay_encoder
        self._encoder_dataset_type = dataset_type
        self._encoder_initializer_mean = initializer_mean
        self._encoder_initializer_deviation = initializer_deviation
        self._encoder_number_neurons_encoder = number_neurons_encoder
        self._encoder_number_samples_per_class = number_samples_per_class
        self._encoder_attention_embed_dim = attention_embed_dim
        self._encoder_attention_num_heads = attention_num_heads

    def get_encoder(self, input_shape: int) -> nn.Module:
        """
        Creates and returns the encoder model with cross-attention.

        The model uses cross-attention to condition the input features on label information,
        where the input features act as Query and the label embedding provides Key and Value.

        This method constructs the neural network by stacking dense layers with the provided
        configurations (neurons, dropout, and activation).

        Args:
            input_shape (int): The shape of the input data.

        Returns:
            nn.Module: The encoder model with cross-attention which takes input data and labels
                       and outputs the encoded latent representation and labels.
        """

        class EncoderModule(nn.Module):
            def __init__(self, latent_dim, num_classes, input_dim, number_neurons,
                         init_mean, init_std, dropout_rate, activation_fn, last_activation_fn,
                         get_activation_func, attention_embed_dim, attention_num_heads):
                super().__init__()

                # Store only the necessary configuration, not the parent object
                self.latent_dim = latent_dim
                self.num_classes = num_classes
                self.get_activation_func = get_activation_func
                self.attention_embed_dim = attention_embed_dim
                self.attention_num_heads = attention_num_heads

                # Label embedding layer
                self.label_embedding = nn.Linear(num_classes, attention_embed_dim)
                self._init_weights(self.label_embedding, init_mean, init_std)

                # Project input features to attention embedding dimension
                self.input_projection = nn.Linear(input_dim, attention_embed_dim)
                self._init_weights(self.input_projection, init_mean, init_std)

                # Cross-attention: input features query label information
                self.cross_attention = CrossAttentionLayer(
                    embed_dim=attention_embed_dim,
                    num_heads=attention_num_heads
                )

                # Build layers
                self.layers = nn.ModuleList()
                self.dropouts = nn.ModuleList()
                self.activations = []

                # First layer after attention
                layer = nn.Linear(attention_embed_dim, number_neurons[0])
                self._init_weights(layer, init_mean, init_std)
                self.layers.append(layer)
                self.dropouts.append(nn.Dropout(dropout_rate))
                self.activations.append(activation_fn)

                # Hidden layers
                for i in range(1, len(number_neurons)):
                    prev_neurons = number_neurons[i - 1]
                    curr_neurons = number_neurons[i]

                    layer = nn.Linear(prev_neurons, curr_neurons)
                    self._init_weights(layer, init_mean, init_std)
                    self.layers.append(layer)
                    self.dropouts.append(nn.Dropout(dropout_rate))
                    self.activations.append(activation_fn)

                # Latent layer
                last_neurons = number_neurons[-1]
                self.latent_layer = nn.Linear(last_neurons, latent_dim)
                self._init_weights(self.latent_layer, init_mean, init_std)
                self.latent_activation = last_activation_fn

            def _init_weights(self, layer, mean, std):
                """Initialize layer weights with normal distribution."""
                nn.init.normal_(layer.weight, mean=mean, std=std)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

            def forward(self, x):
                """
                Forward pass through the encoder with cross-attention.

                Args:
                    x: List/tuple of [data_input, label_input]

                Returns:
                    Tuple of (latent_representation, label_input)
                """
                data_input, label_input = x

                # Embed labels
                label_embedded = F.relu(self.label_embedding(label_input))

                # Project input features to attention embedding dimension
                input_projected = self.input_projection(data_input)

                # Cross-attention: input features query label information
                attended_features = self.cross_attention(input_projected, label_embedded)

                # Pass through encoder layers
                x = attended_features
                for layer, dropout, activation_name in zip(self.layers, self.dropouts, self.activations):
                    x = layer(x)
                    x = dropout(x)
                    x = self.get_activation_func(activation_name)(x)

                # Latent layer
                x = self.latent_layer(x)
                x = self.get_activation_func(self.latent_activation)(x)

                return x, label_input

        # Create encoder with configuration values instead of parent reference
        num_classes = self._encoder_number_samples_per_class["number_classes"]

        return EncoderModule(
            latent_dim=self._encoder_latent_dimension,
            num_classes=num_classes,
            input_dim=input_shape,
            number_neurons=self._encoder_number_neurons_encoder,
            init_mean=self._encoder_initializer_mean,
            init_std=self._encoder_initializer_deviation,
            dropout_rate=self._encoder_dropout_decay_rate_encoder,
            activation_fn=self._encoder_activation_function,
            last_activation_fn=self._encoder_last_layer_activation,
            get_activation_func=self._get_activation,
            attention_embed_dim=self._encoder_attention_embed_dim,
            attention_num_heads=self._encoder_attention_num_heads
        )

    def _get_activation(self, activation_name: str):
        """
        Returns the PyTorch activation function based on the name.

        Args:
            activation_name (str): Name of the activation function.

        Returns:
            nn.Module: PyTorch activation function.
        """
        activation_name = activation_name.lower()

        if activation_name == 'relu':
            return nn.ReLU()
        elif activation_name == 'leakyrelu':
            return nn.LeakyReLU(0.2)
        elif activation_name == 'sigmoid':
            return nn.Sigmoid()
        elif activation_name == 'tanh':
            return nn.Tanh()
        elif activation_name == 'swish' or activation_name == 'silu':
            return nn.SiLU()
        elif activation_name == 'elu':
            return nn.ELU()
        elif activation_name == 'softmax':
            return nn.Softmax(dim=1)
        elif activation_name == 'linear' or activation_name == 'none':
            return nn.Identity()
        else:
            # Try to use the Activations parent class method if available
            try:
                return self._add_activation_layer(None, activation_name)
            except:
                raise ValueError(f"Unsupported activation function: {activation_name}")

    @property
    def dropout_decay_rate_encoder(self) -> float:
        """
        Gets the rate of dropout decay for the encoder layers.

        Returns:
            float: The rate of dropout decay applied to the encoder layers.
        """
        return self._encoder_dropout_decay_rate_encoder

    @property
    def number_filters_encoder(self) -> list:
        """
        Gets the number of neurons for each encoder layer.

        Returns:
            list: A list specifying the number of neurons in each encoder layer.
        """
        return self._encoder_number_neurons_encoder

    @property
    def attention_embed_dim(self) -> int:
        """int: Gets the embedding dimension for cross-attention."""
        return self._encoder_attention_embed_dim

    @property
    def attention_num_heads(self) -> int:
        """int: Gets the number of attention heads."""
        return self._encoder_attention_num_heads

    @dropout_decay_rate_encoder.setter
    def dropout_decay_rate_encoder(self, dropout_decay_rate_generator: float) -> None:
        """
        Sets the rate of dropout decay for the encoder layers.

        Args:
            dropout_decay_rate_generator (float): The new dropout decay rate.

        Raises:
            ValueError: If the value is not a float between 0 and 1.
        """
        if not (0 <= dropout_decay_rate_generator <= 1):
            raise ValueError("dropout_decay_rate_encoder must be a float between 0 and 1.")

        self._encoder_dropout_decay_rate_encoder = dropout_decay_rate_generator