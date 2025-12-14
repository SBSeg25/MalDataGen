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
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from typing import List, Dict, Union, Optional
    from Engine.Architectures.QuantizedVAE.Torch.ActivationTorch import ActivationsTorch

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


class SamplingLayer(nn.Module):
    """Sampling layer for VAE reparameterization trick."""

    def forward(self, z_mean, z_log_var):
        """
        Sample from latent distribution using reparameterization trick.

        Args:
            z_mean: Mean of latent distribution
            z_log_var: Log variance of latent distribution

        Returns:
            Sampled latent vector
        """
        batch_size = z_mean.size(0)
        latent_dim = z_mean.size(1)
        epsilon = torch.randn(batch_size, latent_dim, device=z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class EncoderNetwork(nn.Module):
    """Neural network implementation for the encoder with cross-attention."""

    def __init__(self,
                 num_classes,
                 output_shape,
                 latent_dimension,
                 number_neurons_encoder,
                 activation_function,
                 last_layer_activation,
                 dropout_decay_rate,
                 initializer_mean,
                 initializer_deviation,
                 attention_embed_dim=128,
                 attention_num_heads=4):
        super().__init__()

        # Store configuration
        self.num_classes = num_classes
        self.output_shape = output_shape
        self.latent_dimension = latent_dimension
        self.activation_function = activation_function
        self.last_layer_activation = last_layer_activation
        self.attention_embed_dim = attention_embed_dim
        self.attention_num_heads = attention_num_heads

        # Helper function for weight initialization
        def init_weights(layer, mean, std):
            nn.init.normal_(layer.weight, mean=mean, std=std)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # Helper function for activation
        def get_activation(activation_name):
            activations = {
                'relu': nn.ReLU(),
                'tanh': nn.Tanh(),
                'sigmoid': nn.Sigmoid(),
                'leaky_relu': nn.LeakyReLU(),
                'swish': nn.SiLU(),
                'linear': nn.Identity()
            }
            return activations.get(activation_name.lower(), nn.ReLU())

        # Label embedding layer
        self.label_embedding = nn.Linear(num_classes, attention_embed_dim)
        init_weights(self.label_embedding, initializer_mean, initializer_deviation)

        # Project input features to attention embedding dimension
        self.input_projection = nn.Linear(output_shape, attention_embed_dim)
        init_weights(self.input_projection, initializer_mean, initializer_deviation)

        # Cross-attention: input features query label information
        self.cross_attention = CrossAttentionLayer(
            embed_dim=attention_embed_dim,
            num_heads=attention_num_heads
        )

        # First layer after attention
        layers = []
        first_layer = nn.Linear(attention_embed_dim, number_neurons_encoder[0])
        init_weights(first_layer, initializer_mean, initializer_deviation)
        layers.append(first_layer)
        layers.append(nn.Dropout(dropout_decay_rate))
        layers.append(get_activation(activation_function))

        # Hidden layers
        for i in range(1, len(number_neurons_encoder)):
            layer = nn.Linear(
                number_neurons_encoder[i - 1],
                number_neurons_encoder[i]
            )
            init_weights(layer, initializer_mean, initializer_deviation)
            layers.append(layer)
            layers.append(nn.Dropout(dropout_decay_rate))
            layers.append(get_activation(activation_function))

        self.encoder_layers = nn.Sequential(*layers)

        # Final dense layer
        self.final_dense = nn.Linear(
            number_neurons_encoder[-1],
            latent_dimension
        )
        init_weights(self.final_dense, initializer_mean, initializer_deviation)

        # Latent space layers
        self.z_mean = nn.Linear(latent_dimension, latent_dimension)
        init_weights(self.z_mean, initializer_mean, initializer_deviation)

        self.z_log_var = nn.Linear(latent_dimension, latent_dimension)
        init_weights(self.z_log_var, initializer_mean, initializer_deviation)

        # Sampling layer
        self.sampling_layer = SamplingLayer()

        # Store activation getter
        self._get_activation = get_activation

    def forward(self, x, label=None):
        """
        Forward pass through the encoder with cross-attention.

        Args:
            x: Input features
            label: One-hot encoded labels (optional)

        Returns:
            Tuple of (z_mean, z_log_var, z_sample, label)
        """
        # If label not provided, create zeros
        if label is None:
            batch_size = x.size(0)
            label = torch.zeros((batch_size, self.num_classes), device=x.device)

        # Embed labels
        label_embedded = F.relu(self.label_embedding(label))

        # Project input features to attention embedding dimension
        input_projected = self.input_projection(x)

        # Cross-attention: input features query label information
        attended_features = self.cross_attention(input_projected, label_embedded)

        # Pass through encoder layers
        encoded = self.encoder_layers(attended_features)

        # Final dense layer
        encoded = self.final_dense(encoded)
        encoded = self._get_activation(self.last_layer_activation)(encoded)

        # Generate mean and log variance
        z_mean = self.z_mean(encoded)
        z_log_var = self.z_log_var(encoded)

        # Sample from latent space
        z = self.sampling_layer(z_mean, z_log_var)

        return z_mean, z_log_var, z, label


class VanillaEncoderTorch(nn.Module, ActivationsTorch):
    """
    VanillaEncoder with Cross-Attention

    Implements a fully connected conditional variational encoder (CVAE) model with cross-attention
    mechanism for label conditioning. Instead of simple concatenation, the encoder uses
    cross-attention where the input features query the label embedding information.

    The architecture consists of:
    1. Label embedding layer
    2. Cross-attention layer (input features query label)
    3. Sequence of fully connected layers with activation and dropout
    4. Final layers for VAE mean and log variance

    Attributes
    ----------
    @encoder_latent_dimension : int
        Dimensionality of the latent space.
    @encoder_output_shape : int
        Dimensionality of the input data.
    @encoder_activation_function : str
        Activation function applied to intermediate layers.
    @encoder_last_layer_activation : str
        Activation function applied to the final layer before latent variables.
    @encoder_dropout_decay_rate_encoder : float
        Dropout rate applied to the dense layers for regularization.
    @encoder_dataset_type : type
        Data type for inputs and outputs (default: numpy.float32).
    @encoder_initializer_mean : float
        Mean of the normal distribution used for weight initialization.
    @encoder_initializer_deviation : float
        Standard deviation of the normal distribution used for weight initialization.
    @encoder_number_neurons_encoder : List[int]
        List specifying the number of neurons in each dense layer.
    @encoder_number_samples_per_class : Optional[Dict[str, int]]
        Dictionary containing class metadata (e.g., number of samples per class).
    @encoder_attention_embed_dim : int
        Embedding dimension for cross-attention mechanism.
    @encoder_attention_num_heads : int
        Number of attention heads in cross-attention layer.
    @encoder_model : Optional[nn.Module]
        Placeholder for the actual encoder network (built later).

    Examples
    --------
    >>> encoder = VanillaEncoderTorch(
    ...     latent_dimension=128,
    ...     output_shape=784,
    ...     activation_function='relu',
    ...     initializer_mean=0.0,
    ...     initializer_deviation=0.02,
    ...     dropout_decay_encoder=0.3,
    ...     last_layer_activation='linear',
    ...     number_neurons_encoder=[256, 128, 64],
    ...     dataset_type=numpy.float32,
    ...     number_samples_per_class={"number_classes": 10},
    ...     attention_embed_dim=128,
    ...     attention_num_heads=4
    ... )
    >>> encoder_model = encoder.get_encoder()
    """

    def __init__(self,
                 latent_dimension: int,
                 output_shape: int,
                 activation_function: str,
                 initializer_mean: float,
                 initializer_deviation: float,
                 dropout_decay_encoder: float,
                 last_layer_activation: str,
                 number_neurons_encoder: List[int],
                 dataset_type: Union[numpy.dtype, type] = numpy.float32,
                 number_samples_per_class: Optional[Dict[str, int]] = None,
                 attention_embed_dim: int = 128,
                 attention_num_heads: int = 4) -> None:
        """
        Initializes the VanillaEncoder with cross-attention.

        Parameters
        ----------
        @latent_dimension : int
            Dimensionality of the latent space.
        @output_shape : int
            Dimensionality of the input data.
        @activation_function : str
            Activation function for intermediate layers.
        @initializer_mean : float
            Mean of the normal distribution for weight initialization.
        @initializer_deviation : float
            Standard deviation for weight initialization.
        @dropout_decay_encoder : float
            Dropout rate applied to intermediate layers.
        @last_layer_activation : str
            Activation function for the layer before latent variables.
        @number_neurons_encoder : list of int
            Number of neurons in each fully connected layer.
        @dataset_type : type, optional
            Data type for inputs and outputs (default: numpy.float32).
        @number_samples_per_class : dict, optional
            Dictionary with class information (must contain 'number_classes' key).
        @attention_embed_dim : int, optional
            Embedding dimension for cross-attention (default: 128).
        @attention_num_heads : int, optional
            Number of attention heads (default: 4).
        """
        nn.Module.__init__(self)
        ActivationsTorch.__init__(self)

        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("Latent dimension must be a positive integer.")

        if not isinstance(output_shape, int) or output_shape <= 0:
            raise ValueError("Output shape must be a positive integer.")

        if not isinstance(activation_function, str):
            raise ValueError("Activation function must be a string.")

        if not isinstance(initializer_mean, (int, float)):
            raise ValueError("Initializer mean must be a numerical value.")

        if not isinstance(initializer_deviation, (int, float)) or initializer_deviation <= 0:
            raise ValueError("Initializer deviation must be a positive numerical value.")

        if not (0.0 <= dropout_decay_encoder <= 1.0):
            raise ValueError("Dropout decay rate must be between 0 and 1.")

        if not isinstance(last_layer_activation, str):
            raise ValueError("Last layer activation must be a string.")

        if not isinstance(number_neurons_encoder, list) or not all(
                isinstance(n, int) and n > 0 for n in number_neurons_encoder):
            raise ValueError("Number of neurons per encoder layer must be a list of positive integers.")

        if dataset_type not in [numpy.float16, numpy.float32, numpy.float64]:
            raise ValueError("Datasets type must be one of numpy.float16, numpy.float32, or numpy.float64.")

        if not isinstance(attention_embed_dim, int) or attention_embed_dim <= 0:
            raise ValueError("attention_embed_dim must be a positive integer.")

        if not isinstance(attention_num_heads, int) or attention_num_heads <= 0:
            raise ValueError("attention_num_heads must be a positive integer.")

        if attention_embed_dim % attention_num_heads != 0:
            raise ValueError("attention_embed_dim must be divisible by attention_num_heads.")

        # Initialize instance variables
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
        self._encoder_model = None

        # Sampling layer (standalone, not part of network to avoid circular ref)
        self._sampling_layer = SamplingLayer()

    def Sampling(self):
        """Returns the sampling layer for reparameterization trick."""
        return self._sampling_layer

    def get_encoder(self, input_shape=None):
        """
        Constructs and returns the encoder model with cross-attention.

        The model uses cross-attention to condition the input features on label information,
        where the input features act as Query and the label embedding provides Key and Value.

        Args:
            input_shape: Optional input shape (for API compatibility, not used)

        Returns:
            nn.Module: The constructed encoder model with cross-attention conditioning.

        Raises:
            ValueError: If the number of classes is not specified in number_samples_per_class.
        """
        # Ensure the number of classes is provided in the configuration
        if not self._encoder_number_samples_per_class or "number_classes" not in self._encoder_number_samples_per_class:
            raise ValueError("The number of classes must be specified in 'number_samples_per_class'.")

        # Create encoder network with configuration (no parent reference)
        if self._encoder_model is None:
            self._encoder_model = EncoderNetwork(
                num_classes=self._encoder_number_samples_per_class["number_classes"],
                output_shape=self._encoder_output_shape,
                latent_dimension=self._encoder_latent_dimension,
                number_neurons_encoder=self._encoder_number_neurons_encoder,
                activation_function=self._encoder_activation_function,
                last_layer_activation=self._encoder_last_layer_activation,
                dropout_decay_rate=self._encoder_dropout_decay_rate_encoder,
                initializer_mean=self._encoder_initializer_mean,
                initializer_deviation=self._encoder_initializer_deviation,
                attention_embed_dim=self._encoder_attention_embed_dim,
                attention_num_heads=self._encoder_attention_num_heads
            )

        return self._encoder_model

    @property
    def dropout_decay_rate_encoder(self) -> float:
        """float: Dropout rate for encoder regularization."""
        return self._encoder_dropout_decay_rate_encoder

    @property
    def number_filters_encoder(self) -> List[int]:
        """List[int]: Number of neurons in each encoder layer."""
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
    def dropout_decay_rate_encoder(self, dropout_decay_rate_encoder: float) -> None:
        """
        Set the dropout rate for encoder regularization.

        Args:
            dropout_decay_rate_encoder (float): Dropout rate to set.

        Raises:
            ValueError: If the dropout rate is not between 0 and 1.
        """
        if not (0.0 <= dropout_decay_rate_encoder <= 1.0):
            raise ValueError("Dropout decay rate must be between 0 and 1.")

        self._encoder_dropout_decay_rate_encoder = dropout_decay_rate_encoder