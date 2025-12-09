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
    from typing import List, Dict, Union, Optional
    from Engine.Architectures.QuantizedVAE.Torch.ActivationTorch import ActivationsTorch

except ImportError as error:
    print(error)
    sys.exit(-1)


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
    """Neural network implementation for the encoder."""

    def __init__(self,
                 num_classes,
                 output_shape,
                 latent_dimension,
                 number_neurons_encoder,
                 activation_function,
                 last_layer_activation,
                 dropout_decay_rate,
                 initializer_mean,
                 initializer_deviation):
        super().__init__()

        # Store configuration (not parent reference)
        self.num_classes = num_classes
        self.output_shape = output_shape
        self.latent_dimension = latent_dimension
        self.activation_function = activation_function
        self.last_layer_activation = last_layer_activation

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
        self.label_embedding = nn.Linear(num_classes, 8)
        init_weights(self.label_embedding, initializer_mean, initializer_deviation)

        # First layer after concatenation
        input_size = output_shape + 8
        layers = []

        # First layer
        first_layer = nn.Linear(input_size, number_neurons_encoder[0])
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
        Forward pass through the encoder.

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
        label_embedded = torch.relu(self.label_embedding(label))

        # Concatenate features and embedded labels
        combined = torch.cat([x, label_embedded], dim=1)

        # Pass through encoder layers
        encoded = self.encoder_layers(combined)

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
    VanillaEncoder

    Implements a fully connected conditional variational encoder (CVAE) model.
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
                 number_samples_per_class: Optional[Dict[str, int]] = None) -> None:
        """
        Initializes the VanillaEncoder with user-defined hyperparameters.
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
        self._encoder_model = None

        # Sampling layer (standalone, not part of network to avoid circular ref)
        self._sampling_layer = SamplingLayer()

    def Sampling(self):
        """Returns the sampling layer for reparameterization trick."""
        return self._sampling_layer

    def get_encoder(self, input_shape=None):
        """
        Constructs and returns the encoder model.

        Args:
            input_shape: Optional input shape (for API compatibility, not used)

        Returns:
            nn.Module: The constructed encoder model.

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
                initializer_deviation=self._encoder_initializer_deviation
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