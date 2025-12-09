#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

from Engine.Architectures.QuantizedVAE.Torch.ActivationTorch import ActivationsTorch

# MIT License - Copyright (c) 2025 Synthetic Ocean AI

try:
    import sys
    import numpy
    import torch
    import torch.nn as nn

except ImportError as error:
    print(error)
    sys.exit(-1)


class DecoderNetwork(nn.Module):
    """Neural network implementation for the decoder."""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        # Label embedding layer
        self.label_embedding = nn.Linear(
            parent._decoder_number_samples_per_class["number_classes"],
            8
        )
        parent._init_weights(self.label_embedding, parent._decoder_initializer_mean,
                             parent._decoder_initializer_deviation)

        # First layer after concatenation
        input_size = parent._decoder_latent_dimension + 8
        layers = []

        # First layer
        first_layer = nn.Linear(input_size, parent._decoder_number_neurons_decoder[0])
        parent._init_weights(first_layer, parent._decoder_initializer_mean,
                             parent._decoder_initializer_deviation)
        layers.append(first_layer)
        layers.append(parent._get_activation(parent._decoder_intermediary_activation_function))
        layers.append(nn.Dropout(parent._decoder_dropout_decay_rate_decoder))

        # Hidden layers
        for i in range(1, len(parent._decoder_number_neurons_decoder)):
            layer = nn.Linear(
                parent._decoder_number_neurons_decoder[i - 1],
                parent._decoder_number_neurons_decoder[i]
            )
            parent._init_weights(layer, parent._decoder_initializer_mean,
                                 parent._decoder_initializer_deviation)
            layers.append(layer)
            layers.append(nn.Dropout(parent._decoder_dropout_decay_rate_decoder))
            layers.append(parent._get_activation(parent._decoder_intermediary_activation_function))

        self.decoder_layers = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(
            parent._decoder_number_neurons_decoder[-1],
            parent._decoder_output_shape
        )
        parent._init_weights(self.output_layer, parent._decoder_initializer_mean,
                             parent._decoder_initializer_deviation)

    def forward(self, latent, label):
        """
        Forward pass through the decoder.

        Args:
            latent: Latent vector
            label: One-hot encoded labels

        Returns:
            Reconstructed output
        """
        # Embed labels
        label_embedded = torch.relu(self.label_embedding(label))

        # Concatenate latent and embedded labels
        combined = torch.cat([latent, label_embedded], dim=1)

        # Pass through decoder layers
        decoded = self.decoder_layers(combined)

        # Output layer
        output = self.output_layer(decoded)
        output = self.parent._get_activation(self.parent._decoder_last_layer_activation)(output)

        return output


class VanillaDecoderTorch(nn.Module, ActivationsTorch):
    """
    VanillaDecoder

    This class implements a conditional fully connected decoder network.
    """

    def __init__(self,
                 latent_dimension,
                 output_shape,
                 activation_function,
                 initializer_mean,
                 initializer_deviation,
                 dropout_decay_decoder,
                 last_layer_activation,
                 number_neurons_decoder,
                 dataset_type=numpy.float32,
                 number_samples_per_class=None):
        """
        Initializes the VanillaDecoder with the given hyperparameters.
        """
        nn.Module.__init__(self)
        ActivationsTorch.__init__(self)

        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")

        if not isinstance(output_shape, int) or output_shape <= 0:
            raise ValueError("output_shape must be a positive integer.")

        if not isinstance(activation_function, (str, callable)):
            raise ValueError("activation_function must be a string or a callable function.")

        if not isinstance(last_layer_activation, (str, callable)):
            raise ValueError("last_layer_activation must be a string or a callable function.")

        if not isinstance(initializer_mean, (int, float)):
            raise ValueError("initializer_mean must be a numeric value.")

        if not isinstance(initializer_deviation, (int, float)) or initializer_deviation <= 0:
            raise ValueError("initializer_deviation must be a positive numeric value.")

        if not isinstance(dropout_decay_decoder, (int, float)) or not (0.0 <= dropout_decay_decoder <= 1.0):
            raise ValueError("dropout_decay_decoder must be a float between 0 and 1.")

        if not isinstance(number_neurons_decoder, list) or not all(
                isinstance(n, int) and n > 0 for n in number_neurons_decoder):
            raise ValueError("number_neurons_decoder must be a list of positive integers.")

        if number_samples_per_class is not None:

            if not isinstance(number_samples_per_class, dict):
                raise ValueError("number_samples_per_class must be a dictionary or None.")

            if "number_classes" not in number_samples_per_class or not isinstance(
                    number_samples_per_class["number_classes"], int):
                raise ValueError("number_samples_per_class must contain a key 'number_classes' with an integer value.")

        self._decoder_latent_dimension = latent_dimension
        self._decoder_output_shape = output_shape
        self._decoder_intermediary_activation_function = activation_function
        self._decoder_last_layer_activation = last_layer_activation
        self._decoder_dropout_decay_rate_decoder = dropout_decay_decoder
        self._decoder_dataset_type = dataset_type
        self._decoder_initializer_mean = initializer_mean
        self._decoder_initializer_deviation = initializer_deviation
        self._decoder_number_neurons_decoder = number_neurons_decoder
        self._decoder_number_samples_per_class = number_samples_per_class
        self._decoder_model = None

    def _init_weights(self, layer, mean, std):
        """Initialize layer weights with normal distribution."""
        nn.init.normal_(layer.weight, mean=mean, std=std)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def _get_activation(self, activation_name):
        """Get activation function by name."""
        if callable(activation_name):
            return activation_name

        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'swish': nn.SiLU(),
            'linear': nn.Identity()
        }
        return activations.get(activation_name.lower(), nn.ReLU())

    def get_decoder(self, input_shape=None):
        """
        Builds and returns the decoder model.

        Args:
            input_shape: Optional input shape (for API compatibility, not used)

        Returns:
            nn.Module: The decoder model.
        """
        # FIX: Create and cache the decoder network instance
        if self._decoder_model is None:
            self._decoder_model = DecoderNetwork(self)

        return self._decoder_model

    @property
    def dropout_decay_rate_decoder(self):
        """float: Gets or sets the dropout decay rate for the decoder."""
        return self._decoder_dropout_decay_rate_decoder

    @property
    def number_filters_decoder(self):
        """list[int]: Gets the number of neurons for each layer in the decoder."""
        return self._decoder_number_neurons_decoder

    @dropout_decay_rate_decoder.setter
    def dropout_decay_rate_decoder(self, dropout_decay_rate_discriminator):
        """
        Sets the dropout decay rate for the decoder.

        Args:
            dropout_decay_rate_discriminator (float): New dropout decay rate.
        """
        self._decoder_dropout_decay_rate_decoder = dropout_decay_rate_discriminator