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
    from Engine.Activations.Activations import Activations

except ImportError as error:
    print(error)
    sys.exit(-1)


class QuantizedVAEVanillaDecoderTorch(Activations, nn.Module):
    """
    VanillaDecoder

    A class representing a conditional decoder model with support for customized dense layers,
    activation functions, dropout, and label-conditioned input. The decoder is designed to process
    a latent representation and output the desired shape. This class is typically used in tasks such
    as generative models, autoencoders, and conditional models that generate data from a latent space.

    Attributes:
        @decoder_latent_dimension (int):
            The dimensionality of the latent space input, which the decoder will use to generate outputs.
        @decoder_output_shape (int):
            The dimensionality of the output layer, specifying the shape of the decoder's output.
        @decoder_activation_function (str):
            The activation function applied to each layer of the decoder (e.g., 'ReLU', 'LeakyReLU').
        @decoder_last_layer_activation (str):
            The activation function applied to the final output layer.
        @decoder_dropout_decay_rate_decoder (float):
            The rate of dropout applied during decoding to improve generalization (must be between 0 and 1).
        @decoder_number_neurons_decoder (list):
            A list specifying the number of neurons (or units) in each layer of the decoder network.
        @decoder_dataset_type (dtype):
            The data type of the input dataset, default is numpy.float32.
        @decoder_initializer_mean (float):
            The mean for the normal distribution used to initialize the weights.
        @decoder_initializer_deviation (float):
            The standard deviation for the normal distribution used to initialize the weights.
        @decoder_number_samples_per_class (Optional[dict]):
            An optional dictionary containing metadata about the number of classes for label input.

    Raises:
        ValueError:
            Raised when the following invalid arguments are passed during initialization:
            - `latent_dimension` or `output_shape` are not positive integers.
            - `activation_function`, `last_layer_activation` are not strings.
            - `initializer_mean` or `initializer_deviation` are not numbers.
            - `dropout_decay_decoder` is outside the valid range [0, 1].
            - `number_neurons_decoder` is not a list of positive integers.
            - `dataset_type` is not a valid type.
            - `number_samples_per_class` is provided but is not a dictionary containing 'number_classes'.

    Example:
        >>> decoder = VanillaDecoder(
        ...     latent_dimension=128,
        ...     output_shape=64,
        ...     activation_function='ReLU',
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_decoder=0.5,
        ...     last_layer_activation='sigmoid',
        ...     number_neurons_decoder=[512, 256, 128],
        ...     dataset_type=numpy.float32,
        ...     number_samples_per_class={"number_classes": 10}
        ... )
    """

    def __init__(self, latent_dimension: int, output_shape: int, activation_function: str, initializer_mean: float,
                 initializer_deviation: float, dropout_decay_decoder: float, last_layer_activation: str,
                 number_neurons_decoder: list[int], dataset_type: type = numpy.float32,
                 number_samples_per_class: dict = None):
        """
        Initializes the VanillaDecoder class with the given configuration.

        Args:
            latent_dimension (int): Dimensionality of the latent space input.
            output_shape (int): Dimensionality of the output layer.
            activation_function (str): Activation function name.
            initializer_mean (float): Mean for the initializer.
            initializer_deviation (float): Standard deviation for the initializer.
            dropout_decay_decoder (float): Dropout rate for decoder layers.
            last_layer_activation (str): Activation function for the output layer.
            number_neurons_decoder (list[int]): Number of neurons in decoder layers.
            dataset_type (type): Data type for inputs/outputs (default is numpy.float32).
            number_samples_per_class (dict, optional): Number of classes for label input.

        Raises:
            ValueError: If any of the provided parameters are invalid.
        """
        Activations.__init__(self)
        nn.Module.__init__(self)

        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError(f"Invalid value for latent_dimension: {latent_dimension}. It must be a positive integer.")

        if not isinstance(output_shape, int) or output_shape <= 0:
            raise ValueError(f"Invalid value for output_shape: {output_shape}. It must be a positive integer.")

        if not isinstance(activation_function, str):
            raise ValueError(f"Invalid value for activation_function: {activation_function}. It must be a string.")

        if not isinstance(initializer_mean, (int, float)):
            raise ValueError(f"Invalid value for initializer_mean: {initializer_mean}. It must be a number.")

        if not isinstance(initializer_deviation, (int, float)):
            raise ValueError(f"Invalid value for initializer_deviation: {initializer_deviation}. It must be a number.")

        if not isinstance(dropout_decay_decoder, (int, float)) or not (0 <= dropout_decay_decoder <= 1):
            raise ValueError(
                f"Invalid value for dropout_decay_decoder: {dropout_decay_decoder}. It must be a number between 0 and 1.")

        if not isinstance(last_layer_activation, str):
            raise ValueError(f"Invalid value for last_layer_activation: {last_layer_activation}. It must be a string.")

        if not isinstance(number_neurons_decoder, list) or not all(
                isinstance(x, int) and x > 0 for x in number_neurons_decoder):
            raise ValueError(
                f"Invalid value for number_neurons_decoder: {number_neurons_decoder}. It must be a list of positive integers.")

        if not isinstance(dataset_type, type):
            raise ValueError(f"Invalid value for dataset_type: {dataset_type}. It must be a valid type.")

        if number_samples_per_class is not None and (
                not isinstance(number_samples_per_class, dict) or "number_classes" not in number_samples_per_class):
            raise ValueError(
                f"Invalid value for number_samples_per_class: {number_samples_per_class}. It must be a dictionary with 'number_classes'.")

        self._decoder_latent_dimension = latent_dimension
        self._decoder_output_shape = output_shape
        self._decoder_activation_function = activation_function
        self._decoder_last_layer_activation = last_layer_activation
        self._decoder_dropout_decay_rate_decoder = dropout_decay_decoder
        self._decoder_dataset_type = dataset_type
        self._decoder_initializer_mean = initializer_mean
        self._decoder_initializer_deviation = initializer_deviation
        self._decoder_number_neurons_decoder = number_neurons_decoder
        self._decoder_number_samples_per_class = number_samples_per_class

    def get_decoder_model(self, output_shape: int):
        """
        Constructs and returns the decoder model.

        Args:
            output_shape (int): The output dimensionality of the decoder.

        Returns:
            nn.Module: The constructed decoder model.

        Raises:
            ValueError: If the output shape is invalid.
        """
        if not isinstance(output_shape, int) or output_shape <= 0:
            raise ValueError(f"Invalid value for output_shape: {output_shape}. It must be a positive integer.")

        return DecoderModel(
            latent_dimension=self._decoder_latent_dimension,
            output_shape=output_shape,
            number_classes=self._decoder_number_samples_per_class["number_classes"],
            number_neurons=self._decoder_number_neurons_decoder,
            activation_function=self._decoder_activation_function,
            last_layer_activation=self._decoder_last_layer_activation,
            dropout_rate=self._decoder_dropout_decay_rate_decoder,
            initializer_mean=self._decoder_initializer_mean,
            initializer_deviation=self._decoder_initializer_deviation,
            add_activation_layer=self._add_activation_layer
        )

    @property
    def dropout_decay_rate_decoder(self) -> float:
        """float: Gets or sets the dropout decay rate for decoder layers."""
        return self._decoder_dropout_decay_rate_decoder

    @property
    def number_filters_decoder(self) -> list[int]:
        """list[int]: Gets the number of neurons in decoder layers."""
        return self._decoder_number_neurons_decoder

    @dropout_decay_rate_decoder.setter
    def dropout_decay_rate_decoder(self, dropout_decay_rate_discriminator: float) -> None:
        """
        Sets the dropout decay rate for the decoder layers.

        Args:
            dropout_decay_rate_discriminator (float): The dropout rate for the decoder layers (between 0 and 1).

        Raises:
            ValueError: If the dropout rate is not a valid number between 0 and 1.
        """
        if not isinstance(dropout_decay_rate_discriminator, (int, float)) or not (
                0 <= dropout_decay_rate_discriminator <= 1):
            raise ValueError(
                f"Invalid value for dropout_decay_rate_discriminator: {dropout_decay_rate_discriminator}. It must be a number between 0 and 1.")
        self._decoder_dropout_decay_rate_decoder = dropout_decay_rate_discriminator


class DecoderModel(nn.Module):
    """PyTorch implementation of the decoder network"""

    def __init__(self, latent_dimension, output_shape, number_classes, number_neurons,
                 activation_function, last_layer_activation, dropout_rate,
                 initializer_mean, initializer_deviation, add_activation_layer):
        super(DecoderModel, self).__init__()

        self.add_activation_layer = add_activation_layer
        self.activation_function = activation_function
        self.last_layer_activation = last_layer_activation

        # Calculate input size after concatenation
        concat_size = latent_dimension + number_classes

        # Build layers
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # First layer
        first_layer = nn.Linear(concat_size, number_neurons[0])
        nn.init.normal_(first_layer.weight, mean=initializer_mean, std=initializer_deviation)
        nn.init.zeros_(first_layer.bias)
        self.layers.append(first_layer)
        self.dropouts.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for i in range(len(number_neurons) - 1):
            layer = nn.Linear(number_neurons[i], number_neurons[i + 1])
            nn.init.normal_(layer.weight, mean=initializer_mean, std=initializer_deviation)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)
            self.dropouts.append(nn.Dropout(dropout_rate))

        # Output layer
        output_layer = nn.Linear(number_neurons[-1], output_shape)
        nn.init.normal_(output_layer.weight, mean=initializer_mean, std=initializer_deviation)
        nn.init.zeros_(output_layer.bias)
        self.output_layer = output_layer

    def forward(self, x):
        """
        Forward pass through decoder

        Args:
            x: List/tuple of [latent_vector, labels]

        Returns:
            Reconstructed output
        """
        latent_vector, labels = x

        # Concatenate latent vector and labels
        x = torch.cat([latent_vector, labels], dim=1)

        # Pass through layers
        for layer, dropout in zip(self.layers, self.dropouts):
            x = layer(x)
            x = dropout(x)
            x = self.add_activation_layer(x, self.activation_function)

        # Output layer
        x = self.output_layer(x)
        x = self.add_activation_layer(x, self.last_layer_activation)

        return x