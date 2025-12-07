#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

from Engine.Models.QuantizedVAE.Torch.ActivationTorch import ActivationsTorch

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
    from typing import Any, Dict, Optional

except ImportError as error:
    print(error)
    sys.exit(-1)


class QuantizedVAEVanillaEncoderTorch(ActivationsTorch, nn.Module):
    """
    VanillaEncoder

    A class representing a Vanilla Encoder model for deep learning applications. The encoder
    is designed to process inputs and labels, apply a series of dense layers with activations
    and dropout, and output a latent representation of the input data. This model is typically
    used in applications such as autoencoders, variational autoencoders, or other generative models.

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

    Raises:
        ValueError:
            Raised when the following invalid arguments are passed during initialization:
            - `latent_dimension` is not a positive integer.
            - `initializer_mean` or `initializer_deviation` is not a number.
            - `dropout_decay_encoder` is outside the valid range [0, 1].
            - `number_neurons_encoder` is not a non-empty list or contains non-positive integers.
            - `number_samples_per_class` is provided but is not a dictionary.

    Example:
        >>> encoder = VanillaEncoder(
        ...     latent_dimension=128,
        ...     output_shape=(64, 64, 1),
        ...     activation_function='ReLU',
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_encoder=0.5,
        ...     last_layer_activation='sigmoid',
        ...     number_neurons_encoder=[512, 256, 128],
        ...     dataset_type=numpy.float32,
        ...     number_samples_per_class={"number_classes": 10}
        ... )
    """

    def __init__(self, latent_dimension: int, output_shape: tuple, activation_function: str, initializer_mean: float,
                 initializer_deviation: float, dropout_decay_encoder: float, last_layer_activation: str,
                 number_neurons_encoder: list, dataset_type: Any = numpy.float32,
                 number_samples_per_class: Optional[Dict[str, Any]] = None):
        """
        Initializes the VanillaEncoder with the provided parameters.

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
        """
        ActivationsTorch.__init__(self)
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

    def get_encoder_model(self, input_shape: int):
        """
        Creates and returns the encoder model.

        This method constructs the neural network by stacking dense layers with the provided
        configurations (neurons, dropout, and activation). It also concatenates the input data
        and labels before passing through the layers.

        Args:
            input_shape (tuple): The shape of the input data.

        Returns:
            nn.Module: The encoder model which takes input data and labels and outputs the
                          encoded latent representation and labels.
        """

        if not isinstance(input_shape, int) or input_shape <= 0:
            raise ValueError(f"Invalid value for output_shape: {input_shape}. It must be a positive integer.")

        return EncoderModel(
            input_shape=input_shape,
            latent_dimension=self._encoder_latent_dimension,
            number_classes=self._encoder_number_samples_per_class["number_classes"],
            number_neurons=self._encoder_number_neurons_encoder,
            activation_function=self._encoder_activation_function,
            last_layer_activation=self._encoder_last_layer_activation,
            dropout_rate=self._encoder_dropout_decay_rate_encoder,
            initializer_mean=self._encoder_initializer_mean,
            initializer_deviation=self._encoder_initializer_deviation,
            add_activation_layer=self._add_activation_layer
        )

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


class EncoderModel(nn.Module):
    """PyTorch implementation of the encoder network"""

    def __init__(self, input_shape, latent_dimension, number_classes, number_neurons,
                 activation_function, last_layer_activation, dropout_rate,
                 initializer_mean, initializer_deviation, add_activation_layer):
        super(EncoderModel, self).__init__()

        self.add_activation_layer = add_activation_layer
        self.activation_function = activation_function
        self.last_layer_activation = last_layer_activation

        # Calculate input size after concatenation
        concat_size = input_shape + number_classes

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
        output_layer = nn.Linear(number_neurons[-1], latent_dimension)
        nn.init.normal_(output_layer.weight, mean=initializer_mean, std=initializer_deviation)
        nn.init.zeros_(output_layer.bias)
        self.output_layer = output_layer

    def forward(self, x):
        """
        Forward pass through encoder

        Args:
            x: List/tuple of [input_data, labels]

        Returns:
            List of [latent_representation, labels]
        """
        input_data, labels = x

        # Concatenate input and labels
        x = torch.cat([input_data, labels], dim=1)

        # Pass through layers
        for i, (layer, dropout) in enumerate(zip(self.layers, self.dropouts)):
            x = layer(x)
            x = dropout(x)
            x = self.add_activation_layer(x, self.activation_function)

        # Output layer
        x = self.output_layer(x)
        x = self.add_activation_layer(x, self.last_layer_activation)

        return [x, labels]