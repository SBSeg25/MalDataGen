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

    from Engine.activations.Activations import Activations

except ImportError as error:
    print(error)
    sys.exit(-1)


class VanillaEncoderTorch(Activations, nn.Module):
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
        @encoder_optimizer (str):
            Type of architecture to use: 'dense' (default) or 'convolutional' for Conv1D.

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
        ...     number_samples_per_class={"number_classes": 10},
        ...     optimizer='convolutional'
        ... )
    """

    def __init__(self, latent_dimension: int, output_shape: tuple, activation_function: str, initializer_mean: float,
                 initializer_deviation: float, dropout_decay_encoder: float, last_layer_activation: str,
                 number_neurons_encoder: list, dataset_type: Any = numpy.float32,
                 number_samples_per_class: Optional[Dict[str, Any]] = None, optimizer: str = 'dense'):
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
            optimizer (str, optional): Type of architecture: 'dense' for fully-connected (default) or
                'convolutional' for Conv1D architecture.
        """
        # Initialize activations first (if it's a class with __init__)
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
        self._encoder_optimizer = optimizer

    def get_encoder(self, input_shape: int) -> nn.Module:
        """
        Creates and returns the encoder model.

        This method constructs the neural network by stacking dense layers with the provided
        configurations (neurons, dropout, and activation). It also concatenates the input data
        and labels before passing through the layers.

        Args:
            input_shape (int): The shape of the input data.

        Returns:
            nn.Module: The encoder model which takes input data and labels and outputs the
                       encoded latent representation and labels.
        """
        # Choose architecture based on optimizer
        if self._encoder_optimizer == 'convolutional':
            return self._build_convolutional_encoder(input_shape)
        else:
            return self._build_dense_encoder(input_shape)

    def _build_dense_encoder(self, input_shape: int) -> nn.Module:
        """
        Builds a fully-connected (dense) encoder architecture.

        Args:
            input_shape (int): The shape of the input data.

        Returns:
            nn.Module: The constructed dense encoder model.
        """
        class DenseEncoderModule(nn.Module):
            def __init__(self, latent_dim, num_classes, input_dim, number_neurons,
                         init_mean, init_std, dropout_rate, activation_fn, last_activation_fn,
                         get_activation_func):
                super().__init__()

                self.latent_dim = latent_dim
                self.num_classes = num_classes
                self.get_activation_func = get_activation_func

                # Build layers
                self.layers = nn.ModuleList()
                self.dropouts = nn.ModuleList()
                self.activations = []

                # First layer
                layer = nn.Linear(input_dim, number_neurons[0])
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
                Forward pass through the encoder.

                Args:
                    x: List/tuple of [data_input, label_input]

                Returns:
                    Tuple of (latent_representation, label_input)
                """
                data_input, label_input = x

                # Concatenate data and labels
                x = torch.cat([data_input, label_input], dim=1)

                # Pass through layers
                for layer, dropout, activation_name in zip(self.layers, self.dropouts, self.activations):
                    x = layer(x)
                    x = dropout(x)
                    x = self.get_activation_func(activation_name)(x)

                # Latent layer
                x = self.latent_layer(x)
                x = self.get_activation_func(self.latent_activation)(x)

                return x, label_input

        num_classes = self._encoder_number_samples_per_class["number_classes"]
        input_dim = input_shape + num_classes

        return DenseEncoderModule(
            latent_dim=self._encoder_latent_dimension,
            num_classes=num_classes,
            input_dim=input_dim,
            number_neurons=self._encoder_number_neurons_encoder,
            init_mean=self._encoder_initializer_mean,
            init_std=self._encoder_initializer_deviation,
            dropout_rate=self._encoder_dropout_decay_rate_encoder,
            activation_fn=self._encoder_activation_function,
            last_activation_fn=self._encoder_last_layer_activation,
            get_activation_func=self._get_activation
        )

    def _build_convolutional_encoder(self, input_shape: int) -> nn.Module:
        """
        Builds a 1D convolutional encoder architecture with downsampling.

        Args:
            input_shape (int): The shape of the input data.

        Returns:
            nn.Module: The constructed convolutional encoder model.
        """
        class ConvEncoderModule(nn.Module):
            def __init__(self, latent_dim, num_classes, input_dim, number_neurons,
                         init_mean, init_std, dropout_rate, activation_fn, last_activation_fn,
                         get_activation_func):
                super().__init__()

                self.latent_dim = latent_dim
                self.num_classes = num_classes
                self.input_dim = input_dim
                self.get_activation_func = get_activation_func

                # Initial dense layer to expand to spatial dimension
                initial_spatial_dim = input_dim
                self.initial_dense = nn.Linear(input_dim, initial_spatial_dim)
                self._init_weights(self.initial_dense, init_mean, init_std)
                self.initial_activation_name = activation_fn

                self.initial_spatial_dim = initial_spatial_dim

                # Build convolutional layers
                self.conv_layers = nn.ModuleList()
                self.dropouts = nn.ModuleList()
                self.activations = []

                in_channels = 1
                for i, filters in enumerate(number_neurons):
                    spatial_size = initial_spatial_dim // (2 ** i)
                    kernel_size = min(5, max(3, spatial_size // 4))

                    conv = nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=filters,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2
                    )
                    self._init_weights(conv, init_mean, init_std)
                    self.conv_layers.append(conv)
                    self.dropouts.append(nn.Dropout(dropout_rate))
                    self.activations.append(activation_fn)

                    in_channels = filters

                # Flatten and final dense layers
                self.flatten_dense = nn.LazyLinear(number_neurons[-1])
                self.flatten_activation_name = activation_fn

                # Latent layer
                self.latent_layer = nn.Linear(number_neurons[-1], latent_dim)
                self._init_weights(self.latent_layer, init_mean, init_std)
                self.latent_activation_name = last_activation_fn

            def _init_weights(self, layer, mean, std):
                """Initialize layer weights with normal distribution."""
                if isinstance(layer, (nn.Linear, nn.Conv1d)):
                    nn.init.normal_(layer.weight, mean=mean, std=std)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

            def forward(self, x):
                """
                Forward pass through the convolutional encoder.

                Args:
                    x: List/tuple of [data_input, label_input]

                Returns:
                    Tuple of (latent_representation, label_input)
                """
                data_input, label_input = x

                # Concatenate data and labels
                x = torch.cat([data_input, label_input], dim=1)

                # Initial dense expansion
                x = self.initial_dense(x)
                x = self.get_activation_func(self.initial_activation_name)(x)

                # Reshape to (batch, channels, length)
                x = x.unsqueeze(1)  # Add channel dimension

                # Conv layers with downsampling
                for i, (conv, dropout, activation_name) in enumerate(
                        zip(self.conv_layers, self.dropouts, self.activations)):
                    x = conv(x)
                    x = self.get_activation_func(activation_name)(x)
                    x = dropout(x)

                    # Downsample (except for last layer)
                    if i < len(self.conv_layers) - 1:
                        x = nn.functional.avg_pool1d(x, kernel_size=2, stride=2)

                # Flatten
                x = x.view(x.size(0), -1)

                # Final dense layer
                x = self.flatten_dense(x)
                x = self.get_activation_func(self.flatten_activation_name)(x)

                # Latent layer
                x = self.latent_layer(x)
                x = self.get_activation_func(self.latent_activation_name)(x)

                return x, label_input

        num_classes = self._encoder_number_samples_per_class["number_classes"]
        input_dim = input_shape + num_classes

        return ConvEncoderModule(
            latent_dim=self._encoder_latent_dimension,
            num_classes=num_classes,
            input_dim=input_dim,
            number_neurons=self._encoder_number_neurons_encoder,
            init_mean=self._encoder_initializer_mean,
            init_std=self._encoder_initializer_deviation,
            dropout_rate=self._encoder_dropout_decay_rate_encoder,
            activation_fn=self._encoder_activation_function,
            last_activation_fn=self._encoder_last_layer_activation,
            get_activation_func=self._get_activation
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
            # Try to use the activations parent class method if available
            try:
                return self._add_activation_layer(None, activation_name)
            except:
                raise ValueError(f"Unsupported activation function: {activation_name}")

    def get_optimizer(self) -> str:
        """
        Get the current optimizer/architecture type.

        Returns:
            str: The optimizer type ('dense' or 'convolutional').
        """
        return self._encoder_optimizer

    def set_optimizer(self, optimizer: str):
        """
        Set the optimizer/architecture type.

        Args:
            optimizer (str): The optimizer type ('dense' or 'convolutional').
        """
        self._encoder_optimizer = optimizer

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