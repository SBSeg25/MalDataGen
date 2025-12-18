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

    from Engine.activations.Activations import Activations

except ImportError as error:
    print(error)
    sys.exit(-1)


class VanillaDecoderTorch(Activations, nn.Module):
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
        @decoder_optimizer (str):
            Type of architecture to use: 'dense' (default) or 'convolutional' for Conv1D.

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
        ...     number_samples_per_class={"number_classes": 10},
        ...     optimizer='convolutional'
        ... )
    """

    def __init__(self, latent_dimension: int, output_shape: int, activation_function: str, initializer_mean: float,
                 initializer_deviation: float, dropout_decay_decoder: float, last_layer_activation: str,
                 number_neurons_decoder: list[int], dataset_type: type = numpy.float32,
                 number_samples_per_class: dict = None, optimizer: str = 'dense'):
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
            optimizer (str, optional): Type of architecture: 'dense' for fully-connected (default) or
                'convolutional' for Conv1D architecture.

        Raises:
            ValueError: If any of the provided parameters are invalid.
        """
        # Initialize activations first (if it's a class with __init__)
        try:
            Activations.__init__(self)
        except:
            pass

        # Initialize nn.Module
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
        self._decoder_optimizer = optimizer

    def get_decoder(self, output_shape: int):
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

        # Choose architecture based on optimizer
        if self._decoder_optimizer == 'convolutional':
            return self._build_convolutional_decoder(output_shape)
        else:
            return self._build_dense_decoder(output_shape)

    def _build_dense_decoder(self, output_shape: int):
        """
        Builds a fully-connected (dense) decoder architecture.

        Args:
            output_shape (int): The output dimensionality of the decoder.

        Returns:
            nn.Module: The constructed dense decoder model.
        """
        class DenseDecoderModule(nn.Module):
            def __init__(self, latent_dim, num_classes, out_shape, number_neurons,
                         init_mean, init_std, dropout_rate, activation_fn, last_activation_fn,
                         get_activation_func):
                super().__init__()

                self.latent_dim = latent_dim
                self.num_classes = num_classes
                self.out_shape = out_shape
                self.get_activation_func = get_activation_func

                # Calculate input dimension (latent + labels)
                input_dim = latent_dim + num_classes

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

                # Output layer
                last_neurons = number_neurons[-1]
                self.output_layer = nn.Linear(last_neurons, out_shape)
                self._init_weights(self.output_layer, init_mean, init_std)
                self.output_activation = last_activation_fn

            def _init_weights(self, layer, mean, std):
                """Initialize layer weights with normal distribution."""
                nn.init.normal_(layer.weight, mean=mean, std=std)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

            def forward(self, x):
                if isinstance(x, (list, tuple)):
                    latent_input, label_input = x
                    x = torch.cat([latent_input, label_input], dim=1)

                for layer, dropout, activation_name in zip(self.layers, self.dropouts, self.activations):
                    x = layer(x)
                    x = dropout(x)
                    x = self.get_activation_func(activation_name)(x)

                x = self.output_layer(x)
                x = self.get_activation_func(self.output_activation)(x)

                return x

        num_classes = self._decoder_number_samples_per_class["number_classes"]

        return DenseDecoderModule(
            latent_dim=self._decoder_latent_dimension,
            num_classes=num_classes,
            out_shape=output_shape,
            number_neurons=self._decoder_number_neurons_decoder,
            init_mean=self._decoder_initializer_mean,
            init_std=self._decoder_initializer_deviation,
            dropout_rate=self._decoder_dropout_decay_rate_decoder,
            activation_fn=self._decoder_activation_function,
            last_activation_fn=self._decoder_last_layer_activation,
            get_activation_func=self._get_activation
        )

    def _build_convolutional_decoder(self, output_shape: int):
        """
        Builds a 1D convolutional decoder architecture with upsampling.

        Args:
            output_shape (int): The output dimensionality of the decoder.

        Returns:
            nn.Module: The constructed convolutional decoder model.
        """
        class ConvDecoderModule(nn.Module):
            def __init__(self, latent_dim, num_classes, out_shape, number_neurons,
                         init_mean, init_std, dropout_rate, activation_fn, last_activation_fn,
                         get_activation_func):
                super().__init__()

                self.latent_dim = latent_dim
                self.num_classes = num_classes
                self.out_shape = out_shape
                self.get_activation_func = get_activation_func

                # Calculate input dimension (latent + labels)
                input_dim = latent_dim + num_classes

                # Calculate initial spatial dimension
                initial_spatial_dim = out_shape // (2 ** len(number_neurons))
                initial_spatial_dim = max(4, initial_spatial_dim)

                # Initial dense layer to expand
                expanded_size = initial_spatial_dim * number_neurons[0]
                self.initial_dense = nn.Linear(input_dim, expanded_size)
                self._init_weights(self.initial_dense, init_mean, init_std)
                self.initial_activation_name = activation_fn

                self.initial_spatial_dim = initial_spatial_dim
                self.initial_channels = number_neurons[0]

                # Build convolutional layers
                self.conv_layers = nn.ModuleList()
                self.dropouts = nn.ModuleList()
                self.activations = []

                in_channels = number_neurons[0]
                for i, filters in enumerate(number_neurons):
                    kernel_size = min(5, initial_spatial_dim * (2 ** i))
                    kernel_size = max(3, kernel_size)

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

                # Final conv layer
                self.final_conv = nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=1,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
                self._init_weights(self.final_conv, init_mean, init_std)
                self.final_activation_name = last_activation_fn

                # Final dense layer to ensure exact output shape
                self.final_dense = nn.LazyLinear(out_shape)
                self.output_activation_name = last_activation_fn

            def _init_weights(self, layer, mean, std):
                """Initialize layer weights with normal distribution."""
                if isinstance(layer, (nn.Linear, nn.Conv1d)):
                    nn.init.normal_(layer.weight, mean=mean, std=std)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

            def forward(self, x):
                if isinstance(x, (list, tuple)):
                    latent_input, label_input = x
                    x = torch.cat([latent_input, label_input], dim=1)

                # Initial dense expansion
                x = self.initial_dense(x)
                x = self.get_activation_func(self.initial_activation_name)(x)

                # Reshape to (batch, channels, length)
                x = x.view(x.size(0), self.initial_channels, self.initial_spatial_dim)

                # Conv layers with upsampling
                for i, (conv, dropout, activation_name) in enumerate(
                        zip(self.conv_layers, self.dropouts, self.activations)):
                    x = conv(x)
                    x = self.get_activation_func(activation_name)(x)
                    x = dropout(x)

                    # Upsample (except for last layer)
                    if i < len(self.conv_layers) - 1:
                        x = nn.functional.interpolate(x, scale_factor=2, mode='linear', align_corners=False)

                # Final conv
                x = self.final_conv(x)
                x = self.get_activation_func(self.final_activation_name)(x)

                # Flatten
                x = x.view(x.size(0), -1)

                # Final dense to exact output shape
                x = self.final_dense(x)
                x = self.get_activation_func(self.output_activation_name)(x)

                return x

        num_classes = self._decoder_number_samples_per_class["number_classes"]

        return ConvDecoderModule(
            latent_dim=self._decoder_latent_dimension,
            num_classes=num_classes,
            out_shape=output_shape,
            number_neurons=self._decoder_number_neurons_decoder,
            init_mean=self._decoder_initializer_mean,
            init_std=self._decoder_initializer_deviation,
            dropout_rate=self._decoder_dropout_decay_rate_decoder,
            activation_fn=self._decoder_activation_function,
            last_activation_fn=self._decoder_last_layer_activation,
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
        return self._decoder_optimizer

    def set_optimizer(self, optimizer: str):
        """
        Set the optimizer/architecture type.

        Args:
            optimizer (str): The optimizer type ('dense' or 'convolutional').
        """
        self._decoder_optimizer = optimizer

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