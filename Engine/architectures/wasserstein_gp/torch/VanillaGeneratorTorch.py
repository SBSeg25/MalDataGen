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


class VanillaGeneratorTorch(Activations, nn.Module):
    """
    VanillaGenerator

    Implements a fully connected (dense) generator model for use in generative models,
    such as GANs. This generator is designed to work with label conditioning and
    supports customization of activation functions, layer sizes, initialization, and
    other hyperparameters.

    This class now supports both dense and convolutional (Conv1D) architectures through
    the optimizer parameter.

    Attributes:
        @generator_latent_dimension (int):
            Dimensionality of the input latent space.
        @generator_output_shape (int):
            Dimensionality of the generated output data.
        @generator_activation_function (Callable):
            Activation function applied to all hidden layers.
        @generator_last_layer_activation (Callable):
            Activation function applied to the final output layer.
        @generator_dropout_decay_rate_g (float):
            Dropout rate applied to dense layers to improve generalization.
        @generator_dense_layer_sizes_g (List[int]):
            List of integers specifying the number of units in each dense layer.
        @generator_dataset_type (type):
            Data type of the input dataset (default: numpy.float32).
        @generator_initializer_mean (float):
            Mean of the normal distribution used for weight initialization.
        @generator_initializer_deviation (float):
            Standard deviation of the normal distribution used for weight initialization.
        @generator_number_samples_per_class (Optional[Dict[str, int]]):
            Optional dictionary containing metadata about class distribution.
            Must include a key "number_classes" if provided.
        @generator_optimizer (str):
            Type of architecture to use: 'dense' (default) or 'convolutional' for Conv1D.
        @generator_model_dense (Optional[Model]):
            Placeholder for the compiled model after build().

    Raises:
        ValueError:
            Raised if invalid arguments are passed during initialization, such as:
            - Non-positive `latent_dimension` or `output_shape`
            - Dropout rate outside the range [0, 1]
            - Empty or invalid `dense_layer_sizes_g`
            - Missing required key "number_classes" in `number_samples_per_class`, if provided

    References:
        - Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014).
          Generative adversarial Networks. arXiv preprint arXiv:1406.2661.
          Available at: https://arxiv.org/abs/1406.2661

    Example:
        >>> generator = VanillaGeneratorTorch(
        ...     latent_dimension=100,
        ...     output_shape=784,
        ...     activation_function=leaky_relu,
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_rate_g=0.3,
        ...     last_layer_activation=tanh,
        ...     dense_layer_sizes_g=[256, 512, 1024],
        ...     dataset_type=numpy.float32,
        ...     number_samples_per_class={"number_classes": 10},
        ...     optimizer='convolutional'
        ... )
        >>> generator.build()  # Example method call if present
    """

    def __init__(self, latent_dimension: int,
                 output_shape: int,
                 activation_function: callable,
                 initializer_mean: float,
                 initializer_deviation: float,
                 dropout_decay_rate_g: float,
                 last_layer_activation: callable,
                 dense_layer_sizes_g: list[int],
                 dataset_type: type = numpy.float32,
                 number_samples_per_class: dict | None = None,
                 optimizer: str = 'dense'):
        """
        Initializes the VanillaGenerator class with the provided parameters.

        Args:
            @latent_dimension (int):
                Dimensionality of the latent space.
            @output_shape (int):
                Dimensionality of the generated output data.
            @activation_function (Callable):
                Activation function for all hidden layers.
            @initializer_mean (float):
                Mean of the normal distribution used to initialize weights.
            @initializer_deviation (float):
                Standard deviation of the normal distribution used to initialize weights.
            @dropout_decay_rate_g (float):
                Dropout rate applied to dense layers (0 to 1).
            @last_layer_activation (Callable):
                Activation function applied to the final output layer.
            @dense_layer_sizes_g (List[int]):
                List of integers specifying the number of units per dense layer.
            @dataset_type (type, optional):
                Data type of the input data (default: numpy.float32).
            @number_samples_per_class (Optional[Dict[str, int]], optional):
                Optional dictionary containing the number of samples per class. If provided, it must contain the key "number_classes".
            @optimizer (str, optional):
                Type of architecture: 'dense' for fully-connected (default) or 'convolutional' for Conv1D architecture.

        Raises:
            ValueError:
                If `latent_dimension` or `output_shape` is <= 0.
                If `dropout_decay_rate_g` is not within [0, 1].
                If `dense_layer_sizes_g` is empty or contains non-positive values.
                If `number_samples_per_class` is provided but does not contain the key "number_classes".

        """
        nn.Module.__init__(self)
        Activations.__init__(self)

        if latent_dimension <= 0:
            raise ValueError("Latent dimension must be a positive integer.")

        if output_shape <= 0:
            raise ValueError("Output shape must be a positive integer.")

        if initializer_mean < 0:
            raise ValueError("Initializer mean must be non-negative.")

        if initializer_deviation <= 0:
            raise ValueError("Initializer deviation must be positive.")

        if dropout_decay_rate_g < 0 or dropout_decay_rate_g > 1:
            raise ValueError("Dropout decay rate must be in the range [0, 1].")

        if not dense_layer_sizes_g or any(size <= 0 for size in dense_layer_sizes_g):
            raise ValueError("Dense layer sizes must be a list of positive integers.")

        self._generator_latent_dimension = latent_dimension
        self._generator_output_shape = output_shape
        self._generator_activation_function = activation_function
        self._generator_last_layer_activation = last_layer_activation
        self._generator_dropout_decay_rate_g = dropout_decay_rate_g
        self._generator_dense_layer_sizes_g = dense_layer_sizes_g
        self._generator_dataset_type = dataset_type
        self._generator_initializer_mean = initializer_mean
        self._generator_initializer_deviation = initializer_deviation
        self._generator_number_samples_per_class = number_samples_per_class
        self._generator_optimizer = optimizer
        self._generator_model_dense = None

    def get_generator(self):
        """
        Constructs and returns the generator model using either dense or Conv1D layers.

        Returns:
        --------
        nn.Module
            A PyTorch module implementing the generator with latent and label inputs.

        Raises:
        -------
        ValueError
            If number_samples_per_class is not properly defined.
        """
        if not self._generator_number_samples_per_class or "number_classes" not in self._generator_number_samples_per_class:
            raise ValueError("Number of samples per class must include 'number_classes'.")

        # Build generator based on optimizer type
        if self._generator_optimizer == 'convolutional':
            self._generator_model_dense = self._build_convolutional_generator()
        else:
            self._generator_model_dense = self._build_dense_generator()

        # Complete generator with label conditioning
        class Generator(nn.Module):
            def __init__(self, latent_dim, num_classes, dense_model, activation_fn):
                super().__init__()
                self.dense_model = dense_model
                self.activation_fn = activation_fn
                self.label_embedding = nn.Linear(latent_dim + num_classes, latent_dim)

                nn.init.normal_(self.label_embedding.weight, mean=0.0, std=0.02)
                if self.label_embedding.bias is not None:
                    nn.init.zeros_(self.label_embedding.bias)

            def forward(self, inputs):
                latent_input, label_input = inputs
                concatenated = torch.cat([latent_input, label_input], dim=1)
                embedded = self.label_embedding(concatenated)

                if callable(self.activation_fn):
                    embedded = self.activation_fn(embedded)
                else:
                    embedded = self._apply_activation_string(embedded, self.activation_fn)

                output = self.dense_model(embedded)
                return output

            def _apply_activation_string(self, x, activation_name):
                if isinstance(activation_name, str):
                    if activation_name.lower() == 'leakyrelu':
                        return torch.nn.functional.leaky_relu(x)
                    elif activation_name.lower() == 'relu':
                        return torch.nn.functional.relu(x)
                    elif activation_name.lower() == 'tanh':
                        return torch.tanh(x)
                    elif activation_name.lower() == 'sigmoid':
                        return torch.sigmoid(x)
                return x

        generator = Generator(
            self._generator_latent_dimension,
            self._generator_number_samples_per_class["number_classes"],
            self._generator_model_dense,
            self._generator_activation_function
        )

        return generator

    def _build_dense_generator(self) -> nn.Module:
        """
        Build a fully-connected (dense) generator architecture.

        Returns:
            nn.Module: A PyTorch Module with dense layers.
        """

        class DenseGenerator(nn.Module):
            def __init__(self, input_dim, layer_sizes, output_dim, dropout_rate,
                         activation_fn, last_activation_fn, init_mean, init_std):
                super().__init__()
                self.layers = nn.ModuleList()
                self.dropouts = nn.ModuleList()
                self.activation_fn = activation_fn
                self.last_activation_fn = last_activation_fn

                # First layer
                self.layers.append(nn.Linear(input_dim, layer_sizes[0]))
                self.dropouts.append(nn.Dropout(dropout_rate))

                # Hidden layers
                for i in range(1, len(layer_sizes)):
                    self.layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
                    self.dropouts.append(nn.Dropout(dropout_rate))

                # Output layer
                self.output_layer = nn.Linear(layer_sizes[-1], output_dim)

                # Initialize weights
                for layer in self.layers:
                    nn.init.normal_(layer.weight, mean=init_mean, std=init_std)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                nn.init.normal_(self.output_layer.weight, mean=init_mean, std=init_std)
                if self.output_layer.bias is not None:
                    nn.init.zeros_(self.output_layer.bias)

            def forward(self, x):
                for layer, dropout in zip(self.layers, self.dropouts):
                    x = layer(x)
                    x = dropout(x)
                    if callable(self.activation_fn):
                        x = self.activation_fn(x)
                    else:
                        x = self._apply_activation_string(x, self.activation_fn)

                x = self.output_layer(x)
                if callable(self.last_activation_fn):
                    x = self.last_activation_fn(x)
                else:
                    x = self._apply_activation_string(x, self.last_activation_fn)
                return x

            def _apply_activation_string(self, x, activation_name):
                if isinstance(activation_name, str):
                    if activation_name.lower() == 'leakyrelu':
                        return torch.nn.functional.leaky_relu(x)
                    elif activation_name.lower() == 'relu':
                        return torch.nn.functional.relu(x)
                    elif activation_name.lower() == 'tanh':
                        return torch.tanh(x)
                    elif activation_name.lower() == 'sigmoid':
                        return torch.sigmoid(x)
                return x

        return DenseGenerator(
            self._generator_latent_dimension,
            self._generator_dense_layer_sizes_g,
            self._generator_output_shape,
            self._generator_dropout_decay_rate_g,
            self._generator_activation_function,
            self._generator_last_layer_activation,
            self._generator_initializer_mean,
            self._generator_initializer_deviation
        )

    def _build_convolutional_generator(self) -> nn.Module:
        """
        Build a 1D convolutional generator architecture.

        Returns:
            nn.Module: A PyTorch Module with Conv1d layers.
        """

        class ConvolutionalGenerator(nn.Module):
            def __init__(self, input_dim, layer_sizes, output_dim, dropout_rate,
                         activation_fn, last_activation_fn, init_mean, init_std):
                super().__init__()
                self.activation_fn = activation_fn
                self.last_activation_fn = last_activation_fn
                self.output_dim = output_dim

                # Calculate initial size
                initial_size = output_dim // (2 ** len(layer_sizes))
                initial_size = max(4, initial_size)

                # Initial dense layer to project latent to initial size
                self.initial_dense = nn.Linear(input_dim, initial_size * layer_sizes[0])
                nn.init.normal_(self.initial_dense.weight, mean=init_mean, std=init_std)
                if self.initial_dense.bias is not None:
                    nn.init.zeros_(self.initial_dense.bias)

                self.initial_dropout = nn.Dropout(dropout_rate)
                self.initial_size = initial_size
                self.initial_channels = layer_sizes[0]

                # Build convolutional layers with upsampling
                self.upsample_layers = nn.ModuleList()
                self.conv_layers = nn.ModuleList()
                self.dropouts = nn.ModuleList()

                in_channels = layer_sizes[0]
                for i, filters in enumerate(layer_sizes[1:]):
                    self.upsample_layers.append(nn.Upsample(scale_factor=2, mode='linear', align_corners=False))

                    conv = nn.Conv1d(in_channels, filters, kernel_size=3, stride=1, padding=1)
                    nn.init.normal_(conv.weight, mean=init_mean, std=init_std)
                    if conv.bias is not None:
                        nn.init.zeros_(conv.bias)

                    self.conv_layers.append(conv)
                    self.dropouts.append(nn.Dropout(dropout_rate))
                    in_channels = filters

                # Final conv to single channel
                self.final_conv = nn.Conv1d(in_channels, 1, kernel_size=3, stride=1, padding=1)
                nn.init.normal_(self.final_conv.weight, mean=init_mean, std=init_std)
                if self.final_conv.bias is not None:
                    nn.init.zeros_(self.final_conv.bias)

                # Final dense layer to ensure exact output shape
                self.final_dense = nn.Linear(initial_size * (2 ** len(layer_sizes[1:])), output_dim)
                nn.init.normal_(self.final_dense.weight, mean=init_mean, std=init_std)
                if self.final_dense.bias is not None:
                    nn.init.zeros_(self.final_dense.bias)

            def forward(self, x):
                # Project to initial size
                x = self.initial_dense(x)
                if callable(self.activation_fn):
                    x = self.activation_fn(x)
                else:
                    x = self._apply_activation_string(x, self.activation_fn)
                x = self.initial_dropout(x)

                # Reshape to (batch, channels, sequence_length)
                x = x.view(-1, self.initial_channels, self.initial_size)

                # Apply convolutional layers with upsampling
                for upsample, conv, dropout in zip(self.upsample_layers, self.conv_layers, self.dropouts):
                    x = upsample(x)
                    x = conv(x)
                    if callable(self.activation_fn):
                        x = self.activation_fn(x)
                    else:
                        x = self._apply_activation_string(x, self.activation_fn)
                    x = dropout(x)

                # Final conv to single channel
                x = self.final_conv(x)
                if callable(self.activation_fn):
                    x = self.activation_fn(x)
                else:
                    x = self._apply_activation_string(x, self.activation_fn)

                # Flatten
                x = x.view(x.size(0), -1)

                # Final dense layer to match exact output shape
                x = self.final_dense(x)
                if callable(self.last_activation_fn):
                    x = self.last_activation_fn(x)
                else:
                    x = self._apply_activation_string(x, self.last_activation_fn)

                return x

            def _apply_activation_string(self, x, activation_name):
                if isinstance(activation_name, str):
                    if activation_name.lower() == 'leakyrelu':
                        return torch.nn.functional.leaky_relu(x)
                    elif activation_name.lower() == 'relu':
                        return torch.nn.functional.relu(x)
                    elif activation_name.lower() == 'tanh':
                        return torch.tanh(x)
                    elif activation_name.lower() == 'sigmoid':
                        return torch.sigmoid(x)
                return x

        return ConvolutionalGenerator(
            self._generator_latent_dimension,
            self._generator_dense_layer_sizes_g,
            self._generator_output_shape,
            self._generator_dropout_decay_rate_g,
            self._generator_activation_function,
            self._generator_last_layer_activation,
            self._generator_initializer_mean,
            self._generator_initializer_deviation
        )

    @property
    def dense_generator_model(self):
        """Property that retrieves the dense generator submodel without label conditioning."""
        return self._generator_model_dense

    @property
    def dropout_decay_rate_generator(self) -> float:
        """Property to get the dropout decay rate for the generator."""
        return self._generator_dropout_decay_rate_g

    @property
    def dense_layer_sizes_generator(self) -> list[int]:
        """Property to get the dense layer sizes for the generator."""
        return self._generator_dense_layer_sizes_g

    @property
    def optimizer(self) -> str:
        """Property to get the current optimizer/architecture type."""
        return self._generator_optimizer

    @dropout_decay_rate_generator.setter
    def dropout_decay_rate_generator(self, dropout_decay_rate_generator: float):
        """Property to set the dropout decay rate for the generator."""

        if dropout_decay_rate_generator < 0 or dropout_decay_rate_generator > 1:
            raise ValueError("Dropout decay rate must be in the range [0, 1].")

        self._generator_dropout_decay_rate_g = dropout_decay_rate_generator

    @dense_layer_sizes_generator.setter
    def dense_layer_sizes_generator(self, dense_layer_sizes_generator: list[int]):
        """Property to set the dense layer sizes for the generator."""

        if not dense_layer_sizes_generator or any(size <= 0 for size in dense_layer_sizes_generator):
            raise ValueError("Dense layer sizes must be a list of positive integers.")

        self._generator_dense_layer_sizes_g = dense_layer_sizes_generator

    @optimizer.setter
    def optimizer(self, optimizer: str):
        """Property to set the optimizer/architecture type."""
        if optimizer not in ['dense', 'convolutional']:
            raise ValueError("optimizer must be either 'dense' or 'convolutional'.")
        self._generator_optimizer = optimizer