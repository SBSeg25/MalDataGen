#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/07'
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
    from typing import Optional, List

    import torch
    import torch.nn as nn

except ImportError as error:
    print(error)
    sys.exit(-1)


class VanillaGeneratorTorch(nn.Module):
    """
    VanillaGeneratorTorch

    Implements a dense generator model for generating synthetic data using a
    vanilla architecture in PyTorch. This class is designed for generating
    synthetic data from a latent space using a fully connected neural network.
    It supports flexible configurations for the generator layers, activations,
    and dropout rates, with the option for conditional generation based on the
    number of samples per class.

    Attributes:
        @latent_dimension (int):
            The dimensionality of the latent space, which serves as the input to the generator.
        @output_shape (int):
            The desired dimension of the generated output data.
        @activation_function (str):
            The activation function used in intermediate layers (e.g., 'relu', 'leaky_relu').
        @initializer_mean (float):
            The mean for the weight initialization.
        @initializer_deviation (float):
            The standard deviation for the weight initialization.
        @dropout_decay_rate_g (float):
            The rate at which the dropout is applied in generator layers, should be between 0.0 and 1.0.
        @last_layer_activation (str):
            The activation function to be applied in the last layer (e.g., 'sigmoid' or 'tanh').
        @dense_layer_sizes_g (List[int]):
            A list of integers representing the number of units in each dense layer of the generator.
        @dataset_type (torch.dtype):
            The data type for the input tensors (default is torch.float32).
        @number_samples_per_class (Optional[dict]):
            An optional dictionary indicating the number of samples per class for conditional data generation.

    Raises:
        ValueError:
            Raised if invalid arguments are passed during initialization, such as:
            - Non-positive `latent_dimension` or `output_shape`
            - `dropout_decay_rate_g` outside the range [0.0, 1.0]
            - Invalid values in `dense_layer_sizes_g`

    Example:
        >>> generator = VanillaGeneratorTorch(
        ...     latent_dimension=100,
        ...     output_shape=784,
        ...     activation_function="relu",
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_rate_g=0.3,
        ...     last_layer_activation="sigmoid",
        ...     dense_layer_sizes_g=[128, 256, 512],
        ...     number_samples_per_class={"number_classes": 10}
        ... )
        >>> model = generator.get_generator()
        >>> output = model([latent_vectors, labels])
    """

    def __init__(self,
                 latent_dimension: int,
                 output_shape: int,
                 activation_function: str,
                 initializer_mean: float,
                 initializer_deviation: float,
                 dropout_decay_rate_g: float,
                 last_layer_activation: str,
                 dense_layer_sizes_g: List[int],
                 dataset_type: torch.dtype = torch.float32,
                 number_samples_per_class: Optional[dict] = None):
        """
        Initializes the VanillaGeneratorTorch class with the specified parameters.

        Args:
            latent_dimension (int): Dimension of the latent space.
            output_shape (int): Dimension of the output data.
            activation_function (str): Activation function for intermediate layers.
            initializer_mean (float): Mean for weight initialization.
            initializer_deviation (float): Standard deviation for weight initialization.
            dropout_decay_rate_g (float): Dropout rate for generator layers.
            last_layer_activation (str): Activation function for the output layer.
            dense_layer_sizes_g (List[int]): List of dense layer sizes.
            dataset_type (torch.dtype): Data type for the input tensors.
            number_samples_per_class (Optional[dict]): Dictionary with the number of samples per class.

        Raises:
            ValueError: If any parameter validation fails.
        """
        super().__init__()

        # Validation
        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")

        if not isinstance(output_shape, int) or output_shape <= 0:
            raise ValueError("output_shape must be a positive integer.")

        if not isinstance(activation_function, str):
            raise ValueError("activation_function must be a string.")

        if not isinstance(initializer_mean, (float, int)):
            raise ValueError("initializer_mean must be a float or an integer.")

        if not isinstance(initializer_deviation, (float, int)) or initializer_deviation <= 0:
            raise ValueError("initializer_deviation must be a positive float or integer.")

        if not isinstance(last_layer_activation, str):
            raise ValueError("last_layer_activation must be a string.")

        if not isinstance(dropout_decay_rate_g, (float, int)) or not (0.0 <= dropout_decay_rate_g <= 1.0):
            raise ValueError("dropout_decay_rate_g must be between 0.0 and 1.0.")

        if not isinstance(dense_layer_sizes_g, list) or not all(
                isinstance(s, int) and s > 0 for s in dense_layer_sizes_g):
            raise ValueError("dense_layer_sizes_g must be a list of positive integers.")

        if number_samples_per_class is not None and not isinstance(number_samples_per_class, dict):
            raise ValueError("number_samples_per_class must be a dictionary if provided.")

        # Store parameters
        self._generator_number_samples_per_class = number_samples_per_class
        self._generator_latent_dimension = latent_dimension
        self._generator_output_shape = output_shape
        self._generator_activation_function = activation_function
        self._generator_last_layer_activation = last_layer_activation
        self._generator_dropout_decay_rate_g = dropout_decay_rate_g
        self._generator_dense_layer_sizes_g = dense_layer_sizes_g
        self._generator_dataset_type = dataset_type
        self._generator_initializer_mean = initializer_mean
        self._generator_initializer_deviation = initializer_deviation
        self._generator_model_dense = None

    def _add_activation_layer(self, activation_name: str) -> nn.Module:
        """
        Returns the appropriate activation function module.

        Args:
            activation_name (str): Name of the activation function.

        Returns:
            nn.Module: PyTorch activation module.
        """
        activation_name = activation_name.lower()

        activation_map = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax(dim=1),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'gelu': nn.GELU(),
        }

        if activation_name in activation_map:
            return activation_map[activation_name]
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")

    def _initialize_weights(self, module):
        """
        Initialize weights using normal distribution with specified mean and std.

        Args:
            module: PyTorch module to initialize.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight,
                            mean=self._generator_initializer_mean,
                            std=self._generator_initializer_deviation)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_generator(self):
        """
        Builds and returns the conditional generator model.

        Returns:
            ConditionalGenerator: A PyTorch module with inputs for latent vectors
                                 and conditional labels, and an output for generated data.

        Raises:
            ValueError: If `number_samples_per_class` is not provided for conditional generation.
        """
        if not self._generator_number_samples_per_class or "number_classes" not in self._generator_number_samples_per_class:
            raise ValueError(
                "`number_samples_per_class` must include a 'number_classes' key for conditional generation.")

        # Build the core dense generator model
        layers = []

        # First layer
        layers.append(nn.Linear(self._generator_latent_dimension,
                                self._generator_dense_layer_sizes_g[0]))
        layers.append(nn.Dropout(self._generator_dropout_decay_rate_g))
        layers.append(self._add_activation_layer(self._generator_activation_function))

        # Hidden layers
        for i in range(len(self._generator_dense_layer_sizes_g) - 1):
            layers.append(nn.Linear(self._generator_dense_layer_sizes_g[i],
                                    self._generator_dense_layer_sizes_g[i + 1]))
            layers.append(nn.Dropout(self._generator_dropout_decay_rate_g))
            layers.append(self._add_activation_layer(self._generator_activation_function))

        # Output layer
        layers.append(nn.Linear(self._generator_dense_layer_sizes_g[-1],
                                self._generator_output_shape))
        layers.append(self._add_activation_layer(self._generator_last_layer_activation))

        self._generator_model_dense = nn.Sequential(*layers)

        # Initialize weights
        self._generator_model_dense.apply(self._initialize_weights)

        # Build the label embedding network
        num_classes = self._generator_number_samples_per_class["number_classes"]
        label_embedding_layers = [
            nn.Linear(self._generator_latent_dimension + num_classes,
                      self._generator_latent_dimension),
            self._add_activation_layer(self._generator_activation_function)
        ]
        label_embedding = nn.Sequential(*label_embedding_layers)
        label_embedding.apply(self._initialize_weights)

        # Return the complete conditional generator
        return ConditionalGenerator(
            generator_core=self._generator_model_dense,
            label_embedding=label_embedding
        )

    def get_dense_generator_model(self) -> Optional[nn.Module]:
        """
        Returns the standalone dense generator model.

        Returns:
            Optional[nn.Module]: A PyTorch module without label conditioning, or None if not built.
        """
        return self._generator_model_dense

    def set_dropout_decay_rate_generator(self, dropout_decay_rate_generator: float) -> None:
        """
        Updates the dropout rate of the generator.

        Args:
            dropout_decay_rate_generator (float): New dropout rate.

        Raises:
            ValueError: If the dropout rate is not between 0.0 and 1.0.
        """
        if not (0.0 <= dropout_decay_rate_generator <= 1.0):
            raise ValueError("`dropout_decay_rate_generator` must be between 0.0 and 1.0.")
        self._generator_dropout_decay_rate_g = dropout_decay_rate_generator

    def set_dense_layer_sizes_generator(self, dense_layer_sizes_generator: List[int]) -> None:
        """
        Updates the dense layer sizes of the generator.

        Args:
            dense_layer_sizes_generator (List[int]): New dense layer sizes.

        Raises:
            ValueError: If the list is empty or contains non-positive integers.
        """
        if not dense_layer_sizes_generator or any(size <= 0 for size in dense_layer_sizes_generator):
            raise ValueError("`dense_layer_sizes_generator` must be a list of positive integers.")
        self._generator_dense_layer_sizes_g = dense_layer_sizes_generator


class ConditionalGenerator(nn.Module):
    """
    Conditional Generator wrapper that combines latent vectors and labels.

    This module takes latent vectors and labels as input, concatenates them,
    processes through label embedding, and then through the generator core
    to produce synthetic data.
    """

    def __init__(self, generator_core: nn.Module, label_embedding: nn.Module):
        """
        Initialize the conditional generator.

        Args:
            generator_core: The main generator network.
            label_embedding: Network that processes concatenated [latent, labels].
        """
        super().__init__()
        self.generator_core = generator_core
        self.label_embedding = label_embedding

    def forward(self, inputs):
        """
        Forward pass through the conditional generator.

        Args:
            inputs: Can be either:
                - A list/tuple of [latent_vector, labels]
                - A single tensor (for non-conditional use)

        Returns:
            torch.Tensor: Generated synthetic data.
        """
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            latent, labels = inputs
            # Concatenate latent vector and labels
            concatenated = torch.cat([latent, labels], dim=1)
            # Process through label embedding
            embedded = self.label_embedding(concatenated)
            # Process through generator core
            generated = self.generator_core(embedded)
        else:
            # Non-conditional case: just pass through generator
            generated = self.generator_core(inputs)

        return generated