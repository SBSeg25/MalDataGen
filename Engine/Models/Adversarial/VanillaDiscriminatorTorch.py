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
    from typing import List, Dict, Optional

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

except ImportError as error:
    print(error)
    sys.exit(-1)


class VanillaDiscriminatorTorch(nn.Module):
    """
    VanillaDiscriminatorTorch

    Implements a fully-connected (dense) discriminator network for use in generative models,
    such as Generative Adversarial Networks (GANs). This PyTorch implementation provides
    flexibility in the design of the architecture, including customizable latent dimensions,
    output shapes, activation functions, dropout rates, and layer sizes.

    This class focuses on defining the model architecture and does not directly handle training
    or loss computation.

    Attributes:
        @discriminator_latent_dimension (int):
            Dimensionality of the input latent space for the discriminator network.
        @discriminator_output_shape (int):
            The output shape of the network, typically used to define the shape of input data.
        @discriminator_activation_function (str):
            The activation function applied to all hidden layers (e.g., 'relu', 'leaky_relu').
        @discriminator_last_layer_activation (str):
            The activation function applied to the last layer (e.g., 'sigmoid').
        @discriminator_dropout_decay_rate_d (float):
            Dropout rate applied to layers in the network to help prevent overfitting.
        @discriminator_dense_layer_sizes_d (List[int]):
            List of integers defining the number of units in each dense layer.
        @discriminator_dataset_type (torch.dtype):
            The data type of the dataset (default: torch.float32).
        @discriminator_initializer_mean (float):
            Mean of the normal distribution used for weight initialization.
        @discriminator_initializer_deviation (float):
            Standard deviation of the normal distribution used for weight initialization.
        @discriminator_number_samples_per_class (Optional[Dict[str, int]]):
            Optional dictionary containing the number of samples per class.

    Raises:
        ValueError:
            Raised if invalid arguments are passed during initialization, such as:
            - Non-positive `latent_dimension`
            - Dropout rate outside the range [0, 1]
            - Empty or invalid `dense_layer_sizes_d`
            - Missing required key "number_classes" in `number_samples_per_class`, if provided

    Example:
        >>> discriminator = VanillaDiscriminatorTorch(
        ...     latent_dimension=100,
        ...     output_shape=784,
        ...     activation_function='leaky_relu',
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_rate_d=0.3,
        ...     last_layer_activation='sigmoid',
        ...     dense_layer_sizes_d=[512, 256, 128],
        ...     dataset_type=torch.float32,
        ...     number_samples_per_class={"number_classes": 10}
        ... )
        >>> model = discriminator.get_discriminator()
        >>> output = model([data, labels])
    """

    def __init__(self,
                 latent_dimension: int,
                 output_shape: int,
                 activation_function: str,
                 initializer_mean: float,
                 initializer_deviation: float,
                 dropout_decay_rate_d: float,
                 last_layer_activation: str,
                 dense_layer_sizes_d: List[int],
                 dataset_type: torch.dtype = torch.float32,
                 number_samples_per_class: Optional[Dict[str, int]] = None):
        """
        Initializes the VanillaDiscriminatorTorch class with the provided parameters.

        Args:
            latent_dimension (int):
                The dimensionality of the input latent space.
            output_shape (int):
                The shape of the expected output data (e.g., flattened image size).
            activation_function (str):
                The activation function to apply to all hidden layers.
            initializer_mean (float):
                The mean for weight initialization.
            initializer_deviation (float):
                The standard deviation for weight initialization.
            dropout_decay_rate_d (float):
                Dropout rate for dense layers (should be between 0 and 1).
            last_layer_activation (str):
                The activation function for the last output layer.
            dense_layer_sizes_d (List[int]):
                A list of integers specifying the number of units in each dense layer.
            dataset_type (torch.dtype, optional):
                The data type of the input data (default is torch.float32).
            number_samples_per_class (Optional[Dict[str, int]], optional):
                A dictionary containing metadata about class distribution.

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

        if not isinstance(dropout_decay_rate_d, (float, int)) or not (0 <= dropout_decay_rate_d <= 1):
            raise ValueError("dropout_decay_rate_d must be a float between 0 and 1.")

        if not isinstance(last_layer_activation, str):
            raise ValueError("last_layer_activation must be a string.")

        if not isinstance(dense_layer_sizes_d, list) or not all(
                isinstance(n, int) and n > 0 for n in dense_layer_sizes_d):
            raise ValueError("dense_layer_sizes_d must be a list of positive integers.")

        if number_samples_per_class is not None and not isinstance(number_samples_per_class, dict):
            raise ValueError("number_samples_per_class must be a dictionary if provided.")

        # Store parameters
        self._discriminator_number_samples_per_class = number_samples_per_class
        self._discriminator_latent_dimension = latent_dimension
        self._discriminator_output_shape = output_shape
        self._discriminator_activation_function = activation_function
        self._discriminator_last_layer_activation = last_layer_activation
        self._discriminator_dropout_decay_rate_d = dropout_decay_rate_d
        self._discriminator_dense_layer_sizes_d = dense_layer_sizes_d
        self._discriminator_dataset_type = dataset_type
        self._discriminator_initializer_mean = initializer_mean
        self._discriminator_initializer_deviation = initializer_deviation
        self._discriminator_model_dense = None

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
                            mean=self._discriminator_initializer_mean,
                            std=self._discriminator_initializer_deviation)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_discriminator(self):
        """
        Build and return the complete discriminator model as a conditional discriminator.

        This method constructs a neural network model using dense layers with dropout
        and activation functions. The model accepts both data and labels as input.

        Returns:
            ConditionalDiscriminator: A PyTorch Module representing the conditional discriminator.
        """
        # Build the core discriminator network (processes combined input)
        layers = []

        # First layer
        layers.append(nn.Linear(self._discriminator_output_shape,
                                self._discriminator_dense_layer_sizes_d[0]))
        layers.append(nn.Dropout(self._discriminator_dropout_decay_rate_d))
        layers.append(self._add_activation_layer(self._discriminator_activation_function))

        # Hidden layers
        for i in range(len(self._discriminator_dense_layer_sizes_d) - 1):
            layers.append(nn.Linear(self._discriminator_dense_layer_sizes_d[i],
                                    self._discriminator_dense_layer_sizes_d[i + 1]))
            layers.append(nn.Dropout(self._discriminator_dropout_decay_rate_d))
            layers.append(self._add_activation_layer(self._discriminator_activation_function))

        # Output layer
        layers.append(nn.Linear(self._discriminator_dense_layer_sizes_d[-1], 1))
        layers.append(self._add_activation_layer(self._discriminator_last_layer_activation))

        self._discriminator_model_dense = nn.Sequential(*layers)

        # Initialize weights
        self._discriminator_model_dense.apply(self._initialize_weights)

        # Build the label embedding network
        num_classes = self._discriminator_number_samples_per_class["number_classes"]
        label_embedding_layers = [
            nn.Linear(self._discriminator_output_shape + num_classes,
                      self._discriminator_output_shape),
            self._add_activation_layer(self._discriminator_activation_function)
        ]
        label_embedding = nn.Sequential(*label_embedding_layers)
        label_embedding.apply(self._initialize_weights)

        # Return the complete conditional discriminator
        return ConditionalDiscriminator(
            discriminator_core=self._discriminator_model_dense,
            label_embedding=label_embedding
        )

    def get_dense_discriminator_model(self) -> Optional[nn.Module]:
        """
        Retrieve the dense discriminator model.

        Returns:
            Optional[nn.Module]: The dense discriminator model, or None if not set.
        """
        return self._discriminator_model_dense

    def set_dropout_decay_rate_discriminator(self, dropout_decay_rate_discriminator: float):
        """
        Set the dropout decay rate for the discriminator network.

        Args:
            dropout_decay_rate_discriminator (float): The new dropout decay rate.
        """
        if not (0 <= dropout_decay_rate_discriminator <= 1):
            raise ValueError("dropout_decay_rate_discriminator must be between 0 and 1.")
        self._discriminator_dropout_decay_rate_d = dropout_decay_rate_discriminator

    def set_dense_layer_sizes_discriminator(self, dense_layer_sizes_discriminator: List[int]):
        """
        Set the sizes for the dense layers in the discriminator network.

        Args:
            dense_layer_sizes_discriminator (List[int]): A list of integers specifying the layer sizes.
        """
        if not all(isinstance(n, int) and n > 0 for n in dense_layer_sizes_discriminator):
            raise ValueError("All layer sizes must be positive integers.")
        self._discriminator_dense_layer_sizes_d = dense_layer_sizes_discriminator


class ConditionalDiscriminator(nn.Module):
    """
    Conditional Discriminator wrapper that combines data and labels.

    This module takes data and labels as input, concatenates them,
    processes through label embedding, and then through the discriminator core.
    """

    def __init__(self, discriminator_core: nn.Module, label_embedding: nn.Module):
        """
        Initialize the conditional discriminator.

        Args:
            discriminator_core: The main discriminator network.
            label_embedding: Network that processes concatenated [data, labels].
        """
        super().__init__()
        self.discriminator_core = discriminator_core
        self.label_embedding = label_embedding

    def forward(self, inputs):
        """
        Forward pass through the conditional discriminator.

        Args:
            inputs: Can be either:
                - A list/tuple of [data, labels]
                - A single tensor (for non-conditional use)

        Returns:
            torch.Tensor: Discriminator output (validity score).
        """
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            data, labels = inputs
            # Concatenate data and labels
            concatenated = torch.cat([data, labels], dim=1)
            # Process through label embedding
            embedded = self.label_embedding(concatenated)
            # Process through discriminator core
            validity = self.discriminator_core(embedded)
        else:
            # Non-conditional case: just pass through discriminator
            validity = self.discriminator_core(inputs)

        return validity