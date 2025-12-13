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
    import torch
    import torch.nn as nn
    import numpy
    from typing import Optional, Dict, List, Callable

except ImportError as error:
    print(error)
    sys.exit(-1)


class DenseGeneratorModule(nn.Module):
    """
    Dense generator module without label conditioning.
    """

    def __init__(self, latent_dimension: int, output_shape: int,
                 dense_layer_sizes: List[int], dropout_rate: float,
                 activation_fn, last_activation_fn,
                 initializer_mean: float, initializer_deviation: float):
        super(DenseGeneratorModule, self).__init__()

        layers = []
        current_dim = latent_dimension

        # Build hidden layers
        for layer_size in dense_layer_sizes:
            layers.append(nn.Linear(current_dim, layer_size))
            # Initialize weights
            nn.init.normal_(layers[-1].weight, mean=initializer_mean, std=initializer_deviation)
            nn.init.zeros_(layers[-1].bias)

            layers.append(nn.Dropout(dropout_rate))
            layers.append(activation_fn)
            current_dim = layer_size

        # Output layer
        layers.append(nn.Linear(current_dim, output_shape))
        nn.init.normal_(layers[-1].weight, mean=initializer_mean, std=initializer_deviation)
        nn.init.zeros_(layers[-1].bias)

        layers.append(last_activation_fn)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class GeneratorWithLabelModule(nn.Module):
    """
    Generator module with label conditioning.
    """

    def __init__(self, latent_dimension: int, number_classes: int,
                 dense_generator: nn.Module, activation_fn,
                 initializer_mean: float, initializer_deviation: float):
        super(GeneratorWithLabelModule, self).__init__()

        self.dense_generator = dense_generator
        self.latent_dimension = latent_dimension
        self.number_classes = number_classes

        # Label embedding layer - input size is latent_dimension + number_classes
        self.label_embedding = nn.Sequential(
            nn.Linear(latent_dimension + number_classes, latent_dimension),
            activation_fn
        )

        # Initialize weights
        for layer in self.label_embedding:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=initializer_mean, std=initializer_deviation)
                nn.init.zeros_(layer.bias)

    def forward(self, latent_input, label_input):
        """
        Args:
            latent_input: (batch, latent_dimension)
            label_input: (batch, number_classes) - one-hot encoded labels

        Returns:
            Generated output of shape (batch, output_shape)
        """
        # Debug prints to check shapes
        if torch.isnan(latent_input).any() or torch.isnan(label_input).any():
            print(f"WARNING: NaN detected in inputs!")
            print(f"Latent input shape: {latent_input.shape}, contains NaN: {torch.isnan(latent_input).any()}")
            print(f"Label input shape: {label_input.shape}, contains NaN: {torch.isnan(label_input).any()}")

        # Ensure label_input has the correct shape
        if label_input.shape[1] != self.number_classes:
            raise ValueError(
                f"Label input has {label_input.shape[1]} classes but expected {self.number_classes}. "
                f"Full shape: {label_input.shape}"
            )

        # Concatenate latent and label
        concatenated = torch.cat([latent_input, label_input], dim=-1)

        # Verify concatenated shape
        expected_size = self.latent_dimension + self.number_classes
        if concatenated.shape[1] != expected_size:
            raise ValueError(
                f"Concatenated tensor has shape {concatenated.shape} but expected "
                f"(batch_size, {expected_size}). Latent: {latent_input.shape}, Label: {label_input.shape}"
            )

        # Process through label embedding
        embedded = self.label_embedding(concatenated)

        # Generate output
        output = self.dense_generator(embedded)

        return output


class VanillaGeneratorTorch(nn.Module):
    """
    VanillaGeneratorTorch (PyTorch version)

    Implements a fully connected (dense) generator model for use in generative models,
    such as GANs. This generator is designed to work with label conditioning and
    supports customization of activation functions, layer sizes, initialization, and
    other hyperparameters.

    Attributes:
        generator_latent_dimension (int):
            Dimensionality of the input latent space.
        generator_output_shape (int):
            Dimensionality of the generated output data.
        generator_activation_function (str):
            Activation function applied to all hidden layers.
        generator_last_layer_activation (str):
            Activation function applied to the final output layer.
        generator_dropout_decay_rate_g (float):
            Dropout rate applied to dense layers to improve generalization.
        generator_dense_layer_sizes_g (List[int]):
            List of integers specifying the number of units in each dense layer.
        generator_dataset_type (torch.dtype):
            Data type of the input dataset (default: torch.float32).
        generator_initializer_mean (float):
            Mean of the normal distribution used for weight initialization.
        generator_initializer_deviation (float):
            Standard deviation of the normal distribution used for weight initialization.
        generator_number_samples_per_class (Optional[Dict[str, int]]):
            Optional dictionary containing metadata about class distribution.
            Must include a key "number_classes" if provided.

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
        ...     activation_function='leaky_relu',
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_rate_g=0.3,
        ...     last_layer_activation='tanh',
        ...     dense_layer_sizes_g=[256, 512, 1024],
        ...     dataset_type=torch.float32,
        ...     number_samples_per_class={"number_classes": 10}
        ... )
        >>> generator_model = generator.get_generator()
    """

    def __init__(self, latent_dimension: int,
                 output_shape: int,
                 activation_function: str,
                 initializer_mean: float,
                 initializer_deviation: float,
                 dropout_decay_rate_g: float,
                 last_layer_activation: str,
                 dense_layer_sizes_g: List[int],
                 dataset_type: torch.dtype = torch.float32,
                 number_samples_per_class: Optional[Dict[str, int]] = None):
        """
        Initializes the VanillaGeneratorTorch class with the provided parameters.

        Args:
            latent_dimension (int):
                Dimensionality of the latent space.
            output_shape (int):
                Dimensionality of the generated output data.
            activation_function (str):
                Activation function for all hidden layers.
            initializer_mean (float):
                Mean of the normal distribution used to initialize weights.
            initializer_deviation (float):
                Standard deviation of the normal distribution used to initialize weights.
            dropout_decay_rate_g (float):
                Dropout rate applied to dense layers (0 to 1).
            last_layer_activation (str):
                Activation function applied to the final output layer.
            dense_layer_sizes_g (List[int]):
                List of integers specifying the number of units per dense layer.
            dataset_type (torch.dtype, optional):
                Data type of the input data (default: torch.float32).
            number_samples_per_class (Optional[Dict[str, int]], optional):
                Optional dictionary containing the number of samples per class.

        Raises:
            ValueError:
                If `latent_dimension` or `output_shape` is <= 0.
                If `dropout_decay_rate_g` is not within [0, 1].
                If `dense_layer_sizes_g` is empty or contains non-positive values.
                If `number_samples_per_class` is provided but does not contain "number_classes".
        """
        super(VanillaGeneratorTorch, self).__init__()

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
        self._generator_model_dense = None
        self._generator_model_with_labels = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def get_activation(activation_name: str, alpha: float = 0.2) -> nn.Module:
        """
        Returns the appropriate activation function module.

        Args:
            activation_name (str): Name of the activation function.
            alpha (float): Alpha parameter for LeakyReLU (default: 0.2).

        Returns:
            nn.Module: PyTorch activation module.
        """
        activation_name = activation_name.lower().replace('_', '')

        activation_map = {
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU(alpha),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax(dim=1),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'gelu': nn.GELU(),
            'linear': nn.Identity(),  # Linear activation = no activation
            'none': nn.Identity(),  # Alternative name for no activation
            'identity': nn.Identity(),  # Another alternative
        }

        if activation_name in activation_map:
            return activation_map[activation_name]
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")

    @staticmethod
    def _add_activation_layer(activation_name: str, alpha: float = 0.2):
        """
        Returns the appropriate activation layer for the given activation name.

        Args:
            activation_name (str): Name of the activation function.
            alpha (float): Alpha parameter for LeakyReLU (default: 0.2).

        Returns:
            nn.Module: PyTorch activation module.

        Raises:
            ValueError: If the activation function is not supported.
        """
        activation_name = activation_name.strip().lower().replace('_', '')
        activation_map = {
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU(alpha),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax(dim=1),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'gelu': nn.GELU(),
            'linear': nn.Identity(),  # Linear/no activation
            'none': nn.Identity(),  # Alternative for no activation
            'identity': nn.Identity(),  # Another alternative
        }

        if activation_name in activation_map:
            return activation_map[activation_name]
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}. "
                             f"Supported activations: {list(activation_map.keys())}")

    def get_generator(self) -> nn.Module:
        """
        Constructs and returns the generator model, including the latent space input and label conditioning.

        Returns:
            nn.Module: A PyTorch module implementing the generator with latent and label inputs.

        Raises:
            ValueError: If number_samples_per_class is not properly defined.
        """
        if not self._generator_number_samples_per_class or "number_classes" not in self._generator_number_samples_per_class:
            raise ValueError("Number of samples per class must include 'number_classes'.")

        # Get activation functions
        activation_fn = self.get_activation(
            self._generator_activation_function,
            alpha=0.2
        )
        last_activation_fn = self.get_activation(
            self._generator_last_layer_activation,
            alpha=0.2
        )

        # Build dense generator (without label conditioning)
        self._generator_model_dense = DenseGeneratorModule(
            latent_dimension=self._generator_latent_dimension,
            output_shape=self._generator_output_shape,
            dense_layer_sizes=self._generator_dense_layer_sizes_g,
            dropout_rate=self._generator_dropout_decay_rate_g,
            activation_fn=activation_fn,
            last_activation_fn=last_activation_fn,
            initializer_mean=self._generator_initializer_mean,
            initializer_deviation=self._generator_initializer_deviation
        ).to(self.device)

        # Build generator with label conditioning
        self._generator_model_with_labels = GeneratorWithLabelModule(
            latent_dimension=self._generator_latent_dimension,
            number_classes=self._generator_number_samples_per_class["number_classes"],
            dense_generator=self._generator_model_dense,
            activation_fn=activation_fn,
            initializer_mean=self._generator_initializer_mean,
            initializer_deviation=self._generator_initializer_deviation
        ).to(self.device)

        return self._generator_model_with_labels

    def generate(self, latent_input: torch.Tensor, label_input: torch.Tensor) -> torch.Tensor:
        """
        Generate samples using the generator.

        Args:
            latent_input: (batch, latent_dimension) tensor
            label_input: (batch, number_classes) tensor

        Returns:
            Generated samples of shape (batch, output_shape)
        """
        if self._generator_model_with_labels is None:
            raise ValueError("Generator has not been built. Call get_generator() first.")

        self._generator_model_with_labels.eval()
        with torch.no_grad():
            latent_input = latent_input.to(self.device)
            label_input = label_input.to(self.device)
            output = self._generator_model_with_labels(latent_input, label_input)

        return output

    @property
    def dense_generator_model(self) -> Optional[nn.Module]:
        """Property that retrieves the dense generator submodel without label conditioning."""
        return self._generator_model_dense

    @property
    def generator_model_with_labels(self) -> Optional[nn.Module]:
        """Property that retrieves the full generator model with label conditioning."""
        return self._generator_model_with_labels

    @property
    def dropout_decay_rate_generator(self) -> float:
        """Property to get the dropout decay rate for the generator."""
        return self._generator_dropout_decay_rate_g

    @property
    def dense_layer_sizes_generator(self) -> List[int]:
        """Property to get the dense layer sizes for the generator."""
        return self._generator_dense_layer_sizes_g

    @property
    def latent_dimension(self) -> int:
        """Property to get the latent dimension."""
        return self._generator_latent_dimension

    @property
    def output_shape(self) -> int:
        """Property to get the output shape."""
        return self._generator_output_shape

    @property
    def activation_function(self) -> str:
        """Property to get the activation function."""
        return self._generator_activation_function

    @property
    def last_layer_activation(self) -> str:
        """Property to get the last layer activation function."""
        return self._generator_last_layer_activation

    @property
    def initializer_mean(self) -> float:
        """Property to get the initializer mean."""
        return self._generator_initializer_mean

    @property
    def initializer_deviation(self) -> float:
        """Property to get the initializer deviation."""
        return self._generator_initializer_deviation

    @property
    def number_samples_per_class(self) -> Optional[Dict[str, int]]:
        """Property to get the number of samples per class."""
        return self._generator_number_samples_per_class

    @dropout_decay_rate_generator.setter
    def dropout_decay_rate_generator(self, dropout_decay_rate_generator: float):
        """Property to set the dropout decay rate for the generator."""
        if dropout_decay_rate_generator < 0 or dropout_decay_rate_generator > 1:
            raise ValueError("Dropout decay rate must be in the range [0, 1].")
        self._generator_dropout_decay_rate_g = dropout_decay_rate_generator

    @dense_layer_sizes_generator.setter
    def dense_layer_sizes_generator(self, dense_layer_sizes_generator: List[int]):
        """Property to set the dense layer sizes for the generator."""
        if not dense_layer_sizes_generator or any(size <= 0 for size in dense_layer_sizes_generator):
            raise ValueError("Dense layer sizes must be a list of positive integers.")
        self._generator_dense_layer_sizes_g = dense_layer_sizes_generator

    @latent_dimension.setter
    def latent_dimension(self, value: int):
        """Property to set the latent dimension."""
        if value <= 0:
            raise ValueError("Latent dimension must be a positive integer.")
        self._generator_latent_dimension = value

    @output_shape.setter
    def output_shape(self, value: int):
        """Property to set the output shape."""
        if value <= 0:
            raise ValueError("Output shape must be a positive integer.")
        self._generator_output_shape = value

    @activation_function.setter
    def activation_function(self, value: str):
        """Property to set the activation function."""
        if not isinstance(value, str):
            raise ValueError("Activation function must be a string.")
        self._generator_activation_function = value

    @last_layer_activation.setter
    def last_layer_activation(self, value: str):
        """Property to set the last layer activation function."""
        if not isinstance(value, str):
            raise ValueError("Last layer activation must be a string.")
        self._generator_last_layer_activation = value

    @initializer_mean.setter
    def initializer_mean(self, value: float):
        """Property to set the initializer mean."""
        if value < 0:
            raise ValueError("Initializer mean must be non-negative.")
        self._generator_initializer_mean = value

    @initializer_deviation.setter
    def initializer_deviation(self, value: float):
        """Property to set the initializer deviation."""
        if value <= 0:
            raise ValueError("Initializer deviation must be positive.")
        self._generator_initializer_deviation = value

    @number_samples_per_class.setter
    def number_samples_per_class(self, value: Optional[Dict[str, int]]):
        """Property to set the number of samples per class."""
        if value is not None and (not isinstance(value, dict) or "number_classes" not in value):
            raise ValueError("number_samples_per_class must be a dictionary containing 'number_classes'.")
        self._generator_number_samples_per_class = value