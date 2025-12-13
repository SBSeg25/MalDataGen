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

try:
    import sys
    import torch
    import torch.nn as nn
    import numpy

    from typing import Dict
    from typing import List
    from typing import Tuple
    from typing import Optional

except ImportError as error:
    print(error)
    sys.exit(-1)


class DenseDiscriminatorModule(nn.Module):
    """
    Dense discriminator module without label conditioning.
    """

    def __init__(self, input_shape: int, dense_layer_sizes: List[int],
                 dropout_rate: float, activation_fn, last_activation_fn,
                 initializer_mean: float, initializer_deviation: float):
        super(DenseDiscriminatorModule, self).__init__()

        layers = []
        current_dim = input_shape

        # Build hidden layers
        for layer_size in dense_layer_sizes:
            layers.append(nn.Linear(current_dim, layer_size))
            # Initialize weights
            nn.init.normal_(layers[-1].weight, mean=initializer_mean, std=initializer_deviation)
            nn.init.zeros_(layers[-1].bias)

            layers.append(nn.Dropout(dropout_rate))
            layers.append(activation_fn)
            current_dim = layer_size

        # Output layer (single value for discriminator)
        layers.append(nn.Linear(current_dim, 1))
        nn.init.normal_(layers[-1].weight, mean=initializer_mean, std=initializer_deviation)
        nn.init.zeros_(layers[-1].bias)

        layers.append(last_activation_fn)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DiscriminatorWithLabelModule(nn.Module):
    """
    Discriminator module with label conditioning.
    """

    def __init__(self, output_shape: int, number_classes: int,
                 dense_discriminator: nn.Module, initializer_mean: float,
                 initializer_deviation: float):
        super(DiscriminatorWithLabelModule, self).__init__()

        self.dense_discriminator = dense_discriminator
        self.output_shape = output_shape
        self.number_classes = number_classes

        # Label embedding layer - input size is output_shape + number_classes
        self.label_embedding = nn.Linear(output_shape + number_classes, output_shape)
        nn.init.normal_(self.label_embedding.weight, mean=initializer_mean, std=initializer_deviation)
        nn.init.zeros_(self.label_embedding.bias)

    def forward(self, discriminator_input, label_input):
        """
        Args:
            discriminator_input: (batch, output_shape)
            label_input: (batch, number_classes) - one-hot encoded labels

        Returns:
            Validity scores of shape (batch, 1)
        """
        # Debug prints to check shapes
        if torch.isnan(discriminator_input).any() or torch.isnan(label_input).any():
            print(f"WARNING: NaN detected in discriminator inputs!")
            print(
                f"Discriminator input shape: {discriminator_input.shape}, contains NaN: {torch.isnan(discriminator_input).any()}")
            print(f"Label input shape: {label_input.shape}, contains NaN: {torch.isnan(label_input).any()}")

        # Ensure label_input has the correct shape
        if label_input.shape[1] != self.number_classes:
            raise ValueError(
                f"Label input has {label_input.shape[1]} classes but expected {self.number_classes}. "
                f"Full shape: {label_input.shape}"
            )

        # Concatenate input and label
        concatenated = torch.cat([discriminator_input, label_input], dim=-1)

        # Verify concatenated shape
        expected_size = self.output_shape + self.number_classes
        if concatenated.shape[1] != expected_size:
            raise ValueError(
                f"Concatenated tensor has shape {concatenated.shape} but expected "
                f"(batch_size, {expected_size}). Input: {discriminator_input.shape}, Label: {label_input.shape}"
            )

        # Process through label embedding
        embedded = self.label_embedding(concatenated)

        # Get validity score
        validity = self.dense_discriminator(embedded)

        return validity


class VanillaDiscriminatorTorch(nn.Module):
    """
    VanillaDiscriminatorTorch (PyTorch version)

    Implements a fully-connected (dense) discriminator network for use in generative models,
    such as GANs or WGANs. This class supports fully customizable layer sizes, activation
    functions, dropout rates, and initialization schemes, allowing it to be adapted to
    various tasks requiring a critic or discriminator network.

    This class does not implement training or loss computation directly, focusing instead
    on the architecture definition and construction.

    Attributes:
        discriminator_latent_dimension (int):
            Dimensionality of the latent space used by the model.
        discriminator_output_shape (int):
            Dimensionality of the expected output data.
        discriminator_activation_function (str):
            Activation function applied to all hidden layers.
        discriminator_last_layer_activation (str):
            Activation function applied to the final output layer.
        discriminator_dropout_decay_rate_d (float):
            Dropout rate applied to dense layers to improve generalization.
        discriminator_dense_layer_sizes_d (List[int]):
            List of integers specifying the number of units in each dense layer.
        discriminator_dataset_type (torch.dtype):
            Data type of the input dataset (default: torch.float32).
        discriminator_initializer_mean (float):
            Mean of the normal distribution used for weight initialization.
        discriminator_initializer_deviation (float):
            Standard deviation of the normal distribution used for weight initialization.
        discriminator_number_samples_per_class (Optional[Dict[str, int]]):
            Optional dictionary containing metadata about class distribution.
            Must include a key "number_classes" if provided.

    Raises:
        ValueError:
            Raised if invalid arguments are passed during initialization, such as:
            - Non-positive `latent_dimension`
            - Dropout rate outside the range [0, 1]
            - Empty or invalid `dense_layer_sizes_d`
            - Missing required key "number_classes" in `number_samples_per_class`, if provided

    References:
        - Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014).
          Generative adversarial Networks. arXiv preprint arXiv:1406.2661.
          Available at: https://arxiv.org/abs/1406.2661

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
        >>> discriminator_model = discriminator.get_discriminator()
    """

    def __init__(
            self,
            latent_dimension: int,
            output_shape: int,
            activation_function: str,
            initializer_mean: float,
            initializer_deviation: float,
            dropout_decay_rate_d: float,
            last_layer_activation: str,
            dense_layer_sizes_d: List[int],
            dataset_type: torch.dtype = torch.float32,
            number_samples_per_class: Optional[Dict[str, int]] = None) -> None:

        """
        Initializes the VanillaDiscriminatorTorch.

        This constructor sets up all internal attributes related to the discriminator
        architecture, including layer sizes, activation functions, initializers, and
        optional class distribution metadata.

        Args:
            latent_dimension (int): Dimensionality of the latent space.
            output_shape (int): Dimensionality of the output data.
            activation_function (str): Activation function for all hidden layers.
            initializer_mean (float): Mean of the normal distribution used to initialize weights.

            initializer_deviation (float): Standard deviation of the normal distribution.
            dropout_decay_rate_d (float): Dropout rate applied to dense layers (0 to 1).
            last_layer_activation (str): Activation function applied to the final output layer.
            dense_layer_sizes_d (List[int]): List of integers specifying units per dense layer.
            dataset_type (torch.dtype, optional): Data type of the input data.
            number_samples_per_class (Optional[Dict[str, int]], optional): Optional dictionary
                containing the number of samples per class.

        Raises:
            ValueError:
                If `latent_dimension` is <= 0.
                If `dropout_decay_rate_d` is not within [0, 1].
                If `dense_layer_sizes_d` is empty or contains non-positive values.
                If `number_samples_per_class` is provided but does not contain "number_classes".
        """
        super(VanillaDiscriminatorTorch, self).__init__()

        if latent_dimension <= 0:
            raise ValueError("The latent_dimension must be a positive integer.")

        if dropout_decay_rate_d < 0 or dropout_decay_rate_d > 1:
            raise ValueError("The dropout_decay_rate_d must be between 0 and 1.")

        if not dense_layer_sizes_d or not all(isinstance(x, int) and x > 0 for x in dense_layer_sizes_d):
            raise ValueError("dense_layer_sizes_d must be a non-empty list of positive integers.")

        if number_samples_per_class and "number_classes" not in number_samples_per_class:
            raise ValueError("number_samples_per_class must include a 'number_classes' key if provided.")

        self._discriminator_latent_dimension: int = latent_dimension
        self._discriminator_output_shape: int = output_shape
        self._discriminator_activation_function: str = activation_function
        self._discriminator_last_layer_activation: str = last_layer_activation
        self._discriminator_dropout_decay_rate_d: float = dropout_decay_rate_d
        self._discriminator_dense_layer_sizes_d: List[int] = dense_layer_sizes_d
        self._discriminator_dataset_type: torch.dtype = dataset_type
        self._discriminator_initializer_mean: float = initializer_mean
        self._discriminator_initializer_deviation: float = initializer_deviation
        self._discriminator_number_samples_per_class: Optional[Dict[str, int]] = number_samples_per_class
        self._discriminator_model_dense: Optional[nn.Module] = None
        self._discriminator_model_with_labels: Optional[nn.Module] = None
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

    def get_discriminator(self) -> nn.Module:
        """
        Constructs the discriminator model using dense layers, dropout, and activation functions.

        Returns:
            nn.Module: A PyTorch Module instance representing the discriminator.

        Raises:
            ValueError: If number_samples_per_class or its "number_classes" key is not properly set.
        """
        if not self._discriminator_number_samples_per_class or "number_classes" not in self._discriminator_number_samples_per_class:
            raise ValueError(
                "number_samples_per_class with a 'number_classes' key must be provided to construct the model.")

        # Get activation functions
        activation_fn = self.get_activation(
            self._discriminator_activation_function,
            alpha=0.2
        )
        last_activation_fn = self.get_activation(
            self._discriminator_last_layer_activation,
            alpha=0.2
        )

        # Build dense discriminator (without label conditioning)
        self._discriminator_model_dense = DenseDiscriminatorModule(
            input_shape=self._discriminator_output_shape,
            dense_layer_sizes=self._discriminator_dense_layer_sizes_d,
            dropout_rate=self._discriminator_dropout_decay_rate_d,
            activation_fn=activation_fn,
            last_activation_fn=last_activation_fn,
            initializer_mean=self._discriminator_initializer_mean,
            initializer_deviation=self._discriminator_initializer_deviation
        ).to(self.device)

        # Build discriminator with label conditioning
        self._discriminator_model_with_labels = DiscriminatorWithLabelModule(
            output_shape=self._discriminator_output_shape,
            number_classes=self._discriminator_number_samples_per_class["number_classes"],
            dense_discriminator=self._discriminator_model_dense,
            initializer_mean=self._discriminator_initializer_mean,
            initializer_deviation=self._discriminator_initializer_deviation
        ).to(self.device)

        return self._discriminator_model_with_labels

    def discriminate(self, discriminator_input: torch.Tensor, label_input: torch.Tensor) -> torch.Tensor:
        """
        Evaluate samples using the discriminator.

        Args:
            discriminator_input: (batch, output_shape) tensor
            label_input: (batch, number_classes) tensor

        Returns:
            Validity scores of shape (batch, 1)
        """
        if self._discriminator_model_with_labels is None:
            raise ValueError("Discriminator has not been built. Call get_discriminator() first.")

        self._discriminator_model_with_labels.eval()
        with torch.no_grad():
            discriminator_input = discriminator_input.to(self.device)
            label_input = label_input.to(self.device)
            output = self._discriminator_model_with_labels(discriminator_input, label_input)

        return output

    @property
    def dense_discriminator_model(self) -> Optional[nn.Module]:
        """Returns the dense part of the discriminator model."""
        return self._discriminator_model_dense

    @property
    def discriminator_model_with_labels(self) -> Optional[nn.Module]:
        """Returns the full discriminator model with label conditioning."""
        return self._discriminator_model_with_labels

    @property
    def dropout_decay_rate_discriminator(self) -> float:
        """Gets the dropout rate for the discriminator."""
        return self._discriminator_dropout_decay_rate_d

    @property
    def dense_layer_sizes_discriminator(self) -> List[int]:
        """Gets the sizes of the dense layers for the discriminator."""
        return self._discriminator_dense_layer_sizes_d

    @property
    def latent_dimension(self) -> int:
        """Gets the latent dimension."""
        return self._discriminator_latent_dimension

    @property
    def output_shape(self) -> int:
        """Gets the output shape."""
        return self._discriminator_output_shape

    @property
    def activation_function(self) -> str:
        """Gets the activation function."""
        return self._discriminator_activation_function

    @property
    def last_layer_activation(self) -> str:
        """Gets the last layer activation function."""
        return self._discriminator_last_layer_activation

    @property
    def initializer_mean(self) -> float:
        """Gets the initializer mean."""
        return self._discriminator_initializer_mean

    @property
    def initializer_deviation(self) -> float:
        """Gets the initializer deviation."""
        return self._discriminator_initializer_deviation

    @property
    def number_samples_per_class(self) -> Optional[Dict[str, int]]:
        """Gets the number of samples per class."""
        return self._discriminator_number_samples_per_class

    @dropout_decay_rate_discriminator.setter
    def dropout_decay_rate_discriminator(self, dropout_decay_rate_discriminator: float) -> None:
        """Sets the dropout rate for the discriminator."""
        if dropout_decay_rate_discriminator < 0 or dropout_decay_rate_discriminator > 1:
            raise ValueError("The dropout_decay_rate_discriminator must be between 0 and 1.")
        self._discriminator_dropout_decay_rate_d = dropout_decay_rate_discriminator

    @dense_layer_sizes_discriminator.setter
    def dense_layer_sizes_discriminator(self, dense_layer_sizes_discriminator: List[int]) -> None:
        """Sets the sizes of the dense layers for the discriminator."""
        if not dense_layer_sizes_discriminator or not all(
                isinstance(x, int) and x > 0 for x in dense_layer_sizes_discriminator):
            raise ValueError("dense_layer_sizes_discriminator must be a non-empty list of positive integers.")
        self._discriminator_dense_layer_sizes_d = dense_layer_sizes_discriminator

    @latent_dimension.setter
    def latent_dimension(self, value: int) -> None:
        """Sets the latent dimension."""
        if value <= 0:
            raise ValueError("The latent_dimension must be a positive integer.")
        self._discriminator_latent_dimension = value

    @output_shape.setter
    def output_shape(self, value: int) -> None:
        """Sets the output shape."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("output_shape must be a positive integer.")
        self._discriminator_output_shape = value

    @activation_function.setter
    def activation_function(self, value: str) -> None:
        """Sets the activation function."""
        if not isinstance(value, str):
            raise ValueError("activation_function must be a string.")
        self._discriminator_activation_function = value

    @last_layer_activation.setter
    def last_layer_activation(self, value: str) -> None:
        """Sets the last layer activation function."""
        if not isinstance(value, str):
            raise ValueError("last_layer_activation must be a string.")
        self._discriminator_last_layer_activation = value

    @initializer_mean.setter
    def initializer_mean(self, value: float) -> None:
        """Sets the initializer mean."""
        if not isinstance(value, (int, float)):
            raise ValueError("initializer_mean must be a number.")
        self._discriminator_initializer_mean = value

    @initializer_deviation.setter
    def initializer_deviation(self, value: float) -> None:
        """Sets the initializer deviation."""
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("initializer_deviation must be a positive number.")
        self._discriminator_initializer_deviation = value

    @number_samples_per_class.setter
    def number_samples_per_class(self, value: Optional[Dict[str, int]]) -> None:
        """Sets the number of samples per class."""
        if value is not None and (not isinstance(value, dict) or "number_classes" not in value):
            raise ValueError("number_samples_per_class must be a dictionary containing 'number_classes'.")
        self._discriminator_number_samples_per_class = value