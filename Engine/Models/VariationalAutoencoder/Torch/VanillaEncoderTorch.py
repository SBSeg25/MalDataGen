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
    from typing import List, Dict, Union, Optional
    from Engine.Activations.Activations import Activations

except ImportError as error:
    print(error)
    sys.exit(-1)


class SamplingLayer(nn.Module):
    """Sampling layer for VAE reparameterization trick."""

    def forward(self, z_mean, z_log_var):
        """
        Sample from latent distribution using reparameterization trick.

        Args:
            z_mean: Mean of latent distribution
            z_log_var: Log variance of latent distribution

        Returns:
            Sampled latent vector
        """
        batch_size = z_mean.size(0)
        latent_dim = z_mean.size(1)
        epsilon = torch.randn(batch_size, latent_dim, device=z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class VanillaEncoderTorch(nn.Module, Activations):
    """
    VanillaEncoder

    Implements a fully connected conditional variational encoder (CVAE) model designed for probabilistic generative tasks.
    This encoder maps input data to a structured latent space while incorporating conditional information, enhancing control
    over latent representations. The model supports various activation functions, dropout-based regularization, and custom
    weight initialization.

    Attributes:
        @encoder_latent_dimension (int):
            Dimensionality of the latent space, defining the encoded feature representation.
        @encoder_output_shape (int):
            Dimensionality of the input data that will be encoded.
        @encoder_activation_function (str):
            Activation function applied to all hidden layers (e.g., ReLU, Tanh, LeakyReLU).
        @encoder_last_layer_activation (str):
            Activation function applied to the final layer of the encoder.
        @encoder_dropout_decay_rate_encoder (float):
            Dropout rate applied to dense layers to improve generalization (must be between 0 and 1).
        @encoder_dataset_type (Union[numpy.dtype, type]):
            Data type of the input tensors (default: numpy.float32).
        @encoder_initializer_mean (float):
            Mean of the normal distribution used for weight initialization.
        @encoder_initializer_deviation (float):
            Standard deviation of the normal distribution used for weight initialization.
        @encoder_number_neurons_encoder (List[int]):
            List of integers specifying the number of units per dense layer, defining model complexity.
        @encoder_number_samples_per_class (Optional[Dict[str, int]]):
            Dictionary specifying the number of samples per class in conditional scenarios, if provided.
        @encoder_model (Optional[nn.Module]):
            The actual encoder network.

    Raises:
        ValueError:
            Raised if invalid arguments are passed during initialization, such as:
            - Non-positive `latent_dimension` or `output_shape`
            - Dropout rate outside the range [0, 1]
            - Empty or invalid `number_neurons_encoder`

    References:
        - Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv preprint arXiv:1312.6114.
          Available at: https://arxiv.org/abs/1312.6114

    Example:
        >>> encoder = VanillaEncoder(
        ...     latent_dimension=64,
        ...     output_shape=784,
        ...     activation_function='relu',
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_encoder=0.3,
        ...     last_layer_activation='linear',
        ...     number_neurons_encoder=[512, 256, 128],
        ...     dataset_type=numpy.float32,
        ...     number_samples_per_class={"number_classes": 10}
        ... )
    """

    def __init__(self,
                 latent_dimension: int,
                 output_shape: int,
                 activation_function: str,
                 initializer_mean: float,
                 initializer_deviation: float,
                 dropout_decay_encoder: float,
                 last_layer_activation: str,
                 number_neurons_encoder: List[int],
                 dataset_type: Union[numpy.dtype, type] = numpy.float32,
                 number_samples_per_class: Optional[Dict[str, int]] = None) -> None:
        """
        Initializes the VanillaEncoder with user-defined hyperparameters.

        Args:
            @latent_dimension (int): Dimensionality of the latent space, defining the encoded feature representation.
            @output_shape (int): Dimensionality of the input data that will be encoded.
            @activation_function (str): Non-linear activation function applied in each encoder layer (e.g., ReLU, Tanh, LeakyReLU).
            @initializer_mean (float): Mean value for the Gaussian distribution used in weight initialization.
            @initializer_deviation (float): Standard deviation for the Gaussian distribution used in weight initialization.
            @dropout_decay_encoder (float): Dropout rate applied for regularization, preventing overfitting (must be between 0 and 1).
            @last_layer_activation (str): Activation function applied to the final layer of the encoder, defining latent space properties.
            @number_neurons_encoder (List[int]): List specifying the number of neurons per encoder layer, defining model complexity.
            @dataset_type (Union[numpy.dtype, type], optional): Data type of the input tensors (default: numpy.float32).
            @number_samples_per_class (Optional[Dict[str, int]], optional): Dictionary specifying the number of samples
            per class in conditional scenarios.

        Raises:
            ValueError: If latent_dimension, output_shape, or dropout_decay_encoder have invalid values.
        """
        nn.Module.__init__(self)
        Activations.__init__(self)

        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("Latent dimension must be a positive integer.")

        if not isinstance(output_shape, int) or output_shape <= 0:
            raise ValueError("Output shape must be a positive integer.")

        if not isinstance(activation_function, str):
            raise ValueError("Activation function must be a string.")

        if not isinstance(initializer_mean, (int, float)):
            raise ValueError("Initializer mean must be a numerical value.")

        if not isinstance(initializer_deviation, (int, float)) or initializer_deviation <= 0:
            raise ValueError("Initializer deviation must be a positive numerical value.")

        if not (0.0 <= dropout_decay_encoder <= 1.0):
            raise ValueError("Dropout decay rate must be between 0 and 1.")

        if not isinstance(last_layer_activation, str):
            raise ValueError("Last layer activation must be a string.")

        if not isinstance(number_neurons_encoder, list) or not all(
                isinstance(n, int) and n > 0 for n in number_neurons_encoder):
            raise ValueError("Number of neurons per encoder layer must be a list of positive integers.")

        if dataset_type not in [numpy.float16, numpy.float32, numpy.float64]:
            raise ValueError("Datasets type must be one of numpy.float16, numpy.float32, or numpy.float64.")

        # Initialize instance variables
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
        self._encoder_model = None

        # Sampling layer
        self._sampling_layer = SamplingLayer()

    def Sampling(self):
        """Returns the sampling layer for reparameterization trick."""
        return self._sampling_layer

    def get_encoder(self):
        """
        Constructs and returns the encoder model.

        The encoder combines input features and labels into a conditional representation
        that maps to a latent space. The model uses variational layers for mean and log variance,
        enabling sampling in the latent space.

        Returns:
            nn.Module: The constructed encoder model.

        Raises:
            ValueError: If the number of classes is not specified in number_samples_per_class.
        """
        # Ensure the number of classes is provided in the configuration
        if not self._encoder_number_samples_per_class or "number_classes" not in self._encoder_number_samples_per_class:
            raise ValueError("The number of classes must be specified in 'number_samples_per_class'.")

        class EncoderNetwork(nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.parent = parent

                # Label embedding layer
                self.label_embedding = nn.Linear(
                    parent._encoder_number_samples_per_class["number_classes"],
                    8
                )

                # First layer after concatenation
                input_size = parent._encoder_output_shape + 8
                layers = []

                # First layer
                layers.append(nn.Linear(input_size, parent._encoder_number_neurons_encoder[0]))
                self._init_weights(layers[-1], parent._encoder_initializer_mean, parent._encoder_initializer_deviation)
                layers.append(nn.Dropout(parent._encoder_dropout_decay_rate_encoder))
                layers.append(self._get_activation(parent._encoder_activation_function))

                # Hidden layers
                for i in range(1, len(parent._encoder_number_neurons_encoder)):
                    layers.append(nn.Linear(
                        parent._encoder_number_neurons_encoder[i - 1],
                        parent._encoder_number_neurons_encoder[i]
                    ))
                    self._init_weights(layers[-1], parent._encoder_initializer_mean,
                                       parent._encoder_initializer_deviation)
                    layers.append(nn.Dropout(parent._encoder_dropout_decay_rate_encoder))
                    layers.append(self._get_activation(parent._encoder_activation_function))

                self.encoder_layers = nn.Sequential(*layers)

                # Final dense layer
                self.final_dense = nn.Linear(
                    parent._encoder_number_neurons_encoder[-1],
                    parent._encoder_latent_dimension
                )
                self._init_weights(self.final_dense, parent._encoder_initializer_mean,
                                   parent._encoder_initializer_deviation)

                # Latent space layers
                self.z_mean = nn.Linear(parent._encoder_latent_dimension, parent._encoder_latent_dimension)
                self.z_log_var = nn.Linear(parent._encoder_latent_dimension, parent._encoder_latent_dimension)

            def _init_weights(self, layer, mean, std):
                """Initialize layer weights with normal distribution."""
                nn.init.normal_(layer.weight, mean=mean, std=std)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

            def _get_activation(self, activation_name):
                """Get activation function by name."""
                activations = {
                    'relu': nn.ReLU(),
                    'tanh': nn.Tanh(),
                    'sigmoid': nn.Sigmoid(),
                    'leaky_relu': nn.LeakyReLU(),
                    'swish': nn.SiLU(),
                    'linear': nn.Identity()
                }
                return activations.get(activation_name.lower(), nn.ReLU())

            def forward(self, x, label=None):
                """
                Forward pass through the encoder.

                Args:
                    x: Input features
                    label: One-hot encoded labels (optional)

                Returns:
                    Tuple of (z_mean, z_log_var, z_sample, label)
                """
                # Handle combined input (x + label concatenated)
                if label is None:
                    # Assume label is concatenated at the end of x
                    num_classes = self.parent._encoder_number_samples_per_class["number_classes"]
                    features = x[:, :-num_classes]
                    label = x[:, -num_classes:]
                else:
                    features = x

                # Embed labels
                label_embedded = torch.relu(self.label_embedding(label))

                # Concatenate features and embedded labels
                combined = torch.cat([features, label_embedded], dim=1)

                # Pass through encoder layers
                encoded = self.encoder_layers(combined)

                # Final dense layer
                encoded = self.final_dense(encoded)
                encoded = self._get_activation(self.parent._encoder_last_layer_activation)(encoded)

                # Generate mean and log variance
                z_mean = self.z_mean(encoded)
                z_log_var = self.z_log_var(encoded)

                # Sample from latent space
                z = self.parent._sampling_layer(z_mean, z_log_var)

                return z_mean, z_log_var, z, label

        self._encoder_model = EncoderNetwork(self)
        return self._encoder_model

    @property
    def dropout_decay_rate_encoder(self) -> float:
        """float: Dropout rate for encoder regularization."""
        return self._encoder_dropout_decay_rate_encoder

    @property
    def number_filters_encoder(self) -> List[int]:
        """List[int]: Number of neurons in each encoder layer."""
        return self._encoder_number_neurons_encoder

    @dropout_decay_rate_encoder.setter
    def dropout_decay_rate_encoder(self, dropout_decay_rate_encoder: float) -> None:
        """
        Set the dropout rate for encoder regularization.

        Args:
            dropout_decay_rate_encoder (float): Dropout rate to set.

        Raises:
            ValueError: If the dropout rate is not between 0 and 1.
        """
        if not (0.0 <= dropout_decay_rate_encoder <= 1.0):
            raise ValueError("Dropout decay rate must be between 0 and 1.")

        self._encoder_dropout_decay_rate_encoder = dropout_decay_rate_encoder