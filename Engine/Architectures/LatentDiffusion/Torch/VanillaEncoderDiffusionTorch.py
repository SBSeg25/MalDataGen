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
    import torch.nn.functional as F

    from typing import List
    from typing import Dict
    from typing import Union
    from typing import Optional

    from Engine.Layers.Torch.Activations import Activations
    from Engine.Layers.Torch.SamplingLayer import LayerSampling

except ImportError as error:
    print(error)
    sys.exit(-1)


class VanillaEncoderDiffusionTorch(nn.Module, Activations, LayerSampling):
    """
    VanillaEncoder - PyTorch Implementation

    Implements a fully connected conditional variational encoder (CVAE) model designed
    for probabilistic generative tasks. This encoder maps input data to a structured
    latent space while incorporating conditional information, enhancing control over
    latent representations. The model supports various activation functions,
    dropout-based regularization, and custom weight initialization.

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
        @encoder_model (Optional[Model]):
            Placeholder for the compiled model after build().

    References:
        - Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv preprint arXiv:1312.6114.
          Available at: https://arxiv.org/abs/1312.6114

    Example:
        >>> encoder = VanillaEncoderDiffusionTorch(
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
            @latent_dimension (int): Dimensionality of the latent space, defining
             the encoded feature representation.
            @output_shape (int): Dimensionality of the input data that will be encoded.
            @activation_function (str): Non-linear activation function applied in
             each encoder layer (e.g., ReLU, Tanh, LeakyReLU).
            @initializer_mean (float): Mean value for the Gaussian distribution
             used in weight initialization.
            @initializer_deviation (float): Standard deviation for the Gaussian
             distribution used in weight initialization.
            @dropout_decay_encoder (float): Dropout rate applied for regularization,
             preventing overfitting (must be between 0 and 1).
            @last_layer_activation (str): Activation function applied to the final
             layer of the encoder, defining latent space properties.
            @number_neurons_encoder (List[int]): List specifying the number of neurons
             per encoder layer, defining model complexity.
            @dataset_type (Union[numpy.dtype, type], optional): Data type of the input
             tensors (default: numpy.float32).
            @number_samples_per_class (Optional[Dict[str, int]], optional): Dictionary
             specifying the number of samples per class in conditional scenarios.

        Raises:
            ValueError: If latent_dimension, output_shape, or dropout_decay_encoder have invalid values.
        """
        super(VanillaEncoderDiffusionTorch, self).__init__()

        # Validate inputs to ensure valid model configuration
        if latent_dimension <= 0:
            raise ValueError("Latent dimension must be greater than 0.")

        if output_shape <= 0:
            raise ValueError("Output shape must be greater than 0.")

        if not (0.0 <= dropout_decay_encoder <= 1.0):
            raise ValueError("Dropout decay rate must be between 0 and 1.")

        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")

        if not isinstance(output_shape, int) or output_shape <= 0:
            raise ValueError("output_shape must be a positive integer.")

        if not isinstance(activation_function, (str, callable)):
            raise ValueError("activation_function must be a string or a callable function.")

        if not isinstance(initializer_mean, (float, int)):
            raise ValueError("initializer_mean must be a float or an integer.")

        if not isinstance(initializer_deviation, (float, int)) or initializer_deviation <= 0:
            raise ValueError("initializer_deviation must be a positive float or integer.")

        if not isinstance(last_layer_activation, (str, callable)):
            raise ValueError("last_layer_activation must be a string or a callable function.")

        if not isinstance(dataset_type, type):
            raise ValueError("dataset_type must be a valid data type.")

        if number_samples_per_class is not None:
            if not isinstance(number_samples_per_class, dict) or "number_classes" not in number_samples_per_class:
                raise ValueError("number_samples_per_class must be a dictionary containing the key 'number_classes'.")

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

        # Build the encoder network
        self._build_encoder()

    def _build_encoder(self):
        """Constructs the encoder network layers."""
        if not self._encoder_number_samples_per_class or "number_classes" not in self._encoder_number_samples_per_class:
            raise ValueError("The number of classes must be specified in 'number_samples_per_class'.")

        # Input size is feature size + number of classes
        input_size = self._encoder_output_shape + self._encoder_number_samples_per_class["number_classes"]

        # Build encoder layers
        self.encoder_layers = nn.ModuleList()
        self.encoder_dropouts = nn.ModuleList()

        # First layer
        layer = nn.Linear(input_size, self._encoder_number_neurons_encoder[0])
        nn.init.normal_(layer.weight, mean=self._encoder_initializer_mean, std=self._encoder_initializer_deviation)
        self.encoder_layers.append(layer)
        self.encoder_dropouts.append(nn.Dropout(self._encoder_dropout_decay_rate_encoder))

        # Additional layers
        for i in range(1, len(self._encoder_number_neurons_encoder)):
            layer = nn.Linear(self._encoder_number_neurons_encoder[i - 1], self._encoder_number_neurons_encoder[i])
            nn.init.normal_(layer.weight, mean=self._encoder_initializer_mean, std=self._encoder_initializer_deviation)
            self.encoder_layers.append(layer)
            self.encoder_dropouts.append(nn.Dropout(self._encoder_dropout_decay_rate_encoder))

        # Final layer before latent
        self.final_layer = nn.Linear(self._encoder_number_neurons_encoder[-1], self._encoder_latent_dimension)
        nn.init.normal_(self.final_layer.weight, mean=self._encoder_initializer_mean,
                        std=self._encoder_initializer_deviation)

        # Latent mean and log variance layers
        self.latent_mean_layer = nn.Linear(self._encoder_latent_dimension, self._encoder_latent_dimension)
        self.latent_log_var_layer = nn.Linear(self._encoder_latent_dimension, self._encoder_latent_dimension)

    def get_encoder_trained(self):
        """Returns the encoder model."""
        return self

    def create_embedding(self, data, labels, batch_size=32):
        """
        Generates latent space embeddings using the trained encoder.

        Args:
            data (Tensor): Input data to encode.
            labels (Tensor): Class labels.
            batch_size (int): Batch size for processing.

        Returns:
            Tensor: Latent space representations (mean).
        """
        self.eval()
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch_data = data[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]
                z_mean, _, _, _ = self.forward(batch_data, batch_labels)
                embeddings.append(z_mean)

        return torch.cat(embeddings, dim=0)

    def forward(self, neural_model_inputs, label_input):
        """
        Forward pass through the encoder.

        Args:
            neural_model_inputs (Tensor): Input features [batch, output_shape]
            label_input (Tensor): Class labels [batch, number_classes]

        Returns:
            Tuple: (latent_mean, latent_log_var, latent_sample, label_input)
        """
        # Concatenate inputs and labels
        x = torch.cat([neural_model_inputs, label_input], dim=-1)

        # Pass through encoder layers
        for layer, dropout in zip(self.encoder_layers, self.encoder_dropouts):
            x = layer(x)
            x = dropout(x)
            x = self._get_activation(self._encoder_activation_function)(x)

        # Final layer with specified activation
        x = self.final_layer(x)
        x = self._get_activation(self._encoder_last_layer_activation)(x)

        # Generate latent mean and log variance
        latent_mean = self.latent_mean_layer(x)
        latent_log_var = self.latent_log_var_layer(x)

        # Sample from latent space
        latent = self.sampling(latent_mean, latent_log_var)

        return latent_mean, latent_log_var, latent, label_input

    def sampling(self, z_mean, z_log_var):
        """
        Reparameterization trick for sampling from latent distribution.

        Args:
            z_mean (Tensor): Mean of latent distribution
            z_log_var (Tensor): Log variance of latent distribution

        Returns:
            Tensor: Sampled latent vector
        """
        batch_size = z_mean.shape[0]
        latent_dim = z_mean.shape[1]
        epsilon = torch.randn(batch_size, latent_dim, device=z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

    def _get_activation(self, name):
        """Get activation function by name."""
        if callable(name):
            return name

        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'swish': nn.SiLU(),
            'silu': nn.SiLU(),
            'linear': nn.Identity(),
            'elu': nn.ELU(),
        }
        return activations.get(name.lower(), nn.Identity())

    def get_encoder(self):
        """
        Returns the encoder model (for compatibility with original interface).

        Returns:
            self: The encoder model itself.
        """
        return self

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