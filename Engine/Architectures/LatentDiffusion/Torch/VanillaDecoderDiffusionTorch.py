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

    from Engine.Layers.Torch.Activations import Activations

except ImportError as error:
    print(error)
    sys.exit(-1)


class VanillaDecoderDiffusionTorch(nn.Module, Activations):
    """
      VanillaDecoder - PyTorch Implementation

      This class implements a conditional fully connected decoder network, which reconstructs data from a
      latent representation. It extends from the `Activations` base class to inherit common activation
      utilities. The decoder can be conditioned on class labels or other side information, making it
      suitable for use in conditional autoencoders or conditional generative models.

      The architecture consists of a sequence of fully connected layers, each followed by an
      activation function and optional dropout for regularization. The final output layer can use
      a customizable activation function (e.g., sigmoid, tanh) depending on the desired output format.

      Attributes
      ----------
      @decoder_latent_dimension : int
          Dimensionality of the latent space (input to the decoder).

      @decoder_output_shape : int
          Dimensionality of the output data (reconstructed data).

      @decoder_intermediary_activation_function : Callable
          Activation function applied to intermediate layers.

      @decoder_last_layer_activation : Callable
          Activation function applied to the final layer.

      @decoder_dropout_decay_rate_decoder : float
          Dropout rate applied to the dense layers for regularization.

      @decoder_dataset_type : type
          Data type for inputs and outputs (default: numpy.float32).

      @decoder_initializer_mean : float
          Mean of the normal distribution used for weight initialization.

      @decoder_initializer_deviation : float
          Standard deviation of the normal distribution used for weight initialization.

      @decoder_number_neurons_decoder : List[int]
          List specifying the number of neurons in each dense layer.

      @decoder_number_samples_per_class : Optional[Dict[str, int]]
          Dictionary containing class metadata (e.g., number of samples per class). This allows
          the decoder to incorporate label conditioning if provided.

      References
      ----------
      Kingma, D.P., & Welling, M. (2013). Auto-Encoding Variational Bayes.
      https://arxiv.org/abs/1312.6114
      """

    def __init__(self,
                 latent_dimension,
                 output_shape,
                 activation_function,
                 initializer_mean,
                 initializer_deviation,
                 dropout_decay_decoder,
                 last_layer_activation,
                 number_neurons_decoder,
                 dataset_type=numpy.float32,
                 number_samples_per_class=None):
        """
        Initializes the VanillaDecoder with the given hyperparameters.

        Parameters
        ----------
        @latent_dimension : int
            Dimensionality of the input latent space.

        @output_shape : int
            Dimensionality of the output (typically the same as the input to the encoder).

        @activation_function : str or callable
            Activation function for intermediate layers.

        @initializer_mean : float
            Mean of the normal distribution for weight initialization.

        @initializer_deviation : float
            Standard deviation of the normal distribution for weight initialization.

        @dropout_decay_decoder : float
            Dropout rate applied to intermediate layers.

        @last_layer_activation : str or callable
            Activation function for the final layer (e.g., 'sigmoid' for normalized outputs).

        @number_neurons_decoder : list of int
            Number of neurons in each fully connected layer.

        @dataset_type : type, optional
            Data type for inputs and outputs (default: numpy.float32).

        @number_samples_per_class : dict, optional
            Optional metadata dictionary containing information about class counts
            (used for label conditioning, if applicable).

        Raises
        ------
        ValueError
            If any provided parameter has an invalid value (e.g., non-positive layer sizes,
            invalid dropout rates).
        """
        super(VanillaDecoderDiffusionTorch, self).__init__()

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

        if not isinstance(dropout_decay_decoder, (float, int)) or not (0 <= dropout_decay_decoder <= 1):
            raise ValueError("dropout_decay_decoder must be a float between 0 and 1.")

        if not isinstance(last_layer_activation, (str, callable)):
            raise ValueError("last_layer_activation must be a string or a callable function.")

        if not isinstance(number_neurons_decoder, list) or not all(
                isinstance(n, int) and n > 0 for n in number_neurons_decoder):
            raise ValueError("number_neurons_decoder must be a list of positive integers.")

        if not isinstance(dataset_type, type):
            raise ValueError("dataset_type must be a valid data type.")

        if number_samples_per_class is not None:
            if not isinstance(number_samples_per_class, dict) or "number_classes" not in number_samples_per_class:
                raise ValueError("number_samples_per_class must be a dictionary containing the key 'number_classes'.")

        self._decoder_latent_dimension = latent_dimension
        self._decoder_output_shape = output_shape
        self._decoder_intermediary_activation_function = activation_function
        self._decoder_last_layer_activation = last_layer_activation
        self._decoder_dropout_decay_rate_decoder = dropout_decay_decoder
        self._decoder_dataset_type = dataset_type
        self._decoder_initializer_mean = initializer_mean
        self._decoder_initializer_deviation = initializer_deviation
        self._decoder_number_neurons_decoder = number_neurons_decoder
        self._decoder_number_samples_per_class = number_samples_per_class
        self._decoder_model = None

        # Build the decoder network
        self._build_decoder()

    def _build_decoder(self):
        """Constructs the decoder network layers."""
        # Input size is latent dimension + number of classes
        input_size = self._decoder_latent_dimension + self._decoder_number_samples_per_class["number_classes"]

        # Build decoder layers
        self.decoder_layers = nn.ModuleList()
        self.decoder_dropouts = nn.ModuleList()

        # First layer
        layer = nn.Linear(input_size, self._decoder_number_neurons_decoder[0])
        nn.init.normal_(layer.weight, mean=self._decoder_initializer_mean, std=self._decoder_initializer_deviation)
        self.decoder_layers.append(layer)
        self.decoder_dropouts.append(nn.Dropout(self._decoder_dropout_decay_rate_decoder))

        # Hidden layers
        for i in range(1, len(self._decoder_number_neurons_decoder)):
            layer = nn.Linear(self._decoder_number_neurons_decoder[i - 1], self._decoder_number_neurons_decoder[i])
            nn.init.normal_(layer.weight, mean=self._decoder_initializer_mean, std=self._decoder_initializer_deviation)
            self.decoder_layers.append(layer)
            self.decoder_dropouts.append(nn.Dropout(self._decoder_dropout_decay_rate_decoder))

        # Output layer
        self.output_layer = nn.Linear(self._decoder_number_neurons_decoder[-1], self._decoder_output_shape)
        nn.init.normal_(self.output_layer.weight, mean=self._decoder_initializer_mean,
                        std=self._decoder_initializer_deviation)

    def get_decoder_trained(self):
        """Returns the decoder model."""
        return self

    def forward(self, neural_model_inputs, label_input):
        """
        Forward pass through the decoder.

        Args:
            neural_model_inputs (Tensor): Latent vector [batch, latent_dimension]
            label_input (Tensor): Class labels [batch, number_classes]

        Returns:
            Tensor: Reconstructed output [batch, output_shape]
        """
        # Concatenate latent vector with conditional labels
        x = torch.cat([neural_model_inputs, label_input], dim=-1)

        # Pass through decoder layers
        for i, (layer, dropout) in enumerate(zip(self.decoder_layers, self.decoder_dropouts)):
            x = layer(x)
            x = self._get_activation(self._decoder_intermediary_activation_function)(x)
            x = dropout(x)

        # Output layer
        x = self.output_layer(x)
        x = self._get_activation(self._decoder_last_layer_activation)(x)

        return x

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

    def get_decoder(self):
        """
        Returns the decoder model (for compatibility with original interface).

        Returns:
            self: The decoder model itself.
        """
        return self

    @property
    def dropout_decay_rate_decoder(self):
        """float: Gets or sets the dropout decay rate for the decoder."""
        return self._decoder_dropout_decay_rate_decoder

    @property
    def number_filters_decoder(self):
        """list[int]: Gets the number of neurons for each layer in the decoder."""
        return self._decoder_number_neurons_decoder

    @dropout_decay_rate_decoder.setter
    def dropout_decay_rate_decoder(self, dropout_decay_rate_discriminator):
        """
        Sets the dropout decay rate for the decoder.

        Args:
            dropout_decay_rate_discriminator (float): New dropout decay rate.
        """
        self._decoder_dropout_decay_rate_decoder = dropout_decay_rate_discriminator