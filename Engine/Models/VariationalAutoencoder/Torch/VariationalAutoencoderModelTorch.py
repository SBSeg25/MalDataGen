#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

from Engine.Models.VariationalAutoencoder.Torch.VanillaDecoderTorch import VanillaDecoderTorch
from Engine.Models.VariationalAutoencoder.Torch.VanillaEncoderTorch import VanillaEncoderTorch

# MIT License - Copyright (c) 2025 Synthetic Ocean AI

try:
    import sys
    import numpy
    import torch.nn as nn

except ImportError as error:
    print(error)
    sys.exit(-1)

DEFAULT_VARIATIONAL_AUTOENCODER_LATENT_DIMENSION = 32
DEFAULT_VARIATIONAL_AUTOENCODER_ACTIVATION_INTERMEDIARY = "swish"
DEFAULT_VARIATIONAL_AUTOENCODER_DROPOUT_DECAY_RATE_ENCODER = 0.25
DEFAULT_VARIATIONAL_AUTOENCODER_DROPOUT_DECAY_RATE_DECODER = 0.25
DEFAULT_VARIATIONAL_AUTOENCODER_DENSE_LAYERS_SETTINGS_ENCODER = [320, 160]
DEFAULT_VARIATIONAL_AUTOENCODER_DENSE_LAYERS_SETTINGS_DECODER = [160, 320]
DEFAULT_VARIATIONAL_AUTOENCODER_LAST_ACTIVATION_LAYER = "sigmoid"
DEFAULT_VARIATIONAL_AUTOENCODER_INITIALIZER_MEAN = 0
DEFAULT_VARIATIONAL_AUTOENCODER_INITIALIZER_DEVIATION = 0.125


class VariationalModelTorch(nn.Module):
    """
    A Variational Model that integrates both VanillaEncoder and VanillaDecoder
    functionalities using composition instead of multiple inheritance.

    FIX: Changed from multiple inheritance to composition pattern to avoid
    PyTorch module registration issues.
    """

    def __init__(self,
                 latent_dimension: int = DEFAULT_VARIATIONAL_AUTOENCODER_LATENT_DIMENSION,
                 output_shape=None,
                 activation_function: str = DEFAULT_VARIATIONAL_AUTOENCODER_ACTIVATION_INTERMEDIARY,
                 initializer_mean: float = DEFAULT_VARIATIONAL_AUTOENCODER_INITIALIZER_MEAN,
                 initializer_deviation: float = DEFAULT_VARIATIONAL_AUTOENCODER_INITIALIZER_DEVIATION,
                 dropout_decay_encoder: float = DEFAULT_VARIATIONAL_AUTOENCODER_DROPOUT_DECAY_RATE_ENCODER,
                 dropout_decay_decoder: float = DEFAULT_VARIATIONAL_AUTOENCODER_DROPOUT_DECAY_RATE_DECODER,
                 last_layer_activation: str = DEFAULT_VARIATIONAL_AUTOENCODER_LAST_ACTIVATION_LAYER,
                 number_neurons_encoder=None,
                 number_neurons_decoder=None,
                 dataset_type=numpy.float32,
                 number_samples_per_class=None):
        """
        Initializes the VariationalModel with user-defined encoder and decoder configurations.
        """
        super(VariationalModelTorch, self).__init__()

        if number_neurons_decoder is None:
            number_neurons_decoder = DEFAULT_VARIATIONAL_AUTOENCODER_DENSE_LAYERS_SETTINGS_DECODER

        if number_neurons_encoder is None:
            number_neurons_encoder = DEFAULT_VARIATIONAL_AUTOENCODER_DENSE_LAYERS_SETTINGS_ENCODER

        # FIX: Use composition - create encoder and decoder as components
        self._encoder_component = VanillaEncoderTorch(
            latent_dimension,
            output_shape,
            activation_function,
            initializer_mean,
            initializer_deviation,
            dropout_decay_encoder,
            last_layer_activation,
            number_neurons_encoder,
            dataset_type,
            number_samples_per_class
        )

        self._decoder_component = VanillaDecoderTorch(
            latent_dimension,
            output_shape,
            activation_function,
            initializer_mean,
            initializer_deviation,
            dropout_decay_decoder,
            last_layer_activation,
            number_neurons_decoder,
            dataset_type,
            number_samples_per_class
        )

        # Initialize the actual encoder and decoder models
        self._encoder_model = None
        self._decoder_model = None

    def get_encoder(self, input_shape):
        """
        Returns the encoder model, creating it if necessary.

        Returns:
            nn.Module: The constructed encoder model.
        """
        if self._encoder_model is None:
            # FIX: Pass input_shape=None explicitly to match the method signature
            self._encoder_model = self._encoder_component.get_encoder(input_shape)
        return self._encoder_model

    def get_decoder(self, input_shape):
        """
        Returns the decoder model, creating it if necessary.

        Returns:
            nn.Module: The constructed decoder model.
        """
        if self._decoder_model is None:
            # FIX: Pass input_shape=None explicitly to match the method signature
            self._decoder_model = self._decoder_component.get_decoder(input_shape)
        return self._decoder_model

    def latent_dimension(self, latent_dimension):
        """
        Sets the latent dimension for both encoder and decoder.

        Args:
            latent_dimension (int): The dimensionality of the latent space.
        """
        self._encoder_component._encoder_latent_dimension = latent_dimension
        self._decoder_component._decoder_latent_dimension = latent_dimension

    def output_shape(self, output_shape):
        """
        Configures the output shape for both encoder and decoder.

        Args:
            output_shape (tuple): The desired output shape.
        """
        self._encoder_component._encoder_output_shape = output_shape
        self._decoder_component._decoder_output_shape = output_shape

    def intermediary_activation_function(self, activation_function):
        """
        Configures the activation function for intermediary layers in both encoder and decoder.

        Args:
            activation_function (str or callable): The activation function to be used.
        """
        self._encoder_component._encoder_activation_function = activation_function
        self._decoder_component._decoder_intermediary_activation_function = activation_function

    def last_layer_activation(self, last_layer_activation):
        """
        Configures the activation function for the last layer in both encoder and decoder.

        Args:
            last_layer_activation (str or callable): The activation function for the last layer.
        """
        self._encoder_component._encoder_last_layer_activation = last_layer_activation
        self._decoder_component._decoder_last_layer_activation = last_layer_activation