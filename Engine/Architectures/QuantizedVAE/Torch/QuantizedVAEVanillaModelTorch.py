#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

from Engine.Architectures.QuantizedVAE.Torch.QuantizedVAEVanillaDecoderTorch import QuantizedVAEVanillaDecoderTorch
from Engine.Architectures.QuantizedVAE.Torch.QuantizedVAEVanillaEncoderTorch import QuantizedVAEVanillaEncoderTorch
from Engine.Architectures.QuantizedVAE.Torch.VectorQuantizerTorch import VectorQuantizerTorch

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

    from typing import List
    from typing import Tuple
    from typing import Optional


except ImportError as error:
    print(error)
    sys.exit(-1)

DEFAULT_QUANTIZED_VAE_LATENT_DIMENSION = 16
DEFAULT_QUANTIZED_VAE_NUMBER_EMBEDDING = 16
DEFAULT_QUANTIZED_VAE_ACTIVATION_INTERMEDIARY = "swish"
DEFAULT_QUANTIZED_VAE_DROPOUT_DECAY_RATE_ENCODER = 0.25
DEFAULT_QUANTIZED_VAE_DROPOUT_DECAY_RATE_DECODER = 0.25
DEFAULT_QUANTIZED_VAE_DENSE_LAYERS_SETTINGS_ENCODER = [320, 160]
DEFAULT_QUANTIZED_VAE_DENSE_LAYERS_SETTINGS_DECODER = [160, 320]
DEFAULT_QUANTIZED_VAE_LAST_ACTIVATION_LAYER = "sigmoid"
DEFAULT_QUANTIZED_VAE_INITIALIZER_MEAN = 0
DEFAULT_QUANTIZED_VAE_INITIALIZER_DEVIATION = 0.125


class QuantizedVAEModelTorch(QuantizedVAEVanillaEncoderTorch, QuantizedVAEVanillaDecoderTorch):

    def __init__(self,
                 latent_dimension: int = DEFAULT_QUANTIZED_VAE_LATENT_DIMENSION,
                 number_embeddings: int = DEFAULT_QUANTIZED_VAE_NUMBER_EMBEDDING,
                 output_shape: int = 128,
                 activation_function: str = DEFAULT_QUANTIZED_VAE_ACTIVATION_INTERMEDIARY,
                 initializer_mean: float = DEFAULT_QUANTIZED_VAE_INITIALIZER_MEAN,
                 initializer_deviation: float = DEFAULT_QUANTIZED_VAE_INITIALIZER_DEVIATION,
                 dropout_decay_encoder: float = DEFAULT_QUANTIZED_VAE_DROPOUT_DECAY_RATE_ENCODER,
                 dropout_decay_decoder: float = DEFAULT_QUANTIZED_VAE_DROPOUT_DECAY_RATE_DECODER,
                 last_layer_activation: str = DEFAULT_QUANTIZED_VAE_LAST_ACTIVATION_LAYER,
                 number_neurons_encoder=None,
                 number_neurons_decoder=None,
                 dataset_type: numpy.dtype = numpy.float32,
                 number_samples_per_class: Optional[int] = None):

        if number_neurons_decoder is None:
            number_neurons_decoder = DEFAULT_QUANTIZED_VAE_DENSE_LAYERS_SETTINGS_DECODER

        if number_neurons_encoder is None:
            number_neurons_encoder = DEFAULT_QUANTIZED_VAE_DENSE_LAYERS_SETTINGS_ENCODER

        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")

        if not isinstance(activation_function, str):
            raise ValueError("activation_function must be a string.")

        if not isinstance(initializer_mean, (float, int)):
            raise ValueError("initializer_mean must be a float or integer.")

        if not isinstance(initializer_deviation, (float, int)):
            raise ValueError("initializer_deviation must be a float or integer.")

        if not isinstance(dropout_decay_encoder, (float, int)) or not (0 <= dropout_decay_encoder <= 1):
            raise ValueError("dropout_decay_encoder must be a float or integer between 0 and 1.")

        if not isinstance(dropout_decay_decoder, (float, int)) or not (0 <= dropout_decay_decoder <= 1):
            raise ValueError("dropout_decay_decoder must be a float or integer between 0 and 1.")

        if not isinstance(last_layer_activation, str):
            raise ValueError("last_layer_activation must be a string.")

        if not isinstance(number_neurons_encoder, list) or not all(isinstance(x, int) for x in number_neurons_encoder):
            raise ValueError("number_neurons_encoder must be a list of integers.")

        if not isinstance(number_neurons_decoder, list) or not all(isinstance(x, int) for x in number_neurons_decoder):
            raise ValueError("number_neurons_decoder must be a list of integers.")

        QuantizedVAEVanillaDecoderTorch.__init__(self,
                                            latent_dimension,
                                            output_shape,
                                            activation_function,
                                            initializer_mean,
                                            initializer_deviation,
                                            dropout_decay_decoder,
                                            last_layer_activation,
                                            number_neurons_decoder,
                                            dataset_type,
                                            number_samples_per_class)

        QuantizedVAEVanillaEncoderTorch.__init__(self,
                                            latent_dimension,
                                            output_shape,
                                            activation_function,
                                            initializer_mean,
                                            initializer_deviation,
                                            dropout_decay_encoder,
                                            last_layer_activation,
                                            number_neurons_encoder,
                                            dataset_type,
                                            number_samples_per_class)
        self.output_shape_encoder = output_shape
        self._latent_dimension = latent_dimension
        self._vector_quantizer_layer = VectorQuantizerTorch(number_embeddings, latent_dimension, name="vector_quantizer")
        self._encoder_model = None
        self._decoder_model = None

    def get_encoder(self):
        return self._encoder_model

    def get_decoder(self):
        return self._decoder_model

    def get_quantized_model(self):
        """
        Builds and returns the complete VQ-VAE model

        Returns:
            nn.Module: Complete VQ-VAE model
        """
        self._encoder_model = self.get_encoder_model(self.output_shape_encoder)
        self._decoder_model = self.get_decoder_model(self.output_shape_encoder)

        return VQVAEModule(  # Changed from QuantizedVAEModelTorch
            encoder=self._encoder_model,
            decoder=self._decoder_model,
            vector_quantizer=self._vector_quantizer_layer
        )

    def get_dense_encoder_model(self):
        """
        Returns the encoder model.

        Returns:
            nn.Module: The model representing the encoder.
        """
        return self._encoder_model

    def get_dense_decoder_model(self):
        """
        Returns the decoder model.

        Returns:
            nn.Module: The model representing the decoder.
        """
        return self._decoder_model

    def latent_dimension(self, latent_dimension: int) -> None:
        """
        Sets the latent dimension for both the encoder and decoder.

        Args:
            latent_dimension (int): The dimensionality of the latent space.
        """
        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")

        self._encoder_latent_dimension = latent_dimension
        self._decoder_latent_dimension = latent_dimension

    def output_shape(self, output_shape: Tuple[int, ...]) -> None:
        """
        Sets the output shape for both the encoder and decoder.

        Args:
            output_shape (tuple): The shape of the output to be generated by both the encoder and decoder.
        """
        if not isinstance(output_shape, tuple) or not all(isinstance(x, int) for x in output_shape):
            raise ValueError("output_shape must be a tuple of integers.")

        self._encoder_output_shape = output_shape
        self._decoder_output_shape = output_shape

    def activation_function(self, activation_function: str) -> None:
        """
        Sets the activation function for both the encoder and decoder.

        Args:
            activation_function (str): The activation function to be applied throughout the encoder and decoder.
        """
        if not isinstance(activation_function, str):
            raise ValueError("activation_function must be a string.")

        self._encoder_activation_function = activation_function
        self._decoder_activation_function = activation_function

    def last_layer_activation(self, last_layer_activation: str) -> None:
        """
        Sets the activation function for the last layer of both the encoder and decoder.

        Args:
            last_layer_activation (str): The activation function to be used in the last layer of both the encoder and decoder.
        """
        if not isinstance(last_layer_activation, str):
            raise ValueError("last_layer_activation must be a string.")

        self._encoder_last_layer_activation = last_layer_activation
        self._decoder_last_layer_activation = last_layer_activation


class VQVAEModule(nn.Module):
    """Complete VQ-VAE model combining encoder, quantizer, and decoder"""

    def __init__(self, encoder, decoder, vector_quantizer):
        super(VQVAEModule, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vector_quantizer = vector_quantizer

    def forward(self, x):
        """
        Forward pass through complete VQ-VAE

        Args:
            x: List/tuple of [input_data, labels]

        Returns:
            Reconstructed output
        """
        # Encode
        encoder_output, labels = self.encoder(x)

        # Quantize
        quantized = self.vector_quantizer(encoder_output)

        # Decode
        reconstruction = self.decoder([quantized, labels])

        return reconstruction

    def get_layer(self, name):
        """Get layer by name for compatibility"""
        if name == "vector_quantizer":
            return self.vector_quantizer
        return None

    @property
    def losses(self):
        """Returns list of VQ losses"""
        return self.vector_quantizer.losses