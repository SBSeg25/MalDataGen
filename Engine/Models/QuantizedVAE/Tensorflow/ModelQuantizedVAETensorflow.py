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
    
    from typing import List
    from typing import Tuple
    from typing import Optional

    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Concatenate

    from Engine.Layers.VectorQuantizerLayer import VectorQuantizer

    from Engine.Models.QuantizedVAE.Tensorflow.QuantizedVAEVanillaDecoderTensorflow import QuantizedVAEVanillaDecoder
    from Engine.Models.QuantizedVAE.Tensorflow.QuantizedVAEVanillaEncoderTensorflow import QuantizedVAEVanillaEncoder


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


class QuantizedVAEModelTensorflow(QuantizedVAEVanillaEncoder, QuantizedVAEVanillaDecoder):

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
                 number_neurons_encoder = None,
                 number_neurons_decoder = None,
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

        QuantizedVAEVanillaDecoder.__init__(self,
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

        QuantizedVAEVanillaEncoder.__init__(self,
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
        self._vector_quantizer_layer = VectorQuantizer(number_embeddings, latent_dimension, name="vector_quantizer")
        self._encoder_model = None
        self._decoder_model = None

    def get_encoder(self):
        return self._encoder_model

    def get_decoder(self):
        return self._decoder_model
    def get_quantized_model(self):

        neural_model_inputs = Input(shape=(self.output_shape_encoder,), dtype=self._encoder_dataset_type, name="first_input")
        label_input = Input(shape=(self._encoder_number_samples_per_class["number_classes"],),
                            dtype=self._encoder_dataset_type, name="second_input")

        self._encoder_model = self.get_encoder_model(self.output_shape_encoder)
        self._decoder_model = self.get_decoder_model(self.output_shape_encoder)
        encoder_flow = self._encoder_model([neural_model_inputs, label_input])
        quantized_latents = self._vector_quantizer_layer(encoder_flow)

        reconstructions = self._decoder_model(quantized_latents)

        return Model(inputs=[neural_model_inputs, label_input], outputs=reconstructions, name="vq_vae")

    def get_dense_encoder_model(self):
        """
        Returns the encoder model.

        Returns:
            tensorflow.keras.Model: The model representing the encoder.
        """
        return self._encoder_model

    def get_dense_decoder_model(self):
        """
        Returns the decoder model.

        Returns:
            tensorflow.keras.Model: The model representing the decoder.
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


