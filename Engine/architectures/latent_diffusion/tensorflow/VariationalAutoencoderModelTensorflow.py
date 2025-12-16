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

    from Engine.architectures.latent_diffusion.tensorflow.VanillaDecoderDiffusionTensorflow import VanillaDecoderDiffusion
    from Engine.architectures.latent_diffusion.tensorflow.VanillaEncoderDiffusionTensorflow import VanillaEncoderDiffusion

except ImportError as error:
    print(error)
    sys.exit(-1)


class VariationalModelDiffusionTensorflow(VanillaDecoderDiffusion, VanillaEncoderDiffusion):
    """
    A Variational Model that integrates both VanillaEncoder and VanillaDecoder
    functionalities. This class enables flexible configuration of encoder and
    decoder parameters, facilitating variational-based learning tasks.

    Attributes:
        @latent_dimension (int): Dimensionality of the latent space.
        @output_shape (tuple): Shape of the output produced by the decoder.
        @activation_function (str or callable): Activation function for intermediary layers.
        @initializer_mean (float): Mean value for weight initialization.
        @initializer_deviation (float): Standard deviation for weight initialization.
        @dropout_decay_encoder (float): Dropout rate for encoder layers.
        @dropout_decay_decoder (float): Dropout rate for decoder layers.
        @last_layer_activation (str or callable): Activation function for the last layer.
        @number_neurons_encoder (list): Number of neurons in each layer of the encoder.
        @number_neurons_decoder (list): Number of neurons in each layer of the decoder.
        @dataset_type (dtype, optional): Data type of the dataset, defaults to numpy.float32.
        @number_samples_per_class (int, optional): Number of samples per class, defaults to None.
    """

    def __init__(self,
                 latent_dimension,
                 output_shape,
                 activation_function,
                 initializer_mean,
                 initializer_deviation,
                 dropout_decay_encoder,
                 dropout_decay_decoder,
                 last_layer_activation,
                 number_neurons_encoder,
                 number_neurons_decoder,
                 dataset_type=numpy.float32,
                 number_samples_per_class=None):
        """
        Initializes the VariationalModel with user-defined encoder and decoder configurations.

        Args:
            @latent_dimension (int): The dimensionality of the latent space.
            @output_shape (tuple): The shape of the output produced by the decoder.
            @activation_function (str or callable): Activation function for intermediary layers.
            @initializer_mean (float): Mean value for weight initialization.
            @initializer_deviation (float): Standard deviation for weight initialization.
            @dropout_decay_encoder (float): Dropout rate for encoder layers.
            @dropout_decay_decoder (float): Dropout rate for decoder layers.
            @last_layer_activation (str or callable): Activation function for the last layer.
            @number_neurons_encoder (list): Number of neurons in each layer of the encoder.
            @number_neurons_decoder (list): Number of neurons in each layer of the decoder.
            @dataset_type (dtype, optional): Data type of the dataset, defaults to numpy.float32.
            @number_samples_per_class (int, optional): Number of samples per class, defaults to None.
        """

        # Initialize the encoder using the VanillaEncoder class
        VanillaDecoderDiffusion.__init__(self,
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

        # Initialize the decoder using the VanillaDecoder class
        VanillaEncoderDiffusion.__init__(self,
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

    def latent_dimension(self, latent_dimension):
        """
        Sets the latent dimension for both encoder and decoder.

        Args:
            latent_dimension (int): The dimensionality of the latent space.
        """
        self._encoder_latent_dimension = latent_dimension
        self._decoder_latent_dimension = latent_dimension

    def output_shape(self, output_shape):
        """
        Configures the output shape for both encoder and decoder.

        Args:
            output_shape (tuple): The desired output shape.
        """
        self._encoder_output_shape = output_shape
        self._decoder_output_shape = output_shape

    def intermediary_activation_function(self, activation_function):
        """
        Configures the activation function for intermediary layers in both encoder and decoder.

        Args:
            activation_function (str or callable): The activation function to be used.
        """
        self._encoder_activation_function = activation_function
        self._decoder_activation_function = activation_function

    def last_layer_activation(self, last_layer_activation):
        """
        Configures the activation function for the last layer in both encoder and decoder.

        Args:
            last_layer_activation (str or callable): The activation function for the last layer.
        """
        self._encoder_last_layer_activation = last_layer_activation
        self._decoder_last_layer_activation = last_layer_activation

