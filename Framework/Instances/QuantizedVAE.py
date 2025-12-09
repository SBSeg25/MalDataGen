#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

from Engine.Models.QuantizedVAE.QuantizedVAEModel import QuantizedVAEModel

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
    import logging
    import torch
    import torch.nn.functional as F
    from Engine.Algorithms.QuantizedVAE.QuantizedVAEAlgorithm import QuantizedVAEAlgorithm
    from Engine.Models.QuantizedVAE.Torch.QuantizedVAEVanillaModelTorch import QuantizedVAEModelTorch

except ImportError as error:
    logging.error(error)
    sys.exit(-1)


class QuantizedVAEInstance:
    """
    A class that instantiates and manages a Vector Quantized Variational Autoencoder (VQ-VAE) model.
    This implementation provides complete configuration, training, and management capabilities
    for quantized latent space learning tasks within the Synthetic Ocean ecosystem.

    Attributes:
        _quantizedVAE_algorithm (QuantizedVAEAlgorithm): Manages the VQ-VAE training process
        _quantizedVAE_model (QuantizedVAEModel): Contains encoder, decoder and quantization components

    Configuration Parameters (with getters/setters):
        _quantized_vae_number_epochs (int): Number of training epochs
        _quantized_vae_batch_size (int): Size of training batches
        _quantized_vae_latent_dimension (int): Size of the latent space
        _quantized_vae_number_embeddings (int): Number of embeddings in the codebook
        _quantized_vae_activation_function (str): Activation function for hidden layers
        _quantized_vae_initializer_mean (float): Mean for weight initialization
        _quantized_vae_initializer_deviation (float): Std dev for weight initialization
        _quantized_vae_dropout_decay_encoder (float): Encoder dropout rate
        _quantized_vae_dropout_decay_decoder (float): Decoder dropout rate
        _quantized_vae_last_layer_activation (str): Last layer activation function
        _quantized_vae_number_neurons_encoder (list): Encoder layer sizes
        _quantized_vae_number_neurons_decoder (list): Decoder layer sizes
        _quantized_vae_train_variance (float): Training variance parameter
        _quantized_vae_file_name_encoder (str): Encoder model filename
        _quantized_vae_file_name_decoder (str): Decoder model filename
        _quantized_vae_path_output_models (str): Path for saving models
    """

    def __init__(self, arguments):
        """
        Initializes the quantized VAE instance with configuration parameters.

        Args:
            arguments (Namespace): Configuration object containing all required parameters:
                - quantized_vae_number_epochs: Training epochs
                - quantized_vae_batch_size: Batch size
                - quantized_vae_latent_dimension: Latent space size
                - quantized_vae_number_embedding: Codebook size
                - [All other parameters matching attribute names]
        """
        self._quantizedVAE_algorithm = None
        self._quantizedVAE_model = None

        # ** Vector Quantized Variational Autoencoder (VQ-VAE) Configuration Parameters **
        self._quantized_vae_number_epochs = arguments.quantized_vae_number_epochs
        self._quantized_vae_batch_size = arguments.quantized_vae_batch_size
        self._quantized_vae_latent_dimension = arguments.quantized_vae_latent_dimension
        self._quantized_vae_number_embeddings = arguments.quantized_vae_number_embedding
        self._quantized_vae_activation_function = arguments.quantized_vae_activation_function
        self._quantized_vae_initializer_mean = arguments.quantized_vae_initializer_mean
        self._quantized_vae_initializer_deviation = arguments.quantized_vae_mean_distribution
        self._quantized_vae_dropout_decay_encoder = arguments.quantized_vae_dropout_decay_rate_encoder
        self._quantized_vae_dropout_decay_decoder = arguments.quantized_vae_dropout_decay_rate_decoder
        self._quantized_vae_last_layer_activation = arguments.quantized_vae_last_activation_layer
        self._quantized_vae_number_neurons_encoder = arguments.quantized_vae_dense_layer_sizes_encoder
        self._quantized_vae_number_neurons_decoder = arguments.quantized_vae_dense_layer_sizes_decoder
        self._quantized_vae_train_variance = arguments.quantized_vae_train_variance
        self._quantized_vae_file_name_encoder = arguments.quantized_vae_file_name_encoder
        self._quantized_vae_file_name_decoder = arguments.quantized_vae_file_name_decoder
        self._quantized_vae_path_output_models = arguments.quantized_vae_path_output_models

    def _get_quantized_vae(self, input_shape):
        """
        Initialize and configure the Quantized Variational Autoencoder (VQ-VAE) model, including encoder, decoder,
        and quantization components.

        This method sets up a Quantized VAE model by configuring the encoder, decoder, and quantization layers using the
        `QuantizedVAEModel` class and links them with the `QuantizedVAEAlgorithm` class. The model is initialized with
        specified configurations such as latent dimension, number of embeddings, activation functions, dropout rates,
        and layer sizes for both the encoder and decoder.

        Args:
            input_shape (tuple):
                The shape of the input data, which determines the output shape for the models.

        Initializes:
            self._quantized_vae_model:
                An instance of the `QuantizedVAEModel` class, including the encoder, decoder, and quantization setup,
                with configurations for activation functions, layer sizes, dropout rates, and more.
            self._quantized_vae_algorithm:
                An instance of the `QuantizedVAEAlgorithm` class, managing the quantized VAE training process, including
                the encoder, decoder, and quantized models, training variance, latent dimension, number of embeddings,
                and model file paths.
        """

        # Quantized VAE Model setup for Encoder, Decoder, and Quantization
        self._quantized_vae_model = QuantizedVAEModel(
            latent_dimension=self._quantized_vae_latent_dimension,
            number_embeddings=self._quantized_vae_number_embeddings,
            output_shape=input_shape,
            activation_function=self._quantized_vae_activation_function,
            initializer_mean=self._quantized_vae_initializer_mean,
            initializer_deviation=self._quantized_vae_initializer_deviation,
            dropout_decay_encoder=self._quantized_vae_dropout_decay_encoder,
            dropout_decay_decoder=self._quantized_vae_dropout_decay_decoder,
            last_layer_activation=self._quantized_vae_last_layer_activation,
            number_neurons_encoder=self._quantized_vae_number_neurons_encoder,
            number_neurons_decoder=self._quantized_vae_number_neurons_decoder,
            dataset_type=numpy.float32,
            number_samples_per_class=self._number_samples_per_class
        )

        quantized_model = self._quantized_vae_model.get_quantized_model()

        # Quantized VAE Algorithm setup for training and model operations
        self._quantized_vae_algorithm = QuantizedVAEAlgorithm(
            encoder_model=self._quantized_vae_model.get_encoder(),
            decoder_model=self._quantized_vae_model.get_decoder(),
            quantized_vae_model=quantized_model,
            train_variance=self._quantized_vae_train_variance,
            latent_dimension=self._quantized_vae_latent_dimension,
            number_embeddings=self._quantized_vae_number_embeddings,
            file_name_encoder=self._quantized_vae_file_name_encoder,
            file_name_decoder=self._quantized_vae_file_name_decoder,
            models_saved_path=self._quantized_vae_path_output_models
        )

    def _training_quantized_VAE_model(self, input_shape, arguments, x_real_samples, y_real_samples):
        """
        Executes the complete quantized VAE training process.

        Args:
            input_shape (tuple): Shape of input data samples
            arguments (Namespace): Configuration parameters
            x_real_samples (ndarray): Training data samples
            y_real_samples (ndarray): Corresponding class labels

        Process:
            1. Initializes model architecture
            2. Configures optimizer and loss functions
            3. Sets up training callbacks
            4. Executes quantized VAE training
            5. Manages model saving and monitoring
        """
        # Initialize the variational autoencoder model
        self._get_quantized_vae(input_shape)

        # Print the model summaries for the encoder and decoder
        print("\nEncoder Model:")
        print(self._quantized_vae_model.get_encoder())
        print("\nDecoder Model:")
        print(self._quantized_vae_model.get_decoder())

        # Compile the variational autoencoder algorithm with the specified optimizer
        optimizer = torch.optim.Adam(
            self._quantized_vae_algorithm._quantized_vae_model.parameters(),
            lr=0.0001
        )
        self._quantized_vae_algorithm.compile(optimizer=optimizer)

        callbacks_list = [self._callback_resources_monitor, self._callback_model_monitor]

        if arguments.use_early_stop:
            callbacks_list.append(self._callback_early_stop)

        # Convert labels to one-hot encoding
        num_classes = self._number_samples_per_class["number_classes"]
        y_one_hot = numpy.zeros((len(y_real_samples), num_classes))
        y_one_hot[numpy.arange(len(y_real_samples)), y_real_samples.astype(int)] = 1

        # Fit the variational autoencoder model
        self._quantized_vae_algorithm.fit(
            (x_real_samples, y_one_hot),
            x_real_samples,
            epochs=self._quantized_vae_number_epochs,
            batch_size=self._quantized_vae_batch_size,
            callbacks=callbacks_list
        )

    @property
    def quantized_vae_number_epochs(self):
        return self._quantized_vae_number_epochs

    @quantized_vae_number_epochs.setter
    def quantized_vae_number_epochs(self, value):
        self._quantized_vae_number_epochs = value

    @property
    def quantized_vae_batch_size(self):
        return self._quantized_vae_batch_size

    @quantized_vae_batch_size.setter
    def quantized_vae_batch_size(self, value):
        self._quantized_vae_batch_size = value

    @property
    def quantized_vae_latent_dimension(self):
        return self._quantized_vae_latent_dimension

    @quantized_vae_latent_dimension.setter
    def quantized_vae_latent_dimension(self, value):
        self._quantized_vae_latent_dimension = value

    @property
    def quantized_vae_number_embeddings(self):
        return self._quantized_vae_number_embeddings

    @quantized_vae_number_embeddings.setter
    def quantized_vae_number_embeddings(self, value):
        self._quantized_vae_number_embeddings = value

    @property
    def quantized_vae_activation_function(self):
        return self._quantized_vae_activation_function

    @quantized_vae_activation_function.setter
    def quantized_vae_activation_function(self, value):
        self._quantized_vae_activation_function = value

    @property
    def quantized_vae_initializer_mean(self):
        return self._quantized_vae_initializer_mean

    @quantized_vae_initializer_mean.setter
    def quantized_vae_initializer_mean(self, value):
        self._quantized_vae_initializer_mean = value

    @property
    def quantized_vae_initializer_deviation(self):
        return self._quantized_vae_initializer_deviation

    @quantized_vae_initializer_deviation.setter
    def quantized_vae_initializer_deviation(self, value):
        self._quantized_vae_initializer_deviation = value

    @property
    def quantized_vae_dropout_decay_encoder(self):
        return self._quantized_vae_dropout_decay_encoder

    @quantized_vae_dropout_decay_encoder.setter
    def quantized_vae_dropout_decay_encoder(self, value):
        self._quantized_vae_dropout_decay_encoder = value

    @property
    def quantized_vae_dropout_decay_decoder(self):
        return self._quantized_vae_dropout_decay_decoder

    @quantized_vae_dropout_decay_decoder.setter
    def quantized_vae_dropout_decay_decoder(self, value):
        self._quantized_vae_dropout_decay_decoder = value

    @property
    def quantized_vae_last_layer_activation(self):
        return self._quantized_vae_last_layer_activation

    @quantized_vae_last_layer_activation.setter
    def quantized_vae_last_layer_activation(self, value):
        self._quantized_vae_last_layer_activation = value

    @property
    def quantized_vae_number_neurons_encoder(self):
        return self._quantized_vae_number_neurons_encoder

    @quantized_vae_number_neurons_encoder.setter
    def quantized_vae_number_neurons_encoder(self, value):
        self._quantized_vae_number_neurons_encoder = value

    @property
    def quantized_vae_number_neurons_decoder(self):
        return self._quantized_vae_number_neurons_decoder

    @quantized_vae_number_neurons_decoder.setter
    def quantized_vae_number_neurons_decoder(self, value):
        self._quantized_vae_number_neurons_decoder = value

    @property
    def quantized_vae_train_variance(self):
        return self._quantized_vae_train_variance

    @quantized_vae_train_variance.setter
    def quantized_vae_train_variance(self, value):
        self._quantized_vae_train_variance = value

    @property
    def quantized_vae_file_name_encoder(self):
        return self._quantized_vae_file_name_encoder

    @quantized_vae_file_name_encoder.setter
    def quantized_vae_file_name_encoder(self, value):
        self._quantized_vae_file_name_encoder = value

    @property
    def quantized_vae_file_name_decoder(self):
        return self._quantized_vae_file_name_decoder

    @quantized_vae_file_name_decoder.setter
    def quantized_vae_file_name_decoder(self, value):
        self._quantized_vae_file_name_decoder = value

    @property
    def quantized_vae_path_output_models(self):
        return self._quantized_vae_path_output_models

    @quantized_vae_path_output_models.setter
    def quantized_vae_path_output_models(self, value):
        self._quantized_vae_path_output_models = value