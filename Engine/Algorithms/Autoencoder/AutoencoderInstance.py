#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

from Engine.Algorithms.Adversarial.AdversarialInstance import AdversarialInstance
from Engine.Algorithms.Autoencoder.AutoencoderAlgorithm import AutoencoderAlgorithm
from Engine.Models.Autoencoder.ModelAutoencoder import AutoencoderModel

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

    import keras
    import numpy

    import logging
    import tensorflow

    from tensorflow.keras.optimizers import Adam

    from tensorflow.keras.utils import to_categorical

    from tensorflow.python.keras.losses import MeanSquaredError
    from Engine.Callbacks.CallbackEarlyStop import EarlyStopping

    from tensorflow.python.keras.losses import BinaryCrossentropy



except ImportError as error:
    logging.error(error)
    sys.exit(-1)

class AutoencoderInstance:
    """
    A class that instantiates and manages an Autoencoder model.
    This implementation provides complete configuration, training, and management capabilities
    for autoencoder-based learning tasks within the Synthetic Ocean ecosystem.

    Attributes:
        _autoencoder_model (AutoencoderModel): Contains encoder and decoder components
        _autoencoder_algorithm (AutoencoderAlgorithm): Manages the autoencoder training process

    Configuration Parameters (with getters/setters):
        _autoencoder_latent_dimension (int): Size of the latent space
        _autoencoder_training_algorithm (str): Training algorithm specification
        _autoencoder_activation_function (str): Activation function for hidden layers
        _autoencoder_dropout_decay_rate_encoder (float): Encoder dropout rate
        _autoencoder_dropout_decay_rate_decoder (float): Decoder dropout rate
        _autoencoder_dense_layer_sizes_encoder (list): Encoder layer sizes
        _autoencoder_dense_layer_sizes_decoder (list): Decoder layer sizes
        _autoencoder_batch_size (int): Size of training batches
        _autoencoder_number_epochs (int): Number of training epochs
        _autoencoder_number_classes (int): Number of output classes
        _autoencoder_loss_function (str): Loss function for reconstruction
        _autoencoder_momentum (float): Momentum parameter for optimization
        _autoencoder_last_activation_layer (str): Last layer activation function
        _autoencoder_initializer_mean (float): Mean for weight initialization
        _autoencoder_initializer_deviation (float): Std dev for weight initialization
        _autoencoder_latent_mean_distribution (float): Latent space mean
        _autoencoder_latent_stander_deviation (float): Latent space std dev
        _autoencoder_file_name_encoder (str): Encoder model filename
        _autoencoder_file_name_decoder (str): Decoder model filename
        _autoencoder_path_output_models (str): Path for saving models
    """
    def __init__(self, arguments):
        """
        Initializes the autoencoder instance with configuration parameters.

        Args:
            arguments (Namespace): Configuration object containing all required parameters:
                - autoencoder_latent_dimension: Latent space size
                - autoencoder_training_algorithm: Training algorithm
                - autoencoder_activation_function: Activation function
                - [All other parameters matching attribute names]
        """
        self._autoencoder_algorithm = None
        self._autoencoder_model = None

        # ** Autoencoder Model Configuration Parameters **
        self._autoencoder_latent_dimension = arguments.autoencoder_latent_dimension
        self._autoencoder_training_algorithm = arguments.autoencoder_training_algorithm
        self._autoencoder_activation_function = arguments.autoencoder_activation_function
        self._autoencoder_dropout_decay_rate_encoder = arguments.autoencoder_dropout_decay_rate_encoder
        self._autoencoder_dropout_decay_rate_decoder = arguments.autoencoder_dropout_decay_rate_decoder
        self._autoencoder_dense_layer_sizes_encoder = arguments.autoencoder_dense_layer_sizes_encoder
        self._autoencoder_dense_layer_sizes_decoder = arguments.autoencoder_dense_layer_sizes_decoder
        self._autoencoder_batch_size = arguments.autoencoder_batch_size
        self._autoencoder_number_epochs = arguments.autoencoder_number_epochs
        self._autoencoder_number_classes = arguments.autoencoder_number_classes
        self._autoencoder_loss_function = arguments.autoencoder_loss_function
        self._autoencoder_momentum = arguments.autoencoder_momentum
        self._autoencoder_last_activation_layer = arguments.autoencoder_last_activation_layer
        self._autoencoder_initializer_mean = arguments.autoencoder_initializer_mean
        self._autoencoder_initializer_deviation = arguments.autoencoder_initializer_deviation
        self._autoencoder_latent_mean_distribution = arguments.autoencoder_latent_mean_distribution
        self._autoencoder_latent_stander_deviation = arguments.autoencoder_latent_stander_deviation
        self._autoencoder_file_name_encoder = arguments.autoencoder_file_name_encoder
        self._autoencoder_file_name_decoder = arguments.autoencoder_file_name_decoder
        self._autoencoder_path_output_models = arguments.autoencoder_path_output_models

    def _get_autoencoder(self, input_shape):
        """
        Initialize and configure the Autoencoder model, including encoder and decoder components.

        This method sets up an Autoencoder model by configuring both the encoder and decoder using the `AutoencoderModel`
        class and links them with the `AutoencoderAlgorithm` class. The model is initialized with specified configurations
        such as latent dimension, activation functions, dropout rates, and layer sizes for both the encoder and decoder.

        Args:
            input_shape (tuple):
                The shape of the input data, which determines the output shape for the models.

        Initializes:
            self._autoencoder_model:
                An instance of the `AutoencoderModel` class, including the encoder and decoder setup, with
                configurations for activation functions, layer sizes, dropout rates, and more.
            self._autoencoder_algorithm:
                An instance of the `AutoencoderAlgorithm` class, managing the autoencoder training process, including
                the encoder and decoder models, loss function, latent distributions, and model file paths.

        """

        # Autoencoder Model setup for Encoder and Decoder
        self._autoencoder_model = AutoencoderModel(latent_dimension=self._autoencoder_latent_dimension,
                                                   output_shape=input_shape,
                                                   activation_function=self._autoencoder_activation_function,
                                                   initializer_mean=self._autoencoder_initializer_mean,
                                                   initializer_deviation=self._autoencoder_initializer_deviation,
                                                   dropout_decay_encoder=self._autoencoder_dropout_decay_rate_encoder,
                                                   dropout_decay_decoder=self._autoencoder_dropout_decay_rate_decoder,
                                                   last_layer_activation=self._autoencoder_last_activation_layer,
                                                   number_neurons_encoder=self._autoencoder_dense_layer_sizes_encoder,
                                                   number_neurons_decoder=self._autoencoder_dense_layer_sizes_decoder,
                                                   dataset_type=numpy.float32,
                                                   number_samples_per_class = self._number_samples_per_class)

        # Autoencoder Algorithm setup for training and model operations
        self._autoencoder_algorithm = AutoencoderAlgorithm(encoder_model=self._autoencoder_model.get_encoder(input_shape),
                                                           decoder_model=self._autoencoder_model.get_decoder(input_shape),
                                                           loss_function=self._autoencoder_loss_function,
                                                           file_name_encoder=self._autoencoder_file_name_encoder,
                                                           file_name_decoder=self._autoencoder_file_name_decoder,
                                                           models_saved_path=self._autoencoder_path_output_models,
                                                           latent_mean_distribution=self._autoencoder_latent_mean_distribution,
                                                           latent_stander_deviation=self._autoencoder_latent_stander_deviation,
                                                           latent_dimension=self._autoencoder_latent_dimension)

    def _training_autoencoder_model(self, input_shape, arguments, x_real_samples, y_real_samples):
        """
        Executes the complete autoencoder training process.

        Args:
            input_shape (tuple): Shape of input data samples
            arguments (Namespace): Configuration parameters
            x_real_samples (ndarray): Training data samples
            y_real_samples (ndarray): Corresponding class labels

        Process:
            1. Initializes model architecture
            2. Configures loss function
            3. Sets up training callbacks
            4. Executes autoencoder training
            5. Manages model saving and monitoring
        """
        # Initialize the autoencoder model
        self._get_autoencoder(input_shape)

        # Print the model summaries for the encoder and decoder
        self._autoencoder_model.get_encoder(input_shape).summary()
        self._autoencoder_model.get_decoder(input_shape).summary()

        # Compile the autoencoder algorithm with the specified loss function
        self._autoencoder_algorithm.compile(loss=arguments.autoencoder_loss_function)

        # callbacks_list = [self._callback_resources_monitor, self._callback_model_monitor]
        callbacks_list = [self._callback_model_monitor]

        if arguments.use_early_stop:
            callbacks_list.append(self._callback_early_stop)

        # Fit the autoencoder model
        self._autoencoder_algorithm.fit((
            x_real_samples, to_categorical(y_real_samples,
                                           num_classes=self._number_samples_per_class["number_classes"])),
            x_real_samples, epochs=self._autoencoder_number_epochs, batch_size=self._autoencoder_batch_size,
            callbacks=callbacks_list)

    # Getter and setter for autoencoder_latent_dimension
    @property
    def autoencoder_latent_dimension(self):
        return self._autoencoder_latent_dimension

    @autoencoder_latent_dimension.setter
    def autoencoder_latent_dimension(self, value):
        self._autoencoder_latent_dimension = value

    # Getter and setter for autoencoder_training_algorithm
    @property
    def autoencoder_training_algorithm(self):
        return self._autoencoder_training_algorithm

    @autoencoder_training_algorithm.setter
    def autoencoder_training_algorithm(self, value):
        self._autoencoder_training_algorithm = value

    # Getter and setter for autoencoder_activation_function
    @property
    def autoencoder_activation_function(self):
        return self._autoencoder_activation_function

    @autoencoder_activation_function.setter
    def autoencoder_activation_function(self, value):
        self._autoencoder_activation_function = value

    # Getter and setter for autoencoder_dropout_decay_rate_encoder
    @property
    def autoencoder_dropout_decay_rate_encoder(self):
        return self._autoencoder_dropout_decay_rate_encoder

    @autoencoder_dropout_decay_rate_encoder.setter
    def autoencoder_dropout_decay_rate_encoder(self, value):
        self._autoencoder_dropout_decay_rate_encoder = value

    # Getter and setter for autoencoder_dropout_decay_rate_decoder
    @property
    def autoencoder_dropout_decay_rate_decoder(self):
        return self._autoencoder_dropout_decay_rate_decoder

    @autoencoder_dropout_decay_rate_decoder.setter
    def autoencoder_dropout_decay_rate_decoder(self, value):
        self._autoencoder_dropout_decay_rate_decoder = value

    # Getter and setter for autoencoder_dense_layer_sizes_encoder
    @property
    def autoencoder_dense_layer_sizes_encoder(self):
        return self._autoencoder_dense_layer_sizes_encoder

    @autoencoder_dense_layer_sizes_encoder.setter
    def autoencoder_dense_layer_sizes_encoder(self, value):
        self._autoencoder_dense_layer_sizes_encoder = value

    # Getter and setter for autoencoder_dense_layer_sizes_decoder
    @property
    def autoencoder_dense_layer_sizes_decoder(self):
        return self._autoencoder_dense_layer_sizes_decoder

    @autoencoder_dense_layer_sizes_decoder.setter
    def autoencoder_dense_layer_sizes_decoder(self, value):
        self._autoencoder_dense_layer_sizes_decoder = value

    # Getter and setter for autoencoder_batch_size
    @property
    def autoencoder_batch_size(self):
        return self._autoencoder_batch_size

    @autoencoder_batch_size.setter
    def autoencoder_batch_size(self, value):
        self._autoencoder_batch_size = value

    # Getter and setter for autoencoder_number_classes
    @property
    def autoencoder_number_classes(self):
        return self._autoencoder_number_classes

    @autoencoder_number_classes.setter
    def autoencoder_number_classes(self, value):
        self._autoencoder_number_classes = value

    # Getter and setter for autoencoder_loss_function
    @property
    def autoencoder_loss_function(self):
        return self._autoencoder_loss_function

    @autoencoder_loss_function.setter
    def autoencoder_loss_function(self, value):
        self._autoencoder_loss_function = value

    # Getter and setter for autoencoder_momentum
    @property
    def autoencoder_momentum(self):
        return self._autoencoder_momentum

    @autoencoder_momentum.setter
    def autoencoder_momentum(self, value):
        self._autoencoder_momentum = value

    # Getter and setter for autoencoder_last_activation_layer
    @property
    def autoencoder_last_activation_layer(self):
        return self._autoencoder_last_activation_layer

    @autoencoder_last_activation_layer.setter
    def autoencoder_last_activation_layer(self, value):
        self._autoencoder_last_activation_layer = value

    # Getter and setter for autoencoder_initializer_mean
    @property
    def autoencoder_initializer_mean(self):
        return self._autoencoder_initializer_mean

    @autoencoder_initializer_mean.setter
    def autoencoder_initializer_mean(self, value):
        self._autoencoder_initializer_mean = value

    # Getter and setter for autoencoder_initializer_deviation
    @property
    def autoencoder_initializer_deviation(self):
        return self._autoencoder_initializer_deviation

    @autoencoder_initializer_deviation.setter
    def autoencoder_initializer_deviation(self, value):
        self._autoencoder_initializer_deviation = value

    # Getter and setter for autoencoder_latent_mean_distribution
    @property
    def autoencoder_latent_mean_distribution(self):
        return self._autoencoder_latent_mean_distribution

    @autoencoder_latent_mean_distribution.setter
    def autoencoder_latent_mean_distribution(self, value):
        self._autoencoder_latent_mean_distribution = value

    # Getter and setter for autoencoder_latent_stander_deviation
    @property
    def autoencoder_latent_stander_deviation(self):
        return self._autoencoder_latent_stander_deviation

    @autoencoder_latent_stander_deviation.setter
    def autoencoder_latent_stander_deviation(self, value):
        self._autoencoder_latent_stander_deviation = value

    # Getter and setter for autoencoder_file_name_encoder
    @property
    def autoencoder_file_name_encoder(self):
        return self._autoencoder_file_name_encoder

    @autoencoder_file_name_encoder.setter
    def autoencoder_file_name_encoder(self, value):
        self._autoencoder_file_name_encoder = value

    # Getter and setter for autoencoder_file_name_decoder
    @property
    def autoencoder_file_name_decoder(self):
        return self._autoencoder_file_name_decoder

    @autoencoder_file_name_decoder.setter
    def autoencoder_file_name_decoder(self, value):
        self._autoencoder_file_name_decoder = value

    # Getter and setter for autoencoder_path_output_models
    @property
    def autoencoder_path_output_models(self):
        return self._autoencoder_path_output_models

    @autoencoder_path_output_models.setter
    def autoencoder_path_output_models(self, value):
        self._autoencoder_path_output_models = value
