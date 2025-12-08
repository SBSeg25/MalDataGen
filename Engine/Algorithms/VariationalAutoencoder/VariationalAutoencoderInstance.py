#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

from Engine.Algorithms.VariationalAutoencoder.AlgorithmVariationalAutoencoder import VariationalAlgorithm
from Engine.Models.VariationalAutoencoder.Tensorflow.VariationalAutoencoderModelTensorflow import VariationalModel

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

except ImportError as error:
    logging.error(error)
    sys.exit(-1)

class VariationalAutoencoderInstance:
    """
    A class that implements a Variational Autoencoder (VAE) for probabilistic generative modeling.
    This implementation combines an encoder-decoder architecture with variational inference to learn
    a compressed latent representation of input data while enabling efficient sampling and generation.

    Key Components:
    - Encoder network that maps inputs to a latent distribution
    - Decoder network that reconstructs inputs from latent samples
    - KL divergence regularization for latent space structure
    - Flexible architecture configuration via arguments
    - Complete training pipeline with monitoring

    Attributes:
        _variation_model: Contains the encoder and decoder networks
        _variational_algorithm: Manages the VAE training process

        # VAE Architecture Parameters
        _variational_autoencoder_latent_dimension: Dimensionality of latent space
        _variational_autoencoder_training_algorithm: Training methodology
        _variational_autoencoder_activation_function: Activation for hidden layers
        _variational_autoencoder_dropout_decay_rate_encoder: Dropout rate for encoder
        _variational_autoencoder_dropout_decay_rate_decoder: Dropout rate for decoder
        _variational_autoencoder_dense_layer_sizes_encoder: Layer sizes for encoder
        _variational_autoencoder_dense_layer_sizes_decoder: Layer sizes for decoder
        _variational_autoencoder_batch_size: Training batch size
        _variational_autoencoder_number_classes: Number of output classes
        _variational_autoencoder_loss_function: Composite loss (reconstruction + KL)
        _variational_autoencoder_momentum: Optimizer momentum parameter
        _variational_autoencoder_number_epochs: Training epochs
        _variational_autoencoder_last_activation_layer: Output layer activation
        _variational_autoencoder_initializer_mean: Weight init mean
        _variational_autoencoder_initializer_deviation: Weight init std dev

        # Latent Space Parameters
        _variational_autoencoder_mean_distribution: Distribution type for latent mean
        _variational_autoencoder_stander_deviation: Std dev for latent distribution

        # Model Persistence
        _variational_autoencoder_file_name_encoder: Encoder save filename
        _variational_autoencoder_file_name_decoder: Decoder save filename
        _variational_autoencoder_path_output_models: Model save directory
    """

    def __init__(self, arguments):
        """
        Initializes the VAE instance with configuration parameters.

        Args:
            arguments (Namespace): Configuration object containing:
                - Encoder/decoder architecture parameters
                - Training hyperparameters
                - Latent space configuration
                - Model persistence settings
        """
        self._variation_model = None
        self._variational_algorithm = None

        # ** Variational Autoencoder (VAE) Configuration Parameters **
        self._variational_autoencoder_latent_dimension = arguments.variational_autoencoder_latent_dimension
        self._variational_autoencoder_training_algorithm = arguments.variational_autoencoder_training_algorithm
        self._variational_autoencoder_activation_function = arguments.variational_autoencoder_activation_function
        self._variational_autoencoder_dropout_decay_rate_encoder = arguments.variational_autoencoder_dropout_decay_rate_encoder
        self._variational_autoencoder_dropout_decay_rate_decoder = arguments.variational_autoencoder_dropout_decay_rate_decoder
        self._variational_autoencoder_dense_layer_sizes_encoder = arguments.variational_autoencoder_dense_layer_sizes_encoder
        self._variational_autoencoder_dense_layer_sizes_decoder = arguments.variational_autoencoder_dense_layer_sizes_decoder
        self._variational_autoencoder_batch_size = arguments.variational_autoencoder_batch_size
        self._variational_autoencoder_number_classes = arguments.variational_autoencoder_number_classes
        self._variational_autoencoder_loss_function = arguments.variational_autoencoder_loss_function
        self._variational_autoencoder_momentum = arguments.variational_autoencoder_momentum
        self._variational_autoencoder_number_epochs = arguments.variational_autoencoder_number_epochs
        self._variational_autoencoder_last_activation_layer = arguments.variational_autoencoder_last_activation_layer

        # Latent Space Parameters
        self._variational_autoencoder_initializer_mean = arguments.variational_autoencoder_initializer_mean
        self._variational_autoencoder_initializer_deviation = arguments.variational_autoencoder_initializer_deviation
        self._variational_autoencoder_mean_distribution = arguments.variational_autoencoder_mean_distribution
        self._variational_autoencoder_stander_deviation = arguments.variational_autoencoder_stander_deviation

        # Model Persistence
        self._variational_autoencoder_file_name_encoder = arguments.variational_autoencoder_file_name_encoder
        self._variational_autoencoder_file_name_decoder = arguments.variational_autoencoder_file_name_decoder
        self._variational_autoencoder_path_output_models = arguments.variational_autoencoder_path_output_models


    def _get_variational_autoencoder(self, input_shape):
        """
        Initializes and sets up a Variational Autoencoder (VAE) model.

        This method creates an instance of a Variational Autoencoder (VAE) by configuring its encoder and decoder
        components. It uses a custom `VariationalModel` class to define and manage these components, and a `VariationalAlgorithm`
        to handle the training and operations of the VAE model. The VAE is designed for probabilistic inference and data generation.

        Args:
            input_shape (tuple):
                The shape of the input data, which is used to define the output shape of the model.

        Initializes:
            self._variation_model:
                An instance of the `VariationalModel` class that includes the encoder and decoder setup with
                configurations like latent dimension, activation functions, dropout rates, and neural network sizes.
            self._variational_algorithm:
                An instance of the `VariationalAlgorithm` class that handles the VAE's training process, loss function,
                and model parameters, including latent mean and standard deviation distributions.

        """

        # Variational Model setup for the VAE's encoder and decoder
        self._variation_model = VariationalModel(latent_dimension=self._variational_autoencoder_latent_dimension,
                                                 output_shape=input_shape,
                                                 activation_function=self._variational_autoencoder_activation_function,
                                                 initializer_mean=self._variational_autoencoder_initializer_mean,
                                                 initializer_deviation=self._variational_autoencoder_initializer_deviation,
                                                 dropout_decay_encoder=self._variational_autoencoder_dropout_decay_rate_encoder,
                                                 dropout_decay_decoder=self._variational_autoencoder_dropout_decay_rate_decoder,
                                                 last_layer_activation=self._variational_autoencoder_last_activation_layer,
                                                 number_neurons_encoder=self._variational_autoencoder_dense_layer_sizes_encoder,
                                                 number_neurons_decoder=self._variational_autoencoder_dense_layer_sizes_decoder,
                                                 dataset_type=numpy.float32, number_samples_per_class = self._number_samples_per_class)

        # Variational Algorithm setup for training and model operations
        self._variational_algorithm = VariationalAlgorithm(encoder_model=self._variation_model.get_encoder(),
                                                           decoder_model=self._variation_model.get_decoder(),
                                                           loss_function=self._variational_autoencoder_loss_function,
                                                           latent_dimension=self._variational_autoencoder_latent_dimension,
                                                           decoder_latent_dimension = self._variational_autoencoder_latent_dimension,
                                                           latent_mean_distribution=self._variational_autoencoder_mean_distribution,
                                                           latent_stander_deviation=self._variational_autoencoder_stander_deviation,
                                                           file_name_encoder=self._variational_autoencoder_file_name_encoder,
                                                           file_name_decoder=self._variational_autoencoder_file_name_decoder,
                                                           models_saved_path=self._variational_autoencoder_path_output_models)


    def _training_variational_autoencoder_model(self, input_shape, arguments, x_real_samples, y_real_samples):
        """
        Executes the complete VAE training pipeline.

        The training process:
        1. Initializes encoder and decoder models
        2. Configures the composite loss (reconstruction + KL divergence)
        3. Sets up optimizer with specified parameters
        4. Trains using minibatch gradient descent
        5. Manages training callbacks and monitoring

        Args:
            input_shape (tuple): Input data dimensions
            arguments (Namespace): Training configuration parameters
            x_real_samples (ndarray): Training dataset samples
            y_real_samples (ndarray): Corresponding sample labels
        """
        # Initialize the variational autoencoder model
        self._get_variational_autoencoder(input_shape)

        # Print the model summaries for the encoder and decoder
        self._variation_model.get_encoder().summary()
        self._variation_model.get_decoder().summary()

        variational_optimizer = keras.optimizers.Adam()
        # Compile the variational autoencoder algorithm with the specified loss function
        self._variational_algorithm.compile(loss=self._variational_autoencoder_loss_function,
                                            optimizer=variational_optimizer)

        # callbacks_list = [self._callback_resources_monitor, self._callback_model_monitor]
        callbacks_list = [self._callback_model_monitor]

        if arguments.use_early_stop:
            callbacks_list.append(self._callback_early_stop)

        # Fit the variational autoencoder model
        self._variational_algorithm.fit((x_real_samples, to_categorical(y_real_samples,
                                           num_classes=self._number_samples_per_class["number_classes"])),
                                        x_real_samples, epochs=self._variational_autoencoder_number_epochs,
                                        batch_size=self._variational_autoencoder_batch_size,
                                        callbacks=callbacks_list)


    # Getter and setter for variational_autoencoder_latent_dimension
    @property
    def variational_autoencoder_latent_dimension(self):
        return self._variational_autoencoder_latent_dimension

    @variational_autoencoder_latent_dimension.setter
    def variational_autoencoder_latent_dimension(self, value):
        self._variational_autoencoder_latent_dimension = value

    # Getter and setter for variational_autoencoder_training_algorithm
    @property
    def variational_autoencoder_training_algorithm(self):
        return self._variational_autoencoder_training_algorithm

    @variational_autoencoder_training_algorithm.setter
    def variational_autoencoder_training_algorithm(self, value):
        self._variational_autoencoder_training_algorithm = value

    # Getter and setter for variational_autoencoder_activation_function
    @property
    def variational_autoencoder_activation_function(self):
        return self._variational_autoencoder_activation_function

    @variational_autoencoder_activation_function.setter
    def variational_autoencoder_activation_function(self, value):
        self._variational_autoencoder_activation_function = value

    # Getter and setter for variational_autoencoder_dropout_decay_rate_encoder
    @property
    def variational_autoencoder_dropout_decay_rate_encoder(self):
        return self._variational_autoencoder_dropout_decay_rate_encoder

    @variational_autoencoder_dropout_decay_rate_encoder.setter
    def variational_autoencoder_dropout_decay_rate_encoder(self, value):
        self._variational_autoencoder_dropout_decay_rate_encoder = value

    # Getter and setter for variational_autoencoder_dropout_decay_rate_decoder
    @property
    def variational_autoencoder_dropout_decay_rate_decoder(self):
        return self._variational_autoencoder_dropout_decay_rate_decoder

    @variational_autoencoder_dropout_decay_rate_decoder.setter
    def variational_autoencoder_dropout_decay_rate_decoder(self, value):
        self._variational_autoencoder_dropout_decay_rate_decoder = value

    # Getter and setter for variational_autoencoder_dense_layer_sizes_encoder
    @property
    def variational_autoencoder_dense_layer_sizes_encoder(self):
        return self._variational_autoencoder_dense_layer_sizes_encoder

    @variational_autoencoder_dense_layer_sizes_encoder.setter
    def variational_autoencoder_dense_layer_sizes_encoder(self, value):
        self._variational_autoencoder_dense_layer_sizes_encoder = value

    # Getter and setter for variational_autoencoder_dense_layer_sizes_decoder
    @property
    def variational_autoencoder_dense_layer_sizes_decoder(self):
        return self._variational_autoencoder_dense_layer_sizes_decoder

    @variational_autoencoder_dense_layer_sizes_decoder.setter
    def variational_autoencoder_dense_layer_sizes_decoder(self, value):
        self._variational_autoencoder_dense_layer_sizes_decoder = value

    # Getter and setter for variational_autoencoder_batch_size
    @property
    def variational_autoencoder_batch_size(self):
        return self._variational_autoencoder_batch_size

    @variational_autoencoder_batch_size.setter
    def variational_autoencoder_batch_size(self, value):
        self._variational_autoencoder_batch_size = value

    # Getter and setter for variational_autoencoder_number_classes
    @property
    def variational_autoencoder_number_classes(self):
        return self._variational_autoencoder_number_classes

    @variational_autoencoder_number_classes.setter
    def variational_autoencoder_number_classes(self, value):
        self._variational_autoencoder_number_classes = value

    # Getter and setter for variational_autoencoder_loss_function
    @property
    def variational_autoencoder_loss_function(self):
        return self._variational_autoencoder_loss_function

    @variational_autoencoder_loss_function.setter
    def variational_autoencoder_loss_function(self, value):
        self._variational_autoencoder_loss_function = value

    # Getter and setter for variational_autoencoder_momentum
    @property
    def variational_autoencoder_momentum(self):
        return self._variational_autoencoder_momentum

    @variational_autoencoder_momentum.setter
    def variational_autoencoder_momentum(self, value):
        self._variational_autoencoder_momentum = value

    # Getter and setter for variational_autoencoder_last_activation_layer
    @property
    def variational_autoencoder_last_activation_layer(self):
        return self._variational_autoencoder_last_activation_layer

    @variational_autoencoder_last_activation_layer.setter
    def variational_autoencoder_last_activation_layer(self, value):
        self._variational_autoencoder_last_activation_layer = value

    # Getter and setter for variational_autoencoder_initializer_mean
    @property
    def variational_autoencoder_initializer_mean(self):
        return self._variational_autoencoder_initializer_mean

    @variational_autoencoder_initializer_mean.setter
    def variational_autoencoder_initializer_mean(self, value):
        self._variational_autoencoder_initializer_mean = value

    # Getter and setter for variational_autoencoder_initializer_deviation
    @property
    def variational_autoencoder_initializer_deviation(self):
        return self._variational_autoencoder_initializer_deviation

    @variational_autoencoder_initializer_deviation.setter
    def variational_autoencoder_initializer_deviation(self, value):
        self._variational_autoencoder_initializer_deviation = value

    # Getter and setter for variational_autoencoder_mean_distribution
    @property
    def variational_autoencoder_mean_distribution(self):
        return self._variational_autoencoder_mean_distribution

    @variational_autoencoder_mean_distribution.setter
    def variational_autoencoder_mean_distribution(self, value):
        self._variational_autoencoder_mean_distribution = value

    # Getter and setter for variational_autoencoder_stander_deviation
    @property
    def variational_autoencoder_stander_deviation(self):
        return self._variational_autoencoder_stander_deviation

    @variational_autoencoder_stander_deviation.setter
    def variational_autoencoder_stander_deviation(self, value):
        self._variational_autoencoder_stander_deviation = value

    # Getter and setter for variational_autoencoder_file_name_encoder
    @property
    def variational_autoencoder_file_name_encoder(self):
        return self._variational_autoencoder_file_name_encoder

    @variational_autoencoder_file_name_encoder.setter
    def variational_autoencoder_file_name_encoder(self, value):
        self._variational_autoencoder_file_name_encoder = value

    # Getter and setter for variational_autoencoder_file_name_decoder
    @property
    def variational_autoencoder_file_name_decoder(self):
        return self._variational_autoencoder_file_name_decoder

    @variational_autoencoder_file_name_decoder.setter
    def variational_autoencoder_file_name_decoder(self, value):
        self._variational_autoencoder_file_name_decoder = value

    # Getter and setter for variational_autoencoder_path_output_models
    @property
    def variational_autoencoder_path_output_models(self):
        return self._variational_autoencoder_path_output_models

    @variational_autoencoder_path_output_models.setter
    def variational_autoencoder_path_output_models(self, value):
        self._variational_autoencoder_path_output_models = value
