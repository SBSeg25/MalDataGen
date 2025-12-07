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

    import keras
    import numpy

    import logging
    import tensorflow

    from tensorflow.keras.optimizers import Adam

    from tensorflow.keras.utils import to_categorical

    from tensorflow.python.keras.losses import MeanSquaredError

    from tensorflow.python.keras.losses import BinaryCrossentropy

    from Engine.Algorithms.Adversarial.AdversarialAlgorithm import AdversarialAlgorithm
    from Engine.Models.Adversarial.AdversarialModel import AdversarialModel

except ImportError as error:
    logging.error(error)
    sys.exit(-1)



class AdversarialInstance:
    """
    A class that instantiates and manages a Conditional Generative Adversarial Network (CGAN) model.
    This implementation provides complete configuration, training, and management capabilities
    for adversarial learning tasks within the Synthetic Ocean ecosystem.

    Attributes:
        _adversarial_algorithm (AdversarialAlgorithm): Manages the adversarial training process
        _adversarial_model (AdversarialModel): Contains generator and discriminator components

    Configuration Parameters (with getters/setters):
        _adversarial_number_epochs (int): Number of training epochs
        _adversarial_batch_size (int): Size of training batches
        _adversarial_initializer_mean (float): Mean for weight initialization
        _adversarial_initializer_deviation (float): Std dev for weight initialization
        _adversarial_latent_dimension (int): Size of the latent space
        _adversarial_training_algorithm (str): Training algorithm specification
        _adversarial_activation_function (str): Activation function for hidden layers
        _adversarial_dropout_decay_rate_g (float): Generator dropout rate
        _adversarial_dropout_decay_rate_d (float): Discriminator dropout rate
        _adversarial_dense_layer_sizes_g (list): Generator layer sizes
        _adversarial_dense_layer_sizes_d (list): Discriminator layer sizes
        _adversarial_loss_generator (str): Generator loss function
        _adversarial_loss_discriminator (str): Discriminator loss function
        _adversarial_smoothing_rate (float): Label smoothing rate
        _adversarial_latent_mean_distribution (float): Latent space mean
        _adversarial_latent_stander_deviation (float): Latent space std dev
        _adversarial_file_name_discriminator (str): Discriminator model filename
        _adversarial_file_name_generator (str): Generator model filename
        _adversarial_path_output_models (str): Path for saving models
        _adversarial_last_layer_activation (str): Last layer activation function
        _variational_autoencoder_number_epochs (int): Epochs for VAE pre-training
    """

    def __init__(self, arguments):
        """
        Initializes the adversarial instance with configuration parameters.

        Args:
            arguments (Namespace): Configuration object containing all required parameters:
                - adversarial_number_epochs: Training epochs
                - adversarial_batch_size: Batch size
                - adversarial_initializer_mean: Weight init mean
                - adversarial_initializer_deviation: Weight init std dev
                - [All other parameters matching attribute names]
        """
        self._adversarial_algorithm = None
        self._adversarial_model = None

        # ** Adversarial Model (GAN) Configuration Parameters **
        self._adversarial_number_epochs = arguments.adversarial_number_epochs
        self._adversarial_batch_size = arguments.adversarial_batch_size
        self._adversarial_initializer_mean = arguments.adversarial_initializer_mean
        self._adversarial_initializer_deviation = arguments.adversarial_initializer_deviation
        self._adversarial_latent_dimension = arguments.adversarial_latent_dimension
        self._adversarial_training_algorithm = arguments.adversarial_training_algorithm
        self._adversarial_activation_function = arguments.adversarial_activation_function
        self._adversarial_dropout_decay_rate_g = arguments.adversarial_dropout_decay_rate_g
        self._adversarial_dropout_decay_rate_d = arguments.adversarial_dropout_decay_rate_d
        self._adversarial_dense_layer_sizes_g = arguments.adversarial_dense_layer_sizes_g
        self._adversarial_dense_layer_sizes_d = arguments.adversarial_dense_layer_sizes_d
        self._adversarial_loss_generator = arguments.adversarial_loss_generator
        self._adversarial_loss_discriminator = arguments.adversarial_loss_discriminator
        self._adversarial_smoothing_rate = arguments.adversarial_smoothing_rate
        self._adversarial_latent_mean_distribution = arguments.adversarial_latent_mean_distribution
        self._adversarial_latent_stander_deviation = arguments.adversarial_latent_stander_deviation
        self._adversarial_file_name_discriminator = arguments.adversarial_file_name_discriminator
        self._adversarial_file_name_generator = arguments.adversarial_file_name_generator
        self._adversarial_path_output_models = arguments.adversarial_path_output_models
        self._adversarial_last_layer_activation = arguments.adversarial_last_layer_activation
        self._variational_autoencoder_number_epochs = arguments.variational_autoencoder_number_epochs

    def _get_adversarial_model(self, input_shape):
        """
        Initialize and configure the Adversarial model, including both the generator and discriminator components.

        This method sets up an Adversarial model by configuring both the generator and discriminator using the
        `AdversarialModel` class and linking them with the `AdversarialAlgorithm` class. The model is initialized
        with specified configurations such as latent dimension, activation functions, dropout rates, and layer sizes
        for both the generator and discriminator.

        Args:
            input_shape (tuple):
                The shape of the input data, which determines the output shape for the models.

        Initializes:
            self._adversarial_model:
                An instance of the `AdversarialModel` class, including the generator and discriminator setup, with
                configurations for activation functions, layer sizes, dropout rates, and more.
            self._adversarial_algorithm:
                An instance of the `AdversarialAlgorithm` class, managing the adversarial training process, including
                the generator and discriminator models, loss functions, latent distributions, and model file paths.

        """

        # Adversarial Model setup for Generator and Discriminator
        self._adversarial_model = AdversarialModel(latent_dimension=self._adversarial_latent_dimension,
                                                   output_shape=input_shape,
                                                   activation_function=self._adversarial_activation_function,
                                                   initializer_mean=self._adversarial_initializer_mean,
                                                   initializer_deviation=self._adversarial_initializer_deviation,
                                                   dropout_decay_rate_g=self._adversarial_dropout_decay_rate_g,
                                                   dropout_decay_rate_d=self._adversarial_dropout_decay_rate_d,
                                                   last_layer_activation=self._adversarial_last_layer_activation,
                                                   dense_layer_sizes_g=self._adversarial_dense_layer_sizes_g,
                                                   dense_layer_sizes_d=self._adversarial_dense_layer_sizes_d,
                                                   dataset_type=numpy.float32,
                                                   number_samples_per_class = self._number_samples_per_class)

        # Adversarial Algorithm setup for training and model operations
        self._adversarial_algorithm = AdversarialAlgorithm(generator_model=self._adversarial_model.get_generator(),
                                                           discriminator_model=self._adversarial_model.get_discriminator(),
                                                           latent_dimension=self._adversarial_latent_dimension,
                                                           loss_generator=self._adversarial_loss_generator,
                                                           loss_discriminator=self._adversarial_loss_discriminator,
                                                           file_name_discriminator=self._adversarial_file_name_discriminator,
                                                           file_name_generator=self._adversarial_file_name_generator,
                                                           models_saved_path=self._adversarial_path_output_models,
                                                           latent_mean_distribution=self._adversarial_latent_mean_distribution,
                                                           latent_stander_deviation=self._adversarial_latent_stander_deviation,
                                                           smoothing_rate=self._adversarial_smoothing_rate)


    def _training_adversarial_modelo(self, input_shape, arguments, x_real_samples, y_real_samples):
        """
        Executes the complete adversarial training process.

        Args:
            input_shape (tuple): Shape of input data samples
            arguments (Namespace): Configuration parameters
            x_real_samples (ndarray): Training data samples
            y_real_samples (ndarray): Corresponding class labels

        Process:
            1. Initializes model architecture
            2. Configures optimizers and loss functions
            3. Sets up training callbacks
            4. Executes adversarial training
            5. Manages model saving and monitoring
        """

        # Initialize the adversarial model
        self._get_adversarial_model(input_shape)

        # Print the model summaries for the generator and discriminator
        self._adversarial_model.get_generator().summary()
        self._adversarial_model.get_discriminator().summary()

        # Set up optimizers for the generator and discriminator
        generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        # Compile the adversarial algorithm with binary cross-entropy loss
        self._adversarial_algorithm.compile(generator_optimizer,
                                            discriminator_optimizer,
                                            BinaryCrossentropy(),
                                            BinaryCrossentropy())

        # callbacks_list = [self._callback_resources_monitor, self._callback_model_monitor]
        callbacks_list = [self._callback_model_monitor]

        if arguments.use_early_stop:
            callbacks_list.append(self._callback_early_stop)

        # Fit the model with real samples and the corresponding labels
        self._adversarial_algorithm.fit(
            x_real_samples,
            to_categorical(y_real_samples, num_classes=self._number_samples_per_class["number_classes"]),
            epochs=self._adversarial_number_epochs, batch_size=self._adversarial_batch_size,
            callbacks=callbacks_list)


    # Getter and setter for adversarial_number_epochs
    @property
    def adversarial_number_epochs(self):
        return self._adversarial_number_epochs

    @adversarial_number_epochs.setter
    def adversarial_number_epochs(self, value):
        self._adversarial_number_epochs = value

    # Getter and setter for adversarial_initializer_mean
    @property
    def adversarial_initializer_mean(self):
        return self._adversarial_initializer_mean

    @adversarial_initializer_mean.setter
    def adversarial_initializer_mean(self, value):
        self._adversarial_initializer_mean = value

    # Getter and setter for adversarial_initializer_deviation
    @property
    def adversarial_initializer_deviation(self):
        return self._adversarial_initializer_deviation

    @adversarial_initializer_deviation.setter
    def adversarial_initializer_deviation(self, value):
        self._adversarial_initializer_deviation = value

    # Getter and setter for adversarial_latent_dimension
    @property
    def adversarial_latent_dimension(self):
        return self._adversarial_latent_dimension

    @adversarial_latent_dimension.setter
    def adversarial_latent_dimension(self, value):
        self._adversarial_latent_dimension = value

    # Getter and setter for adversarial_training_algorithm
    @property
    def adversarial_training_algorithm(self):
        return self._adversarial_training_algorithm

    @adversarial_training_algorithm.setter
    def adversarial_training_algorithm(self, value):
        self._adversarial_training_algorithm = value

    # Getter and setter for adversarial_activation_function
    @property
    def adversarial_activation_function(self):
        return self._adversarial_activation_function

    @adversarial_activation_function.setter
    def adversarial_activation_function(self, value):
        self._adversarial_activation_function = value

    # Getter and setter for adversarial_dropout_decay_rate_g
    @property
    def adversarial_dropout_decay_rate_g(self):
        return self._adversarial_dropout_decay_rate_g

    @adversarial_dropout_decay_rate_g.setter
    def adversarial_dropout_decay_rate_g(self, value):
        self._adversarial_dropout_decay_rate_g = value

    # Getter and setter for adversarial_dropout_decay_rate_d
    @property
    def adversarial_dropout_decay_rate_d(self):
        return self._adversarial_dropout_decay_rate_d

    @adversarial_dropout_decay_rate_d.setter
    def adversarial_dropout_decay_rate_d(self, value):
        self._adversarial_dropout_decay_rate_d = value

    # Getter and setter for adversarial_dense_layer_sizes_g
    @property
    def adversarial_dense_layer_sizes_g(self):
        return self._adversarial_dense_layer_sizes_g

    @adversarial_dense_layer_sizes_g.setter
    def adversarial_dense_layer_sizes_g(self, value):
        self._adversarial_dense_layer_sizes_g = value

    # Getter and setter for adversarial_dense_layer_sizes_d
    @property
    def adversarial_dense_layer_sizes_d(self):
        return self._adversarial_dense_layer_sizes_d

    @adversarial_dense_layer_sizes_d.setter
    def adversarial_dense_layer_sizes_d(self, value):
        self._adversarial_dense_layer_sizes_d = value

    # Getter and setter for adversarial_loss_generator
    @property
    def adversarial_loss_generator(self):
        return self._adversarial_loss_generator

    @adversarial_loss_generator.setter
    def adversarial_loss_generator(self, value):
        self._adversarial_loss_generator = value

    # Getter and setter for adversarial_loss_discriminator
    @property
    def adversarial_loss_discriminator(self):
        return self._adversarial_loss_discriminator

    @adversarial_loss_discriminator.setter
    def adversarial_loss_discriminator(self, value):
        self._adversarial_loss_discriminator = value

    # Getter and setter for adversarial_smoothing_rate
    @property
    def adversarial_smoothing_rate(self):
        return self._adversarial_smoothing_rate

    @adversarial_smoothing_rate.setter
    def adversarial_smoothing_rate(self, value):
        self._adversarial_smoothing_rate = value

    # Getter and setter for adversarial_latent_mean_distribution
    @property
    def adversarial_latent_mean_distribution(self):
        return self._adversarial_latent_mean_distribution

    @adversarial_latent_mean_distribution.setter
    def adversarial_latent_mean_distribution(self, value):
        self._adversarial_latent_mean_distribution = value

    # Getter and setter for adversarial_latent_stander_deviation
    @property
    def adversarial_latent_stander_deviation(self):
        return self._adversarial_latent_stander_deviation

    @adversarial_latent_stander_deviation.setter
    def adversarial_latent_stander_deviation(self, value):
        self._adversarial_latent_stander_deviation = value

    # Getter and setter for adversarial_file_name_discriminator
    @property
    def adversarial_file_name_discriminator(self):
        return self._adversarial_file_name_discriminator

    @adversarial_file_name_discriminator.setter
    def adversarial_file_name_discriminator(self, value):
        self._adversarial_file_name_discriminator = value

    # Getter and setter for adversarial_file_name_generator
    @property
    def adversarial_file_name_generator(self):
        return self._adversarial_file_name_generator

    @adversarial_file_name_generator.setter
    def adversarial_file_name_generator(self, value):
        self._adversarial_file_name_generator = value

    # Getter and setter for adversarial_path_output_models
    @property
    def adversarial_path_output_models(self):
        return self._adversarial_path_output_models

    @adversarial_path_output_models.setter
    def adversarial_path_output_models(self, value):
        self._adversarial_path_output_models = value
