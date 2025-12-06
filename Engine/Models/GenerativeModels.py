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
    from Engine.Callbacks.CallbackEarlyStop import EarlyStopping

    from tensorflow.python.keras.losses import BinaryCrossentropy

    from Engine.Algorithms.Copy.CopyAlgorithm import CopyAlgorithm

    from Engine.Callbacks.CallbackModel import ModelMonitorCallback
    from Engine.Models.LatentDiffusion.DiffusionModelUnet import UNetModel

    from Engine.Algorithms.SMOTE.AlgorithmSMOTE import SMOTEAlgorithm

    from Engine.Callbacks.CallbackResources import ResourceMonitorCallback

    from Engine.Models.Adversarial.AdversarialModel import AdversarialModel
    from Engine.Models.Autoencoder.ModelAutoencoder import AutoencoderModel

    from Engine.Models.WassersteinGP.ModelWassersteinGPGAN import WassersteinGPModel
    from Engine.Models.QuantizedVAE.ModelQuantizedVAE import QuantizedVAEModel
    from Engine.Models.DenoisingDiffusion.DiffusionModelUnet import UNetDenoisingModel
    from Engine.Algorithms.LatentDiffusion.GaussianLatentDiffusion import GaussianDiffusion
    from Engine.Models.DiffusionKernel.DiffusionModelUnet import UNetModelKernel

    from Engine.Algorithms.Wasserstein.AlgorithmWassersteinGAN import WassersteinAlgorithm
    from Engine.Models.Wasserstein.ModelWassersteinGAN import WassersteinModel

    from Engine.Algorithms.RandomNoise.AlgorithmRandomNoise import RandomNoiseAlgorithm
    from Engine.Algorithms.Adversarial.AdversarialAlgorithm import AdversarialAlgorithm
    from Engine.Algorithms.Autoencoder.AutoencoderAlgorithm import AutoencoderAlgorithm

    from Engine.Algorithms.WassersteinGP.AlgorithmWassersteinGANGP import WassersteinGPAlgorithm
    from Engine.Algorithms.QuantizedVAE.AlgorithmQuantizedVAE import QuantizedVAEAlgorithm

    from Engine.Models.LatentDiffusion.VariationalAutoencoderModel import VariationalModelDiffusion

    from Engine.Models.VariationalAutoencoder.VariationalAutoencoderModel import VariationalModel
    from Engine.Algorithms.LatentDiffusion.AlgorithmLatentDiffusion import LatentDiffusionAlgorithm

    from Engine.Algorithms.LatentDiffusion.AlgorithmVAELatentDiffusion import VAELatentDiffusionAlgorithm
    from Engine.Algorithms.DenoisingDiffusion.AlgorithmDenoisingDiffusion import DenoisingDiffusionAlgorithm
    from Engine.Algorithms.VariationalAutoencoder.AlgorithmVariationalAutoencoder import VariationalAlgorithm



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
        self._quantized_vae_model = QuantizedVAEModel(latent_dimension=self._quantized_vae_latent_dimension,
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
                                                  number_samples_per_class=self._number_samples_per_class)

        quantized_model = self._quantized_vae_model.get_quantized_model()

        # Quantized VAE Algorithm setup for training and model operations
        self._quantized_vae_algorithm = QuantizedVAEAlgorithm(encoder_model = self._quantized_vae_model.get_encoder(),
                                                            decoder_model = self._quantized_vae_model.get_decoder(),
                                                            quantized_vae_model = quantized_model,
                                                            train_variance = self._quantized_vae_train_variance,
                                                            latent_dimension = self._quantized_vae_latent_dimension,
                                                            number_embeddings = self._quantized_vae_number_embeddings,
                                                            file_name_encoder = self._quantized_vae_file_name_encoder,
                                                            file_name_decoder = self._quantized_vae_file_name_decoder,
                                                            models_saved_path = self._quantized_vae_path_output_models)


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
        self._quantized_vae_model.get_encoder().summary()
        self._quantized_vae_model.get_decoder().summary()

        # Compile the variational autoencoder algorithm with the specified loss function
        self._quantized_vae_algorithm.compile(optimizer=Adam(learning_rate=0.0001))

        callbacks_list = [self._callback_resources_monitor, self._callback_model_monitor]

        if arguments.use_early_stop:
            callbacks_list.append(self._callback_early_stop)

        # Fit the variational autoencoder model
        self._quantized_vae_algorithm.fit((
            x_real_samples,
            to_categorical(y_real_samples, num_classes=self._number_samples_per_class["number_classes"])),
            x_real_samples, epochs=self._quantized_vae_number_epochs, batch_size=self._quantized_vae_batch_size,
            callbacks=callbacks_list)


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




class LatentDiffusionInstance:
    """
    A class that implements a Latent Denoising Probabilistic Diffusion (LDPD) model for generative tasks.
    This implementation combines variational autoencoders with diffusion models in latent space for
    high-quality sample generation.

    Key Components:
    - Two UNet models for the diffusion process
    - Variational Autoencoder for latent space representation
    - Gaussian diffusion utilities for noise scheduling
    - Complete training pipeline for both VAE and diffusion components
    - Highly configurable architecture via arguments for research experimentation

    Attributes:
        _latent_variational_algorithm_diffusion: Orchestrates training of the VAE within the diffusion context
        _latent_variation_model_diffusion: Stores the encoder and decoder of the VAE
        _latent_autoencoder_diffusion: Core autoencoder used for latent embedding and reconstruction
        _latent_gaussian_diffusion_util: Utility object for beta schedules and diffusion parameters
        _latent_second_unet_model: The second-stage UNet used in the denoising chain
        _latent_first_unet_model: The initial UNet used in early-stage denoising

        # Latent Diffusion - UNet Parameters
        _latent_diffusion_unet_last_layer_activation: Activation function used in UNet's final layer
        _latent_diffusion_latent_dimension: Dimensionality of latent space
        _latent_diffusion_unet_num_embedding_channels: Number of channels for positional/time embeddings
        _latent_diffusion_unet_channels_per_level: Channel config per U-Net level
        _latent_diffusion_unet_batch_size: Batch size used during UNet training
        _latent_diffusion_unet_attention_mode: Attention mechanism used in UNet (e.g., multi-head, cross-attn)
        _latent_diffusion_unet_num_residual_blocks: Number of residual blocks per level
        _latent_diffusion_unet_group_normalization: Whether to apply group norm in UNet layers
        _latent_diffusion_unet_intermediary_activation: Activation function for intermediate layers
        _latent_diffusion_unet_intermediary_activation_alpha: Alpha value (if using LeakyReLU, etc.)
        _latent_diffusion_unet_epochs: Number of epochs for UNet training

        # Latent Diffusion - VAE Parameters
        _latent_diffusion_VAE_mean_distribution: Type of distribution for latent mean (e.g., normal)
        _latent_diffusion_VAE_stander_deviation: Std deviation for latent distribution
        _latent_diffusion_VAE_file_name_encoder: File path to save/load the encoder
        _latent_diffusion_VAE_file_name_decoder: File path to save/load the decoder
        _latent_diffusion_VAE_path_output_models: Directory to store trained autoencoder components
        _latent_diffusion_VAE_loss_function: Loss used to optimize VAE (e.g., MSE + KL)
        _latent_diffusion_VAE_encoder_filters: Conv filter settings for encoder
        _latent_diffusion_VAE_decoder_filters: Conv filter settings for decoder
        _latent_diffusion_VAE_last_layer_activation: Output activation of decoder
        _latent_diffusion_VAE_latent_dimension: Size of compressed latent vector
        _latent_diffusion_VAE_batch_size_create_embedding: Batch size used for embedding generation
        _latent_diffusion_VAE_batch_size_training: Batch size during VAE training
        _latent_diffusion_VAE_epochs: Training epochs for the VAE
        _latent_diffusion_VAE_intermediary_activation_function: Activation in intermediate layers
        _latent_diffusion_VAE_intermediary_activation_alpha: Alpha parameter for the activation
        _latent_diffusion_VAE_activation_output_encoder: Activation at output of encoder

        # Latent Diffusion - Noise and Training Parameters
        _latent_diffusion_margin: Margin used in contrastive or reconstruction objectives
        _latent_diffusion_ema: Use of Exponential Moving Average in parameter updates
        _latent_diffusion_time_steps: Number of time steps for forward/reverse diffusion

        # Gaussian Diffusion - Scheduling and Initializer
        _latent_diffusion_gaussian_beta_start: Initial β value for the schedule
        _latent_diffusion_gaussian_beta_end: Final β value for the schedule
        _latent_diffusion_gaussian_time_steps: Number of diffusion steps
        _latent_diffusion_gaussian_clip_min: Minimum value for scheduled noise
        _latent_diffusion_gaussian_clip_max: Maximum value for scheduled noise
        _latent_diffusion_VAE_initializer_mean: Initial mean for model weight initialization
        _latent_diffusion_VAE_initializer_deviation: Initial std deviation for model weights
        _latent_diffusion_VAE_dropout_decay_rate_encoder: Dropout decay schedule for encoder
        _latent_diffusion_VAE_dropout_decay_rate_decoder: Dropout decay schedule for decoder

    """

    def __init__(self, arguments):
        """
        Initializes the latent diffusion instance with configuration parameters.

        Args:
            arguments (Namespace): Configuration object containing:
                - UNet architecture parameters
                - VAE configuration
                - Gaussian diffusion settings
                - Training hyperparameters
                - Model saving paths
        """

        self._latent_variational_algorithm_diffusion = None
        self._latent_variation_model_diffusion = None

        self._latent_autoencoder_diffusion = None
        self._latent_gaussian_diffusion_util = None

        self._latent_second_unet_model = None
        self._latent_first_unet_model = None

        # ** Latent Denoising Probabilistic LatentDiffusion (LDPD) Configuration Parameters **
        self._latent_diffusion_unet_last_layer_activation = arguments.latent_diffusion_unet_last_layer_activation
        self._latent_diffusion_latent_dimension = arguments.latent_diffusion_latent_dimension
        self._latent_diffusion_unet_num_embedding_channels = arguments.latent_diffusion_unet_num_embedding_channels
        self._latent_diffusion_unet_channels_per_level = arguments.latent_diffusion_unet_channels_per_level
        self._latent_diffusion_unet_batch_size = arguments.latent_diffusion_unet_batch_size
        self._latent_diffusion_unet_attention_mode = arguments.latent_diffusion_unet_attention_mode
        self._latent_diffusion_unet_num_residual_blocks = arguments.latent_diffusion_unet_num_residual_blocks
        self._latent_diffusion_unet_group_normalization = arguments.latent_diffusion_unet_group_normalization
        self._latent_diffusion_unet_intermediary_activation = arguments.latent_diffusion_unet_intermediary_activation
        self._latent_diffusion_unet_intermediary_activation_alpha = arguments.latent_diffusion_unet_intermediary_activation_alpha
        self._latent_diffusion_unet_epochs = arguments.latent_diffusion_unet_epochs

        self._latent_diffusion_VAE_mean_distribution = arguments.latent_diffusion_autoencoder_mean_distribution
        self._latent_diffusion_VAE_stander_deviation = arguments.latent_diffusion_autoencoder_stander_deviation
        self._latent_diffusion_VAE_file_name_encoder = arguments.latent_diffusion_autoencoder_file_name_encoder
        self._latent_diffusion_VAE_file_name_decoder = arguments.latent_diffusion_autoencoder_file_name_decoder
        self._latent_diffusion_VAE_path_output_models = arguments.latent_diffusion_autoencoder_path_output_models

        # Gaussian Diffusion - Scheduling and Initializer
        self._latent_diffusion_gaussian_beta_start = arguments.latent_diffusion_gaussian_beta_start
        self._latent_diffusion_gaussian_beta_end = arguments.latent_diffusion_gaussian_beta_end
        self._latent_diffusion_gaussian_time_steps = arguments.latent_diffusion_gaussian_time_steps
        self._latent_diffusion_gaussian_clip_min = arguments.latent_diffusion_gaussian_clip_min
        self._latent_diffusion_gaussian_clip_max = arguments.latent_diffusion_gaussian_clip_max


        self._latent_diffusion_VAE_loss_function = arguments.latent_diffusion_autoencoder_loss
        self._latent_diffusion_VAE_encoder_filters = arguments.latent_diffusion_autoencoder_encoder_filters
        self._latent_diffusion_VAE_decoder_filters = arguments.latent_diffusion_autoencoder_decoder_filters
        self._latent_diffusion_VAE_last_layer_activation = arguments.latent_diffusion_autoencoder_last_layer_activation
        self._latent_diffusion_VAE_latent_dimension = arguments.latent_diffusion_autoencoder_latent_dimension
        self._latent_diffusion_VAE_batch_size_create_embedding = arguments.latent_diffusion_autoencoder_batch_size_create_embedding
        self._latent_diffusion_VAE_batch_size_training = arguments.latent_diffusion_autoencoder_batch_size_training
        self._latent_diffusion_VAE_epochs = arguments.latent_diffusion_autoencoder_epochs
        self._latent_diffusion_VAE_intermediary_activation_function = arguments.latent_diffusion_autoencoder_intermediary_activation_function
        self._latent_diffusion_VAE_intermediary_activation_alpha = arguments.latent_diffusion_autoencoder_intermediary_activation_alpha
        self._latent_diffusion_VAE_activation_output_encoder = arguments.latent_diffusion_autoencoder_activation_output_encoder
        self._latent_diffusion_margin = arguments.latent_diffusion_margin
        self._latent_diffusion_ema = arguments.latent_diffusion_ema
        self._latent_diffusion_time_steps = arguments.latent_diffusion_time_steps

        # ** Gaussian LatentDiffusion Configuration Parameters **
        self._latent_diffusion_VAE_initializer_mean = arguments.latent_diffusion_autoencoder_initializer_mean
        self._latent_diffusion_VAE_initializer_deviation = arguments.latent_diffusion_autoencoder_initializer_deviation
        self._latent_diffusion_VAE_dropout_decay_rate_encoder = arguments.latent_diffusion_autoencoder_dropout_decay_rate_encoder
        self._latent_diffusion_VAE_dropout_decay_rate_decoder = arguments.latent_diffusion_autoencoder_dropout_decay_rate_decoder


    def _get_latent_diffusion(self, input_shape):
        """
         Initializes and configures the LatentDiffusion model using UNet architecture for image generation.

         This method initializes multiple components required for the diffusion process, including
         two UNet instances, a DiffusionAutoencoderModel, and a GaussianDiffusion utility. The UNet
         instances are configured with the specified hyperparameters for building the model. The
         weights of the second UNet model are synchronized with the first one. Additionally, the method
         sets up the variational model diffusion and the associated variational algorithm diffusion
         for image generation and embedding reconstruction.

         Args:
             input_shape (tuple):
              The shape of the input data, typically the dimensions of the images (height, width, channels).

         Initializes:
             self._first_instance_unet (UNetModel):
                The first instance of the UNet model used for the diffusion process.
             self._second_instance_unet (UNetModel):
                The second instance of the UNet model, which is a copy of the first one.
             self._first_unet_model (Model):
                The compiled UNet model for the first instance.
             self._second_unet_model (Model):
                The compiled UNet model for the second instance, with synchronized weights from the first model.
             self._gaussian_diffusion_util (GaussianDiffusion):
                Utility for managing the diffusion process with Gaussian noise.
             self._variation_model_diffusion (VariationalModelDiffusion):
                The diffusion model with variational autoencoder for latent representation learning.
             self._variational_algorithm_diffusion (VariationalAlgorithmDiffusion):
                The algorithm for variational inference during the diffusion process.
         """

        # Initialize the first instance of UNet for the diffusion model
        self._latent_first_instance_unet = UNetModel(embedding_dimension=self._latent_diffusion_latent_dimension,
                                                     embedding_channels= self._latent_diffusion_unet_num_embedding_channels,
                                                     list_neurons_per_level=self._latent_diffusion_unet_channels_per_level,
                                                     list_attentions=self._latent_diffusion_unet_attention_mode,
                                                     number_residual_blocks=self._latent_diffusion_unet_num_residual_blocks,
                                                     normalization_groups=self._latent_diffusion_unet_group_normalization,
                                                     intermediary_activation_function=self._latent_diffusion_unet_intermediary_activation,
                                                     intermediary_activation_alpha= self._latent_diffusion_unet_intermediary_activation_alpha,
                                                     last_layer_activation=self._latent_diffusion_unet_last_layer_activation,
                                                     number_samples_per_class=self._number_samples_per_class)

        # Initialize the second instance of UNet with the same configuration
        self._latent_second_instance_unet = UNetModel(embedding_dimension=self._latent_diffusion_latent_dimension,
                                                      embedding_channels= self._latent_diffusion_unet_num_embedding_channels,
                                                      list_neurons_per_level=self._latent_diffusion_unet_channels_per_level,
                                                      list_attentions=self._latent_diffusion_unet_attention_mode,
                                                      number_residual_blocks=self._latent_diffusion_unet_num_residual_blocks,
                                                      normalization_groups=self._latent_diffusion_unet_group_normalization,
                                                      intermediary_activation_function=self._latent_diffusion_unet_intermediary_activation,
                                                      intermediary_activation_alpha= self._latent_diffusion_unet_intermediary_activation_alpha,
                                                      last_layer_activation=self._latent_diffusion_unet_last_layer_activation,
                                                      number_samples_per_class=self._number_samples_per_class)

        # Build the models for both UNet instances
        self._latent_first_unet_model = self._latent_first_instance_unet.build_model()
        self._latent_second_unet_model = self._latent_second_instance_unet.build_model()

        # Synchronize the weights of the second UNet model with the first one
        self._latent_second_unet_model.set_weights(self._latent_first_unet_model.get_weights())

        # Initialize the GaussianDiffusion utility for the diffusion process
        self._latent_gaussian_diffusion_util = GaussianDiffusion(beta_start=self._latent_diffusion_gaussian_beta_start,
                                                                 beta_end=self._latent_diffusion_gaussian_beta_end,
                                                                 time_steps=self._latent_diffusion_gaussian_time_steps,
                                                                 clip_min=self._latent_diffusion_gaussian_clip_min,
                                                                 clip_max=self._latent_diffusion_gaussian_clip_max)

        # Initialize the VariationalModelDiffusion for embedding learning and reconstructor
        self._latent_variation_model_diffusion = VariationalModelDiffusion(latent_dimension=self._latent_diffusion_latent_dimension, output_shape=input_shape,
                                                                           activation_function=self._latent_diffusion_VAE_intermediary_activation_function,
                                                                           initializer_mean=self._latent_diffusion_VAE_initializer_mean,
                                                                           initializer_deviation=self._latent_diffusion_VAE_initializer_deviation,
                                                                           dropout_decay_encoder=self._latent_diffusion_VAE_dropout_decay_rate_encoder,
                                                                           dropout_decay_decoder=self._latent_diffusion_VAE_dropout_decay_rate_decoder,
                                                                           last_layer_activation=self._latent_diffusion_VAE_activation_output_encoder,
                                                                           number_neurons_encoder=self._latent_diffusion_VAE_encoder_filters,
                                                                           number_neurons_decoder=self._latent_diffusion_VAE_decoder_filters,
                                                                           dataset_type=numpy.float32,
                                                                           number_samples_per_class = self._number_samples_per_class)

        # Initialize the VariationalAlgorithmDiffusion for the training and diffusion process
        self._latent_variational_algorithm_diffusion = VAELatentDiffusionAlgorithm(encoder_model=self._latent_variation_model_diffusion.get_encoder(),
                                                                                   decoder_model=self._latent_variation_model_diffusion.get_decoder(),
                                                                                   loss_function=self._latent_diffusion_VAE_loss_function,
                                                                                   latent_dimension=self._latent_diffusion_latent_dimension,
                                                                                   decoder_latent_dimension = self._latent_diffusion_latent_dimension,
                                                                                   latent_mean_distribution=self._latent_diffusion_VAE_mean_distribution,
                                                                                   latent_stander_deviation=self._latent_diffusion_VAE_stander_deviation,
                                                                                   file_name_encoder=self._latent_diffusion_VAE_file_name_encoder,
                                                                                   file_name_decoder=self._latent_diffusion_VAE_file_name_decoder,
                                                                                   models_saved_path=self._latent_diffusion_VAE_path_output_models)

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
        self._variation_model = VariationalModel(latent_dimension=self._latent_diffusion_VAE_latent_dimension,
                                                 output_shape=input_shape,
                                                 activation_function=self._latent_diffusion_VAE_intermediary_activation_function,
                                                 initializer_mean=self._latent_diffusion_VAE_initializer_mean,
                                                 initializer_deviation=self._latent_diffusion_VAE_initializer_deviation,
                                                 dropout_decay_encoder=self._latent_diffusion_VAE_dropout_decay_rate_encoder,
                                                 dropout_decay_decoder=self._latent_diffusion_VAE_dropout_decay_rate_decoder,
                                                 last_layer_activation=self._latent_diffusion_VAE_last_layer_activation,
                                                 number_neurons_encoder=self._latent_diffusion_VAE_encoder_filters,
                                                 number_neurons_decoder=self._latent_diffusion_VAE_decoder_filters,
                                                 dataset_type=numpy.float32,
                                                 number_samples_per_class = self._number_samples_per_class)

        # Variational Algorithm setup for training and model operations
        self._variational_algorithm = VariationalAlgorithm(encoder_model=self._variation_model.get_encoder(),
                                                           decoder_model=self._variation_model.get_decoder(),
                                                           loss_function=self._latent_diffusion_VAE_loss_function,
                                                           latent_dimension=self._latent_diffusion_VAE_latent_dimension,
                                                           decoder_latent_dimension = self._latent_diffusion_VAE_latent_dimension,
                                                           latent_mean_distribution=self._latent_diffusion_VAE_mean_distribution,
                                                           latent_stander_deviation=self._latent_diffusion_VAE_stander_deviation,
                                                           file_name_encoder=self._latent_diffusion_VAE_file_name_encoder,
                                                           file_name_decoder=self._latent_diffusion_VAE_file_name_decoder,
                                                           models_saved_path=self._latent_diffusion_VAE_path_output_models)



    def _training_latent_diffusion_model(self, input_shape, arguments, x_real_samples, y_real_samples):
        """
        Executes the complete training pipeline for latent diffusion.

        Process:
        1. Initializes diffusion models
        2. Trains variational autoencoder
        3. Creates latent embeddings
        4. Trains diffusion models on latent space
        5. Manages callbacks and monitoring

        Args:
            input_shape (tuple): Input data shape
            arguments (Namespace): Training configuration
            x_real_samples (ndarray): Training samples
            y_real_samples (ndarray): Corresponding labels
        """
        # Initialize the diffusion model
        self._get_latent_diffusion(input_shape)

        # Print the model summaries for the U-Net models
        self._latent_first_unet_model.summary()
        self._latent_second_unet_model.summary()

        # Initialize the variational autoencoder model for diffusion
        self._get_variational_autoencoder(input_shape)

        self._latent_variation_model_diffusion.get_encoder().summary()
        self._latent_variation_model_diffusion.get_decoder().summary()

        # Compile the variational algorithm for diffusion
        self._latent_variational_algorithm_diffusion.compile(loss=self._latent_diffusion_VAE_loss_function)

        # callbacks_list = [self._callback_resources_monitor, self._callback_model_monitor]
        callbacks_list = [self._callback_model_monitor]

        if arguments.use_early_stop:
            callbacks_list.append(self._callback_early_stop)

        # Fit the diffusion model with the training data
        self._latent_variational_algorithm_diffusion.fit((
            x_real_samples,
            to_categorical(y_real_samples, num_classes=self._number_samples_per_class["number_classes"])),
            x_real_samples, epochs=self._latent_diffusion_VAE_epochs,
            batch_size=self._latent_diffusion_VAE_batch_size_training,
            callbacks=callbacks_list)

        # Retrieve the trained encoder and decoder from the variational algorithm
        self._encoder_latent_diffusion = self._latent_variational_algorithm_diffusion.get_encoder_trained()
        self._decoder_latent_diffusion = self._latent_variational_algorithm_diffusion.get_decoder_trained()

        # Print summaries of the trained encoder and decoder
        self._encoder_latent_diffusion.summary()
        self._decoder_latent_diffusion.summary()

        # Initialize the final diffusion algorithm
        self._latent_diffusion_algorithm = LatentDiffusionAlgorithm(first_unet_model=self._latent_first_unet_model,
                                                                    second_unet_model=self._latent_second_unet_model,
                                                                    encoder_model_image=self._encoder_latent_diffusion,
                                                                    decoder_model_image=self._decoder_latent_diffusion,
                                                                    gdf_util=self._latent_gaussian_diffusion_util,
                                                                    optimizer_autoencoder=Adam(learning_rate=0.0001),
                                                                    optimizer_diffusion=Adam(learning_rate=0.0001),
                                                                    time_steps=self._latent_diffusion_gaussian_time_steps,
                                                                    ema=self._latent_diffusion_ema,
                                                                    margin=self._latent_diffusion_margin,
                                                                    embedding_dimension=self._latent_diffusion_latent_dimension)

        # Compile the diffusion model
        self._latent_diffusion_algorithm.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001))

        # Prepare the data embedding and train the diffusion model
        data_embedding = self._latent_variational_algorithm_diffusion.create_embedding([
            x_real_samples,
            to_categorical(y_real_samples, num_classes=self._number_samples_per_class["number_classes"])])

        data_embedding = numpy.array(data_embedding)
        data_embedding = tensorflow.expand_dims(data_embedding, axis=-1)

        # callbacks_list = [self._callback_resources_monitor, self._callback_model_monitor]
        callbacks_list = [self._callback_model_monitor]

        if arguments.use_early_stop:
            callbacks_list.append(self._callback_early_stop)

        self._latent_diffusion_algorithm.fit(
            data_embedding,
            to_categorical(y_real_samples, num_classes=self._number_samples_per_class["number_classes"]),
            epochs=self._latent_diffusion_unet_epochs, batch_size=self._latent_diffusion_unet_batch_size,
            callbacks=callbacks_list, verbose=2)

    # ** Latent Denoising Probabilistic LatentDiffusion (LDPD) Configuration Parameters **
    @property
    def latent_diffusion_unet_last_layer_activation(self):
        """Getter for UNET last layer activation function"""
        return self._latent_diffusion_unet_last_layer_activation

    @latent_diffusion_unet_last_layer_activation.setter
    def latent_diffusion_unet_last_layer_activation(self, value):
        """Setter for UNET last layer activation function"""
        self._latent_diffusion_unet_last_layer_activation = value

    @property
    def latent_diffusion_latent_dimension(self):
        """Getter for latent dimension size"""
        return self._latent_diffusion_latent_dimension

    @latent_diffusion_latent_dimension.setter
    def latent_diffusion_latent_dimension(self, value):
        """Setter for latent dimension size"""
        self._latent_diffusion_latent_dimension = value

    @property
    def latent_diffusion_unet_num_embedding_channels(self):
        """Getter for number of embedding channels in UNET"""
        return self._latent_diffusion_unet_num_embedding_channels

    @latent_diffusion_unet_num_embedding_channels.setter
    def latent_diffusion_unet_num_embedding_channels(self, value):
        """Setter for number of embedding channels in UNET"""
        self._latent_diffusion_unet_num_embedding_channels = value

    @property
    def latent_diffusion_unet_channels_per_level(self):
        """Getter for channels per level in UNET"""
        return self._latent_diffusion_unet_channels_per_level

    @latent_diffusion_unet_channels_per_level.setter
    def latent_diffusion_unet_channels_per_level(self, value):
        """Setter for channels per level in UNET"""
        self._latent_diffusion_unet_channels_per_level = value

    @property
    def latent_diffusion_unet_batch_size(self):
        """Getter for UNET batch size"""
        return self._latent_diffusion_unet_batch_size

    @latent_diffusion_unet_batch_size.setter
    def latent_diffusion_unet_batch_size(self, value):
        """Setter for UNET batch size"""
        self._latent_diffusion_unet_batch_size = value

    @property
    def latent_diffusion_unet_attention_mode(self):
        """Getter for UNET attention mode"""
        return self._latent_diffusion_unet_attention_mode

    @latent_diffusion_unet_attention_mode.setter
    def latent_diffusion_unet_attention_mode(self, value):
        """Setter for UNET attention mode"""
        self._latent_diffusion_unet_attention_mode = value

    @property
    def latent_diffusion_unet_num_residual_blocks(self):
        """Getter for number of residual blocks in UNET"""
        return self._latent_diffusion_unet_num_residual_blocks

    @latent_diffusion_unet_num_residual_blocks.setter
    def latent_diffusion_unet_num_residual_blocks(self, value):
        """Setter for number of residual blocks in UNET"""
        self._latent_diffusion_unet_num_residual_blocks = value

    @property
    def latent_diffusion_unet_group_normalization(self):
        """Getter for UNET group normalization setting"""
        return self._latent_diffusion_unet_group_normalization

    @latent_diffusion_unet_group_normalization.setter
    def latent_diffusion_unet_group_normalization(self, value):
        """Setter for UNET group normalization setting"""
        self._latent_diffusion_unet_group_normalization = value

    @property
    def latent_diffusion_unet_intermediary_activation(self):
        """Getter for UNET intermediary activation function"""
        return self._latent_diffusion_unet_intermediary_activation

    @latent_diffusion_unet_intermediary_activation.setter
    def latent_diffusion_unet_intermediary_activation(self, value):
        """Setter for UNET intermediary activation function"""
        self._latent_diffusion_unet_intermediary_activation = value

    @property
    def latent_diffusion_unet_intermediary_activation_alpha(self):
        """Getter for alpha parameter of UNET intermediary activation"""
        return self._latent_diffusion_unet_intermediary_activation_alpha

    @latent_diffusion_unet_intermediary_activation_alpha.setter
    def latent_diffusion_unet_intermediary_activation_alpha(self, value):
        """Setter for alpha parameter of UNET intermediary activation"""
        self._latent_diffusion_unet_intermediary_activation_alpha = value

    @property
    def latent_diffusion_unet_epochs(self):
        """Getter for number of training epochs for UNET"""
        return self._latent_diffusion_unet_epochs

    @latent_diffusion_unet_epochs.setter
    def latent_diffusion_unet_epochs(self, value):
        """Setter for number of training epochs for UNET"""
        self._latent_diffusion_unet_epochs = value

    # VAE-related properties
    @property
    def latent_diffusion_VAE_mean_distribution(self):
        """Getter for VAE mean distribution"""
        return self._latent_diffusion_VAE_mean_distribution

    @latent_diffusion_VAE_mean_distribution.setter
    def latent_diffusion_VAE_mean_distribution(self, value):
        """Setter for VAE mean distribution"""
        self._latent_diffusion_VAE_mean_distribution = value

    @property
    def latent_diffusion_VAE_stander_deviation(self):
        """Getter for VAE standard deviation"""
        return self._latent_diffusion_VAE_stander_deviation

    @latent_diffusion_VAE_stander_deviation.setter
    def latent_diffusion_VAE_stander_deviation(self, value):
        """Setter for VAE standard deviation"""
        self._latent_diffusion_VAE_stander_deviation = value

    @property
    def latent_diffusion_VAE_file_name_encoder(self):
        """Getter for VAE encoder filename"""
        return self._latent_diffusion_VAE_file_name_encoder

    @latent_diffusion_VAE_file_name_encoder.setter
    def latent_diffusion_VAE_file_name_encoder(self, value):
        """Setter for VAE encoder filename"""
        self._latent_diffusion_VAE_file_name_encoder = value

    @property
    def latent_diffusion_VAE_file_name_decoder(self):
        """Getter for VAE decoder filename"""
        return self._latent_diffusion_VAE_file_name_decoder

    @latent_diffusion_VAE_file_name_decoder.setter
    def latent_diffusion_VAE_file_name_decoder(self, value):
        """Setter for VAE decoder filename"""
        self._latent_diffusion_VAE_file_name_decoder = value

    @property
    def latent_diffusion_VAE_path_output_models(self):
        """Getter for VAE output models path"""
        return self._latent_diffusion_VAE_path_output_models

    @latent_diffusion_VAE_path_output_models.setter
    def latent_diffusion_VAE_path_output_models(self, value):
        """Setter for VAE output models path"""
        self._latent_diffusion_VAE_path_output_models = value

    # Gaussian diffusion properties
    @property
    def latent_diffusion_gaussian_beta_start(self):
        """Getter for Gaussian diffusion beta start value"""
        return self._latent_diffusion_gaussian_beta_start

    @latent_diffusion_gaussian_beta_start.setter
    def latent_diffusion_gaussian_beta_start(self, value):
        """Setter for Gaussian diffusion beta start value"""
        self._latent_diffusion_gaussian_beta_start = value

    @property
    def latent_diffusion_gaussian_beta_end(self):
        """Getter for Gaussian diffusion beta end value"""
        return self._latent_diffusion_gaussian_beta_end

    @latent_diffusion_gaussian_beta_end.setter
    def latent_diffusion_gaussian_beta_end(self, value):
        """Setter for Gaussian diffusion beta end value"""
        self._latent_diffusion_gaussian_beta_end = value

    @property
    def latent_diffusion_gaussian_time_steps(self):
        """Getter for number of Gaussian diffusion time steps"""
        return self._latent_diffusion_gaussian_time_steps

    @latent_diffusion_gaussian_time_steps.setter
    def latent_diffusion_gaussian_time_steps(self, value):
        """Setter for number of Gaussian diffusion time steps"""
        self._latent_diffusion_gaussian_time_steps = value

    @property
    def latent_diffusion_gaussian_clip_min(self):
        """Getter for Gaussian diffusion minimum clip value"""
        return self._latent_diffusion_gaussian_clip_min

    @latent_diffusion_gaussian_clip_min.setter
    def latent_diffusion_gaussian_clip_min(self, value):
        """Setter for Gaussian diffusion minimum clip value"""
        self._latent_diffusion_gaussian_clip_min = value

    @property
    def latent_diffusion_gaussian_clip_max(self):
        """Getter for Gaussian diffusion maximum clip value"""
        return self._latent_diffusion_gaussian_clip_max

    @latent_diffusion_gaussian_clip_max.setter
    def latent_diffusion_gaussian_clip_max(self, value):
        """Setter for Gaussian diffusion maximum clip value"""
        self._latent_diffusion_gaussian_clip_max = value

    # More VAE properties
    @property
    def latent_diffusion_VAE_loss_function(self):
        """Getter for VAE loss function"""
        return self._latent_diffusion_VAE_loss_function

    @latent_diffusion_VAE_loss_function.setter
    def latent_diffusion_VAE_loss_function(self, value):
        """Setter for VAE loss function"""
        self._latent_diffusion_VAE_loss_function = value

    @property
    def latent_diffusion_VAE_encoder_filters(self):
        """Getter for VAE encoder filters"""
        return self._latent_diffusion_VAE_encoder_filters

    @latent_diffusion_VAE_encoder_filters.setter
    def latent_diffusion_VAE_encoder_filters(self, value):
        """Setter for VAE encoder filters"""
        self._latent_diffusion_VAE_encoder_filters = value

    @property
    def latent_diffusion_VAE_decoder_filters(self):
        """Getter for VAE decoder filters"""
        return self._latent_diffusion_VAE_decoder_filters

    @latent_diffusion_VAE_decoder_filters.setter
    def latent_diffusion_VAE_decoder_filters(self, value):
        """Setter for VAE decoder filters"""
        self._latent_diffusion_VAE_decoder_filters = value

    @property
    def latent_diffusion_VAE_last_layer_activation(self):
        """Getter for VAE last layer activation"""
        return self._latent_diffusion_VAE_last_layer_activation

    @latent_diffusion_VAE_last_layer_activation.setter
    def latent_diffusion_VAE_last_layer_activation(self, value):
        """Setter for VAE last layer activation"""
        self._latent_diffusion_VAE_last_layer_activation = value

    @property
    def latent_diffusion_VAE_latent_dimension(self):
        """Getter for VAE latent dimension"""
        return self._latent_diffusion_VAE_latent_dimension

    @latent_diffusion_VAE_latent_dimension.setter
    def latent_diffusion_VAE_latent_dimension(self, value):
        """Setter for VAE latent dimension"""
        self._latent_diffusion_VAE_latent_dimension = value

    @property
    def latent_diffusion_VAE_batch_size_create_embedding(self):
        """Getter for VAE embedding creation batch size"""
        return self._latent_diffusion_VAE_batch_size_create_embedding

    @latent_diffusion_VAE_batch_size_create_embedding.setter
    def latent_diffusion_VAE_batch_size_create_embedding(self, value):
        """Setter for VAE embedding creation batch size"""
        self._latent_diffusion_VAE_batch_size_create_embedding = value

    @property
    def latent_diffusion_VAE_batch_size_training(self):
        """Getter for VAE training batch size"""
        return self._latent_diffusion_VAE_batch_size_training

    @latent_diffusion_VAE_batch_size_training.setter
    def latent_diffusion_VAE_batch_size_training(self, value):
        """Setter for VAE training batch size"""
        self._latent_diffusion_VAE_batch_size_training = value

    @property
    def latent_diffusion_VAE_epochs(self):
        """Getter for VAE training epochs"""
        return self._latent_diffusion_VAE_epochs

    @latent_diffusion_VAE_epochs.setter
    def latent_diffusion_VAE_epochs(self, value):
        """Setter for VAE training epochs"""
        self._latent_diffusion_VAE_epochs = value

    @property
    def latent_diffusion_VAE_intermediary_activation_function(self):
        """Getter for VAE intermediary activation function"""
        return self._latent_diffusion_VAE_intermediary_activation_function

    @latent_diffusion_VAE_intermediary_activation_function.setter
    def latent_diffusion_VAE_intermediary_activation_function(self, value):
        """Setter for VAE intermediary activation function"""
        self._latent_diffusion_VAE_intermediary_activation_function = value

    @property
    def latent_diffusion_VAE_intermediary_activation_alpha(self):
        """Getter for VAE intermediary activation alpha parameter"""
        return self._latent_diffusion_VAE_intermediary_activation_alpha

    @latent_diffusion_VAE_intermediary_activation_alpha.setter
    def latent_diffusion_VAE_intermediary_activation_alpha(self, value):
        """Setter for VAE intermediary activation alpha parameter"""
        self._latent_diffusion_VAE_intermediary_activation_alpha = value

    @property
    def latent_diffusion_VAE_activation_output_encoder(self):
        """Getter for VAE encoder output activation"""
        return self._latent_diffusion_VAE_activation_output_encoder

    @latent_diffusion_VAE_activation_output_encoder.setter
    def latent_diffusion_VAE_activation_output_encoder(self, value):
        """Setter for VAE encoder output activation"""
        self._latent_diffusion_VAE_activation_output_encoder = value

    @property
    def latent_diffusion_margin(self):
        """Getter for latent diffusion margin parameter"""
        return self._latent_diffusion_margin

    @latent_diffusion_margin.setter
    def latent_diffusion_margin(self, value):
        """Setter for latent diffusion margin parameter"""
        self._latent_diffusion_margin = value

    @property
    def latent_diffusion_ema(self):
        """Getter for EMA (Exponential Moving Average) setting"""
        return self._latent_diffusion_ema

    @latent_diffusion_ema.setter
    def latent_diffusion_ema(self, value):
        """Setter for EMA (Exponential Moving Average) setting"""
        self._latent_diffusion_ema = value

    @property
    def latent_diffusion_time_steps(self):
        """Getter for number of diffusion time steps"""
        return self._latent_diffusion_time_steps

    @latent_diffusion_time_steps.setter
    def latent_diffusion_time_steps(self, value):
        """Setter for number of diffusion time steps"""
        self._latent_diffusion_time_steps = value

    # ** Gaussian LatentDiffusion Configuration Parameters **
    @property
    def latent_diffusion_VAE_initializer_mean(self):
        """Getter for VAE initializer mean"""
        return self._latent_diffusion_VAE_initializer_mean

    @latent_diffusion_VAE_initializer_mean.setter
    def latent_diffusion_VAE_initializer_mean(self, value):
        """Setter for VAE initializer mean"""
        self._latent_diffusion_VAE_initializer_mean = value

    @property
    def latent_diffusion_VAE_initializer_deviation(self):
        """Getter for VAE initializer standard deviation"""
        return self._latent_diffusion_VAE_initializer_deviation

    @latent_diffusion_VAE_initializer_deviation.setter
    def latent_diffusion_VAE_initializer_deviation(self, value):
        """Setter for VAE initializer standard deviation"""
        self._latent_diffusion_VAE_initializer_deviation = value

    @property
    def latent_diffusion_VAE_dropout_decay_rate_encoder(self):
        """Getter for VAE encoder dropout decay rate"""
        return self._latent_diffusion_VAE_dropout_decay_rate_encoder

    @latent_diffusion_VAE_dropout_decay_rate_encoder.setter
    def latent_diffusion_VAE_dropout_decay_rate_encoder(self, value):
        """Setter for VAE encoder dropout decay rate"""
        self._latent_diffusion_VAE_dropout_decay_rate_encoder = value

    @property
    def latent_diffusion_VAE_dropout_decay_rate_decoder(self):
        """Getter for VAE decoder dropout decay rate"""
        return self._latent_diffusion_VAE_dropout_decay_rate_decoder

    @latent_diffusion_VAE_dropout_decay_rate_decoder.setter
    def latent_diffusion_VAE_dropout_decay_rate_decoder(self, value):
        """Setter for VAE decoder dropout decay rate"""
        self._latent_diffusion_VAE_dropout_decay_rate_decoder = value



class WassersteinInstance:
    """
     A class that implements a Wasserstein Generative Adversarial Network (WGAN).
     This implementation follows the Wasserstein GAN framework with improved training stability.

     Key Components:
     - Generator model for synthetic sample generation
     - Critic/Discriminator model (with Wasserstein loss)
     - Custom training loop with critic pre-training steps
     - Flexible architecture configuration via arguments

     Attributes:
         _wasserstein_algorithm: Orchestrates the WGAN-GP training process
         _wasserstein_model: Stores the generator and critic/discriminator models

         # WGAN Architecture Parameters
         _wasserstein_latent_dimension: Dimensionality of the latent space
         _wasserstein_training_algorithm: Type of training algorithm used
         _wasserstein_activation_function: Activation function for hidden layers
         _wasserstein_dropout_decay_rate_g: Dropout rate decay for generator
         _wasserstein_dropout_decay_rate_d: Dropout rate decay for discriminator
         _wasserstein_dense_layer_sizes_generator: Layer sizes for generator
         _wasserstein_dense_layer_sizes_discriminator: Layer sizes for discriminator
         _wasserstein_batch_size: Batch size for training
         _wasserstein_number_epochs: Number of training epochs
         _wasserstein_number_classes: Number of output classes
         _wasserstein_loss_function: Loss function used for optimization
         _wasserstein_momentum: Momentum parameter for optimizers
         _wasserstein_last_activation_layer: Activation for final layer
         _wasserstein_initializer_mean: Mean for weight initialization
         _wasserstein_initializer_deviation: Std dev for weight initialization

         # Optimization Parameters
         _wasserstein_optimizer_generator_learning_rate: Generator learning rate
         _wasserstein_optimizer_discriminator_learning_rate: Discriminator learning rate
         _wasserstein_optimizer_generator_beta: Beta parameter for generator optimizer
         _wasserstein_optimizer_discriminator_beta: Beta parameter for discriminator optimizer
         _wasserstein_discriminator_steps: Number of critic steps per generator step

         _wasserstein_smoothing_rate: Label smoothing rate
         _wasserstein_latent_mean_distribution: Distribution type for latent space
         _wasserstein_latent_stander_deviation: Std dev for latent distribution
         _wasserstein_file_name_discriminator: Filename for saving critic
         _wasserstein_file_name_generator: Filename for saving generator
         _wasserstein_path_output_models: Path for saving models
     """

    def __init__(self, arguments):
        """
        Initializes the Wasserstein GAN instance with configuration parameters.

        Args:
            arguments (Namespace): Configuration object containing:
                - Generator and discriminator architecture parameters
                - Training hyperparameters
                - Optimization settings
                - WGAN specific configurations
                - Model saving paths
        """
        self._wasserstein_algorithm = None
        self._wasserstein_model = None

        # ** Wasserstein GAN Configuration Parameters **
        self._wasserstein_latent_dimension = arguments.wasserstein_latent_dimension
        self._wasserstein_training_algorithm = arguments.wasserstein_training_algorithm
        self._wasserstein_activation_function = arguments.wasserstein_activation_function
        self._wasserstein_dropout_decay_rate_g = arguments.wasserstein_dropout_decay_rate_g
        self._wasserstein_dropout_decay_rate_d = arguments.wasserstein_dropout_decay_rate_d
        self._wasserstein_dense_layer_sizes_generator = arguments.wasserstein_dense_layer_sizes_generator
        self._wasserstein_dense_layer_sizes_discriminator = arguments.wasserstein_dense_layer_sizes_discriminator
        self._wasserstein_batch_size = arguments.wasserstein_batch_size
        self._wasserstein_number_epochs = arguments.wasserstein_number_epochs
        self._wasserstein_number_classes = arguments.wasserstein_number_classes
        self._wasserstein_loss_function = arguments.wasserstein_loss_function
        self._wasserstein_momentum = arguments.wasserstein_momentum
        self._wasserstein_last_activation_layer = arguments.wasserstein_last_activation_layer
        self._wasserstein_initializer_mean = arguments.wasserstein_initializer_mean
        self._wasserstein_initializer_deviation = arguments.wasserstein_initializer_deviation
        self._wasserstein_optimizer_generator_learning_rate = arguments.wasserstein_optimizer_generator_learning_rate
        self._wasserstein_optimizer_discriminator_learning_rate = arguments.wasserstein_optimizer_discriminator_learning_rate
        self._wasserstein_optimizer_generator_beta = arguments.wasserstein_optimizer_generator_beta
        self._wasserstein_optimizer_discriminator_beta = arguments.wasserstein_optimizer_discriminator_beta
        self._wasserstein_discriminator_steps = arguments.wasserstein_discriminator_steps
        self._wasserstein_smoothing_rate = arguments.wasserstein_smoothing_rate
        self._wasserstein_latent_mean_distribution = arguments.wasserstein_latent_mean_distribution
        self._wasserstein_latent_stander_deviation = arguments.wasserstein_latent_stander_deviation
        self._wasserstein_file_name_discriminator = arguments.wasserstein_file_name_discriminator
        self._wasserstein_file_name_generator = arguments.wasserstein_file_name_generator
        self._wasserstein_path_output_models = arguments.wasserstein_path_output_models


    def _get_wasserstein(self, input_shape):
        """
        Initializes and sets up a Wasserstein GAN model.

        This method sets up a Wasserstein Generative Adversarial Network (WGAN) by configuring the generator and discriminator
        models using custom `WassersteinModel` and `WassersteinAlgorithm` classes. The generator and discriminator are created
        and configured with their respective parameters, including latent dimensions, activation functions, loss functions,
        and other hyperparameters specific to the WassersteinGP GAN architecture.

        Args:
            input_shape (tuple): The shape of the input data, which determines the output shape for the models.

        Initializes:
            self._wasserstein_model: An instance of the `WassersteinModel` class, which includes the generator and discriminator
                                     setup with configurations like latent dimension, activation functions, dropout rates,
                                     and dense layer sizes.
            self._wasserstein_algorithm: An instance of the `WassersteinAlgorithm` class that manages the training process
                                         of the Wasserstein GAN, including generator and discriminator loss functions,
                                         gradient penalty, and model parameters such as file names for saving and latent
                                         distributions.

        """

        # Wasserstein Model setup for the Generator and Discriminator
        self._wasserstein_model = WassersteinModel(latent_dimension=self._wasserstein_latent_dimension,
                                                     output_shape=input_shape,
                                                     activation_function=self._wasserstein_activation_function,
                                                     initializer_mean=self._wasserstein_initializer_mean,
                                                     initializer_deviation=self._wasserstein_initializer_deviation,
                                                     dropout_decay_rate_g=self._wasserstein_dropout_decay_rate_g,
                                                     dropout_decay_rate_d=self._wasserstein_dropout_decay_rate_d,
                                                     last_layer_activation=self._wasserstein_last_activation_layer,
                                                     dense_layer_sizes_g=self._wasserstein_dense_layer_sizes_generator,
                                                     dense_layer_sizes_d=self._wasserstein_dense_layer_sizes_discriminator,
                                                     dataset_type=numpy.float32,
                                                     number_samples_per_class = self._number_samples_per_class)

        # Wasserstein Algorithm setup for training and model operations
        self._wasserstein_algorithm = WassersteinAlgorithm(generator_model=self._wasserstein_model.get_generator(),
                                                                discriminator_model=self._wasserstein_model.get_discriminator(),
                                                                latent_dimension=self._wasserstein_latent_dimension,
                                                                generator_loss_fn=self._wasserstein_loss_function,
                                                                discriminator_loss_fn=self._wasserstein_loss_function,
                                                                file_name_discriminator=self._wasserstein_file_name_discriminator,
                                                                file_name_generator=self._wasserstein_file_name_generator,
                                                                models_saved_path=self._wasserstein_path_output_models,
                                                                latent_mean_distribution=self._wasserstein_latent_mean_distribution,
                                                                latent_standard_deviation=self._wasserstein_latent_stander_deviation,
                                                                smoothing_rate=self._wasserstein_smoothing_rate,
                                                                discriminator_steps=self._wasserstein_discriminator_steps,
                                                                clip_value=0.01)

    def _training_wasserstein_model(self, input_shape, arguments, x_real_samples, y_real_samples):
        """
        Executes the complete training pipeline for Wasserstein GAN with Gradient Penalty.

        Process:
        1. Initializes generator and critic models
        2. Configures custom Wasserstein loss functions
        3. Sets up optimizers with specified parameters
        4. Trains using alternating critic/generator updates
        5. Manages callbacks and monitoring

        Args:
            input_shape (tuple): Input data shape
            arguments (Namespace): Training configuration
            x_real_samples (ndarray): Training samples
            y_real_samples (ndarray): Corresponding labels
        """
        # Initialize the WassersteinGP model
        self._get_wasserstein(input_shape)

        # Print the model summaries for the generator and discriminator
        self._wasserstein_model.get_generator().summary()
        self._wasserstein_model.get_discriminator().summary()

        # Define the custom loss functions for the discriminator and generator
        def discriminator_loss(real_img, fake_img):
            return tensorflow.reduce_mean(fake_img) - tensorflow.reduce_mean(real_img)

        def generator_loss(fake_img):
            return -tensorflow.reduce_mean(fake_img)

        generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

        # Compile the Wasserstein GAN algorithm
        self._wasserstein_algorithm.compile(generator_optimizer,
                                            discriminator_optimizer,
                                            generator_loss,
                                            discriminator_loss)

        callbacks_list = [self._callback_model_monitor]

        if arguments.use_early_stop:
            callbacks_list.append(self._callback_early_stop)

        # Fit the WassersteinGP GAN model
        self._wasserstein_algorithm.fit(
            x_real_samples,
            to_categorical(y_real_samples, num_classes=self._number_samples_per_class["number_classes"]),
            epochs=self._wasserstein_number_epochs, batch_size=self._wasserstein_batch_size,
            callbacks=callbacks_list)

    # Getter and setter for wasserstein_latent_dimension
    @property
    def wasserstein_latent_dimension(self):
        return self._wasserstein_latent_dimension

    @wasserstein_latent_dimension.setter
    def wasserstein_latent_dimension(self, value):
        self._wasserstein_latent_dimension = value

    # Getter and setter for wasserstein_training_algorithm
    @property
    def wasserstein_training_algorithm(self):
        return self._wasserstein_training_algorithm

    @wasserstein_training_algorithm.setter
    def wasserstein_training_algorithm(self, value):
        self._wasserstein_training_algorithm = value

    # Getter and setter for wasserstein_activation_function
    @property
    def wasserstein_activation_function(self):
        return self._wasserstein_activation_function

    @wasserstein_activation_function.setter
    def wasserstein_activation_function(self, value):
        self._wasserstein_activation_function = value

    # Getter and setter for wasserstein_dropout_decay_rate_g
    @property
    def wasserstein_dropout_decay_rate_g(self):
        return self._wasserstein_dropout_decay_rate_g

    @wasserstein_dropout_decay_rate_g.setter
    def wasserstein_dropout_decay_rate_g(self, value):
        self._wasserstein_dropout_decay_rate_g = value

    # Getter and setter for wasserstein_dropout_decay_rate_d
    @property
    def wasserstein_dropout_decay_rate_d(self):
        return self._wasserstein_dropout_decay_rate_d

    @wasserstein_dropout_decay_rate_d.setter
    def wasserstein_dropout_decay_rate_d(self, value):
        self._wasserstein_dropout_decay_rate_d = value

    # Getter and setter for wasserstein_dense_layer_sizes_generator
    @property
    def wasserstein_dense_layer_sizes_generator(self):
        return self._wasserstein_dense_layer_sizes_generator

    @wasserstein_dense_layer_sizes_generator.setter
    def wasserstein_dense_layer_sizes_generator(self, value):
        self._wasserstein_dense_layer_sizes_generator = value

    # Getter and setter for wasserstein_dense_layer_sizes_discriminator
    @property
    def wasserstein_dense_layer_sizes_discriminator(self):
        return self._wasserstein_dense_layer_sizes_discriminator

    @wasserstein_dense_layer_sizes_discriminator.setter
    def wasserstein_dense_layer_sizes_discriminator(self, value):
        self._wasserstein_dense_layer_sizes_discriminator = value

    # Getter and setter for wasserstein_batch_size
    @property
    def wasserstein_batch_size(self):
        return self._wasserstein_batch_size

    @wasserstein_batch_size.setter
    def wasserstein_batch_size(self, value):
        self._wasserstein_batch_size = value

    # Getter and setter for wasserstein_number_classes
    @property
    def wasserstein_number_classes(self):
        return self._wasserstein_number_classes

    @wasserstein_number_classes.setter
    def wasserstein_number_classes(self, value):
        self._wasserstein_number_classes = value

    # Getter and setter for wasserstein_loss_function
    @property
    def wasserstein_loss_function(self):
        return self._wasserstein_loss_function

    @wasserstein_loss_function.setter
    def wasserstein_loss_function(self, value):
        self._wasserstein_loss_function = value

    # Getter and setter for wasserstein_momentum
    @property
    def wasserstein_momentum(self):
        return self._wasserstein_momentum

    @wasserstein_momentum.setter
    def wasserstein_momentum(self, value):
        self._wasserstein_momentum = value

    # Getter and setter for wasserstein_last_activation_layer
    @property
    def wasserstein_last_activation_layer(self):
        return self._wasserstein_last_activation_layer

    @wasserstein_last_activation_layer.setter
    def wasserstein_last_activation_layer(self, value):
        self._wasserstein_last_activation_layer = value

    # Getter and setter for wasserstein_initializer_mean
    @property
    def wasserstein_initializer_mean(self):
        return self._wasserstein_initializer_mean

    @wasserstein_initializer_mean.setter
    def wasserstein_initializer_mean(self, value):
        self._wasserstein_initializer_mean = value

    # Getter and setter for wasserstein_initializer_deviation
    @property
    def wasserstein_initializer_deviation(self):
        return self._wasserstein_initializer_deviation

    @wasserstein_initializer_deviation.setter
    def wasserstein_initializer_deviation(self, value):
        self._wasserstein_initializer_deviation = value

    # Getter and setter for wasserstein_optimizer_generator_learning_rate
    @property
    def wasserstein_optimizer_generator_learning_rate(self):
        return self._wasserstein_optimizer_generator_learning_rate

    @wasserstein_optimizer_generator_learning_rate.setter
    def wasserstein_optimizer_generator_learning_rate(self, value):
        self._wasserstein_optimizer_generator_learning_rate = value

    # Getter and setter for wasserstein_optimizer_discriminator_learning_rate
    @property
    def wasserstein_optimizer_discriminator_learning_rate(self):
        return self._wasserstein_optimizer_discriminator_learning_rate

    @wasserstein_optimizer_discriminator_learning_rate.setter
    def wasserstein_optimizer_discriminator_learning_rate(self, value):
        self._wasserstein_optimizer_discriminator_learning_rate = value

    # Getter and setter for wasserstein_optimizer_generator_beta
    @property
    def wasserstein_optimizer_generator_beta(self):
        return self._wasserstein_optimizer_generator_beta

    @wasserstein_optimizer_generator_beta.setter
    def wasserstein_optimizer_generator_beta(self, value):
        self._wasserstein_optimizer_generator_beta = value

    # Getter and setter for wasserstein_optimizer_discriminator_beta
    @property
    def wasserstein_optimizer_discriminator_beta(self):
        return self._wasserstein_optimizer_discriminator_beta

    @wasserstein_optimizer_discriminator_beta.setter
    def wasserstein_optimizer_discriminator_beta(self, value):
        self._wasserstein_optimizer_discriminator_beta = value

    # Getter and setter for wasserstein_discriminator_steps
    @property
    def wasserstein_discriminator_steps(self):
        return self._wasserstein_discriminator_steps

    @wasserstein_discriminator_steps.setter
    def wasserstein_discriminator_steps(self, value):
        self._wasserstein_discriminator_steps = value

    # Getter and setter for wasserstein_smoothing_rate
    @property
    def wasserstein_smoothing_rate(self):
        return self._wasserstein_smoothing_rate

    @wasserstein_smoothing_rate.setter
    def wasserstein_smoothing_rate(self, value):
        self._wasserstein_smoothing_rate = value

    # Getter and setter for wasserstein_latent_mean_distribution
    @property
    def wasserstein_latent_mean_distribution(self):
        return self._wasserstein_latent_mean_distribution

    @wasserstein_latent_mean_distribution.setter
    def wasserstein_latent_mean_distribution(self, value):
        self._wasserstein_latent_mean_distribution = value

    # Getter and setter for wasserstein_latent_stander_deviation
    @property
    def wasserstein_latent_stander_deviation(self):
        return self._wasserstein_latent_stander_deviation

    @wasserstein_latent_stander_deviation.setter
    def wasserstein_latent_stander_deviation(self, value):
        self._wasserstein_latent_stander_deviation = value

    # Getter and setter for wasserstein_file_name_discriminator
    @property
    def wasserstein_file_name_discriminator(self):
        return self._wasserstein_file_name_discriminator

    @wasserstein_file_name_discriminator.setter
    def wasserstein_file_name_discriminator(self, value):
        self._wasserstein_file_name_discriminator = value

    # Getter and setter for wasserstein_file_name_generator
    @property
    def wasserstein_file_name_generator(self):
        return self._wasserstein_file_name_generator

    @wasserstein_file_name_generator.setter
    def wasserstein_file_name_generator(self, value):
        self._wasserstein_file_name_generator = value

    # Getter and setter for wasserstein_path_output_models
    @property
    def wasserstein_path_output_models(self):
        return self._wasserstein_path_output_models

    @wasserstein_path_output_models.setter
    def wasserstein_path_output_models(self, value):
        self._wasserstein_path_output_models = value



class WassersteinGPInstance:
    """
    A class that implements a Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP).
    This version improves upon standard WGAN by using gradient penalty instead of weight clipping
    to enforce the Lipschitz constraint, leading to more stable training and higher quality results.

    Key Components:
    - Generator model for synthetic sample generation
    - Critic model (Wasserstein discriminator) with gradient penalty
    - Custom training loop with critic pre-training steps
    - Gradient penalty for Lipschitz constraint enforcement
    - Flexible architecture configuration via arguments

    Attributes:
        _wasserstein_gp_algorithm: Orchestrates the WGAN-GP training process
        _wasserstein_gp_model: Stores the generator and critic models

        # WGAN-GP Architecture Parameters
        _wasserstein_gp_latent_dimension: Dimensionality of the latent space
        _wasserstein_gp_training_algorithm: Type of training algorithm used
        _wasserstein_gp_activation_function: Activation function for hidden layers
        _wasserstein_gp_dropout_decay_rate_g: Dropout rate decay for generator
        _wasserstein_gp_dropout_decay_rate_d: Dropout rate decay for critic
        _wasserstein_gp_dense_layer_sizes_generator: Layer sizes for generator
        _wasserstein_gp_dense_layer_sizes_discriminator: Layer sizes for critic
        _wasserstein_gp_batch_size: Batch size for training
        _wasserstein_gp_number_epochs: Number of training epochs
        _wasserstein_gp_number_classes: Number of output classes
        _wasserstein_gp_loss_function: Base loss function used
        _wasserstein_gp_momentum: Momentum parameter for optimizers
        _wasserstein_gp_last_activation_layer: Activation for final layer
        _wasserstein_gp_initializer_mean: Mean for weight initialization
        _wasserstein_gp_initializer_deviation: Std dev for weight initialization

        # Optimization Parameters
        _wasserstein_gp_optimizer_generator_learning_rate: Generator learning rate
        _wasserstein_gp_optimizer_discriminator_learning_rate: Critic learning rate
        _wasserstein_gp_optimizer_generator_beta: Beta1 for generator optimizer
        _wasserstein_gp_optimizer_discriminator_beta: Beta1 for critic optimizer
        _wasserstein_gp_discriminator_steps: Number of critic steps per generator step

        # WGAN-GP Specific Parameters
        _wasserstein_gp_smoothing_rate: Label smoothing rate
        _wasserstein_gp_latent_mean_distribution: Distribution type for latent space
        _wasserstein_gp_latent_stander_deviation: Std dev for latent distribution
        _wasserstein_gp_gradient_penalty: Weight for gradient penalty term
        _wasserstein_gp_file_name_discriminator: Filename for saving critic
        _wasserstein_gp_file_name_generator: Filename for saving generator
        _wasserstein_gp_path_output_models: Path for saving models
    """

    def __init__(self, arguments):
        """
        Initializes the WGAN-GP instance with configuration parameters.

        Args:
            arguments (Namespace): Configuration object containing:
                - Generator and critic architecture parameters
                - Training hyperparameters
                - Optimization settings
                - WGAN-GP specific configurations
                - Model saving paths
        """

        self._wasserstein_gp_algorithm = None
        self._wasserstein_gp_model = None

        # ** WassersteinGP GAN with Gradient Penalty (WGAN-GP) Configuration Parameters **
        self._wasserstein_gp_latent_dimension = arguments.wasserstein_gp_latent_dimension
        self._wasserstein_gp_training_algorithm = arguments.wasserstein_gp_training_algorithm
        self._wasserstein_gp_activation_function = arguments.wasserstein_gp_activation_function
        self._wasserstein_gp_dropout_decay_rate_g = arguments.wasserstein_gp_dropout_decay_rate_g
        self._wasserstein_gp_dropout_decay_rate_d = arguments.wasserstein_gp_dropout_decay_rate_d
        self._wasserstein_gp_dense_layer_sizes_generator = arguments.wasserstein_gp_dense_layer_sizes_generator
        self._wasserstein_gp_dense_layer_sizes_discriminator = arguments.wasserstein_gp_dense_layer_sizes_discriminator
        self._wasserstein_gp_batch_size = arguments.wasserstein_gp_batch_size
        self._wasserstein_gp_number_epochs = arguments.wasserstein_gp_number_epochs
        self._wasserstein_gp_number_classes = arguments.wasserstein_gp_number_classes
        self._wasserstein_gp_loss_function = arguments.wasserstein_gp_loss_function
        self._wasserstein_gp_momentum = arguments.wasserstein_gp_momentum
        self._wasserstein_gp_last_activation_layer = arguments.wasserstein_gp_last_activation_layer
        self._wasserstein_gp_initializer_mean = arguments.wasserstein_gp_initializer_mean
        self._wasserstein_gp_initializer_deviation = arguments.wasserstein_gp_initializer_deviation
        self._wasserstein_gp_optimizer_generator_learning_rate = arguments.wasserstein_gp_optimizer_generator_learning_rate
        self._wasserstein_gp_optimizer_discriminator_learning_rate = arguments.wasserstein_gp_optimizer_discriminator_learning_rate
        self._wasserstein_gp_optimizer_generator_beta = arguments.wasserstein_gp_optimizer_generator_beta
        self._wasserstein_gp_optimizer_discriminator_beta = arguments.wasserstein_gp_optimizer_discriminator_beta
        self._wasserstein_gp_discriminator_steps = arguments.wasserstein_gp_discriminator_steps

        # WGAN-GP Specific Parameters
        self._wasserstein_gp_smoothing_rate = arguments.wasserstein_gp_smoothing_rate
        self._wasserstein_gp_latent_mean_distribution = arguments.wasserstein_gp_latent_mean_distribution
        self._wasserstein_gp_latent_stander_deviation = arguments.wasserstein_gp_latent_stander_deviation
        self._wasserstein_gp_gradient_penalty = arguments.wasserstein_gp_gradient_penalty

        # Model Persistence
        self._wasserstein_gp_file_name_discriminator = arguments.wasserstein_gp_file_name_discriminator
        self._wasserstein_gp_file_name_generator = arguments.wasserstein_gp_file_name_generator
        self._wasserstein_gp_path_output_models = arguments.wasserstein_gp_path_output_models

    def _get_wasserstein_gp(self, input_shape):
        """
        Initializes and sets up a WassersteinGP GAN model.

        This method sets up a WassersteinGP Generative Adversarial Network (WGAN) by configuring the generator and discriminator
        models using custom `WassersteinModel` and `WassersteinAlgorithm` classes. The generator and discriminator are created
        and configured with their respective parameters, including latent dimensions, activation functions, loss functions,
        and other hyperparameters specific to the WassersteinGP GAN architecture.

        Args:
            input_shape (tuple): The shape of the input data, which determines the output shape for the models.

        Initializes:
            self._wasserstein_model: An instance of the `WassersteinModel` class, which includes the generator and discriminator
                                     setup with configurations like latent dimension, activation functions, dropout rates,
                                     and dense layer sizes.
            self._wasserstein_algorithm: An instance of the `WassersteinAlgorithm` class that manages the training process
                                         of the WassersteinGP GAN, including generator and discriminator loss functions,
                                         gradient penalty, and model parameters such as file names for saving and latent
                                         distributions.

        """

        # WassersteinGP Model setup for the Generator and Discriminator
        self._wasserstein_gp_model = WassersteinGPModel(latent_dimension=self._wasserstein_gp_latent_dimension,
                                                      output_shape=input_shape,
                                                      activation_function=self._wasserstein_gp_activation_function,
                                                      initializer_mean=self._wasserstein_gp_initializer_mean,
                                                      initializer_deviation=self._wasserstein_gp_initializer_deviation,
                                                      dropout_decay_rate_g=self._wasserstein_gp_dropout_decay_rate_g,
                                                      dropout_decay_rate_d=self._wasserstein_gp_dropout_decay_rate_d,
                                                      last_layer_activation=self._wasserstein_gp_last_activation_layer,
                                                      dense_layer_sizes_g=self._wasserstein_gp_dense_layer_sizes_generator,
                                                      dense_layer_sizes_d=self._wasserstein_gp_dense_layer_sizes_discriminator,
                                                      dataset_type=numpy.float32,
                                                      number_samples_per_class = self._number_samples_per_class)

        # WassersteinGP Algorithm setup for training and model operations
        self._wasserstein_gp_algorithm = WassersteinGPAlgorithm(generator_model=self._wasserstein_gp_model.get_generator(),
                                                                discriminator_model=self._wasserstein_gp_model.get_discriminator(),
                                                                latent_dimension=self._wasserstein_gp_latent_dimension,
                                                                generator_loss_fn=self._wasserstein_gp_loss_function,
                                                                discriminator_loss_fn=self._wasserstein_gp_loss_function,
                                                                file_name_discriminator=self._wasserstein_gp_file_name_discriminator,
                                                                file_name_generator=self._wasserstein_gp_file_name_generator,
                                                                models_saved_path=self._wasserstein_gp_path_output_models,
                                                                latent_mean_distribution=self._wasserstein_gp_latent_mean_distribution,
                                                                latent_stander_deviation=self._wasserstein_gp_latent_stander_deviation,
                                                                smoothing_rate=self._wasserstein_gp_smoothing_rate,
                                                                gradient_penalty_weight=self._wasserstein_gp_gradient_penalty,
                                                                discriminator_steps=self._wasserstein_gp_discriminator_steps)

    def _training_wasserstein_gp_model(self, input_shape, arguments, x_real_samples, y_real_samples):
        """
        Executes the complete WGAN-GP training pipeline.

        The training process:
        1. Initializes generator and critic models
        2. Configures custom Wasserstein loss with gradient penalty
        3. Sets up optimizers with specified parameters
        4. Alternates between critic and generator updates
        5. Applies gradient penalty during critic training
        6. Manages training callbacks and monitoring

        Args:
            input_shape (tuple): Input data dimensions
            arguments (Namespace): Training configuration parameters
            x_real_samples (ndarray): Training dataset samples
            y_real_samples (ndarray): Corresponding sample labels
        """

        # Initialize the WassersteinGP model
        self._get_wasserstein_gp(input_shape)

        # Print the model summaries for the generator and discriminator
        self._wasserstein_gp_model.get_generator().summary()
        self._wasserstein_gp_model.get_discriminator().summary()

        # Define the custom loss functions for the discriminator and generator
        def discriminator_loss(real_img, fake_img):
            return tensorflow.reduce_mean(fake_img) - tensorflow.reduce_mean(real_img)

        def generator_loss(fake_img):
            return -tensorflow.reduce_mean(fake_img)

        generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

        # Compile the WassersteinGP GAN algorithm
        self._wasserstein_gp_algorithm.compile(generator_optimizer,
                                               discriminator_optimizer,
                                               generator_loss,
                                               discriminator_loss)

        # callbacks_list = [self._callback_resources_monitor, self._callback_model_monitor]
        callbacks_list = [self._callback_model_monitor]

        if arguments.use_early_stop:
            callbacks_list.append(self._callback_early_stop)

        # Fit the WassersteinGP GAN model
        self._wasserstein_gp_algorithm.fit(
            x_real_samples,
            to_categorical(y_real_samples, num_classes=self._number_samples_per_class["number_classes"]),
            epochs=self._wasserstein_gp_number_epochs, batch_size=self._wasserstein_gp_batch_size,
            callbacks=callbacks_list)


    # Getter and setter for wasserstein_latent_dimension
    @property
    def wasserstein_gp_latent_dimension(self):
        return self._wasserstein_gp_latent_dimension

    @wasserstein_gp_latent_dimension.setter
    def wasserstein_gp_latent_dimension(self, value):
        self._wasserstein_gp_latent_dimension = value

    # Getter and setter for wasserstein_training_algorithm
    @property
    def wasserstein_gp_training_algorithm(self):
        return self._wasserstein_gp_training_algorithm

    @wasserstein_gp_training_algorithm.setter
    def wasserstein_gp_training_algorithm(self, value):
        self._wasserstein_gp_training_algorithm = value

    # Getter and setter for wasserstein_activation_function
    @property
    def wasserstein_gp_activation_function(self):
        return self._wasserstein_gp_activation_function

    @wasserstein_gp_activation_function.setter
    def wasserstein_gp_activation_function(self, value):
        self._wasserstein_gp_activation_function = value

    # Getter and setter for wasserstein_dropout_decay_rate_g
    @property
    def wasserstein_gp_dropout_decay_rate_g(self):
        return self._wasserstein_gp_dropout_decay_rate_g

    @wasserstein_gp_dropout_decay_rate_g.setter
    def wasserstein_gp_dropout_decay_rate_g(self, value):
        self._wasserstein_gp_dropout_decay_rate_g = value

    # Getter and setter for wasserstein_dropout_decay_rate_d
    @property
    def wasserstein_gp_dropout_decay_rate_d(self):
        return self._wasserstein_gp_dropout_decay_rate_d

    @wasserstein_gp_dropout_decay_rate_d.setter
    def wasserstein_gp_dropout_decay_rate_d(self, value):
        self._wasserstein_gp_dropout_decay_rate_d = value

    # Getter and setter for wasserstein_dense_layer_sizes_generator
    @property
    def wasserstein_gp_dense_layer_sizes_generator(self):
        return self._wasserstein_gp_dense_layer_sizes_generator

    @wasserstein_gp_dense_layer_sizes_generator.setter
    def wasserstein_gp_dense_layer_sizes_generator(self, value):
        self._wasserstein_gp_dense_layer_sizes_generator = value

    # Getter and setter for wasserstein_dense_layer_sizes_discriminator
    @property
    def wasserstein_gp_dense_layer_sizes_discriminator(self):
        return self._wasserstein_gp_dense_layer_sizes_discriminator

    @wasserstein_gp_dense_layer_sizes_discriminator.setter
    def wasserstein_gp_dense_layer_sizes_discriminator(self, value):
        self._wasserstein_gp_dense_layer_sizes_discriminator = value

    # Getter and setter for wasserstein_batch_size
    @property
    def wasserstein_gp_batch_size(self):
        return self._wasserstein_gp_batch_size

    @wasserstein_gp_batch_size.setter
    def wasserstein_gp_batch_size(self, value):
        self._wasserstein_gp_batch_size = value

    # Getter and setter for wasserstein_number_classes
    @property
    def wasserstein_gp_number_classes(self):
        return self._wasserstein_gp_number_classes

    @wasserstein_gp_number_classes.setter
    def wasserstein_gp_number_classes(self, value):
        self._wasserstein_gp_number_classes = value

    # Getter and setter for wasserstein_loss_function
    @property
    def wasserstein_gp_loss_function(self):
        return self._wasserstein_gp_loss_function

    @wasserstein_gp_loss_function.setter
    def wasserstein_gp_loss_function(self, value):
        self._wasserstein_gp_loss_function = value

    # Getter and setter for wasserstein_momentum
    @property
    def wasserstein_gp_momentum(self):
        return self._wasserstein_gp_momentum

    @wasserstein_gp_momentum.setter
    def wasserstein_gp_momentum(self, value):
        self._wasserstein_gp_momentum = value

    # Getter and setter for wasserstein_last_activation_layer
    @property
    def wasserstein_gp_last_activation_layer(self):
        return self._wasserstein_gp_last_activation_layer

    @wasserstein_gp_last_activation_layer.setter
    def wasserstein_gp_last_activation_layer(self, value):
        self._wasserstein_gp_last_activation_layer = value

    # Getter and setter for wasserstein_initializer_mean
    @property
    def wasserstein_gp_initializer_mean(self):
        return self._wasserstein_gp_initializer_mean

    @wasserstein_gp_initializer_mean.setter
    def wasserstein_gp_initializer_mean(self, value):
        self._wasserstein_gp_initializer_mean = value

    # Getter and setter for wasserstein_initializer_deviation
    @property
    def wasserstein_gp_initializer_deviation(self):
        return self._wasserstein_gp_initializer_deviation

    @wasserstein_gp_initializer_deviation.setter
    def wasserstein_gp_initializer_deviation(self, value):
        self._wasserstein_gp_initializer_deviation = value

    # Getter and setter for wasserstein_optimizer_generator_learning_rate
    @property
    def wasserstein_gp_optimizer_generator_learning_rate(self):
        return self._wasserstein_gp_optimizer_generator_learning_rate

    @wasserstein_gp_optimizer_generator_learning_rate.setter
    def wasserstein_gp_optimizer_generator_learning_rate(self, value):
        self._wasserstein_gp_optimizer_generator_learning_rate = value

    # Getter and setter for wasserstein_optimizer_discriminator_learning_rate
    @property
    def wasserstein_gp_optimizer_discriminator_learning_rate(self):
        return self._wasserstein_gp_optimizer_discriminator_learning_rate

    @wasserstein_gp_optimizer_discriminator_learning_rate.setter
    def wasserstein_gp_optimizer_discriminator_learning_rate(self, value):
        self._wasserstein_gp_optimizer_discriminator_learning_rate = value

    # Getter and setter for wasserstein_optimizer_generator_beta
    @property
    def wasserstein_gp_optimizer_generator_beta(self):
        return self._wasserstein_gp_optimizer_generator_beta

    @wasserstein_gp_optimizer_generator_beta.setter
    def wasserstein_gp_optimizer_generator_beta(self, value):
        self._wasserstein_gp_optimizer_generator_beta = value

    # Getter and setter for wasserstein_optimizer_discriminator_beta
    @property
    def wasserstein_gp_optimizer_discriminator_beta(self):
        return self._wasserstein_gp_optimizer_discriminator_beta

    @wasserstein_gp_optimizer_discriminator_beta.setter
    def wasserstein_gp_optimizer_discriminator_beta(self, value):
        self._wasserstein_gp_optimizer_discriminator_beta = value

    # Getter and setter for wasserstein_discriminator_steps
    @property
    def wasserstein_gp_discriminator_steps(self):
        return self._wasserstein_gp_discriminator_steps

    @wasserstein_gp_discriminator_steps.setter
    def wasserstein_gp_discriminator_steps(self, value):
        self._wasserstein_gp_discriminator_steps = value

    # Getter and setter for wasserstein_smoothing_rate
    @property
    def wasserstein_gp_smoothing_rate(self):
        return self._wasserstein_gp_smoothing_rate

    @wasserstein_gp_smoothing_rate.setter
    def wasserstein_gp_smoothing_rate(self, value):
        self._wasserstein_gp_smoothing_rate = value

    # Getter and setter for wasserstein_latent_mean_distribution
    @property
    def wasserstein_gp_latent_mean_distribution(self):
        return self._wasserstein_gp_latent_mean_distribution

    @wasserstein_gp_latent_mean_distribution.setter
    def wasserstein_gp_latent_mean_distribution(self, value):
        self._wasserstein_gp_latent_mean_distribution = value

    # Getter and setter for wasserstein_latent_stander_deviation
    @property
    def wasserstein_gp_latent_stander_deviation(self):
        return self._wasserstein_gp_latent_stander_deviation

    @wasserstein_gp_latent_stander_deviation.setter
    def wasserstein_gp_latent_stander_deviation(self, value):
        self._wasserstein_gp_latent_stander_deviation = value

    # Getter and setter for wasserstein_file_name_discriminator
    @property
    def wasserstein_gp_file_name_discriminator(self):
        return self._wasserstein_gp_file_name_discriminator

    @wasserstein_gp_file_name_discriminator.setter
    def wasserstein_gp_file_name_discriminator(self, value):
        self._wasserstein_gp_file_name_discriminator = value

    # Getter and setter for wasserstein_file_name_generator
    @property
    def wasserstein_gp_file_name_generator(self):
        return self._wasserstein_gp_file_name_generator

    @wasserstein_gp_file_name_generator.setter
    def wasserstein_gp_file_name_generator(self, value):
        self._wasserstein_gp_file_name_generator = value

    # Getter and setter for wasserstein_path_output_models
    @property
    def wasserstein_gp_path_output_models(self):
        return self._wasserstein_gp_path_output_models

    @wasserstein_gp_path_output_models.setter
    def wasserstein_gp_path_output_models(self, value):
        self._wasserstein_gp_path_output_models = value



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




class DenoisingDiffusionInstance:
    """
    A class that implements a Denoising Diffusion Probabilistic Model (DDPM) for image generation.
    This implementation uses a dual-UNet architecture with Gaussian diffusion to progressively
    denoise images through a Markov chain of diffusion steps.

    Key Components:
    - Two identical UNet models for the denoising process
    - Gaussian diffusion utilities for noise scheduling
    - Complete training pipeline for diffusion models
    - Exponential Moving Average (EMA) for model stability
    - Configurable architecture via hyperparameters

    Attributes:
        _denoising_gaussian_diffusion_util: Manages noise scheduling and diffusion process
        _denoising_diffusion_algorithm: Orchestrates the training and sampling process
        _denoising_second_unet_model: Second UNet in the denoising chain
        _denoising_first_unet_model: Primary UNet model for denoising

        # UNet Architecture Parameters
        _denoising_diffusion_unet_last_layer_activation: Final layer activation function
        _denoising_diffusion_latent_dimension: Dimensionality of latent space
        _denoising_diffusion_unet_num_embedding_channels: Channels for positional embeddings
        _denoising_diffusion_unet_channels_per_level: Channel configuration per UNet level
        _denoising_diffusion_unet_batch_size: Training batch size
        _denoising_diffusion_unet_attention_mode: Attention mechanism type
        _denoising_diffusion_unet_num_residual_blocks: Residual blocks per level
        _denoising_diffusion_unet_group_normalization: Whether to use group norm
        _denoising_diffusion_unet_intermediary_activation: Intermediate activation function
        _denoising_diffusion_unet_intermediary_activation_alpha: Alpha for activation (LeakyReLU etc.)
        _denoising_diffusion_unet_epochs: Number of training epochs

        # Diffusion Process Parameters
        _denoising_diffusion_gaussian_beta_start: Initial noise schedule value
        _denoising_diffusion_gaussian_beta_end: Final noise schedule value
        _denoising_diffusion_gaussian_time_steps: Number of diffusion steps
        _denoising_diffusion_gaussian_clip_min: Minimum noise value
        _denoising_diffusion_gaussian_clip_max: Maximum noise value
        _denoising_diffusion_margin: Margin for contrastive objectives
        _denoising_diffusion_ema: Whether to use EMA for model weights
        _denoising_diffusion_time_steps: Number of timesteps in diffusion process
    """

    def __init__(self, arguments):
        """
        Initializes the denoising diffusion instance with configuration parameters.

        Args:
            arguments (Namespace): Configuration object containing:
                - UNet architecture parameters
                - Diffusion process settings
                - Training hyperparameters
                - Optimization parameters
        """
        self._denoising_gaussian_diffusion_util = None
        self._denoising_diffusion_algorithm = None

        self._denoising_second_unet_model = None
        self._denoising_first_unet_model = None

        # ** Denoising Probabilistic LatentDiffusion (LDPD) Configuration Parameters **
        self._denoising_diffusion_unet_last_layer_activation = arguments.denoising_diffusion_unet_last_layer_activation
        self._denoising_diffusion_latent_dimension = arguments.denoising_diffusion_latent_dimension
        self._denoising_diffusion_unet_num_embedding_channels = arguments.denoising_diffusion_unet_num_embedding_channels
        self._denoising_diffusion_unet_channels_per_level = arguments.denoising_diffusion_unet_channels_per_level
        self._denoising_diffusion_unet_batch_size = arguments.denoising_diffusion_unet_batch_size
        self._denoising_diffusion_unet_attention_mode = arguments.denoising_diffusion_unet_attention_mode
        self._denoising_diffusion_unet_num_residual_blocks = arguments.denoising_diffusion_unet_num_residual_blocks
        self._denoising_diffusion_unet_group_normalization = arguments.denoising_diffusion_unet_group_normalization
        self._denoising_diffusion_unet_intermediary_activation = arguments.denoising_diffusion_unet_intermediary_activation
        self._denoising_diffusion_unet_intermediary_activation_alpha = arguments.denoising_diffusion_unet_intermediary_activation_alpha
        self._denoising_diffusion_unet_epochs = arguments.denoising_diffusion_unet_epochs

        # Diffusion Process Parameters
        self._denoising_diffusion_gaussian_beta_start = arguments.denoising_diffusion_gaussian_beta_start
        self._denoising_diffusion_gaussian_beta_end = arguments.denoising_diffusion_gaussian_beta_end
        self._denoising_diffusion_gaussian_time_steps = arguments.denoising_diffusion_gaussian_time_steps
        self._denoising_diffusion_gaussian_clip_min = arguments.denoising_diffusion_gaussian_clip_min
        self._denoising_diffusion_gaussian_clip_max = arguments.denoising_diffusion_gaussian_clip_max
        self._denoising_diffusion_margin = arguments.denoising_diffusion_margin
        self._denoising_diffusion_ema = arguments.denoising_diffusion_ema
        self._denoising_diffusion_time_steps = arguments.denoising_diffusion_time_steps

    def _get_denoising_diffusion(self, input_shape):
        """
         Initializes and configures the LatentDiffusion model using UNet architecture for image generation.

         This method initializes multiple components required for the diffusion process, including
         two UNet instances, a DiffusionAutoencoderModel, and a GaussianDiffusion utility. The UNet
         instances are configured with the specified hyperparameters for building the model. The
         weights of the second UNet model are synchronized with the first one. Additionally, the method
         sets up the variational model diffusion and the associated variational algorithm diffusion
         for image generation and embedding reconstruction.

         Args:
             input_shape (tuple):
              The shape of the input data, typically the dimensions of the images (height, width, channels).

         Initializes:
             self._first_instance_unet (UNetModel):
                The first instance of the UNet model used for the diffusion process.
             self._second_instance_unet (UNetModel):
                The second instance of the UNet model, which is a copy of the first one.
             self._first_unet_model (Model):
                The compiled UNet model for the first instance.
             self._second_unet_model (Model):
                The compiled UNet model for the second instance, with synchronized weights from the first model.
             self._gaussian_diffusion_util (GaussianDiffusion):
                Utility for managing the diffusion process with Gaussian noise.
             self._variation_model_diffusion (VariationalModelDiffusion):
                The diffusion model with variational autoencoder for latent representation learning.
             self._variational_algorithm_diffusion (VariationalAlgorithmDiffusion):
                The algorithm for variational inference during the diffusion process.
         """


        # Initialize the first instance of UNet for the diffusion model
        self._denoising_first_instance_unet = UNetDenoisingModel(output_shape=input_shape,
                                                                 embedding_channels= self._denoising_diffusion_unet_num_embedding_channels,
                                                                 list_neurons_per_level=self._denoising_diffusion_unet_channels_per_level,
                                                                 list_attentions=self._denoising_diffusion_unet_attention_mode,
                                                                 number_residual_blocks=self._denoising_diffusion_unet_num_residual_blocks,
                                                                 normalization_groups=self._denoising_diffusion_unet_group_normalization,
                                                                 intermediary_activation_function=self._denoising_diffusion_unet_intermediary_activation,
                                                                 intermediary_activation_alpha= self._denoising_diffusion_unet_intermediary_activation_alpha,
                                                                 last_layer_activation=self._denoising_diffusion_unet_last_layer_activation,
                                                                 number_samples_per_class=self._number_samples_per_class)

        # Initialize the second instance of UNet with the same configuration
        self._denoising_second_instance_unet = UNetDenoisingModel(output_shape=input_shape,
                                                                  embedding_channels= self._denoising_diffusion_unet_num_embedding_channels,
                                                                  list_neurons_per_level=self._denoising_diffusion_unet_channels_per_level,
                                                                  list_attentions=self._denoising_diffusion_unet_attention_mode,
                                                                  number_residual_blocks=self._denoising_diffusion_unet_num_residual_blocks,
                                                                  normalization_groups=self._denoising_diffusion_unet_group_normalization,
                                                                  intermediary_activation_function=self._denoising_diffusion_unet_intermediary_activation,
                                                                  intermediary_activation_alpha= self._denoising_diffusion_unet_intermediary_activation_alpha,
                                                                  last_layer_activation=self._denoising_diffusion_unet_last_layer_activation,
                                                                  number_samples_per_class=self._number_samples_per_class)

        # Build the models for both UNet instances
        self._denoising_first_unet_model = self._denoising_first_instance_unet.build_model()
        self._denoising_second_unet_model = self._denoising_second_instance_unet.build_model()

        # Synchronize the weights of the second UNet model with the first one
        self._denoising_second_unet_model.set_weights(self._denoising_first_unet_model.get_weights())

        # Initialize the GaussianDiffusion utility for the diffusion process
        self._denoising_gaussian_diffusion_util = GaussianDiffusion(beta_start=self._denoising_diffusion_gaussian_beta_start,
                                                                    beta_end=self._denoising_diffusion_gaussian_beta_end,
                                                                    time_steps=self._denoising_diffusion_gaussian_time_steps,
                                                                    clip_min=self._denoising_diffusion_gaussian_clip_min,
                                                                    clip_max=self._denoising_diffusion_gaussian_clip_max)




    def _training_denoising_diffusion_model(self, input_shape, arguments, x_real_samples, y_real_samples):
        """
        Executes the complete denoising diffusion training pipeline.

        The training process:
        1. Initializes dual UNet models and diffusion utilities
        2. Configures the diffusion algorithm with optimizers
        3. Prepares input data with proper dimensionality
        4. Trains using mean squared error on denoising objective
        5. Manages callbacks for monitoring and early stopping

        Args:
            input_shape (tuple): Input data dimensions
            arguments (Namespace): Training configuration
            x_real_samples (ndarray): Training samples
            y_real_samples (ndarray): Corresponding labels
        """
        # Initialize the diffusion model
        self._get_denoising_diffusion(input_shape)

        # Print the model summaries for the U-Net models
        self._denoising_first_unet_model.summary()
        self._denoising_second_unet_model.summary()

        # callbacks_list = [self._callback_resources_monitor, self._callback_model_monitor]
        callbacks_list = [self._callback_model_monitor]

        if arguments.use_early_stop:
            callbacks_list.append(self._callback_early_stop)

        # Initialize the final diffusion algorithm
        self._denoising_diffusion_algorithm = DenoisingDiffusionAlgorithm(output_shape=input_shape,
                                                                          first_unet_model=self._denoising_first_unet_model,
                                                                          second_unet_model=self._denoising_second_unet_model,
                                                                          gdf_util=self._denoising_gaussian_diffusion_util,
                                                                          optimizer_autoencoder=Adam(
                                                                              learning_rate=0.0001),
                                                                          optimizer_diffusion=Adam(
                                                                              learning_rate=0.0001),
                                                                          time_steps=self._denoising_diffusion_gaussian_time_steps,
                                                                          ema=self._denoising_diffusion_ema,
                                                                          margin=self._denoising_diffusion_margin)

        # Compile the diffusion model
        self._denoising_diffusion_algorithm.compile(loss=MeanSquaredError(),
                                                    optimizer=Adam(learning_rate=0.0001))

        # callbacks_list = [self._callback_resources_monitor, self._callback_model_monitor]
        callbacks_list = [self._callback_model_monitor]

        if arguments.use_early_stop:
            callbacks_list.append(self._callback_early_stop)

        x_real_samples = numpy.array(x_real_samples)
        x_real_samples = tensorflow.expand_dims(x_real_samples, axis=-1)

        self._denoising_diffusion_algorithm.fit(
            x_real_samples,
            to_categorical(y_real_samples, num_classes=self._number_samples_per_class["number_classes"]),
            epochs=self._denoising_diffusion_unet_epochs, batch_size=self._denoising_diffusion_unet_batch_size,
            callbacks=callbacks_list)

    # Getter and setter for diffusion_unet_last_layer_activation
    @property
    def denoising_diffusion_unet_last_layer_activation(self):
        return self.denoising_diffusion_unet_last_layer_activation

    @denoising_diffusion_unet_last_layer_activation.setter
    def denoising_diffusion_unet_last_layer_activation(self, value):
        self.denoising_diffusion_unet_last_layer_activation = value

    # Getter and setter for diffusion_latent_dimension
    @property
    def denoising_diffusion_latent_dimension(self):
        return self.denoising_diffusion_latent_dimension

    @denoising_diffusion_latent_dimension.setter
    def denoising_diffusion_latent_dimension(self, value):
        self.denoising_diffusion_latent_dimension = value

    # Getter and setter for diffusion_unet_num_embedding_channels
    @property
    def denoising_diffusion_unet_num_embedding_channels(self):
        return self.denoising_diffusion_unet_num_embedding_channels

    @denoising_diffusion_unet_num_embedding_channels.setter
    def denoising_diffusion_unet_num_embedding_channels(self, value):
        self.denoising_diffusion_unet_num_embedding_channels = value

    # Getter and setter for diffusion_unet_channels_per_level
    @property
    def denoising_diffusion_unet_channels_per_level(self):
        return self.denoising_diffusion_unet_channels_per_level

    @denoising_diffusion_unet_channels_per_level.setter
    def denoising_diffusion_unet_channels_per_level(self, value):
        self.denoising_diffusion_unet_channels_per_level = value

    # Getter and setter for diffusion_unet_batch_size
    @property
    def denoising_diffusion_unet_batch_size(self):
        return self.denoising_diffusion_unet_batch_size

    @denoising_diffusion_unet_batch_size.setter
    def denoising_diffusion_unet_batch_size(self, value):
        self.denoising_diffusion_unet_batch_size = value

    # Getter and setter for diffusion_unet_attention_mode
    @property
    def denoising_diffusion_unet_attention_mode(self):
        return self.denoising_diffusion_unet_attention_mode

    @denoising_diffusion_unet_attention_mode.setter
    def denoising_diffusion_unet_attention_mode(self, value):
        self.denoising_diffusion_unet_attention_mode = value

    # Getter and setter for diffusion_unet_num_residual_blocks
    @property
    def denoising_diffusion_unet_num_residual_blocks(self):
        return self.denoising_diffusion_unet_num_residual_blocks

    @denoising_diffusion_unet_num_residual_blocks.setter
    def denoising_diffusion_unet_num_residual_blocks(self, value):
        self.denoising_diffusion_unet_num_residual_blocks = value

    # Getter and setter for diffusion_unet_group_normalization
    @property
    def denoising_diffusion_unet_group_normalization(self):
        return self.denoising_diffusion_unet_group_normalization

    @denoising_diffusion_unet_group_normalization.setter
    def denoising_diffusion_unet_group_normalization(self, value):
        self.denoising_diffusion_unet_group_normalization = value

    # Getter and setter for diffusion_unet_intermediary_activation
    @property
    def denoising_diffusion_unet_intermediary_activation(self):
        return self.denoising_diffusion_unet_intermediary_activation

    @denoising_diffusion_unet_intermediary_activation.setter
    def denoising_diffusion_unet_intermediary_activation(self, value):
        self.denoising_diffusion_unet_intermediary_activation = value

    # Getter and setter for diffusion_unet_intermediary_activation_alpha
    @property
    def denoising_diffusion_unet_intermediary_activation_alpha(self):
        return self.denoising_diffusion_unet_intermediary_activation_alpha

    @denoising_diffusion_unet_intermediary_activation_alpha.setter
    def denoising_diffusion_unet_intermediary_activation_alpha(self, value):
        self.denoising_diffusion_unet_intermediary_activation_alpha = value

    # Getter and setter for diffusion_unet_epochs
    @property
    def denoising_diffusion_unet_epochs(self):
        return self.denoising_diffusion_unet_epochs

    @denoising_diffusion_unet_epochs.setter
    def denoising_diffusion_unet_epochs(self, value):
        self.denoising_diffusion_unet_epochs = value

    # Getter and setter for diffusion_gaussian_beta_start
    @property
    def denoising_diffusion_gaussian_beta_start(self):
        return self.denoising_diffusion_gaussian_beta_start

    @denoising_diffusion_gaussian_beta_start.setter
    def denoising_diffusion_gaussian_beta_start(self, value):
        self.denoising_diffusion_gaussian_beta_start = value

    # Getter and setter for diffusion_gaussian_beta_end
    @property
    def denoising_diffusion_gaussian_beta_end(self):
        return self.denoising_diffusion_gaussian_beta_end

    @denoising_diffusion_gaussian_beta_end.setter
    def denoising_diffusion_gaussian_beta_end(self, value):
        self.denoising_diffusion_gaussian_beta_end = value

    # Getter and setter for diffusion_gaussian_time_steps
    @property
    def denoising_diffusion_gaussian_time_steps(self):
        return self.denoising_diffusion_gaussian_time_steps

    @denoising_diffusion_gaussian_time_steps.setter
    def denoising_diffusion_gaussian_time_steps(self, value):
        self.denoising_diffusion_gaussian_time_steps = value

    # Getter and setter for diffusion_gaussian_clip_min
    @property
    def denoising_diffusion_gaussian_clip_min(self):
        return self.denoising_diffusion_gaussian_clip_min

    @denoising_diffusion_gaussian_clip_min.setter
    def denoising_diffusion_gaussian_clip_min(self, value):
        self.denoising_diffusion_gaussian_clip_min = value

    # Getter and setter for diffusion_gaussian_clip_max
    @property
    def denoising_diffusion_gaussian_clip_max(self):
        return self.denoising_diffusion_gaussian_clip_max

    @denoising_diffusion_gaussian_clip_max.setter
    def denoising_diffusion_gaussian_clip_max(self, value):
        self.denoising_diffusion_gaussian_clip_max = value

    # Getter and setter for diffusion_autoencoder_loss
    @property
    def denoising_diffusion_autoencoder_loss(self):
        return self.denoising_diffusion_autoencoder_loss

    @denoising_diffusion_autoencoder_loss.setter
    def denoising_diffusion_autoencoder_loss(self, value):
        self.denoising_diffusion_autoencoder_loss = value

    # Getter and setter for diffusion_autoencoder_encoder_filters
    @property
    def denoising_diffusion_autoencoder_encoder_filters(self):
        return self.denoising_diffusion_autoencoder_encoder_filters

    @denoising_diffusion_autoencoder_encoder_filters.setter
    def denoising_diffusion_autoencoder_encoder_filters(self, value):
        self.denoising_diffusion_autoencoder_encoder_filters = value

    # Getter and setter for diffusion_autoencoder_decoder_filters
    @property
    def denoising_diffusion_autoencoder_decoder_filters(self):
        return self.denoising_diffusion_autoencoder_decoder_filters

    @denoising_diffusion_autoencoder_decoder_filters.setter
    def denoising_diffusion_autoencoder_decoder_filters(self, value):
        self.denoising_diffusion_autoencoder_decoder_filters = value

    # Getter and setter for diffusion_autoencoder_last_layer_activation
    @property
    def denoising_diffusion_autoencoder_last_layer_activation(self):
        return self.denoising_diffusion_autoencoder_last_layer_activation

    @denoising_diffusion_autoencoder_last_layer_activation.setter
    def denoising_diffusion_autoencoder_last_layer_activation(self, value):
        self.denoising_diffusion_autoencoder_last_layer_activation = value

    # Getter and setter for diffusion_autoencoder_latent_dimension
    @property
    def denoising_diffusion_autoencoder_latent_dimension(self):
        return self.denoising_diffusion_autoencoder_latent_dimension

    @denoising_diffusion_autoencoder_latent_dimension.setter
    def denoising_diffusion_autoencoder_latent_dimension(self, value):
        self.denoising_diffusion_autoencoder_latent_dimension = value

    # Getter and setter for diffusion_autoencoder_batch_size_create_embedding
    @property
    def denoising_diffusion_autoencoder_batch_size_create_embedding(self):
        return self.denoising_diffusion_autoencoder_batch_size_create_embedding

    @denoising_diffusion_autoencoder_batch_size_create_embedding.setter
    def denoising_diffusion_autoencoder_batch_size_create_embedding(self, value):
        self.denoising_diffusion_autoencoder_batch_size_create_embedding = value

    # Getter and setter for diffusion_autoencoder_batch_size_training
    @property
    def denoising_diffusion_autoencoder_batch_size_training(self):
        return self.denoising_diffusion_autoencoder_batch_size_training

    @denoising_diffusion_autoencoder_batch_size_training.setter
    def denoising_diffusion_autoencoder_batch_size_training(self, value):
        self.denoising_diffusion_autoencoder_batch_size_training = value

    # Getter and setter for diffusion_autoencoder_epochs
    @property
    def denoising_diffusion_autoencoder_epochs(self):
        return self.denoising_diffusion_autoencoder_epochs

    @denoising_diffusion_autoencoder_epochs.setter
    def denoising_diffusion_autoencoder_epochs(self, value):
        self.denoising_diffusion_autoencoder_epochs = value

    # Getter and setter for diffusion_autoencoder_intermediary_activation_function
    @property
    def denoising_diffusion_autoencoder_intermediary_activation_function(self):
        return self.denoising_diffusion_autoencoder_intermediary_activation_function

    @denoising_diffusion_autoencoder_intermediary_activation_function.setter
    def denoising_diffusion_autoencoder_intermediary_activation_function(self, value):
        self.denoising_diffusion_autoencoder_intermediary_activation_function = value

    # Getter and setter for diffusion_autoencoder_intermediary_activation_alpha
    @property
    def denoising_diffusion_autoencoder_intermediary_activation_alpha(self):
        return self.denoising_diffusion_autoencoder_intermediary_activation_alpha

    @denoising_diffusion_autoencoder_intermediary_activation_alpha.setter
    def denoising_diffusion_autoencoder_intermediary_activation_alpha(self, value):
        self.denoising_diffusion_autoencoder_intermediary_activation_alpha = value

    # Getter and setter for diffusion_autoencoder_activation_output_encoder
    @property
    def denoising_diffusion_autoencoder_activation_output_encoder(self):
        return self.denoising_diffusion_autoencoder_activation_output_encoder

    @denoising_diffusion_autoencoder_activation_output_encoder.setter
    def denoising_diffusion_autoencoder_activation_output_encoder(self, value):
        self.denoising_diffusion_autoencoder_activation_output_encoder = value

    # Getter and setter for diffusion_margin
    @property
    def denoising_diffusion_margin(self):
        return self.denoising_diffusion_margin

    @denoising_diffusion_margin.setter
    def denoising_diffusion_margin(self, value):
        self.denoising_diffusion_margin = value

    # Getter and setter for diffusion_ema
    @property
    def denoising_diffusion_ema(self):
        return self.denoising_diffusion_ema

    @denoising_diffusion_ema.setter
    def denoising_diffusion_ema(self, value):
        self.denoising_diffusion_ema = value

    # Getter and setter for diffusion_time_steps
    @property
    def denoising_diffusion_time_steps(self):
        return self.denoising_diffusion_time_steps

    @denoising_diffusion_time_steps.setter
    def denoising_diffusion_time_steps(self, value):
        self.denoising_diffusion_time_steps = value



class SmoteInstance:
    """
    A class that implements the Synthetic Minority Over-sampling Technique (SMOTE) for handling
    class imbalance in datasets. SMOTE generates synthetic samples for minority classes by
    interpolating between existing instances, effectively balancing the class distribution.

    Key Components:
    - SMOTE algorithm implementation for synthetic sample generation
    - Configurable neighborhood size for interpolation
    - Flexible sampling strategy for target class distribution
    - Random state control for reproducibility

    Attributes:
        _smote_algorithm: The core SMOTE algorithm instance
        _smote_sampling_strategy: Target sampling strategy for class balancing
        _smote_random_state: Seed for random number generation
        _smote_k_neighbors: Number of nearest neighbors to consider for interpolation
    """
    def __init__(self, arguments):
        """
        Initializes the SMOTE instance with configuration parameters.

        Args:
            arguments (Namespace): Configuration object containing:
                - smote_sampling_strategy: Target class distribution strategy
                - smote_random_state: Random seed for reproducibility
                - smote_k_neighbors: Number of neighbors for synthetic sample generation
        """
        self._smote_algorithm = None

        # SMOTE Configuration Parameters
        self._smote_sampling_strategy = arguments.smote_sampling_strategy
        self._smote_random_state = arguments.smote_random_state
        self._smote_k_neighbors = arguments.smote_k_neighbors


    def _get_smote(self, input_shape):
        """
        Initializes and configures the SMOTE algorithm with the specified parameters.

        This method creates an instance of the SMOTEAlgorithm with the configured:
        - Sampling strategy for target class distribution
        - Random state for reproducible results
        - Number of nearest neighbors for synthetic sample generation

        Args:
            input_shape (tuple): The shape of the input data (unused in SMOTE but kept for interface consistency)

        Initializes:
            self._smote_algorithm (SMOTEAlgorithm): The configured SMOTE algorithm instance
        """
        self._smote_algorithm = SMOTEAlgorithm(sampling_strategy = self._smote_sampling_strategy,
                                               random_state = self._smote_random_state,
                                               k_neighbors = self._smote_k_neighbors)


    def _training_smote_model(self, input_shape, arguments, x_real_samples, y_real_samples):
        """
        Executes the SMOTE training process to generate synthetic samples.

        The training process:
        1. Initializes the SMOTE algorithm with configured parameters
        2. Fits the SMOTE model to the input data
        3. Generates synthetic samples for minority classes

        Args:
            input_shape (tuple): Input data dimensions (unused but kept for interface consistency)
            arguments (Namespace): Training configuration (unused in this implementation)
            x_real_samples (ndarray): Original feature vectors
            y_real_samples (ndarray): Corresponding class labels

        Note:
            The method converts labels to categorical format internally to handle multi-class scenarios.
        """
        # Initialize the autoencoder model
        self._get_smote(input_shape)

        # Fit the autoencoder model
        self._smote_algorithm.fit(x_real_samples,
                                  to_categorical(y_real_samples,
                                                 num_classes=self._number_samples_per_class["number_classes"]))

    @property
    def smote_sampling_strategy(self):
        """Get the SMOTE sampling strategy."""
        return self._smote_sampling_strategy

    @smote_sampling_strategy.setter
    def smote_sampling_strategy(self, value):
        """Set the SMOTE sampling strategy."""
        self._smote_sampling_strategy = value

    @property
    def smote_random_state(self):
        """Get the SMOTE random state."""
        return self._smote_random_state

    @smote_random_state.setter
    def smote_random_state(self, value):
        """Set the SMOTE random state."""
        self._smote_random_state = value

    @property
    def smote_k_neighbors(self):
        """Get the SMOTE k-neighbors value."""
        return self._smote_k_neighbors

    @smote_k_neighbors.setter
    def smote_k_neighbors(self, value):
        """Set the SMOTE k-neighbors value."""
        self._smote_k_neighbors = value





class GenerativeModels(AdversarialInstance,
                       AutoencoderInstance,
                       QuantizedVAEInstance,
                       LatentDiffusionInstance,
                       WassersteinInstance,
                       WassersteinGPInstance,
                       VariationalAutoencoderInstance,
                       SmoteInstance,
                       DenoisingDiffusionInstance):

    """
    A class to manage and facilitate the training and generation of various types of generative models,
    including Generative Adversarial Networks (GANs), Autoencoders (AEs), Variational Autoencoders (VAEs),
    LatentDiffusion Models, and WassersteinGP GANs (WGANs). This class provides an interface to configure, initialize,
    and manage the training processes for these models, as well as to generate synthetic data from them.

    It supports flexibility in architecture selection and offers detailed configuration options for each model.
    Additionally, it handles various model types, training parameters, and their specific settings to ensure
    a smooth and efficient workflow for deep learning practitioners working with generative models.

    The class enables users to choose and fine-tune the architecture, training procedures, and hyperparameters
    of these models, facilitating experiments with different generative approaches. Each model type is encapsulated
    with distinct algorithms and training strategies, enabling easy experimentation and comparison.

    Supported Models:
    ----------------
    - **Generative Adversarial Networks (GANs)**:
        A class of generative models that consists of a generator and a discriminator, trained in a competitive process.
        The generator creates synthetic data, and the discriminator distinguishes real from fake data.

    - **Autoencoders (AEs)**:
        A type of neural network used to learn efficient codings of input data. Autoencoders are often used
        for data compression and denoising.

    - **Variational Autoencoders (VAEs)**: A probabilistic variant of autoencoders, designed to model the
        data distribution more effectively by learning a latent space with continuous values. This is
        particularly useful for generating new data samples.

    - **LatentDiffusion Models**: A family of generative models that gradually transform noise into data through
        a sequence of steps. They have gained significant attention for image generation tasks.

    - **Wasserstein GAN (WGAN)**: A type of GAN that uses the Wasserstein distance for training,
        which provides more stable training and better convergence than traditional GANs.

    - **Wasserstein GP GANs (WGAN-GP)**: A type of GAN that uses the Wasserstein distance for training
        + Gradient Penalty, which provides more stable training and better convergence than traditional GANs.

    - **Vector Quantizer Variational Autoencoder (VQ-VAE)**: A type of variational autoencoder that incorporates
        vector quantization for discrete latent representations. Unlike traditional VAEs, VQ-VAE maps inputs
        to a fixed set of learned embeddings, improving the quality and interpretability of the latent space.

    The class allows you to experiment with these models using a unified API, where you can easily configure
    each model's architecture, initialize weights, and set training hyperparameters.

    Attributes:
    -----------
        @arguments (dict):
            A dictionary containing configuration parameters necessary for model initialization. This includes
            model-specific hyperparameters like learning rate, batch size, latent dimensions, etc.

        @_callback_model_monitor (object):
            A callback for monitoring model performance during training. This can be used for logging,
            visualization, and tracking metrics.

        @_callback_resources_monitor (object):
            A callback for tracking resource usage (e.g., memory, GPU utilization) during training to ensure
            efficient resource management.

        @_decoder_diffusion (object):
            Instance of the decoder for the diffusion model, responsible for generating data samples by
            reversing the diffusion process.

        @_encoder_diffusion (object):
            Instance of the encoder for the diffusion model, which converts input data into a latent
            representation during the forward diffusion process.

        @_diffusion_algorithm (object):
            The core diffusion model algorithm, which orchestrates the noise process and the reverse diffusion steps.

        @_adversarial_algorithm (object):
            The generative adversarial algorithm for GANs, which includes both the generator and discriminator,
            and their adversarial training.

        @_autoencoder_algorithm (object):
            The autoencoder model algorithm that defines the encoding and decoding processes, used for
            unsupervised learning and data reconstruction.

        @_variational_algorithm_diffusion (object):
            The variational autoencoder algorithm specifically designed for diffusion models, enabling
            the generation of high-quality samples.

        @_wasserstein_algorithm (object):
            The Wasserstein GAN algorithm, which incorporates the Wasserstein distance metric to improve the
            stability and convergence of GAN training.

        @_wasserstein_gp_algorithm (object):
            The WassersteinGP GAN algorithm, which incorporates the WassersteinGP distance metric to improve the
            stability and convergence of GAN training, also include Gradient Penalty.

        @_vector_quantizer_vae (object):
            The Vector Quantizer VAE algorithm, which uses discrete latent embeddings through vector quantization
            to improve reconstruction quality and enable better generative modeling.

        @_copy_algorithm (Copy):
            A helper class for duplicating model configurations or weights, useful for saving model checkpoints,
            or transferring learned weights between models.


    Methods:
    --------
    @__init__(arguments: dict)
        Initializes the class by setting up all model parameters, callbacks, and algorithms according to
        the provided configuration dictionary.

    @build_models()
        Builds the generative models based on the selected architecture. This method initializes each model’s
        components (e.g., encoder/decoder, generator/discriminator) and sets up the training pipeline.

    @train_models()
        Trains the selected models based on the provided training configurations. It manages the entire training
        process, including the optimization and loss calculation.

    @generate_samples(model_type: str, num_samples: int)
        Generates synthetic data samples using the specified model type. This method can be used for generating
        data, texts, or other data formats depending on the model.

    @save_models()
        Saves the trained models to the specified output directory. Models are saved with their current weights,
        training state, and hyperparameters, allowing easy restoration later.

    @load_models(model_directory: str)
        Loads a trained model from the given directory. This method is used to restore previously trained models
        for further evaluation or fine-tuning.

    Example:
    --------
    >>> Sample configuration dictionary
    ...     arguments = {
    ...     "learning_rate": 0.0002,
    ...     "batch_size": 64,
    ...     "latent_dim": 128,
    ...     "epochs": 100,
    ...     "model_type": "GAN",
    ...     }
    ...     # Create and train a GAN model
    ...     generative_model = GenerativeModels(arguments)
    ...     generative_model.build_models()
    ...     generative_model.train_models()
    ...     # Generate synthetic images
    ...     synthetic_images = generative_model.generate_samples(model_type="GAN", num_samples=10)
    ...     # Save the trained model
    ...     generative_model.save_models()
    ...     # Evaluate the trained model
    >>>     evaluation_results = generative_model.evaluate_model(model_type="GAN", evaluation_data=test_data)
    """

    def __init__(self, arguments):
        """
        Initializes the GenerativeModels class with model configuration parameters.

        The constructor accepts a dictionary of configuration arguments that contain necessary parameters
        for initializing different types of generative models such as GANs, AEs, VAEs, and LatentDiffusion Models.
        It sets up placeholders for various components used in each model and algorithm, and it configures
        model-specific attributes based on the provided arguments.

        Args:
            arguments (dict): A dictionary containing configuration settings for model architectures, hyperparameters,
                              training options, and paths for saving model files. This dictionary is expected to
                              include keys for model parameters, batch sizes, epochs, learning rates, and other
                              settings necessary for training.
        """

        AdversarialInstance.__init__(self, arguments)
        AutoencoderInstance.__init__(self, arguments)
        QuantizedVAEInstance.__init__(self, arguments)
        LatentDiffusionInstance.__init__(self, arguments)
        WassersteinInstance.__init__(self, arguments)
        WassersteinGPInstance.__init__(self, arguments)
        VariationalAutoencoderInstance.__init__(self, arguments)
        SmoteInstance.__init__(self, arguments)
        DenoisingDiffusionInstance.__init__(self, arguments)

        self._callback_model_monitor = None
        self._callback_resources_monitor = None
        self._callback_early_stop = None

        self._random_noise_algorithm = None

        self._copy_algorithm = CopyAlgorithm()
        self.arguments = arguments

        self._random_noise_level = arguments.random_noise_level
        self._random_noise_type_noise = arguments.random_noise_type_noise

        self._number_samples_per_class = arguments.number_samples_per_class


    def _get_random_noise(self, input_shape):

        self._random_noise_algorithm = RandomNoiseAlgorithm(noise_level=self._random_noise_level,
                                                            noise_type=self._random_noise_type_noise)


    def training_model(self, arguments, input_shape, x_real_samples, y_real_samples, monitor_path, k_fold):
        """
        Trains a model based on the selected type: adversarial, diffusion, WassersteinGP, variational, or autoencoder.

        This method handles the training process by first initializing the model according to the `model_type`
        specified in `arguments`. Then, it compiles and fits the model using the provided samples. It also
        supports callback functions for monitoring resources during training.

        Parameters:
            - arguments (dict): Dictionary containing configuration options, including the
              model type (e.g., 'adversarial', 'autoencoder', etc.).
            - input_shape (tuple): Shape of the input data (e.g., (height, width, channels)).
            - x_real_samples (array): The real input samples used for training.
            - y_real_samples (array): The target labels corresponding to the real input samples.
            - monitor_path (str): Path to store the monitoring data for the callbacks.
            - k_fold (int): The k-fold cross-validation split number for monitoring.

        This function initializes and trains the model based on the specified model type.
        It supports the following model types:

            1. Adversarial (GAN)
            2. Autoencoder
            3. Variational Autoencoder
            4. Wasserstein + GP GAN
            5. Wasserstein GAN
            6. LatentDiffusion-based model
            7. LatentDiffusion-based model
            8. Denoising Diffusion Kernel-based model
            9. Smote model
            10. Random model
            11. Vector Quantized Variational Autoencoder

        The method also uses resource and model monitoring callbacks during training to track progress.

        """

        # Initialize resource and model monitoring callbacks
        self._callback_resources_monitor = ResourceMonitorCallback(monitor_path, k_fold)
        self._callback_model_monitor = ModelMonitorCallback(monitor_path, k_fold)
        self._callback_early_stop = EarlyStopping(arguments.early_stop_monitor,
                                                  arguments.early_stop_min_delta,
                                                  arguments.early_stop_patience,
                                                  arguments.early_stop_mode,
                                                  arguments.early_stop_baseline,
                                                  arguments.early_stop_restore_best_weights)

        # Adversarial model training
        if arguments.model_type == 'adversarial':
            self._training_adversarial_modelo(input_shape, arguments, x_real_samples, y_real_samples)

        # Autoencoder model training
        elif arguments.model_type == 'autoencoder':
            self._training_autoencoder_model(input_shape, arguments, x_real_samples, y_real_samples)

        # Autoencoder model training
        elif arguments.model_type == 'random':

            # Initialize the autoencoder model
            self._get_random_noise(input_shape)

            # Fit the autoencoder model
            self._random_noise_algorithm.fit(x_real_samples,
                                             to_categorical(y_real_samples,
                                                            num_classes=self._number_samples_per_class["number_classes"]))

        # Smote model training
        elif arguments.model_type == 'smote':

            # Initialize the autoencoder model
            self._get_smote(input_shape)

            # Fit the autoencoder model
            self._smote_algorithm.fit(
                x_real_samples, to_categorical(y_real_samples,
                                               num_classes=self._number_samples_per_class["number_classes"]))

        # Variational Autoencoder (VAE) model training
        elif arguments.model_type == 'variational':
            self._training_variational_autoencoder_model(input_shape, arguments, x_real_samples, y_real_samples)

        # WassersteinGP GAN model training
        elif arguments.model_type == 'wasserstein_gp':
            self._training_wasserstein_gp_model(input_shape, arguments, x_real_samples, y_real_samples)

        # WassersteinGP GAN model training
        elif arguments.model_type == 'wasserstein':
            self._training_wasserstein_model(input_shape, arguments, x_real_samples, y_real_samples)

        # LatentDiffusion model training
        elif arguments.model_type == 'latent_diffusion':
            self._training_latent_diffusion_model(input_shape, arguments, x_real_samples, y_real_samples)

        # LatentDiffusion model training
        elif arguments.model_type == 'denoising_diffusion':
            self._training_denoising_diffusion_model(input_shape, arguments, x_real_samples, y_real_samples)

        # LatentDiffusion Kernel model training
        elif arguments.model_type == 'diffusion_kernel':
            # Support for the 'diffusion_kernel' model type has not been implemented yet.
            # This section is reserved for initializing a generator based on diffusion kernels,
            # which may later leverage integral operators for non-parametric learning of diffusive dynamics.
            # Stay tuned — the diffusion hasn't spread here yet 😉
            pass

        # Vector Quantized Variational Autoencoder (VQ-VAE) model training
        elif arguments.model_type == 'quantized':
            self._training_quantized_VAE_model(input_shape, arguments, x_real_samples, y_real_samples)

        else:
            # If no valid model type is selected, do nothing
            pass


    def get_samples(self, number_samples_per_class):
        """
        Generate and retrieve samples from a trained model.

        This method generates synthetic samples using the trained model specified by the `model_type` argument.
        It supports multiple model types for generating samples, including adversarial, diffusion, WassersteinGP,
        variational, and autoencoder models. Depending on the selected model type, the corresponding algorithm
        is called to generate the samples.

        Args:
            number_samples_per_class (int): The number of samples to generate for each class.

        Supports the following model types:
            - 'adversarial': Uses the `AdversarialAlgorithm` to generate samples.
            - 'diffusion': Uses the `DiffusionAlgorithm` to generate samples.
            - 'wasserstein': Uses the `WassersteinAlgorithm` to generate samples.
            - 'variational': Uses the `VariationalAlgorithm` to generate samples.
            - 'autoencoder': Uses the `AutoencoderAlgorithm` to generate samples.

        Note:
            Additional models may be added in the future, such as a diffusion kernel model, but this is currently under
            implementation. The corresponding algorithm for that model is not yet available.

        """

        if self.arguments.model_type == 'adversarial':
            # Generate samples using the Adversarial algorithm
            self._adversarial_algorithm.get_samples(number_samples_per_class)
            pass

        elif self.arguments.model_type == 'latent_diffusion':
            # Generate samples using the LatentDiffusion algorithm
            self._latent_diffusion_algorithm.get_samples(number_samples_per_class)
            pass

        elif self.arguments.model_type == 'denoising_diffusion':
            # Generate samples using the LatentDiffusion algorithm
            self._denoising_diffusion_algorithm.get_samples(number_samples_per_class)
            pass

        elif self.arguments.model_type == 'wasserstein':
            # Generate samples using the WassersteinGP algorithm
            self._wasserstein_gp_algorithm.get_samples(number_samples_per_class)
            pass

        elif self.arguments.model_type == 'variational':
            # Generate samples using the Variational algorithm
            self._latent_variational_algorithm_diffusion.get_samples(number_samples_per_class)
            pass

        elif self.arguments.model_type == 'autoencoder':
            # Generate samples using the Autoencoder algorithm
            self._autoencoder_algorithm.get_samples(number_samples_per_class)
            pass

        elif self.arguments.model_type == 'random':
            # Generate samples using the Autoencoder algorithm
            self._random_noise_algorithm.get_samples(number_samples_per_class)
            pass

        elif self.arguments.model_type == 'quantized':
            # Generate samples using the Quantized Variational Autoencoder algorithm
            self._quantized_vae_algorithm.get_samples(number_samples_per_class)
            pass

        elif self.arguments.model_type == 'smote':
            # Generate samples using the Autoencoder algorithm
            self._smote_algorithm.get_samples(number_samples_per_class)
            pass


        # Algorithm in implementation process for future models
        # elif self.arguments.model_type == 'diffusion_kernel':
        #    self._diffusion_algorithm_kernel.get_samples(number_samples_per_class)
        #    pass

        else:
            # If the model type is not recognized, do nothing
            pass






def import_models(function):
    """
    Decorator to create an instance of the Metrics class
    before executing the wrapped function.

    Parameters:
        function (callable): The function to be wrapped.

    Returns:
        callable: The wrapped function that initializes Metrics.
    """
    def wrapper(self, *args, **kwargs):
        # Create an instance of Metrics, passing the arguments from the instance
        GenerativeModels.__init__(self, self.arguments)
        # Call the wrapped function with the metrics instance and other arguments
        return function(self, *args, **kwargs)

    return wrapper



