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

    from Engine.Models.Adversarial.Tensorflow.AdversarialModelTensorflow import AdversarialModel
    from Engine.Models.Autoencoder.Tensorflow.ModelAutoencoderTensorflow import AutoencoderModel

    from Engine.Models.WassersteinGP.ModelWassersteinGPGAN import WassersteinGPModel
    from Engine.Models.QuantizedVAE.Tensorflow.ModelQuantizedVAETensorflow import QuantizedVAEModel
    from Engine.Models.DenoisingDiffusion.DiffusionModelUnet import UNetDenoisingModel
    from Engine.Algorithms.LatentDiffusion.GaussianLatentDiffusion import GaussianDiffusion
    from Engine.Models.DiffusionKernel.DiffusionModelUnet import UNetModelKernel

    from Engine.Algorithms.Wasserstein.AlgorithmWassersteinGAN import WassersteinAlgorithm
    from Engine.Models.Wasserstein.Tensorflow.ModelWassersteinGANTensorflow import WassersteinModel

    from Engine.Algorithms.RandomNoise.AlgorithmRandomNoise import RandomNoiseAlgorithm
    from Engine.Algorithms.Adversarial.AdversarialAlgorithm import AdversarialAlgorithm
    from Engine.Algorithms.Autoencoder.AutoencoderAlgorithm import AutoencoderAlgorithm

    from Engine.Algorithms.WassersteinGP.AlgorithmWassersteinGANGP import WassersteinGPAlgorithm
    from Engine.Algorithms.QuantizedVAE.AlgorithmQuantizedVAE import QuantizedVAEAlgorithm

    from Engine.Models.LatentDiffusion.VariationalAutoencoderModel import VariationalModelDiffusion

    from Engine.Models.VariationalAutoencoder.VariationalAutoencoderModel import VariationalModel
    from Engine.Algorithms.LatentDiffusion.AlgorithmLatentDiffusion import LatentDiffusionAlgorithm

    from Engine.Algorithms.LatentDiffusion.AlgorithmVAELatentDiffusion import VAELatentDiffusionAlgorithm
    from Engine.Algorithms.DenoisingDiffusion.AlgorithmDenoisingDiffusion import AlgorithmDenoisingDiffusion
    from Engine.Algorithms.VariationalAutoencoder.AlgorithmVariationalAutoencoder import VariationalAlgorithm



except ImportError as error:
    logging.error(error)
    sys.exit(-1)


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

