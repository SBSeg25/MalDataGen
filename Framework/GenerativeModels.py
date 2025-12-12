#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

from Engine.Models.LatentDiffusionTorch import LatentDiffusion

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
    from Engine.Models.DenoisingDiffusion import DenoisingDiffusion
    from Engine.Models.Adversarial import Adversarial
    from Engine.Models.Autoencoder import Autoencoder
    from Engine.Models.LatentDiffusion import LatentDiffusionTensorflow
    from Engine.Models.QuantizedVAE import QuantizedVAE
    from Engine.Algorithms.RandomNoise.AlgorithmRandomNoise import RandomNoiseAlgorithm
    from Engine.Callbacks.CallbackModel import ModelMonitorCallback
    from Engine.Callbacks.CallbackResources import ResourceMonitorCallback
    from Engine.Models.Smote import Smote
    from Engine.Models.VariationalAutoencoder import VariationalAutoencoder
    from Engine.Models.Wasserstein import Wasserstein
    from Engine.Models.WassersteinGP import WassersteinGP

except ImportError as error:
    logging.error(error)
    sys.exit(-1)


class GenerativeModels:
    """
    Generative Models Manager - Uses composition instead of multiple inheritance

    This class manages various generative model instances and provides a unified
    interface for training and generating synthetic data.
    """

    def __init__(self, arguments):
        """
        Initialize all generative models as composition instances.

        Args:
            arguments: Configuration object containing all model parameters
        """
        self.arguments = arguments

        # Initialize callback handlers
        self._callback_model_monitor = None
        self._callback_resources_monitor = None
        self._callback_early_stop = None
        self._random_noise_algorithm = None

        # Initialize copy algorithm
        self._copy_algorithm = CopyAlgorithm()

        # Store configuration parameters
        self._random_noise_level = arguments.random_noise_level
        self._random_noise_type_noise = arguments.random_noise_type_noise
        self._number_samples_per_class = arguments.number_samples_per_class

        # Initialize all model instances using composition
        self._initialize_models(arguments)

    def _initialize_models(self, args):
        """
        Initialize all generative model instances.

        Args:
            args: Arguments object containing model configurations
        """
        # Initialize Adversarial Model
        self._adversarial_model = Adversarial(
            args.adversarial_number_epochs,
            args.adversarial_batch_size,
            args.adversarial_initializer_mean,
            args.adversarial_initializer_deviation,
            args.adversarial_latent_dimension,
            args.adversarial_training_algorithm,
            args.adversarial_activation_function,
            args.adversarial_dropout_decay_rate_g,
            args.adversarial_dropout_decay_rate_d,
            args.adversarial_dense_layer_sizes_g,
            args.adversarial_dense_layer_sizes_d,
            args.adversarial_loss_generator,
            args.adversarial_loss_discriminator,
            args.adversarial_smoothing_rate,
            args.adversarial_latent_mean_distribution,
            args.adversarial_latent_stander_deviation,
            args.adversarial_file_name_discriminator,
            args.adversarial_file_name_generator,
            args.adversarial_path_output_models,
            args.adversarial_last_layer_activation,
            args.variational_autoencoder_number_epochs
        )

        # Initialize Autoencoder Model
        self._autoencoder_model = Autoencoder(
            args.autoencoder_latent_dimension,
            args.autoencoder_training_algorithm,
            args.autoencoder_activation_function,
            args.autoencoder_dropout_decay_rate_encoder,
            args.autoencoder_dropout_decay_rate_decoder,
            args.autoencoder_dense_layer_sizes_encoder,
            args.autoencoder_dense_layer_sizes_decoder,
            args.autoencoder_batch_size,
            args.autoencoder_number_epochs,
            args.autoencoder_number_classes,
            args.autoencoder_loss_function,
            args.autoencoder_momentum,
            args.autoencoder_last_activation_layer,
            args.autoencoder_initializer_mean,
            args.autoencoder_initializer_deviation,
            args.autoencoder_latent_mean_distribution,
            args.autoencoder_latent_stander_deviation,
            args.autoencoder_file_name_encoder,
            args.autoencoder_file_name_decoder,
            args.autoencoder_path_output_models
        )

        # Initialize Denoising Diffusion Model
        self._denoising_diffusion_model = DenoisingDiffusion(
            args.denoising_diffusion_unet_last_layer_activation,
            args.denoising_diffusion_latent_dimension,
            args.denoising_diffusion_unet_num_embedding_channels,
            args.denoising_diffusion_unet_channels_per_level,
            args.denoising_diffusion_unet_batch_size,
            args.denoising_diffusion_unet_attention_mode,
            args.denoising_diffusion_unet_num_residual_blocks,
            args.denoising_diffusion_unet_group_normalization,
            args.denoising_diffusion_unet_intermediary_activation,
            args.denoising_diffusion_unet_intermediary_activation_alpha,
            args.denoising_diffusion_unet_epochs,
            args.denoising_diffusion_gaussian_beta_start,
            args.denoising_diffusion_gaussian_beta_end,
            args.denoising_diffusion_gaussian_time_steps,
            args.denoising_diffusion_gaussian_clip_min,
            args.denoising_diffusion_gaussian_clip_max,
            args.denoising_diffusion_margin,
            args.denoising_diffusion_ema,
            args.denoising_diffusion_time_steps
        )

        # Initialize Quantized VAE Model
        self._quantized_vae_model = QuantizedVAE(
            args.quantized_vae_number_epochs,
            args.quantized_vae_batch_size,
            args.quantized_vae_latent_dimension,
            args.quantized_vae_number_embedding,
            args.quantized_vae_activation_function,
            args.quantized_vae_initializer_mean,
            args.quantized_vae_mean_distribution,
            args.quantized_vae_dropout_decay_rate_encoder,
            args.quantized_vae_dropout_decay_rate_decoder,
            args.quantized_vae_last_activation_layer,
            args.quantized_vae_dense_layer_sizes_encoder,
            args.quantized_vae_dense_layer_sizes_decoder,
            args.quantized_vae_train_variance,
            args.quantized_vae_file_name_encoder,
            args.quantized_vae_file_name_decoder,
            args.quantized_vae_path_output_models
        )

        # Initialize Latent Diffusion Model
        self._latent_diffusion_model = LatentDiffusion(
            unet_last_layer_activation=args.latent_diffusion_unet_last_layer_activation,
            latent_dimension=args.latent_diffusion_latent_dimension,
            unet_num_embedding_channels=args.latent_diffusion_unet_num_embedding_channels,
            unet_channels_per_level=args.latent_diffusion_unet_channels_per_level,
            unet_batch_size=args.latent_diffusion_unet_batch_size,
            unet_attention_mode=args.latent_diffusion_unet_attention_mode,
            unet_num_residual_blocks=args.latent_diffusion_unet_num_residual_blocks,
            unet_group_normalization=args.latent_diffusion_unet_group_normalization,
            unet_intermediary_activation=args.latent_diffusion_unet_intermediary_activation,
            unet_intermediary_activation_alpha=args.latent_diffusion_unet_intermediary_activation_alpha,
            unet_epochs=args.latent_diffusion_unet_epochs,
            gaussian_beta_start=args.latent_diffusion_gaussian_beta_start,
            gaussian_beta_end=args.latent_diffusion_gaussian_beta_end,
            gaussian_time_steps=args.latent_diffusion_gaussian_time_steps,
            gaussian_clip_min=args.latent_diffusion_gaussian_clip_min,
            gaussian_clip_max=args.latent_diffusion_gaussian_clip_max,
            VAE_loss_function=args.latent_diffusion_autoencoder_loss,
            VAE_encoder_filters=args.latent_diffusion_autoencoder_encoder_filters,
            VAE_decoder_filters=args.latent_diffusion_autoencoder_decoder_filters,
            VAE_last_layer_activation=args.latent_diffusion_autoencoder_last_layer_activation,
            VAE_latent_dimension=args.latent_diffusion_autoencoder_latent_dimension,
            VAE_batch_size_create_embedding=args.latent_diffusion_autoencoder_batch_size_create_embedding,
            VAE_batch_size_training=args.latent_diffusion_autoencoder_batch_size_training,
            VAE_epochs=args.latent_diffusion_autoencoder_epochs,
            VAE_intermediary_activation_function=args.latent_diffusion_autoencoder_intermediary_activation_function,
            VAE_intermediary_activation_alpha=args.latent_diffusion_autoencoder_intermediary_activation_alpha,
            VAE_activation_output_encoder=args.latent_diffusion_autoencoder_activation_output_encoder,
            VAE_initializer_mean=args.latent_diffusion_autoencoder_initializer_mean,
            VAE_initializer_deviation=args.latent_diffusion_autoencoder_initializer_deviation,
            VAE_dropout_decay_rate_encoder=args.latent_diffusion_autoencoder_dropout_decay_rate_encoder,
            VAE_dropout_decay_rate_decoder=args.latent_diffusion_autoencoder_dropout_decay_rate_decoder,
            VAE_file_name_encoder=args.latent_diffusion_autoencoder_file_name_encoder,
            VAE_file_name_decoder=args.latent_diffusion_autoencoder_file_name_decoder,
            VAE_path_output_models=args.latent_diffusion_autoencoder_path_output_models,
            VAE_mean_distribution=args.latent_diffusion_autoencoder_mean_distribution,
            VAE_stander_deviation=args.latent_diffusion_autoencoder_stander_deviation,
            margin=args.latent_diffusion_margin,
            ema=args.latent_diffusion_ema,
            time_steps=args.latent_diffusion_time_steps
        )

        # Initialize Wasserstein Model
        self._wasserstein_model = Wasserstein(
            args.wasserstein_latent_dimension,
            args.wasserstein_training_algorithm,
            args.wasserstein_activation_function,
            args.wasserstein_dropout_decay_rate_g,
            args.wasserstein_dropout_decay_rate_d,
            args.wasserstein_dense_layer_sizes_generator,
            args.wasserstein_dense_layer_sizes_discriminator,
            args.wasserstein_batch_size,
            args.wasserstein_number_epochs,
            args.wasserstein_number_classes,
            args.wasserstein_loss_function,
            args.wasserstein_momentum,
            args.wasserstein_last_activation_layer,
            args.wasserstein_initializer_mean,
            args.wasserstein_initializer_deviation,
            args.wasserstein_optimizer_generator_learning_rate,
            args.wasserstein_optimizer_discriminator_learning_rate,
            args.wasserstein_optimizer_generator_beta,
            args.wasserstein_optimizer_discriminator_beta,
            args.wasserstein_discriminator_steps,
            args.wasserstein_smoothing_rate,
            args.wasserstein_latent_mean_distribution,
            args.wasserstein_latent_stander_deviation,
            args.wasserstein_file_name_discriminator,
            args.wasserstein_file_name_generator,
            args.wasserstein_path_output_models
        )

        # Initialize Wasserstein GP Model
        self._wasserstein_gp_model = WassersteinGP(
            args.wasserstein_gp_latent_dimension,
            args.wasserstein_gp_training_algorithm,
            args.wasserstein_gp_activation_function,
            args.wasserstein_gp_dropout_decay_rate_g,
            args.wasserstein_gp_dropout_decay_rate_d,
            args.wasserstein_gp_dense_layer_sizes_generator,
            args.wasserstein_gp_dense_layer_sizes_discriminator,
            args.wasserstein_gp_batch_size,
            args.wasserstein_gp_number_epochs,
            args.wasserstein_gp_number_classes,
            args.wasserstein_gp_loss_function,
            args.wasserstein_gp_momentum,
            args.wasserstein_gp_last_activation_layer,
            args.wasserstein_gp_initializer_mean,
            args.wasserstein_gp_initializer_deviation,
            args.wasserstein_gp_optimizer_generator_learning_rate,
            args.wasserstein_gp_optimizer_discriminator_learning_rate,
            args.wasserstein_gp_optimizer_generator_beta,
            args.wasserstein_gp_optimizer_discriminator_beta,
            args.wasserstein_gp_discriminator_steps
        )

        # Initialize Variational Autoencoder Model
        self._variational_autoencoder_model = VariationalAutoencoder(
            args.variational_autoencoder_latent_dimension,
            args.variational_autoencoder_training_algorithm,
            args.variational_autoencoder_activation_function,
            args.variational_autoencoder_dropout_decay_rate_encoder,
            args.variational_autoencoder_dropout_decay_rate_decoder,
            args.variational_autoencoder_dense_layer_sizes_encoder,
            args.variational_autoencoder_dense_layer_sizes_decoder,
            args.variational_autoencoder_batch_size,
            args.variational_autoencoder_number_epochs,
            args.variational_autoencoder_number_classes,
            args.variational_autoencoder_loss_function,
            args.variational_autoencoder_momentum,
            args.variational_autoencoder_last_activation_layer,
            args.variational_autoencoder_initializer_mean,
            args.variational_autoencoder_initializer_deviation,
            args.variational_autoencoder_mean_distribution,
            args.variational_autoencoder_stander_deviation,
            args.variational_autoencoder_file_name_encoder,
            args.variational_autoencoder_file_name_decoder,
            args.variational_autoencoder_path_output_models
        )

        # Initialize SMOTE Model
        self._smote_model = Smote(args)

    def _get_random_noise(self, input_shape):
        """Initialize random noise algorithm."""
        self._random_noise_algorithm = RandomNoiseAlgorithm(
            noise_level=self._random_noise_level,
            noise_type=self._random_noise_type_noise
        )

    def _get_smote(self, input_shape):
        """Initialize SMOTE algorithm."""
        # SMOTE is already initialized in _initialize_models
        pass

    def training_model(self, arguments, input_shape, x_real_samples, y_real_samples, monitor_path, k_fold):
        """
        Trains a model based on the selected type.

        This method delegates training to the appropriate model instance based on model_type.

        Args:
            arguments: Configuration arguments
            input_shape: Shape of input data
            x_real_samples: Real input samples for training
            y_real_samples: Target labels for real samples
            monitor_path: Path to store monitoring data
            k_fold: K-fold cross-validation split number
        """
        # Initialize callbacks
        self._callback_resources_monitor = ResourceMonitorCallback(monitor_path, k_fold)
        self._callback_model_monitor = ModelMonitorCallback(monitor_path, k_fold)
        self._callback_early_stop = EarlyStopping(
            arguments.early_stop_monitor,
            arguments.early_stop_min_delta,
            arguments.early_stop_patience,
            arguments.early_stop_mode,
            arguments.early_stop_baseline,
            arguments.early_stop_restore_best_weights
        )

        # Delegate to appropriate model
        if arguments.model_type == 'adversarial':
            self._adversarial_model.fit_model(input_shape, arguments, x_real_samples, y_real_samples)

        elif arguments.model_type == 'autoencoder':
            self._autoencoder_model.fit_model(input_shape, arguments, x_real_samples, y_real_samples)

        elif arguments.model_type == 'random':
            self._get_random_noise(input_shape)
            self._random_noise_algorithm.fit_model(x_real_samples,
                                                   to_categorical(y_real_samples,
                                                                  num_classes=self._number_samples_per_class["number_classes"]))

        elif arguments.model_type == 'smote':
            self._get_smote(input_shape)
            self._smote_model._smote_algorithm.fit_model(x_real_samples,
                                                         to_categorical(y_real_samples,
                                                                        num_classes=self._number_samples_per_class["number_classes"]))

        elif arguments.model_type == 'variational':
            self._variational_autoencoder_model.fit_model(input_shape, arguments, x_real_samples, y_real_samples)

        elif arguments.model_type == 'wasserstein_gp':
            self._wasserstein_gp_model.fit_model(input_shape, arguments, x_real_samples, y_real_samples)

        elif arguments.model_type == 'wasserstein':
            self._wasserstein_model.fit_model(input_shape, arguments, x_real_samples, y_real_samples)

        elif arguments.model_type == 'latent_diffusion':
            self._latent_diffusion_model.fit_model(input_shape, arguments, x_real_samples, y_real_samples)

        elif arguments.model_type == 'denoising_diffusion':
            self._denoising_diffusion_model.fit_model(input_shape, arguments, x_real_samples, y_real_samples)

        elif arguments.model_type == 'quantized':
            self._quantized_vae_model.fit_model(input_shape, arguments, x_real_samples, y_real_samples)

    def get_samples(self, number_samples_per_class):
        """
        Generate synthetic samples using the trained model.

        Args:
            number_samples_per_class: Number of samples to generate per class
        """
        if self.arguments.model_type == 'adversarial':
            self._adversarial_model.get_samples(number_samples_per_class)

        elif self.arguments.model_type == 'latent_diffusion':
            self._latent_diffusion_model._latent_diffusion_algorithm.get_samples(number_samples_per_class)

        elif self.arguments.model_type == 'denoising_diffusion':
            self._denoising_diffusion_model._denoising_diffusion_algorithm.get_samples(number_samples_per_class)

        elif self.arguments.model_type == 'wasserstein':
            self._wasserstein_model._wasserstein_gp_algorithm.get_samples(number_samples_per_class)

        elif self.arguments.model_type == 'variational':
            self._variational_autoencoder_model._latent_variational_algorithm.get_samples(number_samples_per_class)

        elif self.arguments.model_type == 'autoencoder':
            self._autoencoder_model._autoencoder_algorithm.get_samples(number_samples_per_class)

        elif self.arguments.model_type == 'random':
            self._random_noise_algorithm.get_samples(number_samples_per_class)

        elif self.arguments.model_type == 'quantized':
            self._quantized_vae_model._quantized_vae_algorithm.get_samples(number_samples_per_class)

        elif self.arguments.model_type == 'smote':
            self._smote_model._smote_algorithm.get_samples(number_samples_per_class)


def import_models(function):
    """
    Decorator to create an instance of GenerativeModels class
    before executing the wrapped function.

    Args:
        function: The function to be wrapped

    Returns:
        The wrapped function that initializes GenerativeModels
    """

    def wrapper(self, *args, **kwargs):
        GenerativeModels.__init__(self, self.arguments)
        return function(self, *args, **kwargs)

    return wrapper