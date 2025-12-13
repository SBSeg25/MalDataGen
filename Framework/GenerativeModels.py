#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

# MIT License
# (License text omitted for brevity)

try:
    import sys
    import keras
    import numpy
    import logging
    import tensorflow

    from Engine.Models.Adversarial import Adversarial
    from Engine.Models.Autoencoder import Autoencoder
    from Engine.Models.QuantizedVAE import QuantizedVAE

    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    from Engine.Models.LatentDiffusion import LatentDiffusion
    from tensorflow.python.keras.losses import MeanSquaredError
    from Engine.Callbacks.CallbackEarlyStop import EarlyStopping
    from tensorflow.python.keras.losses import BinaryCrossentropy

    from Engine.Algorithms.Copy.CopyAlgorithm import CopyAlgorithm
    from Engine.Models.DenoisingDiffusion import DenoisingDiffusion
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
    """

    def __init__(self, arguments):
        """Initialize all generative models as composition instances."""
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
        """Initialize all generative model instances."""

        # FIXED: Changed from _model to _algorithm for consistency with SynDataGen

        # Initialize adversarial Model
        self._adversarial_algorithm = Adversarial(
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

        # Initialize autoencoder Model
        self._autoencoder_algorithm = Autoencoder(
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
        self._denoising_diffusion_algorithm = DenoisingDiffusion(
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
        self._quantized_vae_algorithm = QuantizedVAE(
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
        self._latent_diffusion_algorithm = LatentDiffusion(
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


        # Initialize wasserstein Model
        self._wasserstein_algorithm = Wasserstein(
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

        # Initialize wasserstein GP Model
        self._wasserstein_gp_algorithm = WassersteinGP(
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

        # Initialize Variational autoencoder Model
        self._variational_algorithm = VariationalAutoencoder(
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
        self._smote_algorithm = Smote(args)

    def _get_random_noise(self, input_shape):
        """Initialize random noise algorithm."""
        self._random_noise_algorithm = RandomNoiseAlgorithm(
            noise_level=self._random_noise_level,
            noise_type=self._random_noise_type_noise
        )

    def _get_smote(self, input_shape):
        """Initialize SMOTE algorithm."""
        pass

    def training_model(self, arguments, input_shape, x_real_samples, y_real_samples, monitor_path, k_fold):
        """Trains a model based on the selected type."""

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
            self._adversarial_algorithm.fit_model(input_shape, x_real_samples, y_real_samples)

        elif arguments.model_type == 'autoencoder':
            self._autoencoder_algorithm.fit_model(input_shape, x_real_samples, y_real_samples)

        elif arguments.model_type == 'random':
            self._get_random_noise(input_shape)
            self._random_noise_algorithm.fit(x_real_samples,
                                             to_categorical(y_real_samples,
                                                                  num_classes=self._number_samples_per_class[
                                                                      "number_classes"]))

        elif arguments.model_type == 'smote':
            self._get_smote(input_shape)
            self._smote_algorithm._smote_algorithm.fit(x_real_samples,
                                                       to_categorical(y_real_samples,
                                                                            num_classes=self._number_samples_per_class[
                                                                                "number_classes"]))

        elif arguments.model_type == 'variational':
            self._variational_algorithm.fit_model(input_shape, arguments, x_real_samples, y_real_samples)

        elif arguments.model_type == 'wasserstein_gp':
            self._wasserstein_gp_algorithm.fit_model(input_shape, arguments, x_real_samples, y_real_samples)

        elif arguments.model_type == 'wasserstein':
            self._wasserstein_algorithm.fit_model(input_shape, arguments, x_real_samples, y_real_samples)

        elif arguments.model_type == 'latent_diffusion':
            self._latent_diffusion_algorithm.fit_model(input_shape, arguments, x_real_samples, y_real_samples)

        elif arguments.model_type == 'denoising_diffusion':
            self._denoising_diffusion_algorithm.fit_model(input_shape, arguments, x_real_samples, y_real_samples)

        elif arguments.model_type == 'quantized':
            self._quantized_vae_algorithm.fit_model(input_shape, arguments, x_real_samples, y_real_samples)

def import_models(function):
    """Decorator to create an instance of GenerativeModels class."""

    def wrapper(self, *args, **kwargs):
        GenerativeModels.__init__(self, self.arguments)
        return function(self, *args, **kwargs)

    return wrapper