#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

from Engine.Algorithms.LatentDiffusion.Torch.AlgorithmLatentDiffusionTorch import AlgorithmLatentDiffusionTorch
from Engine.Algorithms.LatentDiffusion.Torch.AlgorithmVAELatentDiffusionTorch import VAELatentDiffusionAlgorithmPyTorch
from Engine.Algorithms.LatentDiffusion.Torch.GaussianLatentDiffusionTorch import GaussianDiffusionTorch
from Engine.Architectures.LatentDiffusion.Torch.UNetModelTorch import UNetModelTorch
from Engine.Architectures.LatentDiffusion.Torch.VariationalModelDiffusionTorch import VariationalModelDiffusionTorch

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
    import logging
    import numpy
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam


except ImportError as error:
    logging.error(error)
    sys.exit(-1)


class LatentDiffusionTorch:
    """
    A class that implements a Latent Denoising Probabilistic Diffusion (LDPD) model for generative tasks.
    This implementation combines variational autoencoders with diffusion models in latent space for
    high-quality sample generation using PyTorch.

    Key Components:
    - Two UNet models for the diffusion process
    - Variational Autoencoder for latent space representation
    - Gaussian diffusion utilities for noise scheduling
    - Complete training pipeline for both VAE and diffusion components
    - Highly configurable architecture via arguments for research experimentation

    Attributes:
        _latent_variational_algorithm: Orchestrates training of the VAE within the diffusion context
        _latent_variation_model_diffusion: Stores the encoder and decoder of the VAE
        _latent_gaussian_diffusion_util: Utility object for beta schedules and diffusion parameters
        _latent_second_unet_model: The second-stage UNet used in the denoising chain
        _latent_first_unet_model: The initial UNet used in early-stage denoising

        # Latent Diffusion - UNet Parameters
        _latent_diffusion_unet_last_layer_activation: Activation function used in UNet's final layer
        _latent_diffusion_latent_dimension: Dimensionality of latent space
        _latent_diffusion_unet_num_embedding_channels: Number of channels for positional/time embeddings
        _latent_diffusion_unet_channels_per_level: Channel config per U-Net level
        _latent_diffusion_unet_batch_size: Batch size used during UNet training
        _latent_diffusion_unet_attention_mode: Attention mechanism used in UNet
        _latent_diffusion_unet_num_residual_blocks: Number of residual blocks per level
        _latent_diffusion_unet_group_normalization: Whether to apply group norm in UNet layers
        _latent_diffusion_unet_intermediary_activation: Activation function for intermediate layers
        _latent_diffusion_unet_intermediary_activation_alpha: Alpha value (if using LeakyReLU, etc.)
        _latent_diffusion_unet_epochs: Number of epochs for UNet training

        # Latent Diffusion - VAE Parameters
        _latent_diffusion_VAE_mean_distribution: Type of distribution for latent mean
        _latent_diffusion_VAE_stander_deviation: Std deviation for latent distribution
        _latent_diffusion_VAE_file_name_encoder: File path to save/load the encoder
        _latent_diffusion_VAE_file_name_decoder: File path to save/load the decoder
        _latent_diffusion_VAE_path_output_models: Directory to store trained autoencoder components
        _latent_diffusion_VAE_loss_function: Loss used to optimize VAE
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

    def __init__(self, arguments, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initializes the latent diffusion instance with configuration parameters.

        Args:
            arguments (Namespace): Configuration object containing:
                - UNet architecture parameters
                - VAE configuration
                - Gaussian diffusion settings
                - Training hyperparameters
                - Model saving paths
            device (str): Device to use for training ('cuda' or 'cpu')
        """

        self._device = device
        self._latent_variational_algorithm = None
        self._latent_variation_model_diffusion = None

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
        two UNet instances, a VariationalModelDiffusion, and a GaussianDiffusion utility.

        Args:
            input_shape (int): The shape of the input data (feature dimension).

        Initializes:
            self._latent_first_instance_unet: The first instance of the UNet model
            self._latent_second_instance_unet: The second instance of the UNet model
            self._latent_first_unet_model: The first UNet model
            self._latent_second_unet_model: The second UNet model with synchronized weights
            self._latent_gaussian_diffusion_util: Utility for managing the diffusion process
            self._latent_variation_model_diffusion: The VAE model for latent representation
        """

        # Initialize the first instance of UNet for the diffusion model
        self._latent_first_instance_unet = UNetModelTorch(
            embedding_dimension=self._latent_diffusion_latent_dimension,
            embedding_channels=self._latent_diffusion_unet_num_embedding_channels,
            list_neurons_per_level=self._latent_diffusion_unet_channels_per_level,
            list_attentions=self._latent_diffusion_unet_attention_mode,
            number_residual_blocks=self._latent_diffusion_unet_num_residual_blocks,
            normalization_groups=self._latent_diffusion_unet_group_normalization,
            intermediary_activation_function=self._latent_diffusion_unet_intermediary_activation,
            intermediary_activation_alpha=self._latent_diffusion_unet_intermediary_activation_alpha,
            last_layer_activation=self._latent_diffusion_unet_last_layer_activation,
            number_samples_per_class=self._number_samples_per_class
        ).to(self._device)

        # Initialize the second instance of UNet with the same configuration
        self._latent_second_instance_unet = UNetModelTorch(
            embedding_dimension=self._latent_diffusion_latent_dimension,
            embedding_channels=self._latent_diffusion_unet_num_embedding_channels,
            list_neurons_per_level=self._latent_diffusion_unet_channels_per_level,
            list_attentions=self._latent_diffusion_unet_attention_mode,
            number_residual_blocks=self._latent_diffusion_unet_num_residual_blocks,
            normalization_groups=self._latent_diffusion_unet_group_normalization,
            intermediary_activation_function=self._latent_diffusion_unet_intermediary_activation,
            intermediary_activation_alpha=self._latent_diffusion_unet_intermediary_activation_alpha,
            last_layer_activation=self._latent_diffusion_unet_last_layer_activation,
            number_samples_per_class=self._number_samples_per_class
        ).to(self._device)

        # Build the models for both UNet instances
        self._latent_first_unet_model = self._latent_first_instance_unet.build_model()
        self._latent_second_unet_model = self._latent_second_instance_unet.build_model()

        # Synchronize the weights of the second UNet model with the first one
        self._latent_second_unet_model.load_state_dict(self._latent_first_unet_model.state_dict())

        # Initialize the GaussianDiffusion utility for the diffusion process
        self._latent_gaussian_diffusion_util = GaussianDiffusionTorch(
            beta_start=self._latent_diffusion_gaussian_beta_start,
            beta_end=self._latent_diffusion_gaussian_beta_end,
            time_steps=self._latent_diffusion_gaussian_time_steps,
            clip_min=self._latent_diffusion_gaussian_clip_min,
            clip_max=self._latent_diffusion_gaussian_clip_max
        ).to(self._device)

        # Initialize the VariationalModelDiffusion for embedding learning and reconstruction
        self._latent_variation_model_diffusion = VariationalModelDiffusionTorch(
            latent_dimension=self._latent_diffusion_latent_dimension,
            output_shape=input_shape,
            activation_function=self._latent_diffusion_VAE_intermediary_activation_function,
            initializer_mean=self._latent_diffusion_VAE_initializer_mean,
            initializer_deviation=self._latent_diffusion_VAE_initializer_deviation,
            dropout_decay_encoder=self._latent_diffusion_VAE_dropout_decay_rate_encoder,
            dropout_decay_decoder=self._latent_diffusion_VAE_dropout_decay_rate_decoder,
            last_layer_activation=self._latent_diffusion_VAE_activation_output_encoder,
            number_neurons_encoder=self._latent_diffusion_VAE_encoder_filters,
            number_neurons_decoder=self._latent_diffusion_VAE_decoder_filters,
            dataset_type=numpy.float32,
            number_samples_per_class=self._number_samples_per_class
        ).to(self._device)

        # Initialize the VAE algorithm for training
        self._latent_variational_algorithm = VAELatentDiffusionAlgorithmPyTorch(
            encoder_model=self._latent_variation_model_diffusion.get_encoder(),
            decoder_model=self._latent_variation_model_diffusion.get_decoder(),
            loss_function=self._latent_diffusion_VAE_loss_function,
            latent_dimension=self._latent_diffusion_latent_dimension,
            decoder_latent_dimension=self._latent_diffusion_latent_dimension,
            latent_mean_distribution=self._latent_diffusion_VAE_mean_distribution,
            latent_stander_deviation=self._latent_diffusion_VAE_stander_deviation,
            file_name_encoder=self._latent_diffusion_VAE_file_name_encoder,
            file_name_decoder=self._latent_diffusion_VAE_file_name_decoder,
            models_saved_path=self._latent_diffusion_VAE_path_output_models
        ).to(self._device)

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
            input_shape (int): Input data shape
            arguments (Namespace): Training configuration
            x_real_samples (ndarray): Training samples
            y_real_samples (ndarray): Corresponding labels
        """
        print("---------------------------------------------------------")
        # Initialize the diffusion model
        self._get_latent_diffusion(input_shape)

        # Print model information
        print("First UNet Model:")
        print(self._latent_first_unet_model)
        print("\nSecond UNet Model:")
        print(self._latent_second_unet_model)

        print("\nEncoder Model:")
        print(self._latent_variation_model_diffusion.get_encoder())
        print("\nDecoder Model:")
        print(self._latent_variation_model_diffusion.get_decoder())

        # Convert data to PyTorch tensors
        x_real_samples = torch.from_numpy(x_real_samples).float().to(self._device)
        y_real_samples = torch.from_numpy(y_real_samples).long().to(self._device)

        # One-hot encode labels
        y_real_samples_onehot = F.one_hot(
            y_real_samples,
            num_classes=self._number_samples_per_class["number_classes"]
        ).float()

        # Setup optimizer for VAE
        vae_optimizer = Adam(
            list(self._latent_variation_model_diffusion.get_encoder().parameters()) +
            list(self._latent_variation_model_diffusion.get_decoder().parameters()),
            lr=0.0001
        )

        # Training loop for VAE
        print("\n=== Training VAE ===")
        num_batches = len(x_real_samples) // self._latent_diffusion_VAE_batch_size_training

        for epoch in range(self._latent_diffusion_VAE_epochs):
            epoch_losses = []

            for batch_idx in range(num_batches):
                start_idx = batch_idx * self._latent_diffusion_VAE_batch_size_training
                end_idx = start_idx + self._latent_diffusion_VAE_batch_size_training

                batch_x = x_real_samples[start_idx:end_idx]
                batch_y = y_real_samples_onehot[start_idx:end_idx]

                # Train step

                loss_dict = self._latent_variational_algorithm.train_step(
                    (batch_x, batch_x),
                    batch_y,
                    vae_optimizer
                )

                epoch_losses.append(loss_dict['loss'])

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1}/{self._latent_diffusion_VAE_epochs}], Loss: {avg_loss:.4f}")

            # Early stopping check
            if hasattr(arguments, 'use_early_stop') and arguments.use_early_stop:
                if hasattr(self, '_callback_early_stop') and self._callback_early_stop.should_stop(avg_loss):
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Get trained encoder and decoder
        self._encoder_latent_diffusion = self._latent_variational_algorithm.get_encoder_trained()
        self._decoder_latent_diffusion = self._latent_variational_algorithm.get_decoder_trained()

        print("\nTrained Encoder:")
        print(self._encoder_latent_diffusion)
        print("\nTrained Decoder:")
        print(self._decoder_latent_diffusion)

        # Create embeddings
        print("\n=== Creating Embeddings ===")
        data_embedding = self._latent_variational_algorithm.create_embedding(
            x_real_samples,
            batch_size=self._latent_diffusion_VAE_batch_size_create_embedding
        )

        data_embedding = torch.from_numpy(data_embedding).float().to(self._device)
        if len(data_embedding.shape) == 2:
            data_embedding = data_embedding.unsqueeze(-1)

        # Initialize the final diffusion algorithm
        self._latent_diffusion_algorithm = AlgorithmLatentDiffusionTorch(
            first_unet_model=self._latent_first_unet_model,
            second_unet_model=self._latent_second_unet_model,
            encoder_model_image=self._encoder_latent_diffusion,
            decoder_model_image=self._decoder_latent_diffusion,
            gdf_util=self._latent_gaussian_diffusion_util,
            optimizer_autoencoder=Adam(
                list(self._encoder_latent_diffusion.parameters()) +
                list(self._decoder_latent_diffusion.parameters()),
                lr=0.0001
            ),
            optimizer_diffusion=Adam(self._latent_first_unet_model.parameters(), lr=0.0001),
            time_steps=self._latent_diffusion_gaussian_time_steps,
            ema=self._latent_diffusion_ema,
            margin=self._latent_diffusion_margin,
            embedding_dimension=self._latent_diffusion_latent_dimension
        )

        # Training loop for diffusion model
        print("\n=== Training Diffusion Model ===")
        num_batches = len(data_embedding) // self._latent_diffusion_unet_batch_size

        for epoch in range(self._latent_diffusion_unet_epochs):
            epoch_losses = []

            for batch_idx in range(num_batches):
                start_idx = batch_idx * self._latent_diffusion_unet_batch_size
                end_idx = start_idx + self._latent_diffusion_unet_batch_size

                batch_embedding = data_embedding[start_idx:end_idx]
                batch_labels = y_real_samples_onehot[start_idx:end_idx]

                # Train step
                loss_dict = self._latent_diffusion_algorithm.train_step(
                    (batch_embedding, batch_labels)
                )
                epoch_losses.append(loss_dict['Diffusion_loss'])

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1}/{self._latent_diffusion_unet_epochs}], Diffusion Loss: {avg_loss:.4f}")

            # Early stopping check
            if hasattr(arguments, 'use_early_stop') and arguments.use_early_stop:
                if hasattr(self, '_callback_early_stop') and self._callback_early_stop.should_stop(avg_loss):
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

    # Property getters and setters (same as original, no changes needed)
    @property
    def device(self):
        """Get the device being used (cuda or cpu)"""
        return self._device

    @device.setter
    def device(self, value):
        """Set the device to use"""
        self._device = value

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

    # ... (rest of the properties remain the same as in the original code)