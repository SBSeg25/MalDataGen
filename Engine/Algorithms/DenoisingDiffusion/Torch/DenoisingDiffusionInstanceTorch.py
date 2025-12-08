#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

from Engine.Algorithms.DenoisingDiffusion.Torch.GaussianDenoisingDiffusionTorch import GaussianDiffusionTorch
from Engine.Models.DenoisingDiffusion.DiffusionModelUnetTorch import UNetDenoisingModelTorch

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
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from Engine.Algorithms.DenoisingDiffusion.AlgorithmDenoisingDiffusion import AlgorithmDenoisingDiffusion
    from Engine.Algorithms.DenoisingDiffusion.GaussianDenoisingDiffusion import GaussianDiffusion
    from Engine.Models.DenoisingDiffusion.DiffusionModelUnet import UNetDenoisingModel

except ImportError as error:
    logging.error(error)
    sys.exit(-1)


class DenoisingDiffusionInstanceTorch:
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
        _denoising_diffusion_unet_intermediary_activation_alpha: Alpha for activation
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

        # Device configuration
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _get_denoising_diffusion(self, input_shape):
        """
         Initializes and configures the LatentDiffusion model using UNet architecture for image generation.

         This method initializes multiple components required for the diffusion process, including
         two UNet instances, a DiffusionAutoencoderModel, and a GaussianDiffusion utility. The UNet
         instances are configured with the specified hyperparameters for building the model. The
         weights of the second UNet model are synchronized with the first one.

         Args:
             input_shape (tuple):
              The shape of the input data, typically the dimensions of the images.

         Initializes:
             self._denoising_first_instance_unet: First UNet instance
             self._denoising_second_instance_unet: Second UNet instance
             self._denoising_first_unet_model: First compiled UNet model
             self._denoising_second_unet_model: Second compiled UNet model
             self._denoising_gaussian_diffusion_util: Gaussian diffusion utility
         """

        # Initialize the first instance of UNet for the diffusion model
        self._denoising_first_instance_unet = UNetDenoisingModelTorch(
            output_shape=input_shape,
            embedding_channels=self._denoising_diffusion_unet_num_embedding_channels,
            list_neurons_per_level=self._denoising_diffusion_unet_channels_per_level,
            list_attentions=self._denoising_diffusion_unet_attention_mode,
            number_residual_blocks=self._denoising_diffusion_unet_num_residual_blocks,
            normalization_groups=self._denoising_diffusion_unet_group_normalization,
            intermediary_activation_function=self._denoising_diffusion_unet_intermediary_activation,
            intermediary_activation_alpha=self._denoising_diffusion_unet_intermediary_activation_alpha,
            last_layer_activation=self._denoising_diffusion_unet_last_layer_activation,
            number_samples_per_class=self._number_samples_per_class
        ).to(self._device)

        # Initialize the second instance of UNet with the same configuration
        self._denoising_second_instance_unet = UNetDenoisingModelTorch(
            output_shape=input_shape,
            embedding_channels=self._denoising_diffusion_unet_num_embedding_channels,
            list_neurons_per_level=self._denoising_diffusion_unet_channels_per_level,
            list_attentions=self._denoising_diffusion_unet_attention_mode,
            number_residual_blocks=self._denoising_diffusion_unet_num_residual_blocks,
            normalization_groups=self._denoising_diffusion_unet_group_normalization,
            intermediary_activation_function=self._denoising_diffusion_unet_intermediary_activation,
            intermediary_activation_alpha=self._denoising_diffusion_unet_intermediary_activation_alpha,
            last_layer_activation=self._denoising_diffusion_unet_last_layer_activation,
            number_samples_per_class=self._number_samples_per_class
        ).to(self._device)

        # Store models
        self._denoising_first_unet_model = self._denoising_first_instance_unet
        self._denoising_second_unet_model = self._denoising_second_instance_unet

        # Synchronize weights
        self._denoising_second_unet_model.load_state_dict(
            self._denoising_first_unet_model.state_dict()
        )

        # Initialize the GaussianDiffusion utility
        self._denoising_gaussian_diffusion_util = GaussianDiffusionTorch(
            beta_start=self._denoising_diffusion_gaussian_beta_start,
            beta_end=self._denoising_diffusion_gaussian_beta_end,
            time_steps=self._denoising_diffusion_gaussian_time_steps,
            clip_min=self._denoising_diffusion_gaussian_clip_min,
            clip_max=self._denoising_diffusion_gaussian_clip_max
        )

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
            input_shape (int): Input data dimensions
            arguments (Namespace): Training configuration
            x_real_samples (ndarray): Training samples
            y_real_samples (ndarray): Corresponding labels
        """
        # Initialize the diffusion model
        self._get_denoising_diffusion(input_shape)

        # Print model summaries
        print("First UNet Model:")
        print(self._denoising_first_unet_model)
        print("\nSecond UNet Model:")
        print(self._denoising_second_unet_model)

        # Initialize optimizers
        optimizer_diffusion = torch.optim.Adam(
            self._denoising_first_unet_model.parameters(),
            lr=0.0001
        )
        optimizer_autoencoder = torch.optim.Adam(
            self._denoising_first_unet_model.parameters(),
            lr=0.0001
        )

        # Initialize the diffusion algorithm
        self._denoising_diffusion_algorithm = AlgorithmDenoisingDiffusion(
            output_shape=input_shape,
            first_unet_model=self._denoising_first_unet_model,
            second_unet_model=self._denoising_second_unet_model,
            gdf_util=self._denoising_gaussian_diffusion_util,
            optimizer_autoencoder=optimizer_autoencoder,
            optimizer_diffusion=optimizer_diffusion,
            time_steps=self._denoising_diffusion_gaussian_time_steps,
            ema=self._denoising_diffusion_ema,
            margin=self._denoising_diffusion_margin
        ).to(self._device)

        # Prepare data
        x_real_samples = numpy.array(x_real_samples)
        x_real_samples = torch.from_numpy(x_real_samples).float().unsqueeze(-1)

        # Convert labels to one-hot
        num_classes = self._number_samples_per_class["number_classes"]
        y_one_hot = torch.zeros(len(y_real_samples), num_classes)
        y_one_hot[torch.arange(len(y_real_samples)), y_real_samples] = 1

        # Create dataset and dataloader
        dataset = TensorDataset(x_real_samples, y_one_hot)
        dataloader = DataLoader(
            dataset,
            batch_size=self._denoising_diffusion_unet_batch_size,
            shuffle=True
        )

        # Training loop
        self._denoising_diffusion_algorithm.train()
        for epoch in range(self._denoising_diffusion_unet_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_data, batch_labels in dataloader:
                batch_data = batch_data.to(self._device)
                batch_labels = batch_labels.to(self._device)

                # Training step
                loss_dict = self._denoising_diffusion_algorithm.train_step(
                    batch_data,
                    batch_labels
                )

                epoch_loss += loss_dict["Diffusion_loss"]
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch + 1}/{self._denoising_diffusion_unet_epochs}, "
                  f"Loss: {avg_loss:.6f}")

            # Callbacks
            if hasattr(self, '_callback_model_monitor'):
                self._callback_model_monitor(epoch, avg_loss)

            if hasattr(arguments, 'use_early_stop') and arguments.use_early_stop:
                if hasattr(self, '_callback_early_stop'):
                    if self._callback_early_stop(epoch, avg_loss):
                        print("Early stopping triggered")
                        break

    # Property getters and setters (maintaining same interface)
    @property
    def denoising_diffusion_unet_last_layer_activation(self):
        return self._denoising_diffusion_unet_last_layer_activation

    @denoising_diffusion_unet_last_layer_activation.setter
    def denoising_diffusion_unet_last_layer_activation(self, value):
        self._denoising_diffusion_unet_last_layer_activation = value

    @property
    def denoising_diffusion_latent_dimension(self):
        return self._denoising_diffusion_latent_dimension

    @denoising_diffusion_latent_dimension.setter
    def denoising_diffusion_latent_dimension(self, value):
        self._denoising_diffusion_latent_dimension = value

    @property
    def denoising_diffusion_unet_num_embedding_channels(self):
        return self._denoising_diffusion_unet_num_embedding_channels

    @denoising_diffusion_unet_num_embedding_channels.setter
    def denoising_diffusion_unet_num_embedding_channels(self, value):
        self._denoising_diffusion_unet_num_embedding_channels = value

    @property
    def denoising_diffusion_unet_channels_per_level(self):
        return self._denoising_diffusion_unet_channels_per_level

    @denoising_diffusion_unet_channels_per_level.setter
    def denoising_diffusion_unet_channels_per_level(self, value):
        self._denoising_diffusion_unet_channels_per_level = value

    @property
    def denoising_diffusion_unet_batch_size(self):
        return self._denoising_diffusion_unet_batch_size

    @denoising_diffusion_unet_batch_size.setter
    def denoising_diffusion_unet_batch_size(self, value):
        self._denoising_diffusion_unet_batch_size = value

    @property
    def denoising_diffusion_unet_attention_mode(self):
        return self._denoising_diffusion_unet_attention_mode

    @denoising_diffusion_unet_attention_mode.setter
    def denoising_diffusion_unet_attention_mode(self, value):
        self._denoising_diffusion_unet_attention_mode = value

    @property
    def denoising_diffusion_unet_num_residual_blocks(self):
        return self._denoising_diffusion_unet_num_residual_blocks

    @denoising_diffusion_unet_num_residual_blocks.setter
    def denoising_diffusion_unet_num_residual_blocks(self, value):
        self._denoising_diffusion_unet_num_residual_blocks = value

    @property
    def denoising_diffusion_unet_group_normalization(self):
        return self._denoising_diffusion_unet_group_normalization

    @denoising_diffusion_unet_group_normalization.setter
    def denoising_diffusion_unet_group_normalization(self, value):
        self._denoising_diffusion_unet_group_normalization = value

    @property
    def denoising_diffusion_unet_intermediary_activation(self):
        return self._denoising_diffusion_unet_intermediary_activation

    @denoising_diffusion_unet_intermediary_activation.setter
    def denoising_diffusion_unet_intermediary_activation(self, value):
        self._denoising_diffusion_unet_intermediary_activation = value

    @property
    def denoising_diffusion_unet_intermediary_activation_alpha(self):
        return self._denoising_diffusion_unet_intermediary_activation_alpha

    @denoising_diffusion_unet_intermediary_activation_alpha.setter
    def denoising_diffusion_unet_intermediary_activation_alpha(self, value):
        self._denoising_diffusion_unet_intermediary_activation_alpha = value

    @property
    def denoising_diffusion_unet_epochs(self):
        return self._denoising_diffusion_unet_epochs

    @denoising_diffusion_unet_epochs.setter
    def denoising_diffusion_unet_epochs(self, value):
        self._denoising_diffusion_unet_epochs = value

    @property
    def denoising_diffusion_gaussian_beta_start(self):
        return self._denoising_diffusion_gaussian_beta_start

    @denoising_diffusion_gaussian_beta_start.setter
    def denoising_diffusion_gaussian_beta_start(self, value):
        self._denoising_diffusion_gaussian_beta_start = value

    @property
    def denoising_diffusion_gaussian_beta_end(self):
        return self._denoising_diffusion_gaussian_beta_end

    @denoising_diffusion_gaussian_beta_end.setter
    def denoising_diffusion_gaussian_beta_end(self, value):
        self._denoising_diffusion_gaussian_beta_end = value

    @property
    def denoising_diffusion_gaussian_time_steps(self):
        return self._denoising_diffusion_gaussian_time_steps

    @denoising_diffusion_gaussian_time_steps.setter
    def denoising_diffusion_gaussian_time_steps(self, value):
        self._denoising_diffusion_gaussian_time_steps = value

    @property
    def denoising_diffusion_gaussian_clip_min(self):
        return self._denoising_diffusion_gaussian_clip_min

    @denoising_diffusion_gaussian_clip_min.setter
    def denoising_diffusion_gaussian_clip_min(self, value):
        self._denoising_diffusion_gaussian_clip_min = value

    @property
    def denoising_diffusion_gaussian_clip_max(self):
        return self._denoising_diffusion_gaussian_clip_max

    @denoising_diffusion_gaussian_clip_max.setter
    def denoising_diffusion_gaussian_clip_max(self, value):
        self._denoising_diffusion_gaussian_clip_max = value

    @property
    def denoising_diffusion_margin(self):
        return self._denoising_diffusion_margin

    @denoising_diffusion_margin.setter
    def denoising_diffusion_margin(self, value):
        self._denoising_diffusion_margin = value

    @property
    def denoising_diffusion_ema(self):
        return self._denoising_diffusion_ema

    @denoising_diffusion_ema.setter
    def denoising_diffusion_ema(self, value):
        self._denoising_diffusion_ema = value

    @property
    def denoising_diffusion_time_steps(self):
        return self._denoising_diffusion_time_steps

    @denoising_diffusion_time_steps.setter
    def denoising_diffusion_time_steps(self, value):
        self._denoising_diffusion_time_steps = value