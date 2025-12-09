#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

from Engine.Algorithms.DenoisingDiffusion.Torch.AlgorithmDenoisingDiffusionTorch import \
    AlgorithmDenoisingDiffusionTorch
from Engine.Algorithms.DenoisingDiffusion.Torch.GaussianDenoisingDiffusionTorch import GaussianDiffusionTorch
from Engine.Models.DenoisingDiffusion.Torch.DiffusionModelUnetTorch import UNetDenoisingModelTorch

try:
    import sys
    import numpy
    import logging
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

except ImportError as error:
    logging.error(error)
    sys.exit(-1)


class DenoisingDiffusionInstanceTorch:
    """
    A class that implements a Denoising Diffusion Probabilistic Model (DDPM) for image generation.
    """

    def __init__(self, arguments):
        """
        Initializes the denoising diffusion instance with configuration parameters.
        """
        self._denoising_gaussian_diffusion_util = None
        self._denoising_diffusion_algorithm = None

        self._denoising_second_unet_model = None
        self._denoising_first_unet_model = None

        # Configuration parameters
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

        # Training history
        self._training_history = {
            'epoch': [],
            'loss': [],
            'avg_loss': []
        }

    def _get_denoising_diffusion(self, input_shape):
        """
        Initializes and configures the LatentDiffusion model using UNet architecture.
        """
        # Initialize the first instance of UNet
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

        # Initialize the second instance of UNet
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
        """
        # Initialize the diffusion model
        self._get_denoising_diffusion(input_shape)

        # Print model summaries
        print("\n" + "=" * 80)
        print("DENOISING DIFFUSION MODEL ARCHITECTURE")
        print("=" * 80)
        print(f"\nDevice: {self._device}")
        print(f"Input Shape: {input_shape}")
        print(f"Batch Size: {self._denoising_diffusion_unet_batch_size}")
        print(f"Epochs: {self._denoising_diffusion_unet_epochs}")
        print(f"Time Steps: {self._denoising_diffusion_gaussian_time_steps}")
        print("\nFirst UNet Model:")
        print(self._denoising_first_unet_model)
        print("\nSecond UNet Model (EMA):")
        print(self._denoising_second_unet_model)
        print("=" * 80 + "\n")

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
        self._denoising_diffusion_algorithm = AlgorithmDenoisingDiffusionTorch(
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
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80 + "\n")

        self._denoising_diffusion_algorithm.train()
        best_loss = float('inf')

        for epoch in range(self._denoising_diffusion_unet_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
                batch_data = batch_data.to(self._device)
                batch_labels = batch_labels.to(self._device)

                # Training step
                loss_dict = self._denoising_diffusion_algorithm.train_step(
                    batch_data,
                    batch_labels
                )

                current_loss = loss_dict["Diffusion_loss"]
                epoch_loss += current_loss
                num_batches += 1

            # Calculate average loss for the epoch
            avg_loss = epoch_loss / num_batches

            # Store training history
            self._training_history['epoch'].append(epoch + 1)
            self._training_history['loss'].append(epoch_loss)
            self._training_history['avg_loss'].append(avg_loss)

            # Print epoch summary
            print(f"{'─' * 80}")
            print(f"EPOCH {epoch + 1}/{self._denoising_diffusion_unet_epochs}")
            print(f"{'─' * 80}")
            print(f"  Loss: {avg_loss:.6f}", end="")

            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"  ★ New Best!")
            else:
                print()

            print(f"{'─' * 80}\n")

            # Handle callbacks properly
            if hasattr(self, '_callback_model_monitor'):
                try:
                    # Try calling as a function
                    if callable(self._callback_model_monitor):
                        self._callback_model_monitor(epoch, avg_loss)
                    # Try calling with on_epoch_end method
                    elif hasattr(self._callback_model_monitor, 'on_epoch_end'):
                        self._callback_model_monitor.on_epoch_end(epoch, {'loss': avg_loss})
                    # Try calling with __call__ method
                    elif hasattr(self._callback_model_monitor, '__call__'):
                        self._callback_model_monitor.__call__(epoch, avg_loss)
                except Exception as e:
                    print(f"Warning: Could not call model monitor callback: {e}")

            # Handle early stopping
            if hasattr(arguments, 'use_early_stop') and arguments.use_early_stop:
                if hasattr(self, '_callback_early_stop'):
                    try:
                        should_stop = False
                        # Try different callback interfaces
                        if callable(self._callback_early_stop):
                            should_stop = self._callback_early_stop(epoch, avg_loss)
                        elif hasattr(self._callback_early_stop, 'on_epoch_end'):
                            should_stop = self._callback_early_stop.on_epoch_end(epoch, {'loss': avg_loss})
                        elif hasattr(self._callback_early_stop, '__call__'):
                            should_stop = self._callback_early_stop.__call__(epoch, avg_loss)

                        if should_stop:
                            print("\n" + "=" * 80)
                            print("EARLY STOPPING TRIGGERED")
                            print("=" * 80 + "\n")
                            break
                    except Exception as e:
                        print(f"Warning: Could not call early stop callback: {e}")

        # Print final training summary
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED")
        print("=" * 80)
        print(f"  Total Epochs:   {len(self._training_history['epoch'])}")
        print(f"  Best Loss:      {best_loss:.6f}")
        print(f"  Final Loss:     {avg_loss:.6f}")
        print("=" * 80 + "\n")

    def get_training_history(self):
        """
        Returns the training history.

        Returns:
            dict: Dictionary containing epoch, loss, and avg_loss lists
        """
        return self._training_history

    # Property getters and setters
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