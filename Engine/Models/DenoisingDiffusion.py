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
from Engine.Architectures.DenoisingDiffusion.Torch.DiffusionModelUnetTorch import UNetDenoisingModelTorch

try:
    import sys
    import numpy as np
    import logging
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

except ImportError as error:
    logging.error(error)
    sys.exit(-1)

# Default values from your constants file
DEFAULT_DIFFUSION_UNET_LAST_LAYER_ACTIVATION = 'linear'
DEFAULT_DIFFUSION_LATENT_DIMENSION = 64
DEFAULT_DIFFUSION_UNET_NUMBER_EMBEDDING_CHANNELS = 1
DEFAULT_DIFFUSION_UNET_CHANNELS_PER_LEVEL = [1, 2, 4]
DEFAULT_DIFFUSION_UNET_BATCH_SIZE = 128
DEFAULT_DIFFUSION_UNET_ATTENTION_MODE = [False, True, True]
DEFAULT_DIFFUSION_UNET_NUMBER_RESIDUAL_BLOCKS = 2
DEFAULT_DIFFUSION_UNET_GROUP_NORMALIZATION = 1
DEFAULT_DIFFUSION_UNET_INTERMEDIARY_ACTIVATION = 'swish'
DEFAULT_DIFFUSION_UNET_INTERMEDIARY_ACTIVATION_ALPHA = 0.05
DEFAULT_DIFFUSION_UNET_NUMBER_EPOCHS = 1000
DEFAULT_DIFFUSION_GAUSSIAN_BETA_START = 1e-4
DEFAULT_DIFFUSION_GAUSSIAN_BETA_END = 0.02
DEFAULT_DIFFUSION_GAUSSIAN_TIME_STEPS = 1000
DEFAULT_DIFFUSION_GAUSSIAN_CLIP_MIN = -1.0
DEFAULT_DIFFUSION_GAUSSIAN_CLIP_MAX = 1.0
DEFAULT_DIFFUSION_MARGIN = 0.5
DEFAULT_DIFFUSION_EMA = 0.999
DEFAULT_DIFFUSION_TIME_STEPS = 1000


class DenoisingDiffusionInstanceTorch:
    """
    A class that implements a Denoising Diffusion Probabilistic Model (DDPM) for image generation.
    """

    def __init__(
            self,
            denoising_diffusion_unet_last_layer_activation: str = DEFAULT_DIFFUSION_UNET_LAST_LAYER_ACTIVATION,
            denoising_diffusion_latent_dimension: int = DEFAULT_DIFFUSION_LATENT_DIMENSION,
            denoising_diffusion_unet_num_embedding_channels: int = DEFAULT_DIFFUSION_UNET_NUMBER_EMBEDDING_CHANNELS,
            denoising_diffusion_unet_channels_per_level: list[int] = None,
            denoising_diffusion_unet_batch_size: int = DEFAULT_DIFFUSION_UNET_BATCH_SIZE,
            denoising_diffusion_unet_attention_mode: list[bool] = None,
            denoising_diffusion_unet_num_residual_blocks: int = DEFAULT_DIFFUSION_UNET_NUMBER_RESIDUAL_BLOCKS,
            denoising_diffusion_unet_group_normalization: int = DEFAULT_DIFFUSION_UNET_GROUP_NORMALIZATION,
            denoising_diffusion_unet_intermediary_activation: str = DEFAULT_DIFFUSION_UNET_INTERMEDIARY_ACTIVATION,
            denoising_diffusion_unet_intermediary_activation_alpha: float = DEFAULT_DIFFUSION_UNET_INTERMEDIARY_ACTIVATION_ALPHA,
            denoising_diffusion_unet_epochs: int = DEFAULT_DIFFUSION_UNET_NUMBER_EPOCHS,
            denoising_diffusion_gaussian_beta_start: float = DEFAULT_DIFFUSION_GAUSSIAN_BETA_START,
            denoising_diffusion_gaussian_beta_end: float = DEFAULT_DIFFUSION_GAUSSIAN_BETA_END,
            denoising_diffusion_gaussian_time_steps: int = DEFAULT_DIFFUSION_GAUSSIAN_TIME_STEPS,
            denoising_diffusion_gaussian_clip_min: float = DEFAULT_DIFFUSION_GAUSSIAN_CLIP_MIN,
            denoising_diffusion_gaussian_clip_max: float = DEFAULT_DIFFUSION_GAUSSIAN_CLIP_MAX,
            denoising_diffusion_margin: float = DEFAULT_DIFFUSION_MARGIN,
            denoising_diffusion_ema: float = DEFAULT_DIFFUSION_EMA,
            denoising_diffusion_time_steps: int = DEFAULT_DIFFUSION_TIME_STEPS,
            denoising_first_unet_model: UNetDenoisingModelTorch | None = None,
            denoising_second_unet_model: UNetDenoisingModelTorch | None = None,
            denoising_gaussian_diffusion_util: GaussianDiffusionTorch | None = None,
            denoising_diffusion_algorithm: AlgorithmDenoisingDiffusionTorch | None = None
    ) -> None:
        """
        Initializes the denoising diffusion instance with configuration parameters.

        Args:
            denoising_diffusion_unet_last_layer_activation: Activation for last layer (default: 'linear')
            denoising_diffusion_latent_dimension: Dimension of latent space (default: 64)
            denoising_diffusion_unet_num_embedding_channels: Embedding channels (default: 1)
            denoising_diffusion_unet_channels_per_level: Channels per U-Net level (default: [1, 2, 4])
            denoising_diffusion_unet_batch_size: Batch size (default: 128)
            denoising_diffusion_unet_attention_mode: Attention modes per level (default: [False, True, True])
            denoising_diffusion_unet_num_residual_blocks: Residual blocks (default: 2)
            denoising_diffusion_unet_group_normalization: Group norm groups (default: 1)
            denoising_diffusion_unet_intermediary_activation: Intermediary activation (default: 'swish')
            denoising_diffusion_unet_intermediary_activation_alpha: Activation alpha (default: 0.05)
            denoising_diffusion_unet_epochs: Training epochs (default: 1000)
            denoising_diffusion_gaussian_beta_start: Beta start value (default: 1e-4)
            denoising_diffusion_gaussian_beta_end: Beta end value (default: 0.02)
            denoising_diffusion_gaussian_time_steps: Diffusion time steps (default: 1000)
            denoising_diffusion_gaussian_clip_min: Noise clip minimum (default: -1.0)
            denoising_diffusion_gaussian_clip_max: Noise clip maximum (default: 1.0)
            denoising_diffusion_margin: Margin for diffusion (default: 0.5)
            denoising_diffusion_ema: Exponential moving average (default: 0.999)
            denoising_diffusion_time_steps: Time steps (default: 1000)
            denoising_first_unet_model: Optional pre-initialized first UNet (default: None)
            denoising_second_unet_model: Optional pre-initialized second UNet (default: None)
            denoising_gaussian_diffusion_util: Optional pre-initialized diffusion util (default: None)
            denoising_diffusion_algorithm: Optional pre-initialized algorithm (default: None)
        """
        # Store pre-initialized instances if provided
        self._denoising_first_unet_model: UNetDenoisingModelTorch | None = denoising_first_unet_model
        self._denoising_second_unet_model: UNetDenoisingModelTorch | None = denoising_second_unet_model
        self._denoising_gaussian_diffusion_util: GaussianDiffusionTorch | None = denoising_gaussian_diffusion_util
        self._denoising_diffusion_algorithm: AlgorithmDenoisingDiffusionTorch | None = denoising_diffusion_algorithm

        # Internal instances
        self._denoising_first_instance_unet: UNetDenoisingModelTorch | None = None
        self._denoising_second_instance_unet: UNetDenoisingModelTorch | None = None

        # Configuration parameters
        self._denoising_diffusion_unet_last_layer_activation: str = denoising_diffusion_unet_last_layer_activation
        self._denoising_diffusion_latent_dimension: int = denoising_diffusion_latent_dimension
        self._denoising_diffusion_unet_num_embedding_channels: int = denoising_diffusion_unet_num_embedding_channels

        # Handle mutable default values safely
        self._denoising_diffusion_unet_channels_per_level: list[int] = (
            denoising_diffusion_unet_channels_per_level
            if denoising_diffusion_unet_channels_per_level is not None
            else DEFAULT_DIFFUSION_UNET_CHANNELS_PER_LEVEL.copy()
        )
        self._denoising_diffusion_unet_attention_mode: list[bool] = (
            denoising_diffusion_unet_attention_mode
            if denoising_diffusion_unet_attention_mode is not None
            else DEFAULT_DIFFUSION_UNET_ATTENTION_MODE.copy()
        )

        self._denoising_diffusion_unet_batch_size: int = denoising_diffusion_unet_batch_size
        self._denoising_diffusion_unet_num_residual_blocks: int = denoising_diffusion_unet_num_residual_blocks
        self._denoising_diffusion_unet_group_normalization: int = denoising_diffusion_unet_group_normalization
        self._denoising_diffusion_unet_intermediary_activation: str = denoising_diffusion_unet_intermediary_activation
        self._denoising_diffusion_unet_intermediary_activation_alpha: float = denoising_diffusion_unet_intermediary_activation_alpha
        self._denoising_diffusion_unet_epochs: int = denoising_diffusion_unet_epochs

        # Diffusion Process Parameters
        self._denoising_diffusion_gaussian_beta_start: float = denoising_diffusion_gaussian_beta_start
        self._denoising_diffusion_gaussian_beta_end: float = denoising_diffusion_gaussian_beta_end
        self._denoising_diffusion_gaussian_time_steps: int = denoising_diffusion_gaussian_time_steps
        self._denoising_diffusion_gaussian_clip_min: float = denoising_diffusion_gaussian_clip_min
        self._denoising_diffusion_gaussian_clip_max: float = denoising_diffusion_gaussian_clip_max
        self._denoising_diffusion_margin: float = denoising_diffusion_margin
        self._denoising_diffusion_ema: float = denoising_diffusion_ema
        self._denoising_diffusion_time_steps: int = denoising_diffusion_time_steps

        # Device configuration
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Training history
        self._training_history = {
            'epoch': [],
            'loss': [],
            'avg_loss': []
        }

        # Flags to indicate if instances were provided
        self._has_external_first_unet: bool = denoising_first_unet_model is not None
        self._has_external_second_unet: bool = denoising_second_unet_model is not None
        self._has_external_diffusion_util: bool = denoising_gaussian_diffusion_util is not None
        self._has_external_algorithm: bool = denoising_diffusion_algorithm is not None

    def _get_denoising_diffusion(self, input_shape: tuple[int, ...]) -> None:
        """
        Initializes and configures the LatentDiffusion model using UNet architecture.

        If pre-initialized instances were provided in the constructor, they are used instead of creating new ones.
        """
        # Only create new UNet models if none were provided
        if not self._has_external_first_unet:
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
            self._denoising_first_unet_model = self._denoising_first_instance_unet
        else:
            # Use provided model and move to device
            self._denoising_first_unet_model = self._denoising_first_unet_model.to(self._device)

        if not self._has_external_second_unet:
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
            self._denoising_second_unet_model = self._denoising_second_instance_unet

            # Synchronize weights from first model if both were created
            if not self._has_external_first_unet:
                self._denoising_second_unet_model.load_state_dict(
                    self._denoising_first_unet_model.state_dict()
                )
        else:
            # Use provided model and move to device
            self._denoising_second_unet_model = self._denoising_second_unet_model.to(self._device)

        # Only create new GaussianDiffusion utility if none was provided
        if not self._has_external_diffusion_util:
            # Initialize the GaussianDiffusion utility
            self._denoising_gaussian_diffusion_util = GaussianDiffusionTorch(
                beta_start=self._denoising_diffusion_gaussian_beta_start,
                beta_end=self._denoising_diffusion_gaussian_beta_end,
                time_steps=self._denoising_diffusion_gaussian_time_steps,
                clip_min=self._denoising_diffusion_gaussian_clip_min,
                clip_max=self._denoising_diffusion_gaussian_clip_max
            )

        # Only create new algorithm if none was provided
        if not self._has_external_algorithm:
            # Ensure we have all required components
            if self._denoising_first_unet_model is None:
                raise ValueError("First UNet model is required but was not provided.")
            if self._denoising_second_unet_model is None:
                raise ValueError("Second UNet model is required but was not provided.")
            if self._denoising_gaussian_diffusion_util is None:
                raise ValueError("GaussianDiffusion utility is required but was not provided.")

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
        else:
            # If algorithm was provided externally, update its configuration if needed
            # (assuming AlgorithmDenoisingDiffusionTorch has setters for these properties)
            if hasattr(self._denoising_diffusion_algorithm, 'time_steps'):
                self._denoising_diffusion_algorithm.time_steps = self._denoising_diffusion_gaussian_time_steps
            if hasattr(self._denoising_diffusion_algorithm, 'ema'):
                self._denoising_diffusion_algorithm.ema = self._denoising_diffusion_ema
            if hasattr(self._denoising_diffusion_algorithm, 'margin'):
                self._denoising_diffusion_algorithm.margin = self._denoising_diffusion_margin
            # Move algorithm to device
            self._denoising_diffusion_algorithm = self._denoising_diffusion_algorithm.to(self._device)

    def _training_denoising_diffusion_model(
            self,
            input_shape: tuple[int, ...],
            arguments: 'argparse.Namespace',
            x_real_samples: np.ndarray,
            y_real_samples: np.ndarray
    ) -> None:
        """
        Executes the complete denoising diffusion training pipeline.
        """
        # Initialize the diffusion model (or use provided)
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

        if self._denoising_first_unet_model is not None:
            print("\nFirst UNet Model:")
            print(self._denoising_first_unet_model)

        if self._denoising_second_unet_model is not None:
            print("\nSecond UNet Model (EMA):")
            print(self._denoising_second_unet_model)

        print("=" * 80 + "\n")

        # Ensure we have an algorithm
        if self._denoising_diffusion_algorithm is None:
            raise ValueError("Denoising diffusion algorithm is required but was not provided or created.")

        # Prepare data
        x_real_samples = np.array(x_real_samples)
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

    # Additional getters for the components
    @property
    def denoising_first_unet_model(self) -> UNetDenoisingModelTorch | None:
        """Get the first UNet model instance."""
        return self._denoising_first_unet_model

    @property
    def denoising_second_unet_model(self) -> UNetDenoisingModelTorch | None:
        """Get the second UNet model instance."""
        return self._denoising_second_unet_model

    @property
    def denoising_gaussian_diffusion_util(self) -> GaussianDiffusionTorch | None:
        """Get the Gaussian diffusion utility instance."""
        return self._denoising_gaussian_diffusion_util

    @property
    def denoising_diffusion_algorithm(self) -> AlgorithmDenoisingDiffusionTorch | None:
        """Get the diffusion algorithm instance."""
        return self._denoising_diffusion_algorithm

    def get_training_history(self) -> dict:
        """
        Returns the training history.

        Returns:
            dict: Dictionary containing epoch, loss, and avg_loss lists
        """
        return self._training_history

    # Property getters and setters
    @property
    def denoising_diffusion_unet_last_layer_activation(self) -> str:
        """Get the last layer activation."""
        return self._denoising_diffusion_unet_last_layer_activation

    @denoising_diffusion_unet_last_layer_activation.setter
    def denoising_diffusion_unet_last_layer_activation(self, value: str) -> None:
        """Set the last layer activation."""
        self._denoising_diffusion_unet_last_layer_activation = value

    @property
    def denoising_diffusion_latent_dimension(self) -> int:
        """Get the latent dimension."""
        return self._denoising_diffusion_latent_dimension

    @denoising_diffusion_latent_dimension.setter
    def denoising_diffusion_latent_dimension(self, value: int) -> None:
        """Set the latent dimension."""
        self._denoising_diffusion_latent_dimension = value

    @property
    def denoising_diffusion_unet_num_embedding_channels(self) -> int:
        """Get the number of embedding channels."""
        return self._denoising_diffusion_unet_num_embedding_channels

    @denoising_diffusion_unet_num_embedding_channels.setter
    def denoising_diffusion_unet_num_embedding_channels(self, value: int) -> None:
        """Set the number of embedding channels."""
        self._denoising_diffusion_unet_num_embedding_channels = value

    @property
    def denoising_diffusion_unet_channels_per_level(self) -> list[int]:
        """Get the channels per level."""
        return self._denoising_diffusion_unet_channels_per_level

    @denoising_diffusion_unet_channels_per_level.setter
    def denoising_diffusion_unet_channels_per_level(self, value: list[int]) -> None:
        """Set the channels per level."""
        self._denoising_diffusion_unet_channels_per_level = value

    @property
    def denoising_diffusion_unet_batch_size(self) -> int:
        """Get the batch size."""
        return self._denoising_diffusion_unet_batch_size

    @denoising_diffusion_unet_batch_size.setter
    def denoising_diffusion_unet_batch_size(self, value: int) -> None:
        """Set the batch size."""
        self._denoising_diffusion_unet_batch_size = value

    @property
    def denoising_diffusion_unet_attention_mode(self) -> list[bool]:
        """Get the attention mode."""
        return self._denoising_diffusion_unet_attention_mode

    @denoising_diffusion_unet_attention_mode.setter
    def denoising_diffusion_unet_attention_mode(self, value: list[bool]) -> None:
        """Set the attention mode."""
        self._denoising_diffusion_unet_attention_mode = value

    @property
    def denoising_diffusion_unet_num_residual_blocks(self) -> int:
        """Get the number of residual blocks."""
        return self._denoising_diffusion_unet_num_residual_blocks

    @denoising_diffusion_unet_num_residual_blocks.setter
    def denoising_diffusion_unet_num_residual_blocks(self, value: int) -> None:
        """Set the number of residual blocks."""
        self._denoising_diffusion_unet_num_residual_blocks = value

    @property
    def denoising_diffusion_unet_group_normalization(self) -> int:
        """Get the group normalization value."""
        return self._denoising_diffusion_unet_group_normalization

    @denoising_diffusion_unet_group_normalization.setter
    def denoising_diffusion_unet_group_normalization(self, value: int) -> None:
        """Set the group normalization value."""
        self._denoising_diffusion_unet_group_normalization = value

    @property
    def denoising_diffusion_unet_intermediary_activation(self) -> str:
        """Get the intermediary activation."""
        return self._denoising_diffusion_unet_intermediary_activation

    @denoising_diffusion_unet_intermediary_activation.setter
    def denoising_diffusion_unet_intermediary_activation(self, value: str) -> None:
        """Set the intermediary activation."""
        self._denoising_diffusion_unet_intermediary_activation = value

    @property
    def denoising_diffusion_unet_intermediary_activation_alpha(self) -> float:
        """Get the intermediary activation alpha."""
        return self._denoising_diffusion_unet_intermediary_activation_alpha

    @denoising_diffusion_unet_intermediary_activation_alpha.setter
    def denoising_diffusion_unet_intermediary_activation_alpha(self, value: float) -> None:
        """Set the intermediary activation alpha."""
        self._denoising_diffusion_unet_intermediary_activation_alpha = value

    @property
    def denoising_diffusion_unet_epochs(self) -> int:
        """Get the number of epochs."""
        return self._denoising_diffusion_unet_epochs

    @denoising_diffusion_unet_epochs.setter
    def denoising_diffusion_unet_epochs(self, value: int) -> None:
        """Set the number of epochs."""
        self._denoising_diffusion_unet_epochs = value

    @property
    def denoising_diffusion_gaussian_beta_start(self) -> float:
        """Get the Gaussian beta start value."""
        return self._denoising_diffusion_gaussian_beta_start

    @denoising_diffusion_gaussian_beta_start.setter
    def denoising_diffusion_gaussian_beta_start(self, value: float) -> None:
        """Set the Gaussian beta start value."""
        self._denoising_diffusion_gaussian_beta_start = value

    @property
    def denoising_diffusion_gaussian_beta_end(self) -> float:
        """Get the Gaussian beta end value."""
        return self._denoising_diffusion_gaussian_beta_end

    @denoising_diffusion_gaussian_beta_end.setter
    def denoising_diffusion_gaussian_beta_end(self, value: float) -> None:
        """Set the Gaussian beta end value."""
        self._denoising_diffusion_gaussian_beta_end = value

    @property
    def denoising_diffusion_gaussian_time_steps(self) -> int:
        """Get the Gaussian time steps."""
        return self._denoising_diffusion_gaussian_time_steps

    @denoising_diffusion_gaussian_time_steps.setter
    def denoising_diffusion_gaussian_time_steps(self, value: int) -> None:
        """Set the Gaussian time steps."""
        self._denoising_diffusion_gaussian_time_steps = value

    @property
    def denoising_diffusion_gaussian_clip_min(self) -> float:
        """Get the Gaussian clip minimum."""
        return self._denoising_diffusion_gaussian_clip_min

    @denoising_diffusion_gaussian_clip_min.setter
    def denoising_diffusion_gaussian_clip_min(self, value: float) -> None:
        """Set the Gaussian clip minimum."""
        self._denoising_diffusion_gaussian_clip_min = value

    @property
    def denoising_diffusion_gaussian_clip_max(self) -> float:
        """Get the Gaussian clip maximum."""
        return self._denoising_diffusion_gaussian_clip_max

    @denoising_diffusion_gaussian_clip_max.setter
    def denoising_diffusion_gaussian_clip_max(self, value: float) -> None:
        """Set the Gaussian clip maximum."""
        self._denoising_diffusion_gaussian_clip_max = value

    @property
    def denoising_diffusion_margin(self) -> float:
        """Get the diffusion margin."""
        return self._denoising_diffusion_margin

    @denoising_diffusion_margin.setter
    def denoising_diffusion_margin(self, value: float) -> None:
        """Set the diffusion margin."""
        self._denoising_diffusion_margin = value

    @property
    def denoising_diffusion_ema(self) -> float:
        """Get the EMA value."""
        return self._denoising_diffusion_ema

    @denoising_diffusion_ema.setter
    def denoising_diffusion_ema(self, value: float) -> None:
        """Set the EMA value."""
        self._denoising_diffusion_ema = value

    @property
    def denoising_diffusion_time_steps(self) -> int:
        """Get the time steps."""
        return self._denoising_diffusion_time_steps

    @denoising_diffusion_time_steps.setter
    def denoising_diffusion_time_steps(self, value: int) -> None:
        """Set the time steps."""
        self._denoising_diffusion_time_steps = value
