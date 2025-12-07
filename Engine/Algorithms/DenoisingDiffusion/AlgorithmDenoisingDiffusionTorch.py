#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/07'
__credits__ = ['Kayuã Oleques']

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
    import os
    import sys
    import json
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from typing import Dict, Tuple, Optional
    from torch.utils.data import DataLoader, TensorDataset

except ImportError as error:
    print(error)
    sys.exit(-1)


class History:
    """History object for tracking training metrics."""

    def __init__(self):
        self.history = {}

    def __getitem__(self, key):
        return self.history[key]

    def __setitem__(self, key, value):
        self.history[key] = value

    def keys(self):
        return self.history.keys()


class AlgorithmDenoisingDiffusionTorch(nn.Module):
    """
    PyTorch implementation of a diffusion process using UNet architectures for generating synthetic data.

    This model integrates a diffusion network, enabling controlled generative modeling through 
    Gaussian diffusion. It supports exponential moving average (EMA) updates for stable training,
    multiple training stages, and customizable hyperparameters.

    Attributes:
        @ema (float): Exponential moving average decay rate for stabilizing training.
        @margin (float): Margin parameter for loss computation.
        @gdf_util: Utility object for Gaussian diffusion functions.
        @time_steps (int): Number of time steps in the diffusion process.
        @train_stage (str): Current training stage.
        @network (nn.Module): Primary UNet model for diffusion.
        @second_unet_model (nn.Module): Secondary UNet model for EMA updates.
        @optimizer_diffusion: Optimizer for diffusion model.

    Example:
        >>> diffusion_model = AlgorithmDenoisingDiffusionTorch(
        ...     output_shape=128,
        ...     first_unet_model=primary_unet,
        ...     second_unet_model=ema_unet,
        ...     gdf_util=gaussian_diffusion,
        ...     optimizer_diffusion=torch.optim.Adam(params, lr=2e-4),
        ...     time_steps=1000,
        ...     ema=0.999,
        ...     margin=0.1
        ... )
    """

    def __init__(self,
                 output_shape: int,
                 first_unet_model: nn.Module,
                 second_unet_model: nn.Module,
                 gdf_util,
                 optimizer_autoencoder,
                 optimizer_diffusion,
                 time_steps: int,
                 ema: float,
                 margin: float,
                 train_stage: str = 'all'):
        """
        Initializes the DenoisingDiffusionAlgorithmTorch.

        Args:
            output_shape (int): Shape of the output data.
            first_unet_model (nn.Module): Primary UNet model.
            second_unet_model (nn.Module): Secondary UNet model for EMA.
            gdf_util: Gaussian diffusion utility object.
            optimizer_autoencoder: Optimizer for autoencoder (kept for compatibility).
            optimizer_diffusion: Optimizer for diffusion model.
            time_steps (int): Number of diffusion time steps.
            ema (float): EMA decay factor.
            margin (float): Margin for loss calculation.
            train_stage (str): Training stage identifier.
        """
        super().__init__()

        self._ema = ema
        self._margin = margin
        self._gdf_util = gdf_util
        self._time_steps = time_steps
        self._train_stage = train_stage
        self._network = first_unet_model
        self._output_shape = output_shape
        self._original_shape = output_shape
        self._second_unet_model = second_unet_model
        self._optimizer_diffusion = optimizer_diffusion
        self._optimizer_autoencoder = optimizer_autoencoder

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._network.to(self.device)
        self._second_unet_model.to(self.device)

    def set_stage_training(self, training_stage: str):
        """Sets the current training stage."""
        self._train_stage = training_stage

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        Performs a single training step.

        Args:
            batch (tuple): Tuple containing (data, labels).

        Returns:
            dict: Dictionary with computed losses.
        """
        raw_data, label = batch

        # Move to device
        raw_data = raw_data.to(self.device)
        label = label.to(self.device)

        loss_diffusion = self.train_diffusion_model(raw_data, label)
        self.update_ema_weights()

        return {"Diffusion_loss": loss_diffusion.item() if loss_diffusion is not None else 0}

    def train_diffusion_model(self, data: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Performs a training step for the diffusion model.

        Args:
            data (torch.Tensor): Input data.
            ground_truth (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Computed loss.
        """
        self._network.train()

        embedding_label = ground_truth
        embedding_data_expanded = data
        batch_size = data.shape[0]

        # Pad input tensor if needed
        embedding_data_expanded = self._padding_input_tensor(embedding_data_expanded)
        embedding_data_expanded = embedding_data_expanded.float()

        # Update output shape based on padded tensor
        self._output_shape = embedding_data_expanded.shape[-2]

        # Sample random time steps
        random_time_steps = torch.randint(
            0, self._time_steps, (batch_size,),
            device=self.device, dtype=torch.long
        )

        # Zero gradients
        self._optimizer_diffusion.zero_grad()

        # Sample random noise
        random_noise = torch.randn_like(embedding_data_expanded)

        # Apply forward diffusion
        embedding_with_noise = self._gdf_util.q_sample(
            embedding_data_expanded,
            random_time_steps,
            random_noise
        )

        # Predict noise
        predicted_noise = self._network(embedding_with_noise, random_time_steps, embedding_label)

        # Compute loss
        loss_diffusion = F.mse_loss(predicted_noise.squeeze(-1), random_noise)

        # Backward pass
        loss_diffusion.backward()
        self._optimizer_diffusion.step()

        return loss_diffusion

    def update_ema_weights(self):
        """Updates EMA weights of the second UNet model."""
        with torch.no_grad():
            for weight, ema_weight in zip(self._network.parameters(), self._second_unet_model.parameters()):
                ema_weight.data.mul_(self._ema).add_(weight.data, alpha=1 - self._ema)

    def generate_data(self, labels: torch.Tensor, batch_size: int) -> np.ndarray:
        """
        Generates synthetic data by reversing the diffusion process.

        Args:
            labels (torch.Tensor): Class labels for conditioning.
            batch_size (int): Batch size for generation.

        Returns:
            np.ndarray: Generated synthetic data.
        """
        self._network.eval()

        with torch.no_grad():
            # Start with random noise
            synthetic_data = torch.randn(
                labels.shape[0], self._output_shape, 1,
                device=self.device, dtype=torch.float32
            )

            # Reshape labels
            labels_vector = labels.unsqueeze(-1) if len(labels.shape) == 2 else labels

            # Reverse diffusion process
            for time_step in reversed(range(self._time_steps)):
                # Create time step tensor
                array_time = torch.full(
                    (labels_vector.shape[0],), time_step,
                    device=self.device, dtype=torch.long
                )

                # Predict noise
                predicted_noise = self._network(synthetic_data, array_time, labels_vector)

                # Apply reverse diffusion step
                synthetic_data = self._gdf_util.p_sample(
                    predicted_noise, synthetic_data, array_time, clip_denoised=True
                )

            # Crop to original size
            generated_data = self._crop_tensor_to_original_size(
                synthetic_data.cpu().numpy(), self._original_shape
            )

        return generated_data

    @staticmethod
    def _crop_tensor_to_original_size(tensor: np.ndarray, original_size: int) -> np.ndarray:
        """
        Crops tensor to original size along the second dimension.

        Args:
            tensor (np.ndarray): Input tensor of shape (batch, seq_len, channels).
            original_size (int): Desired sequence length.

        Returns:
            np.ndarray: Cropped tensor.
        """
        if tensor.ndim != 3:
            raise ValueError(f"Expected 3D tensor, got shape: {tensor.shape}")

        current_size = tensor.shape[1]
        if current_size <= original_size:
            return tensor

        return tensor[:, :original_size, :]

    def _padding_input_tensor(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Pads input tensor to match network's expected input shape.

        Args:
            input_tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Padded tensor.
        """
        input_tensor = input_tensor.float()

        # Get target dimension from network
        target_dimension = self._network.input_shape[0][-2] if hasattr(self._network,
                                                                       'input_shape') else self._output_shape
        current_dimension = input_tensor.shape[-2]

        # Calculate padding needed
        padding_needed = max(0, target_dimension - current_dimension)

        if padding_needed > 0:
            # Pad along the sequence dimension (dim=-2)
            # F.pad format: (left, right, top, bottom, front, back) for last 3 dimensions
            pad = (0, 0, 0, padding_needed)  # (channels: 0,0, seq: 0,padding_needed)
            input_tensor = F.pad(input_tensor, pad, mode='constant', value=0)

        return input_tensor

    def fit(self, x=None, y=None, epochs=1, batch_size=32, verbose=1,
            validation_data=None, shuffle=True, **kwargs):
        """
        Train the diffusion model.

        Args:
            x: Training data (DataLoader, tuple, or tensor).
            y: Target labels (optional).
            epochs (int): Number of epochs.
            batch_size (int): Batch size.
            verbose (int): Verbosity level.
            validation_data: Validation data (not implemented).
            shuffle (bool): Whether to shuffle data.

        Returns:
            History: Training history object.
        """
        # Handle different input formats
        if isinstance(x, DataLoader):
            dataloader = x
        elif isinstance(x, tuple) and len(x) == 2:
            x_data, y_data = x
            x_tensor = torch.FloatTensor(x_data) if not isinstance(x_data, torch.Tensor) else x_data
            y_tensor = torch.FloatTensor(y_data) if not isinstance(y_data, torch.Tensor) else y_data
            dataset = TensorDataset(x_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        elif x is not None and y is not None:
            x_tensor = torch.FloatTensor(x) if not isinstance(x, torch.Tensor) else x
            y_tensor = torch.FloatTensor(y) if not isinstance(y, torch.Tensor) else y
            dataset = TensorDataset(x_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        else:
            raise ValueError("Invalid input format.")

        history_obj = History()
        history_obj['Diffusion_loss'] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                loss_dict = self.train_step(batch)
                epoch_loss += loss_dict['Diffusion_loss']
                num_batches += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            history_obj['Diffusion_loss'].append(avg_loss)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - Diffusion_loss: {avg_loss:.4f}")

        return history_obj

    def get_samples(self, number_samples_per_class: Dict) -> Dict:
        """
        Generates synthetic samples for each class.

        Args:
            number_samples_per_class (dict): Dict with 'classes' and 'number_classes' keys.

        Returns:
            dict: Generated samples per class.
        """
        generated_data = {}

        for label_class, number_instances in number_samples_per_class["classes"].items():
            # Create one-hot encoded labels
            label_samples_generated = F.one_hot(
                torch.tensor([label_class] * number_instances),
                num_classes=number_samples_per_class["number_classes"]
            ).float().to(self.device)

            # Generate samples
            generated_samples = self.generate_data(label_samples_generated, batch_size=64)

            # Round and squeeze
            generated_samples = np.rint(np.squeeze(generated_samples, axis=-1))
            generated_data[label_class] = generated_samples

        return generated_data

    def save_model(self, directory: str, file_name: str):
        """
        Save the models to disk.

        Args:
            directory (str): Output directory.
            file_name (str): Base filename.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        first_unet_file = os.path.join(directory, f"{file_name}_first_unet.pth")
        second_unet_file = os.path.join(directory, f"{file_name}_second_unet.pth")

        # Save models
        torch.save({
            'model_state_dict': self._network.state_dict(),
            'optimizer_state_dict': self._optimizer_diffusion.state_dict()
        }, first_unet_file)

        torch.save({
            'model_state_dict': self._second_unet_model.state_dict()
        }, second_unet_file)

    def load_models(self, directory: str, file_name: str):
        """
        Load models from disk.

        Args:
            directory (str): Input directory.
            file_name (str): Base filename.
        """
        first_unet_file = os.path.join(directory, f"{file_name}_first_unet.pth")
        second_unet_file = os.path.join(directory, f"{file_name}_second_unet.pth")

        # Load first UNet
        checkpoint = torch.load(first_unet_file, map_location=self.device)
        self._network.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self._optimizer_diffusion.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load second UNet
        checkpoint = torch.load(second_unet_file, map_location=self.device)
        self._second_unet_model.load_state_dict(checkpoint['model_state_dict'])

        self._network.eval()
        self._second_unet_model.eval()

    # Properties
    @property
    def ema(self) -> float:
        return self._ema

    @ema.setter
    def ema(self, value: float):
        if not 0 < value < 1:
            raise ValueError("EMA must be between 0 and 1")
        self._ema = value

    @property
    def margin(self) -> float:
        return self._margin

    @margin.setter
    def margin(self, value: float):
        if value <= 0:
            raise ValueError("Margin must be positive")
        self._margin = value

    @property
    def gdf_util(self):
        return self._gdf_util

    @gdf_util.setter
    def gdf_util(self, value):
        self._gdf_util = value

    @property
    def time_steps(self) -> int:
        return self._time_steps

    @time_steps.setter
    def time_steps(self, value: int):
        if value <= 0:
            raise ValueError("Time steps must be positive")
        self._time_steps = value

    @property
    def train_stage(self) -> str:
        return self._train_stage

    @train_stage.setter
    def train_stage(self, value: str):
        self._train_stage = value

    @property
    def network(self) -> nn.Module:
        return self._network

    @network.setter
    def network(self, value: nn.Module):
        self._network = value
        self._network.to(self.device)

    @property
    def second_unet_model(self) -> nn.Module:
        return self._second_unet_model

    @second_unet_model.setter
    def second_unet_model(self, value: nn.Module):
        self._second_unet_model = value
        self._second_unet_model.to(self.device)

    @property
    def optimizer_diffusion(self):
        return self._optimizer_diffusion

    @optimizer_diffusion.setter
    def optimizer_diffusion(self, value):
        self._optimizer_diffusion = value

    @property
    def optimizer_autoencoder(self):
        return self._optimizer_autoencoder

    @optimizer_autoencoder.setter
    def optimizer_autoencoder(self, value):
        self._optimizer_autoencoder = value

    def to(self, device: torch.device):
        """Move all models to the specified device."""
        self.device = device
        self._network.to(device)
        self._second_unet_model.to(device)
        if self._gdf_util is not None and hasattr(self._gdf_util, 'to'):
            self._gdf_util.to(device)
        return self