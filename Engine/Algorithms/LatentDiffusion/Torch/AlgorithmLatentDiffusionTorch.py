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
    import os
    import sys
    import json
    import numpy
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from typing import Any

except ImportError as error:
    print(error)
    sys.exit(-1)


class AlgorithmLatentDiffusionTorch(nn.Module):
    """
    PyTorch implementation of a diffusion process using UNet architectures for generating synthetic data.

    This model integrates an autoencoder and a diffusion network, enabling both data
    reconstruction and controlled generative modeling through Gaussian diffusion.

    This class supports exponential moving average (EMA) updates for stable training,
    multiple training stages, and customizable hyperparameters to adapt to different tasks.

    Attributes:
        ema (float):
            Exponential moving average (EMA) decay rate for stabilizing training updates.
        margin (float):
            Margin parameter used for loss computation or regularization purposes.
        gdf_util:
            Utility object for Gaussian diffusion functions, handling noise scheduling.
        time_steps (int):
            Number of time steps used in the diffusion process.
        train_stage (str):
            Defines the current training stage ('all', 'diffusion', etc.).
        network (nn.Module):
            Primary UNet model responsible for the diffusion process.
        second_unet_model (nn.Module):
            Secondary UNet model used for EMA-based weight updates.
        embedding_dimension (int):
            Dimensionality of the latent space used for encoding data.
        encoder_model_data (nn.Module):
            Encoder model responsible for feature extraction.
        decoder_model_data (nn.Module):
            Decoder model used to reconstruct data.
        optimizer_diffusion:
            Optimizer used for training the diffusion model.
        optimizer_autoencoder:
            Optimizer responsible for training the autoencoder.
        device (str):
            Device for computation ('cuda' or 'cpu').
    """

    def __init__(self,
                 first_unet_model,
                 second_unet_model,
                 encoder_model_image,
                 decoder_model_image,
                 gdf_util,
                 optimizer_autoencoder,
                 optimizer_diffusion,
                 time_steps,
                 ema,
                 margin,
                 embedding_dimension,
                 train_stage='all',
                 device='cuda'):
        """
        Initializes the PyTorch Diffusion Model.

        Args:
            first_unet_model (nn.Module): Primary UNet model for diffusion.
            second_unet_model (nn.Module): Secondary UNet model for EMA updates.
            encoder_model_image (nn.Module): Encoder model.
            decoder_model_image (nn.Module): Decoder model.
            gdf_util: Gaussian diffusion utility object.
            optimizer_autoencoder: Optimizer for autoencoder.
            optimizer_diffusion: Optimizer for diffusion model.
            time_steps (int): Number of diffusion time steps.
            ema (float): Exponential moving average decay factor.
            margin (float): Margin value for loss calculations.
            embedding_dimension (int): Dimensionality of embedding space.
            train_stage (str): Current training stage ('all', 'diffusion', etc.).
            device (str): Device for computation ('cuda' or 'cpu').
        """
        super().__init__()

        self._ema = ema
        self._margin = margin
        self._gdf_util = gdf_util
        self._time_steps = time_steps
        self._train_stage = train_stage
        self._network = first_unet_model
        self._second_unet_model = second_unet_model
        self._embedding_dimension = embedding_dimension
        self._encoder_model_data = encoder_model_image
        self._decoder_model_data = decoder_model_image
        self._optimizer_diffusion = optimizer_diffusion
        self._optimizer_autoencoder = optimizer_autoencoder
        self._device = device

    def set_stage_training(self, training_stage):
        """Sets the current training stage."""
        self._train_stage = training_stage

    def train_step(self, data):
        """
        Performs a single training step for the diffusion model.

        Args:
            data (tuple): A tuple containing (input_data, labels).

        Returns:
            dict: Dictionary with computed losses.
        """
        raw_data, label = data

        # Convert to tensors if needed
        if not isinstance(raw_data, torch.Tensor):
            raw_data = torch.from_numpy(raw_data).float().to(self._device)
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(label).float().to(self._device)

        loss_diffusion = self.train_diffusion_model(raw_data, label)
        self.update_ema_weights()

        return {"Diffusion_loss": loss_diffusion.item() if loss_diffusion is not None else 0}

    def train_diffusion_model(self, data, ground_truth):
        """
        Performs a single training step for the diffusion model.

        Args:
            data (torch.Tensor): Input data embeddings.
            ground_truth (torch.Tensor): Corresponding class labels.

        Returns:
            torch.Tensor: The computed loss for this training step.
        """
        self._network.train()

        embedding_label = ground_truth
        embedding_data_expanded = data
        batch_size = data.shape[0]

        # Sample random time steps
        random_time_steps = torch.randint(0, self._time_steps, (batch_size,),
                                          device=self._device, dtype=torch.long)

        # Zero gradients
        self._optimizer_diffusion.zero_grad()

        # Sample random noise
        random_noise = torch.randn_like(embedding_data_expanded)

        # Apply forward diffusion process
        embedding_with_noise = self._gdf_util.q_sample(
            embedding_data_expanded,
            random_time_steps,
            random_noise
        )

        # Predict noise using the diffusion model
        predicted_noise = self._network(embedding_with_noise, random_time_steps, embedding_label)

        # Compute loss
        loss_diffusion = F.mse_loss(predicted_noise, random_noise)

        # Backward pass
        loss_diffusion.backward()
        self._optimizer_diffusion.step()

        return loss_diffusion

    def update_ema_weights(self):
        """Updates the weights of the second UNet model using EMA."""
        with torch.no_grad():
            for param, ema_param in zip(self._network.parameters(),
                                        self._second_unet_model.parameters()):
                ema_param.data.mul_(self._ema).add_(param.data, alpha=1 - self._ema)

    def generate_data(self, labels, batch_size):
        """
        Generates synthetic data by reversing the diffusion process.

        Args:
            labels (torch.Tensor): Class labels for conditioning.
            batch_size (int): Number of samples to generate.

        Returns:
            numpy.ndarray: Generated synthetic data samples.
        """
        self._network.eval()

        with torch.no_grad():
            # Start with random noise
            embedding_diffusion = torch.randn(
                labels.shape[0], self._embedding_dimension, 1,
                device=self._device, dtype=torch.float32
            )

            labels_vector = labels.unsqueeze(-1) if len(labels.shape) == 2 else labels

            # Reverse diffusion process
            for time_step in reversed(range(0, self._time_steps)):
                array_time = torch.full((labels_vector.shape[0],), time_step,
                                        device=self._device, dtype=torch.long)

                # Predict noise
                predicted_noise = self._network(embedding_diffusion, array_time, labels_vector)

                # Apply reverse diffusion step
                embedding_diffusion = self._gdf_util.p_sample(
                    predicted_noise,
                    embedding_diffusion,
                    array_time,
                    clip_denoised=True
                )

            # Decode embeddings to data
            generated_data = self._decoder_model_data(embedding_diffusion, labels_vector)

        return generated_data.cpu().numpy()

    def get_samples(self, number_samples_per_class):
        """
        Generates synthetic data samples for each class.

        Args:
            number_samples_per_class (dict): Dictionary with class info.

        Returns:
            dict: Generated samples for each class.
        """
        generated_data = {}

        for label_class, number_instances in number_samples_per_class["classes"].items():
            # Create one-hot encoded labels
            labels = torch.zeros(number_instances, number_samples_per_class["number_classes"])
            labels[:, label_class] = 1
            labels = labels.float().to(self._device)

            # Generate samples
            generated_samples = self.generate_data(labels, batch_size=64)
            generated_samples = numpy.rint(generated_samples)

            generated_data[label_class] = generated_samples

        return generated_data

    def save_model(self, directory, file_name):
        """
        Save the model components.

        Args:
            directory (str): Directory to save models.
            file_name (str): Base file name.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save model states
        torch.save(self._encoder_model_data.state_dict(),
                   os.path.join(directory, f"{file_name}_encoder.pth"))
        torch.save(self._decoder_model_data.state_dict(),
                   os.path.join(directory, f"{file_name}_decoder.pth"))
        torch.save(self._network.state_dict(),
                   os.path.join(directory, f"{file_name}_first_unet.pth"))
        torch.save(self._second_unet_model.state_dict(),
                   os.path.join(directory, f"{file_name}_second_unet.pth"))

        print(f"Models saved to {directory}")

    # Properties
    @property
    def ema(self) -> float:
        return self._ema

    @ema.setter
    def ema(self, value: float) -> None:
        self._ema = value

    @property
    def margin(self) -> float:
        return self._margin

    @margin.setter
    def margin(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Margin must be positive")
        self._margin = value

    @property
    def gdf_util(self) -> Any:
        return self._gdf_util

    @gdf_util.setter
    def gdf_util(self, value: Any) -> None:
        self._gdf_util = value

    @property
    def time_steps(self) -> int:
        return self._time_steps

    @time_steps.setter
    def time_steps(self, value: int) -> None:
        if value <= 0:
            raise ValueError("Time steps must be positive")
        self._time_steps = value

    @property
    def train_stage(self) -> str:
        return self._train_stage

    @train_stage.setter
    def train_stage(self, value: str) -> None:
        self._train_stage = value

    @property
    def network(self) -> Any:
        return self._network

    @network.setter
    def network(self, value: Any) -> None:
        self._network = value

    @property
    def second_unet_model(self) -> Any:
        return self._second_unet_model

    @second_unet_model.setter
    def second_unet_model(self, value: Any) -> None:
        self._second_unet_model = value

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    @embedding_dimension.setter
    def embedding_dimension(self, value: int) -> None:
        if value <= 0:
            raise ValueError("Embedding dimension must be positive")
        self._embedding_dimension = value

    @property
    def encoder_model_data(self) -> Any:
        return self._encoder_model_data

    @encoder_model_data.setter
    def encoder_model_data(self, value: Any) -> None:
        self._encoder_model_data = value

    @property
    def decoder_model_data(self) -> Any:
        return self._decoder_model_data

    @decoder_model_data.setter
    def decoder_model_data(self, value: Any) -> None:
        self._decoder_model_data = value

    @property
    def optimizer_diffusion(self) -> Any:
        return self._optimizer_diffusion

    @optimizer_diffusion.setter
    def optimizer_diffusion(self, value: Any) -> None:
        self._optimizer_diffusion = value

    @property
    def optimizer_autoencoder(self) -> Any:
        return self._optimizer_autoencoder

    @optimizer_autoencoder.setter
    def optimizer_autoencoder(self, value: Any) -> None:
        self._optimizer_autoencoder = value

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, value: str) -> None:
        self._device = value