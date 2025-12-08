#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
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
    import numpy
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from typing import Any

except ImportError as error:
    print(error)
    sys.exit(-1)


class AlgorithmDenoisingDiffusionTorch(nn.Module):
    """
    Implements a diffusion process using UNet architectures for generating synthetic data.
    This model integrates an autoencoder and a diffusion network, enabling both data
    reconstruction and controlled generative modeling through Gaussian diffusion.

    This class supports exponential moving average (EMA) updates for stable training,
    multiple training stages, and customizable hyperparameters to adapt to different tasks.

    Attributes:
        @ema (float):
            Exponential moving average (EMA) decay rate for stabilizing training updates.
        @margin (float):
            Margin parameter used for loss computation or regularization purposes.
        @gdf_util:
            Utility object for Gaussian diffusion functions, handling noise scheduling and diffusion-related operations.
        @time_steps (int):
            Number of time steps used in the diffusion process.
        @train_stage (str):
            Defines the current training stage ('all', 'diffusion', etc.), determining whether only specific components are updated.
        @network (Model):
            Primary UNet model responsible for the diffusion process.
        @second_unet_model (Model):
            Secondary UNet model used for EMA-based weight updates to enhance training stability.
        @embedding_dimension (int):
            Dimensionality of the latent space used for encoding data.
        @encoder_model_data (Model):
            Encoder model responsible for feature extraction from input data.
        @decoder_model_data (Model):
            Decoder model used to reconstruct data from encoded representations.
        @optimizer_diffusion (Optimizer):
            Optimizer used for training the diffusion model.
        @optimizer_autoencoder (Optimizer):
            Optimizer responsible for training the autoencoder components.
        @ensemble_encoder_decoder (Model):
            Combined encoder-decoder model for data reconstruction.

    Raises:
        ValueError:
            Raised in cases where:
            - The number of time steps is non-positive.
            - The EMA decay rate is outside the range (0,1).
            - The embedding dimension is invalid (<=0).

    References:
        - Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising LatentDiffusion Probabilistic Models."
        Advances in Neural Information Processing Systems (NeurIPS).
        Available at: https://arxiv.org/abs/2006.11239

    Example:
        >>> diffusion_model = AlgorithmDenoisingDiffusionTorch(
        ...     first_unet_model=primary_unet,
        ...     second_unet_model=ema_unet,
        ...     encoder_model_image=encoder,
        ...     decoder_model_image=decoder,
        ...     gdf_util=gaussian_diffusion,
        ...     optimizer_autoencoder=torch.optim.Adam(lr=1e-4),
        ...     optimizer_diffusion=torch.optim.Adam(lr=2e-4),
        ...     time_steps=1000,
        ...     ema=0.999,
        ...     margin=0.1,
        ...     train_stage='all'
        ... )
        >>> diffusion_model.set_stage_training('diffusion')
        >>> diffusion_model.train_step(data)
    """

    def __init__(self,
                 output_shape,
                 first_unet_model,
                 second_unet_model,
                 gdf_util,
                 optimizer_autoencoder,
                 optimizer_diffusion,
                 time_steps,
                 ema,
                 margin,
                 train_stage='all'):

        super().__init__()
        """
        Initializes the DiffusionModel with provided sub-models, optimizers, and hyperparameters.

        This constructor sets up the network structure, including the autoencoder, diffusion
        models, and EMA components, ensuring flexibility for different training strategies.

        Args:
            @first_unet_model (Model):
                Primary UNet model for diffusion-based generation.
            @second_unet_model (Model):
                Secondary UNet model for maintaining EMA-based weight updates.
            @encoder_model_image (Model):
                Encoder model used to extract meaningful feature representations.
            @decoder_model_image (Model):
                Decoder model reconstructing data from encoded embeddings.
            @gdf_util:
                Utility object responsible for Gaussian diffusion operations.
            @optimizer_autoencoder (Optimizer):
                Optimizer handling the training of the encoder-decoder network.
            @optimizer_diffusion (Optimizer):
                Optimizer applied to the diffusion process.
            @time_steps (int):
                Number of discrete time steps for the diffusion process.
            @ema (float):
                Exponential moving average decay factor.
            @margin (float):
                Margin value used in loss calculations or regularization.
            @embedding_dimension (int):
                Dimensionality of the embedding space.
            @train_stage (str, optional):
             Current training stage ('all', 'diffusion', etc.), defaulting to 'all'.

        Raises:
            ValueError:
                If time_steps is <= 0.
                If ema is not within the (0,1) range.
                If embedding_dimension is <= 0.
        """

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
        self._loss_fn = nn.MSELoss()

    def set_stage_training(self, training_stage):
        """
        Sets the current training stage.

        Args:
            training_stage (str): New training stage ('all', 'diffusion', etc.).
        """
        self._train_stage = training_stage

    def train_step(self, data, labels):
        """
        Performs a single training step.

        Args:
            data: Input data tensor
            labels: Label tensor

        Returns:
            dict: A dictionary with the computed loss for diffusion.
        """
        loss_diffusion = self.train_diffusion_model(data, labels)
        self.update_ema_weights()
        return {"Diffusion_loss": loss_diffusion.item() if loss_diffusion is not None else 0}

    def train_diffusion_model(self, data, ground_truth):
        """
        Performs a single training step for the diffusion model.

        This method applies the forward diffusion process (adding noise to the data),
        predicts the noise using the model, computes the loss, and updates the model weights.

        Args:
            data (torch.Tensor): Input data embeddings (e.g., image or text embeddings).
            ground_truth (torch.Tensor): Corresponding class labels or conditioning embeddings.

        Returns:
            torch.Tensor: The computed loss for this training step.
        """
        # Labels (conditioning information) and input data embeddings
        embedding_label = ground_truth
        embedding_data_expanded = data

        # Batch size of the current data batch
        batch_size = data.shape[0]

        embedding_data_expanded = self._padding_input_tensor(embedding_data_expanded)
        embedding_data_expanded = embedding_data_expanded.float()

        static_shape = embedding_data_expanded.shape

        if len(static_shape) >= 2 and static_shape[-2] is not None:
            self._output_shape = static_shape[-2]
        else:
            self._output_shape = embedding_data_expanded.shape[-2]

        # Sample random time steps for each sample in the batch
        random_time_steps = torch.randint(0, self._time_steps, (batch_size,),
                                          dtype=torch.long, device=data.device)

        # Zero gradients
        self._optimizer_diffusion.zero_grad()

        # Sample random noise to add to the data
        random_noise = torch.randn_like(embedding_data_expanded)

        # Apply forward diffusion process
        embedding_with_noise = self._gdf_util.q_sample(embedding_data_expanded,
                                                       random_time_steps,
                                                       random_noise)

        # Predict noise using the diffusion model
        predicted_noise = self._network(embedding_with_noise, random_time_steps, embedding_label)

        # Compute the loss
        loss_diffusion = self._loss_fn(random_noise, predicted_noise.squeeze(-1))

        # Backward pass
        loss_diffusion.backward()

        # Update weights
        self._optimizer_diffusion.step()

        return loss_diffusion

    def update_ema_weights(self):
        """
        Updates the weights of the second UNet model using exponential moving average.
        """
        with torch.no_grad():
            for param, ema_param in zip(self._network.parameters(), self._second_unet_model.parameters()):
                ema_param.data.mul_(self._ema).add_(param.data, alpha=1 - self._ema)

    def generate_data(self, labels, batch_size):
        """
        Generates synthetic data by reversing the diffusion process, starting from pure noise
        and iteratively denoising to create data samples conditioned on class labels.

        Args:
            labels (torch.Tensor): Class labels used to condition the generated data.
            batch_size (int): Number of data samples to generate in a single batch.

        Returns:
            numpy.ndarray: Generated synthetic data samples after reversing the diffusion process.
        """
        device = next(self._network.parameters()).device

        # Start with random noise in the embedding space
        synthetic_data = torch.randn(labels.shape[0], self._output_shape, 1,
                                     dtype=torch.float32, device=device)

        # Reshape labels
        labels_vector = labels.unsqueeze(-1) if len(labels.shape) == 2 else labels

        self._network.eval()
        with torch.no_grad():
            # Reverse the diffusion process
            for time_step in reversed(range(0, self._time_steps)):
                # Create time step tensor
                array_time = torch.full((labels_vector.shape[0],), time_step,
                                        dtype=torch.long, device=device)

                # Predict noise
                predicted_noise = self._network(synthetic_data, array_time, labels_vector)

                # Apply reverse diffusion step
                synthetic_data = self._gdf_util.p_sample(predicted_noise[0] if isinstance(predicted_noise, tuple)
                                                         else predicted_noise,
                                                         synthetic_data, array_time,
                                                         clip_denoised=True)

        self._network.train()

        # Crop to original size
        generated_data = self._crop_tensor_to_original_size(synthetic_data.cpu().numpy(),
                                                            self._original_shape)

        return generated_data

    @staticmethod
    def _crop_tensor_to_original_size(tensor: numpy.ndarray, original_size: int) -> numpy.ndarray:
        """
        Crops the input tensor along the second dimension (axis=1) to match the original size.

        Args:
            tensor (np.ndarray): A 3D NumPy array of shape (X, Y, Z)
            original_size (int): The desired size for the second dimension (Y)

        Returns:
            np.ndarray: A cropped 3D tensor with shape (X, original_size, Z)
        """
        if tensor.ndim != 3:
            raise ValueError(f"Expected 3D tensor (X, Y, Z), got shape: {tensor.shape}")

        current_size = tensor.shape[1]

        if current_size <= original_size:
            return tensor

        return tensor[:, :original_size, :]

    def _padding_input_tensor(self, input_tensor):
        """
        Pads the input tensor along the feature dimension to match the expected input shape.

        Args:
            input_tensor (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Padded tensor
        """
        input_tensor = input_tensor.float()

        # Get target dimension from model
        if hasattr(self._network, 'module'):
            target_dimension = self._network.module._output_shape
        else:
            target_dimension = self._network._output_shape

        current_dimension = input_tensor.shape[-2]

        # Calculate padding needed
        padding_needed = max(0, target_dimension - current_dimension)

        if padding_needed == 0:
            return input_tensor

        # Apply padding: (left, right, top, bottom, front, back)
        # For 3D tensor (batch, seq, channels), pad seq dimension
        pad = (0, 0, 0, padding_needed)  # pad only the second-to-last dimension
        padded_tensor = F.pad(input_tensor, pad, mode='constant', value=0)

        return padded_tensor

    def get_samples(self, number_samples_per_class):
        """
        Generates synthetic data samples for each class.

        Args:
            number_samples_per_class (dict): Dictionary with class information

        Returns:
            dict: Generated samples for each class
        """
        generated_data = {}
        device = next(self._network.parameters()).device

        for label_class, number_instances in number_samples_per_class["classes"].items():
            # Create one-hot encoded labels
            labels = torch.zeros(number_instances, number_samples_per_class["number_classes"],
                                 device=device)
            labels[:, label_class] = 1

            # Generate samples
            generated_samples = self.generate_data(labels, batch_size=64)

            # Round and squeeze
            generated_samples = numpy.rint(numpy.squeeze(generated_samples, axis=-1))

            generated_data[label_class] = generated_samples

        return generated_data

    def save_model(self, directory, file_name):
        """
        Save the model weights.

        Args:
            directory (str): Directory where models will be saved
            file_name (str): Base file name for saving models
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        first_unet_path = os.path.join(directory, f"{file_name}_first_unet.pth")
        second_unet_path = os.path.join(directory, f"{file_name}_second_unet.pth")

        torch.save(self._network.state_dict(), first_unet_path)
        torch.save(self._second_unet_model.state_dict(), second_unet_path)

        print(f"Models saved to {directory}")

    @property
    def ema(self) -> Any:
        return self._ema

    @ema.setter
    def ema(self, value: Any) -> None:
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