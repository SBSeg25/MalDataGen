#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Kayuã Oleques']

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
        self.sync_skip_projections()  # NEW: Sync skip projections before EMA update
        self.update_ema_weights()
        return {"Diffusion_loss": loss_diffusion.item() if loss_diffusion is not None else 0}

    def sync_skip_projections(self):
        """
        Synchronizes dynamically created skip projection layers from the first UNet
        to the second UNet (EMA model). This ensures both models have identical architectures.
        """
        # Copy all skip projections from first model to second model
        for key, projection in self._network._skip_projections.items():
            if key not in self._second_unet_model._skip_projections:
                # Create a new projection in the second model with the same architecture
                self._second_unet_model._skip_projections[key] = nn.Linear(
                    projection.in_features,
                    projection.out_features
                ).to(projection.weight.device)
                # Copy the weights from the first model
                self._second_unet_model._skip_projections[key].load_state_dict(
                    projection.state_dict()
                )

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

        # Convert to float
        embedding_data_expanded = embedding_data_expanded.float()

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

        # Ensure predicted_noise matches random_noise shape exactly
        # Remove any extra channel dimensions if present
        if len(predicted_noise.shape) > len(random_noise.shape):
            predicted_noise = predicted_noise.squeeze(-1)

        # Verify shapes match before computing loss
        if predicted_noise.shape != random_noise.shape:
            raise RuntimeError(
                f"Shape mismatch: predicted_noise {predicted_noise.shape} "
                f"!= random_noise {random_noise.shape}. "
                f"Model output_shape: {self._network._output_shape}, "
                f"Input shape: {embedding_data_expanded.shape}"
            )

        # Compute the loss
        loss_diffusion = self._loss_fn(random_noise, predicted_noise)

        # Backward pass
        loss_diffusion.backward()

        # Update weights
        self._optimizer_diffusion.step()

        return loss_diffusion

    def update_ema_weights(self):
        """
        Updates the weights of the second UNet model using exponential moving average.
        Now safely handles dynamically created layers by only updating matching parameters.
        """
        with torch.no_grad():
            # Get all parameter names from both models
            first_params = dict(self._network.named_parameters())
            second_params = dict(self._second_unet_model.named_parameters())

            # Only update parameters that exist in both models
            for name in first_params.keys():
                if name in second_params:
                    param = first_params[name]
                    ema_param = second_params[name]

                    # Verify shapes match before updating
                    if param.shape == ema_param.shape:
                        ema_param.data.mul_(self._ema).add_(param.data, alpha=1 - self._ema)
                    else:
                        # This shouldn't happen after sync_skip_projections, but log if it does
                        print(f"Warning: Skipping EMA update for {name} due to shape mismatch: "
                              f"{param.shape} vs {ema_param.shape}")

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

        # Use the model's output_shape
        model_output_shape = self._network._output_shape

        # Start with random noise in the embedding space
        synthetic_data = torch.randn(labels.shape[0], model_output_shape, 1,
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
                synthetic_data = self._gdf_util.p_sample(
                    predicted_noise[0] if isinstance(predicted_noise, tuple) else predicted_noise,
                    synthetic_data,
                    array_time,
                    clip_denoised=True
                )

        self._network.fit()

        # Crop to original size
        generated_data = self._crop_tensor_to_original_size(
            synthetic_data.cpu().numpy(),
            self._original_shape
        )

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

        # Apply padding
        pad = (0, 0, 0, padding_needed)
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

    # Properties
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