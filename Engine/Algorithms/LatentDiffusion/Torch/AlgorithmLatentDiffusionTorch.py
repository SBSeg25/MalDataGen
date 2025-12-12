#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'

try:
    import os
    import sys
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
    PyTorch implementation of Latent Diffusion training algorithm.
    Trains diffusion models in latent space using VAE embeddings.
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
                 ema=0.999,
                 margin=0.5,
                 embedding_dimension=64):
        """
        Initialize the latent diffusion algorithm.

        Args:
            first_unet_model: Primary UNet model for training
            second_unet_model: Secondary UNet model for EMA
            encoder_model_image: VAE encoder
            decoder_model_image: VAE decoder
            gdf_util: Gaussian diffusion utility
            optimizer_autoencoder: Optimizer for VAE
            optimizer_diffusion: Optimizer for diffusion model
            time_steps: Number of diffusion timesteps
            ema: Exponential moving average coefficient
            margin: Margin for contrastive loss
            embedding_dimension: Dimension of embeddings
        """
        super().__init__()

        # Store models
        self._network = first_unet_model
        self._second_unet_model = second_unet_model
        self._encoder = encoder_model_image
        self._decoder = decoder_model_image

        # Store utilities
        self._gdf_util = gdf_util

        # Store optimizers
        self._optimizer_autoencoder = optimizer_autoencoder
        self._optimizer_diffusion = optimizer_diffusion

        # Store hyperparameters
        self._time_steps = time_steps
        self._ema = ema
        self._margin = margin
        self._embedding_dimension = embedding_dimension
        self._output_shape = embedding_dimension
        self._original_shape = embedding_dimension

        # Loss function
        self._loss_fn = nn.MSELoss()

    def train_step(self, data):
        """
        Execute one training step.

        Args:
            data: Tuple of (embeddings, labels)

        Returns:
            dict: Dictionary containing loss values
        """
        raw_data, label = data

        # Train the diffusion model
        loss_diffusion = self.train_diffusion_model(raw_data, label)

        # Sync skip projections and update EMA
        self.sync_skip_projections()
        self.update_ema_weights()

        return {"Diffusion_loss": loss_diffusion.item() if loss_diffusion is not None else 0}

    def sync_skip_projections(self):
        """
        Synchronizes dynamically created skip projection layers from the first UNet
        to the second UNet (EMA model).
        """
        if not hasattr(self._network, '_skip_projections'):
            return

        for key, projection in self._network._skip_projections.items():
            if key not in self._second_unet_model._skip_projections:
                # Create a new projection in the second model
                self._second_unet_model._skip_projections[key] = nn.Linear(
                    projection.in_features,
                    projection.out_features
                ).to(projection.weight.device)
                # Copy the weights
                self._second_unet_model._skip_projections[key].load_state_dict(
                    projection.state_dict()
                )

    def train_diffusion_model(self, data, ground_truth):
        """
        Performs a single training step for the diffusion model.

        Args:
            data: Input data embeddings
            ground_truth: Corresponding class labels

        Returns:
            torch.Tensor: The computed loss
        """
        # Set model to training mode
        self._network.train()

        # Labels (conditioning information) and input data embeddings
        embedding_label = ground_truth
        embedding_data_expanded = data

        # Batch size
        batch_size = data.shape[0]

        # Convert to float
        embedding_data_expanded = embedding_data_expanded.float()

        # Sample random time steps
        random_time_steps = torch.randint(
            0, self._time_steps, (batch_size,),
            dtype=torch.long, device=data.device
        )

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
        predicted_noise = self._network(
            embedding_with_noise,
            random_time_steps,
            embedding_label
        )

        # Handle shape mismatches
        if len(predicted_noise.shape) > len(random_noise.shape):
            predicted_noise = predicted_noise.squeeze(-1)

        # Verify shapes match
        if predicted_noise.shape != random_noise.shape:
            raise RuntimeError(
                f"Shape mismatch: predicted_noise {predicted_noise.shape} "
                f"!= random_noise {random_noise.shape}"
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

    def generate_data(self, labels, batch_size):
        """
        Generates synthetic data by reversing the diffusion process.

        Args:
            labels: Class labels for conditioning
            batch_size: Number of samples to generate

        Returns:
            numpy.ndarray: Generated synthetic data
        """
        device = next(self._network.parameters()).device

        # Use the model's output_shape
        model_output_shape = self._network._output_shape

        # Start with random noise
        synthetic_data = torch.randn(
            labels.shape[0], model_output_shape, 1,
            dtype=torch.float32, device=device
        )

        # Reshape labels
        labels_vector = labels.unsqueeze(-1) if len(labels.shape) == 2 else labels

        # Set to evaluation mode
        self._network.eval()

        with torch.no_grad():
            # Reverse the diffusion process
            for time_step in reversed(range(0, self._time_steps)):
                # Create time step tensor
                array_time = torch.full(
                    (labels_vector.shape[0],), time_step,
                    dtype=torch.long, device=device
                )

                # Predict noise
                predicted_noise = self._network(
                    synthetic_data, array_time, labels_vector
                )

                # Apply reverse diffusion step
                synthetic_data = self._gdf_util.p_sample(
                    predicted_noise[0] if isinstance(predicted_noise, tuple) else predicted_noise,
                    synthetic_data,
                    array_time,
                    clip_denoised=True
                )

        # Set back to training mode
        self._network.train()

        # Crop to original size
        generated_data = self._crop_tensor_to_original_size(
            synthetic_data.cpu().numpy(),
            self._original_shape
        )

        return generated_data

    @staticmethod
    def _crop_tensor_to_original_size(tensor: numpy.ndarray, original_size: int) -> numpy.ndarray:
        """Crops tensor to original size."""
        if tensor.ndim != 3:
            raise ValueError(f"Expected 3D tensor, got shape: {tensor.shape}")

        current_size = tensor.shape[1]

        if current_size <= original_size:
            return tensor

        return tensor[:, :original_size, :]

    def get_samples(self, number_samples_per_class):
        """
        Generates synthetic data samples for each class.

        Args:
            number_samples_per_class: Dictionary with class information

        Returns:
            dict: Generated samples for each class
        """
        generated_data = {}
        device = next(self._network.parameters()).device

        for label_class, number_instances in number_samples_per_class["classes"].items():
            # Create one-hot encoded labels
            labels = torch.zeros(
                number_instances,
                number_samples_per_class["number_classes"],
                device=device
            )
            labels[:, label_class] = 1

            # Generate samples
            generated_samples = self.generate_data(labels, batch_size=64)

            # Round and squeeze
            generated_samples = numpy.rint(numpy.squeeze(generated_samples, axis=-1))

            generated_data[label_class] = generated_samples

        return generated_data

    def save_model(self, directory, file_name):
        """Save model weights."""
        if not os.path.exists(directory):
            os.makedirs(directory)

        first_unet_path = os.path.join(directory, f"{file_name}_first_unet.pth")
        second_unet_path = os.path.join(directory, f"{file_name}_second_unet.pth")

        torch.save(self._network.state_dict(), first_unet_path)
        torch.save(self._second_unet_model.state_dict(), second_unet_path)

        print(f"Models saved to {directory}")