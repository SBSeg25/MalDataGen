#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/14'
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
    from torch.utils.data import DataLoader, TensorDataset

except ImportError as error:
    print(error)
    sys.exit(-1)


class AlgorithmLatentDiffusionTorch:
    """
    PyTorch implementation of a diffusion process using UNet architectures for generating synthetic data.

    IMPORTANT: This class does NOT inherit from nn.Module to prevent automatic parameter
    registration which was causing parameter count mismatches between the two UNet models.

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
        Initializes the PyTorch Diffusion Model WITHOUT nn.Module inheritance.

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
        # NOTE: No super().__init__() call - we don't inherit from nn.Module

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

        # Initialize loss tracker
        self._total_loss_tracker = {'sum': 0.0, 'count': 0}

        # Verify models at initialization
        first_param_count = len(list(self._network.parameters()))
        second_param_count = len(list(self._second_unet_model.parameters()))

        if first_param_count != second_param_count:
            raise ValueError(
                f"Models have different parameter counts at initialization: "
                f"{first_param_count} vs {second_param_count}"
            )

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

        loss_value = loss_diffusion.item() if loss_diffusion is not None else 0

        # Update loss tracker
        self._total_loss_tracker['sum'] += loss_value
        self._total_loss_tracker['count'] += 1

        return {
            "Diffusion_loss": loss_value,
            "loss": loss_value
        }

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
            first_params = list(self._network.parameters())
            second_params = list(self._second_unet_model.parameters())

            if len(first_params) != len(second_params):
                raise RuntimeError(
                    f"Parameter count mismatch: {len(first_params)} vs {len(second_params)}. "
                    f"The two UNet models must have identical architectures."
                )

            for idx, (param, ema_param) in enumerate(zip(first_params, second_params)):
                if param.shape != ema_param.shape:
                    raise RuntimeError(
                        f"Parameter shape mismatch at index {idx}: {param.shape} vs {ema_param.shape}"
                    )

                # Perform EMA update
                ema_param.data.mul_(self._ema).add_(param.data, alpha=1 - self._ema)

    def fit(self, x=None, y=None, batch_size=32, epochs=1, verbose=1,
            callbacks=None, validation_data=None, shuffle=True,
            initial_epoch=0, steps_per_epoch=None, validation_steps=None,
            validation_freq=1, optimizer=None, learning_rate=0.001, **kwargs):
        """
        Train the model with a simplified progress bar.

        Args:
            x: Input data (numpy array or torch tensor).
            y: Target data (labels, numpy array or torch tensor).
            batch_size: Number of samples per gradient update.
            epochs: Number of epochs to train.
            verbose: 0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks: List of callbacks to apply during training.
            validation_data: Data for validation (tuple of (x_val, y_val)).
            shuffle: Whether to shuffle data before each epoch.
            initial_epoch: Epoch at which to start training.
            steps_per_epoch: Number of steps per epoch.
            validation_steps: Number of validation steps.
            validation_freq: Validation frequency.
            optimizer: PyTorch optimizer (if provided, replaces current optimizer).
            learning_rate: Learning rate for optimizer (only used if optimizer is provided).

        Returns:
            A History object with training metrics.
        """

        # Set optimizer if provided
        if optimizer is not None:
            self._optimizer_diffusion = optimizer

        # Prepare the dataset
        if isinstance(x, DataLoader):
            train_dataset = x
        else:
            # Convert to tensors if needed
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x).float()
            if y is None:
                y = x
            if not isinstance(y, torch.Tensor):
                y = torch.from_numpy(y).float()

            dataset = TensorDataset(x, y)
            train_dataset = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle
            )

        # Calculate steps per epoch if not provided
        if steps_per_epoch is None:
            steps_per_epoch = len(train_dataset)

        # Prepare validation data
        val_dataset = None
        if validation_data is not None:
            x_val, y_val = validation_data
            if not isinstance(x_val, torch.Tensor):
                x_val = torch.from_numpy(x_val).float()
            if not isinstance(y_val, torch.Tensor):
                y_val = torch.from_numpy(y_val).float()
            val_dataset = DataLoader(
                TensorDataset(x_val, y_val),
                batch_size=batch_size,
                shuffle=False
            )

        # History to store metrics
        history = {'loss': []}

        # Training loop
        for epoch in range(initial_epoch, epochs):
            # Reset loss tracker
            self._total_loss_tracker = {'sum': 0.0, 'count': 0}

            if verbose == 1:
                print(f'\nEpoch {epoch + 1}/{epochs}')

            # Progress tracking
            step = 0
            for batch_data in train_dataset:
                step += 1

                # Perform training step
                metrics = self.train_step(batch_data)
                current_loss = float(metrics['loss'])

                # Simple progress bar
                if verbose == 1:
                    progress = int(50 * step / steps_per_epoch)
                    bar = '=' * progress + '>' + '.' * (50 - progress - 1)
                    print(f'\r[{bar}] {step}/{steps_per_epoch} - loss: {current_loss:.4f}',
                          end='', flush=True)

                if step >= steps_per_epoch:
                    break

            # Store epoch loss
            epoch_loss = (self._total_loss_tracker['sum'] /
                          max(self._total_loss_tracker['count'], 1))
            history['loss'].append(epoch_loss)

            if verbose == 1:
                print(f' - loss: {epoch_loss:.4f}')
            elif verbose == 2:
                print(f'Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.4f}')

            # Validation
            if val_dataset is not None and (epoch + 1) % validation_freq == 0:
                val_loss = self._evaluate_validation(val_dataset, validation_steps)
                if 'val_loss' not in history:
                    history['val_loss'] = []
                history['val_loss'].append(val_loss)

                if verbose >= 1:
                    print(f' - val_loss: {val_loss:.4f}')

            # Callbacks
            if callbacks is not None:
                for callback in callbacks:
                    callback.on_epoch_end(epoch, {'loss': epoch_loss})

        # Return history object
        class History:
            def __init__(self, history_dict):
                self.history = history_dict

        return History(history)

    def _evaluate_validation(self, validation_data, validation_steps=None):
        """
        Evaluate the model on validation data.

        Args:
            validation_data: Validation DataLoader.
            validation_steps: Number of validation steps.

        Returns:
            Average validation loss.
        """
        self._network.eval()
        val_losses = []
        step = 0

        with torch.no_grad():
            for batch_data in validation_data:
                batch_x, batch_y = batch_data

                # Move to device
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)

                # Sample random time steps for validation
                batch_size = batch_x.shape[0]
                random_time_steps = torch.randint(
                    0, self._time_steps, (batch_size,),
                    device=self._device, dtype=torch.long
                )

                # Sample random noise
                random_noise = torch.randn_like(batch_x)

                # Apply forward diffusion
                embedding_with_noise = self._gdf_util.q_sample(
                    batch_x,
                    random_time_steps,
                    random_noise
                )

                # Predict noise
                predicted_noise = self._network(
                    embedding_with_noise,
                    random_time_steps,
                    batch_y
                )

                # Compute loss
                loss = F.mse_loss(predicted_noise, random_noise)
                val_losses.append(float(loss))

                step += 1
                if validation_steps is not None and step >= validation_steps:
                    break

        self._network.train()
        return numpy.mean(val_losses) if val_losses else 0.0

    def generate_data(self, labels, batch_size):
        self._network.eval()

        with torch.no_grad():
            # Start with random noise
            embedding_diffusion = torch.randn(
                labels.shape[0], self._embedding_dimension, 1,
                device=self._device, dtype=torch.float32
            )

            labels_vector = labels

            # Reverse diffusion process
            for time_step in reversed(range(0, self._time_steps)):
                array_time = torch.full((labels_vector.shape[0],), time_step,
                                        device=self._device, dtype=torch.long)

                predicted_noise = self._network(embedding_diffusion, array_time, labels_vector)
                # Apply reverse diffusion step
                embedding_diffusion = self._gdf_util.p_sample(
                    predicted_noise,
                    embedding_diffusion,
                    array_time,
                    clip_denoised=True
                )

            # Decode embeddings to data
            # labels_vector should remain 2D for decoder as well
            generated_data = self._decoder_model_data(embedding_diffusion, labels_vector)

        return generated_data.cpu().numpy()

    def get_samples(self, number_samples_per_class):
        """
        Generates synthetic data samples for each class.

        Args:
            number_samples_per_class (dict): Dictionary with class info.
                Expected format: {
                    "classes": {class_id: num_samples, ...},
                    "number_classes": total_classes
                }

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