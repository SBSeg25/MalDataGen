#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com'
__version__ = '{1}.{0}.{2}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/16'
__credits__ = ['Kayuã Oleques']

# MIT License
#
# Copyright (c) 2025 Synthetic Ocean AI

try:
    import os
    import sys
    import json
    import numpy

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    from abc import ABC
    from typing import Any, Callable, Optional, Tuple, Union

except ImportError as error:
    print(error)
    sys.exit(-1)


def to_categorical(labels, num_classes):
    """Convert class labels to one-hot encoded format."""
    batch_size = len(labels)
    categorical = numpy.zeros((batch_size, num_classes))
    categorical[numpy.arange(batch_size), labels] = 1
    return categorical


class WassersteinGPAlgorithmTorch(nn.Module):
    """
    A Wasserstein GAN with Gradient Penalty (WGAN-GP) model with adaptive input handling.

    This class represents a WGAN-GP consisting of a generator and discriminator model.
    It implements the Wasserstein loss with gradient penalty to improve training stability.

    This implementation automatically adapts to any input shape: (x), (x, y), (x, y, z), etc.

    Reference:
        Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein GAN.
        In Proceedings of the 34th International Conference on Machine Learning (ICML 2017) (Vol. 70, pp. 214-223).
        http://proceedings.mlr.press/v70/arjovsky17a.html

    Example:
        >>> # Example with 1D data
        >>> wgan_gp_1d = WassersteinGPAlgorithmTorch(
        ...     generator_model=generator_1d,
        ...     discriminator_model=discriminator_1d,
        ...     latent_dimension=64,
        ...     input_shape=(100,)
        ... )
        >>> wgan_gp_1d.fit(data_1d, epochs=50)

        >>> # Example with 2D data (images)
        >>> wgan_gp_2d = WassersteinGPAlgorithmTorch(
        ...     generator_model=generator_2d,
        ...     discriminator_model=discriminator_2d,
        ...     latent_dimension=128,
        ...     input_shape=(28, 28)
        ... )
        >>> wgan_gp_2d.fit(data_2d, labels_2d, epochs=100)
    """

    def __init__(self,
                 generator_model,
                 discriminator_model,
                 latent_dimension,
                 generator_loss_fn=None,
                 discriminator_loss_fn=None,
                 file_name_discriminator=None,
                 file_name_generator=None,
                 models_saved_path=None,
                 latent_mean_distribution=0.0,
                 latent_standard_deviation=1.0,
                 smoothing_rate=0.0,
                 gradient_penalty_weight=10.0,
                 discriminator_steps=5,
                 input_shape: Optional[Tuple] = None,
                 auto_adapt_shape: bool = True,
                 *args,
                 **kwargs):
        """
        Initialize the WGAN-GP model.

        Args:
            generator_model: The generator model
            discriminator_model: The discriminator model
            latent_dimension: The dimension of the latent space
            generator_loss_fn: Loss function for generator
            discriminator_loss_fn: Loss function for discriminator
            file_name_discriminator: Filename for saving discriminator
            file_name_generator: Filename for saving generator
            models_saved_path: Path where models are saved
            latent_mean_distribution: Mean of latent distribution
            latent_standard_deviation: Std dev of latent distribution
            smoothing_rate: Rate for label smoothing
            gradient_penalty_weight: Weight for gradient penalty
            discriminator_steps: Number of discriminator updates per generator update
            input_shape: Expected input shape (without batch dimension)
            auto_adapt_shape: Whether to automatically adapt to input data shape
        """
        super().__init__(*args, **kwargs)

        # Initialize instance variables
        self._generator_optimizer = None
        self._discriminator_optimizer = None
        self._generator = generator_model
        self._discriminator = discriminator_model
        self._latent_dimension = latent_dimension
        self._discriminator_loss_fn = discriminator_loss_fn
        self._generator_loss_fn = generator_loss_fn
        self._gradient_penalty_weight = gradient_penalty_weight
        self._smooth_rate = smoothing_rate
        self._latent_mean_distribution = latent_mean_distribution
        self._latent_standard_deviation = latent_standard_deviation
        self._file_name_discriminator = file_name_discriminator
        self._file_name_generator = file_name_generator
        self._models_saved_path = models_saved_path
        self._discriminator_steps = discriminator_steps
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Shape adaptation
        self._input_shape = input_shape
        self._auto_adapt_shape = auto_adapt_shape
        self._inferred_shape = None

    @staticmethod
    def _infer_data_shape(data):
        """
        Infer the shape of input data, excluding the batch dimension.

        Args:
            data: Input data (tensor, array, or tuple/list).

        Returns:
            tuple: Shape of the data excluding batch dimension.
        """
        if isinstance(data, (tuple, list)):
            # If data is a tuple/list, infer from first element
            data = data[0]

        if torch.is_tensor(data):
            shape = tuple(data.shape[1:])
        elif isinstance(data, numpy.ndarray):
            shape = data.shape[1:] if len(data.shape) > 1 else data.shape
        else:
            # Try to convert to tensor and get shape
            try:
                tensor_data = torch.tensor(data)
                shape = tuple(tensor_data.shape[1:])
            except:
                raise ValueError(f"Cannot infer shape from data of type {type(data)}")

        return shape

    def _validate_and_adapt_shape(self, data):
        """
        Validate input data shape and adapt if necessary.

        Args:
            data: Input data.

        Returns:
            bool: True if shape is valid or successfully adapted.
        """
        current_shape = self._infer_data_shape(data)

        if self._inferred_shape is None:
            self._inferred_shape = current_shape
            if self._input_shape is not None and self._input_shape != current_shape:
                print(f"Warning: Specified input_shape {self._input_shape} differs from inferred shape {current_shape}")
                if self._auto_adapt_shape:
                    print(f"Auto-adapting to shape: {current_shape}")
                    self._input_shape = current_shape
            elif self._input_shape is None:
                self._input_shape = current_shape
                print(f"Inferred input shape: {current_shape}")
        else:
            if current_shape != self._inferred_shape:
                if self._auto_adapt_shape:
                    print(f"Warning: Input shape changed from {self._inferred_shape} to {current_shape}")
                    self._inferred_shape = current_shape
                else:
                    raise ValueError(
                        f"Input shape mismatch: expected {self._inferred_shape}, got {current_shape}. "
                        f"Set auto_adapt_shape=True to allow dynamic shape changes."
                    )

        return True

    @staticmethod
    def _prepare_batch(batch):
        """
        Prepare batch data, handling different input formats.

        Args:
            batch: Input batch (can be single tensor, tuple, or list).

        Returns:
            tuple: (batch_x, batch_labels) where batch_x is the feature data.
        """
        if isinstance(batch, (tuple, list)):
            if len(batch) == 1:
                # Single input, no labels
                batch_x = batch[0]
                batch_labels = None
            elif len(batch) == 2:
                # Input and labels provided
                batch_x, batch_labels = batch
            else:
                # Multiple inputs, use first as input, second as labels
                batch_x = batch[0]
                batch_labels = batch[1]
        else:
            # Single tensor, no labels
            batch_x = batch
            batch_labels = None

        return batch_x, batch_labels

    def compile(self, optimizer_generator, optimizer_discriminator,
                loss_generator=None, loss_discriminator=None, *args, **kwargs):
        """
        Compile the WGAN-GP with custom optimizers and loss functions.

        Args:
            optimizer_generator: The optimizer for the generator
            optimizer_discriminator: The optimizer for the discriminator
            loss_generator: The loss function for the generator
            loss_discriminator: The loss function for the discriminator
        """
        import torch.optim as optim

        # Convert Keras optimizers to PyTorch optimizers if needed
        if hasattr(optimizer_generator, 'learning_rate'):
            # It's a Keras optimizer, convert to PyTorch
            lr = float(optimizer_generator.learning_rate.numpy() if hasattr(optimizer_generator.learning_rate,
                                                                            'numpy') else optimizer_generator.learning_rate)
            beta_1 = getattr(optimizer_generator, 'beta_1', 0.9)
            beta_2 = getattr(optimizer_generator, 'beta_2', 0.999)
            self._generator_optimizer = optim.Adam(
                self._generator.parameters(),
                lr=lr,
                betas=(beta_1, beta_2)
            )
        else:
            self._generator_optimizer = optimizer_generator

        if hasattr(optimizer_discriminator, 'learning_rate'):
            # It's a Keras optimizer, convert to PyTorch
            lr = float(optimizer_discriminator.learning_rate.numpy() if hasattr(optimizer_discriminator.learning_rate,
                                                                                'numpy') else optimizer_discriminator.learning_rate)
            beta_1 = getattr(optimizer_discriminator, 'beta_1', 0.9)
            beta_2 = getattr(optimizer_discriminator, 'beta_2', 0.999)
            self._discriminator_optimizer = optim.Adam(
                self._discriminator.parameters(),
                lr=lr,
                betas=(beta_1, beta_2)
            )
        else:
            self._discriminator_optimizer = optimizer_discriminator

        # Wrap loss functions
        if loss_generator is not None:
            self._generator_loss_fn = self._wrap_loss_function(loss_generator, is_discriminator=False)
        if loss_discriminator is not None:
            self._discriminator_loss_fn = self._wrap_loss_function(loss_discriminator, is_discriminator=True)

        # Set default losses if not provided
        if self._generator_loss_fn is None:
            self._generator_loss_fn = lambda fake_img: -torch.mean(fake_img)
        if self._discriminator_loss_fn is None:
            self._discriminator_loss_fn = lambda real_img, fake_img: torch.mean(fake_img) - torch.mean(real_img)

    def _wrap_loss_function(self, loss_fn, is_discriminator=True):
        """
        Wraps a loss function to ensure compatibility between TensorFlow and PyTorch.

        Args:
            loss_fn: Original loss function
            is_discriminator: Whether this is for discriminator (True) or generator (False)

        Returns:
            Wrapped loss function that works with PyTorch tensors
        """

        def wrapped_loss(*args, **kwargs):
            if is_discriminator:
                real_img = kwargs.get('real_img', args[0] if len(args) > 0 else None)
                fake_img = kwargs.get('fake_img', args[1] if len(args) > 1 else None)

                if isinstance(real_img, torch.Tensor) and isinstance(fake_img, torch.Tensor):
                    return torch.mean(fake_img) - torch.mean(real_img)
                else:
                    return loss_fn(*args, **kwargs)
            else:
                fake_img = args[0] if len(args) > 0 else kwargs.get('fake_img')
                if isinstance(fake_img, torch.Tensor):
                    return -torch.mean(fake_img)
                else:
                    return loss_fn(*args, **kwargs)

        return wrapped_loss

    def gradient_penalty(self, batch_size, real_feature, real_label, synthetic_feature):
        """
        Compute the gradient penalty for the Wasserstein GAN with improved numerical stability.

        Args:
            batch_size (int): The batch size of the input data
            real_feature (torch.Tensor): Real data features
            real_label (torch.Tensor): Real data labels
            synthetic_feature (torch.Tensor): Synthetic (generated) data features

        Returns:
            torch.Tensor: Computed gradient penalty value
        """
        # Generate random epsilon for interpolation
        epsilon = torch.rand(batch_size, 1, device=self._device)

        # Interpolate between real and synthetic features
        interpolated_feature = epsilon * real_feature + (1 - epsilon) * synthetic_feature
        interpolated_feature.requires_grad_(True)

        # Get discriminator's output for the interpolated features
        labels_predicted = self.discriminator([interpolated_feature, real_label])

        # Calculate gradients with proper handling
        gradients = torch.autograd.grad(
            outputs=labels_predicted,
            inputs=interpolated_feature,
            grad_outputs=torch.ones_like(labels_predicted),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Compute gradient norm with numerical stability
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-8)

        # Calculate gradient penalty
        gradient_penalty_final = torch.mean((gradients_norm - 1.0) ** 2)

        return gradient_penalty_final

    @staticmethod
    def calculate_samples_per_class(y_labels):
        """
        Calculate the distribution of samples per class from labels.

        Args:
            y_labels (array-like): Labels array

        Returns:
            dict: Dictionary with 'classes' and 'number_classes' keys
        """
        # Convert to numpy if needed
        if isinstance(y_labels, torch.Tensor):
            y_labels = y_labels.cpu().numpy()

        # Handle one-hot encoded labels
        if len(y_labels.shape) == 2 and y_labels.shape[1] > 1:
            y_labels = numpy.argmax(y_labels, axis=1)

        # Count samples per class
        unique, counts = numpy.unique(y_labels, return_counts=True)

        return {
            "classes": dict(zip(unique.tolist(), counts.tolist())),
            "number_classes": len(unique)
        }

    def fit(self, x=None, y=None, batch_size=32, epochs=1, verbose=1,
            callbacks=None, validation_data=None, shuffle=True,
            initial_epoch=0, steps_per_epoch=None, validation_steps=None,
            validation_freq=1, optimizer=None, learning_rate=0.001, **kwargs):
        """
        Train the model with automatic shape adaptation.

        Args:
            x: Input data (any shape) or torch.utils.data.Dataset
            y: Target labels (if None, generates without labels)
            batch_size: Number of samples per gradient update
            epochs: Number of epochs to train
            verbose: 0 = silent, 1 = progress bar, 2 = one line per epoch
            callbacks: List of callbacks to apply during training
            validation_data: Data for validation (tuple of (x_val, y_val))
            shuffle: Whether to shuffle data before each epoch
            initial_epoch: Epoch at which to start training
            steps_per_epoch: Number of steps per epoch
            validation_steps: Number of validation steps
            validation_freq: Validation frequency
            optimizer: Optimizer (dict with 'generator' and 'discriminator' keys)
            learning_rate: Learning rate for optimizers (if optimizer is None)

        Returns:
            History object containing training loss history
        """
        import torch.optim as optim

        # Set optimizers if provided
        if optimizer is not None:
            if isinstance(optimizer, dict):
                self._generator_optimizer = optimizer.get('generator')
                self._discriminator_optimizer = optimizer.get('discriminator')
            else:
                self._generator_optimizer = optimizer
                self._discriminator_optimizer = optimizer
        elif self._generator_optimizer is None or self._discriminator_optimizer is None:
            # Create default optimizers
            self._generator_optimizer = optim.Adam(
                self._generator.parameters(),
                lr=learning_rate
            )
            self._discriminator_optimizer = optim.Adam(
                self._discriminator.parameters(),
                lr=learning_rate
            )

        # Prepare the training dataset
        if isinstance(x, DataLoader):
            train_loader = x
            # Try to infer shape from dataloader
            for batch in train_loader:
                self._validate_and_adapt_shape(batch)
                break
        else:
            # Validate and adapt shape
            self._validate_and_adapt_shape(x)

            # Handle different input formats
            if isinstance(x, tuple):
                if len(x) == 2:
                    x_data, labels = x
                else:
                    x_data = x[0]
                    labels = None
            else:
                x_data = x
                labels = y

            # Convert to tensors
            if isinstance(x_data, numpy.ndarray):
                x_tensor = torch.from_numpy(x_data).float()
            elif not isinstance(x_data, torch.Tensor):
                x_tensor = torch.tensor(x_data, dtype=torch.float32)
            else:
                x_tensor = x_data

            if labels is not None:
                if isinstance(labels, numpy.ndarray):
                    y_tensor = torch.from_numpy(labels).long()
                elif not isinstance(labels, torch.Tensor):
                    y_tensor = torch.tensor(labels, dtype=torch.long)
                else:
                    y_tensor = labels
            else:
                # Create dummy labels
                y_tensor = torch.zeros(x_tensor.shape[0], dtype=torch.long)

            dataset = TensorDataset(x_tensor, y_tensor)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        # Calculate steps per epoch
        if steps_per_epoch is None:
            steps_per_epoch = len(train_loader)

        # Prepare validation dataset
        val_loader = None
        if validation_data is not None:
            if isinstance(validation_data, DataLoader):
                val_loader = validation_data
            else:
                val_x, val_y = validation_data
                if isinstance(val_x, numpy.ndarray):
                    val_x = torch.from_numpy(val_x).float()
                if isinstance(val_y, numpy.ndarray):
                    val_y = torch.from_numpy(val_y).long()
                val_dataset = TensorDataset(val_x, val_y)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Move models to device
        self._generator.to(self._device)
        self._discriminator.to(self._device)

        # History to store metrics
        history = {'loss': [], 'd_loss': [], 'g_loss': []}

        # Training loop
        for epoch in range(initial_epoch, epochs):
            epoch_d_losses = []
            epoch_g_losses = []

            if verbose == 1:
                print(f'\nEpoch {epoch + 1}/{epochs}')

            step = 0
            for batch_data in train_loader:
                step += 1

                # Perform training step
                losses = self.train_step(batch_data)
                current_d_loss = losses['d_loss']
                current_g_loss = losses['g_loss']

                epoch_d_losses.append(current_d_loss)
                epoch_g_losses.append(current_g_loss)

                # Progress bar
                if verbose == 1:
                    progress = int(50 * step / steps_per_epoch)
                    bar = '=' * progress + '>' + '.' * (50 - progress - 1)
                    print(
                        f'\r[{bar}] {step}/{steps_per_epoch} - d_loss: {current_d_loss:.4f} - g_loss: {current_g_loss:.4f}',
                        end='', flush=True)

                if step >= steps_per_epoch:
                    break

            # Store epoch losses
            epoch_loss = numpy.mean(epoch_d_losses) + numpy.mean(epoch_g_losses)
            epoch_d_loss = numpy.mean(epoch_d_losses) if epoch_d_losses else 0.0
            epoch_g_loss = numpy.mean(epoch_g_losses) if epoch_g_losses else 0.0

            history['loss'].append(epoch_loss)
            history['d_loss'].append(epoch_d_loss)
            history['g_loss'].append(epoch_g_loss)

            if verbose == 1:
                print(f' - d_loss: {epoch_d_loss:.4f} - g_loss: {epoch_g_loss:.4f} - total: {epoch_loss:.4f}')
            elif verbose == 2:
                print(f'Epoch {epoch + 1}/{epochs} - d_loss: {epoch_d_loss:.4f} - g_loss: {epoch_g_loss:.4f}')

            # Validation
            if val_loader is not None and (epoch + 1) % validation_freq == 0:
                val_loss = self._evaluate_validation(val_loader, validation_steps)
                if 'val_loss' not in history:
                    history['val_loss'] = []
                history['val_loss'].append(val_loss)

                if verbose >= 1:
                    print(f' - val_loss: {val_loss:.4f}')

            # Callbacks
            if callbacks is not None:
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, {
                            'loss': epoch_loss,
                            'd_loss': epoch_d_loss,
                            'g_loss': epoch_g_loss
                        })

        # Return history object
        class History:
            def __init__(self, history_dict):
                self.history = history_dict

        return History(history)

    def _evaluate_validation(self, val_loader, validation_steps=None):
        """
        Evaluate the model on validation data with automatic shape handling.

        Args:
            val_loader: Validation DataLoader
            validation_steps: Number of validation steps

        Returns:
            Average validation loss
        """
        val_d_losses = []
        val_g_losses = []
        step = 0

        self._generator.eval()
        self._discriminator.eval()

        with torch.no_grad():
            for batch_data in val_loader:
                real_feature, real_samples_label = self._prepare_batch(batch_data)

                # Move to device
                real_feature = real_feature.to(self._device)
                if real_samples_label is None:
                    real_samples_label = torch.zeros((real_feature.shape[0], 1), device=self._device)
                else:
                    real_samples_label = real_samples_label.to(self._device)

                batch_size = real_feature.shape[0]

                # Handle label dimensions
                if len(real_samples_label.shape) == 2 and real_samples_label.shape[1] > 1:
                    labels_for_model = real_samples_label
                else:
                    if len(real_samples_label.shape) == 1:
                        real_samples_label = real_samples_label.unsqueeze(-1)
                    labels_for_model = real_samples_label

                # Generate synthetic samples
                latent_space = torch.randn(batch_size, self._latent_dimension, device=self._device) * \
                               self._latent_standard_deviation + self._latent_mean_distribution

                synthetic_feature = self._generator([latent_space, labels_for_model])

                # Get discriminator predictions
                label_predicted_real = self._discriminator([real_feature, labels_for_model])
                label_predicted_synthetic = self._discriminator([synthetic_feature, labels_for_model])

                # Calculate losses
                d_loss = self._discriminator_loss_fn(
                    real_img=label_predicted_real,
                    fake_img=label_predicted_synthetic
                )
                g_loss = self._generator_loss_fn(label_predicted_synthetic)

                val_d_losses.append(d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss)
                val_g_losses.append(g_loss.item() if isinstance(g_loss, torch.Tensor) else g_loss)

                step += 1
                if validation_steps is not None and step >= validation_steps:
                    break

        self._generator.train()
        self._discriminator.train()

        avg_val_loss = (numpy.mean(val_d_losses) + numpy.mean(val_g_losses)) / 2
        return avg_val_loss if val_d_losses else 0.0

    def train_step(self, batch):
        """
        Executes one training step for the GAN model with automatic batch handling.

        Args:
            batch: Input batch (can be single tensor, tuple, or list)

        Returns:
            dict: Dictionary containing the discriminator and generator loss
        """
        # Prepare batch
        real_feature, real_samples_label = self._prepare_batch(batch)

        # Move to device
        real_feature = real_feature.to(self._device)
        if real_samples_label is None:
            real_samples_label = torch.zeros((real_feature.shape[0], 1), device=self._device)
        else:
            real_samples_label = real_samples_label.to(self._device)

        batch_size = real_feature.shape[0]

        # Handle label dimensions
        if len(real_samples_label.shape) == 2 and real_samples_label.shape[1] > 1:
            labels_for_model = real_samples_label
        else:
            if len(real_samples_label.shape) == 1:
                real_samples_label = real_samples_label.unsqueeze(-1)
            labels_for_model = real_samples_label

        # === Discriminator Training Loop ===
        for _ in range(self._discriminator_steps):
            latent_space = torch.randn(batch_size, self._latent_dimension, device=self._device) * \
                           self._latent_standard_deviation + self._latent_mean_distribution

            self._discriminator_optimizer.zero_grad()

            with torch.no_grad():
                synthetic_feature = self._generator([latent_space, labels_for_model])

            label_predicted_real = self._discriminator([real_feature, labels_for_model])
            label_predicted_synthetic = self._discriminator([synthetic_feature, labels_for_model])

            discriminator_loss_result = self._discriminator_loss_fn(
                real_img=label_predicted_real, fake_img=label_predicted_synthetic)

            gradient_penalty = self.gradient_penalty(batch_size,
                                                     real_feature,
                                                     labels_for_model,
                                                     synthetic_feature)

            all_discriminator_loss = discriminator_loss_result + gradient_penalty * self._gradient_penalty_weight

            all_discriminator_loss.backward()
            self._discriminator_optimizer.step()

        # === Generator Training Step ===
        latent_space = torch.randn(batch_size, self._latent_dimension, device=self._device) * \
                       self._latent_standard_deviation + self._latent_mean_distribution

        self._generator_optimizer.zero_grad()

        synthetic_feature = self._generator([latent_space, labels_for_model])
        predicted_labels = self._discriminator([synthetic_feature, labels_for_model])

        all_generator_loss = self._generator_loss_fn(predicted_labels)

        all_generator_loss.backward()
        self._generator_optimizer.step()

        return {"d_loss": all_discriminator_loss.item(), "g_loss": all_generator_loss.item()}

    def get_input_shape(self):
        """Get the current input shape."""
        return self._inferred_shape if self._inferred_shape is not None else self._input_shape

    @staticmethod
    def reshape_data(data, target_shape):
        """Reshape data to target shape if needed."""
        if isinstance(data, numpy.ndarray):
            if data.shape[1:] != target_shape:
                batch_size = data.shape[0]
                return data.reshape((batch_size,) + target_shape)
        elif torch.is_tensor(data):
            if tuple(data.shape[1:]) != target_shape:
                batch_size = data.shape[0]
                return data.reshape((batch_size,) + target_shape)
        return data

    def get_samples(self, number_samples_per_class):
        """
        Generates synthetic samples for each specified class using the trained generator.

        Args:
            number_samples_per_class (dict): A dictionary containing:
                - "classes" (dict): Mapping of class labels to number of samples
                - "number_classes" (int): Total number of classes

        Returns:
            dict: Dictionary with class labels as keys and generated samples as values
        """
        if number_samples_per_class is None:
            raise ValueError("number_samples_per_class is required for generating samples")

        generated_data = {}
        self._generator.eval()

        with torch.no_grad():
            for label_class, number_instances in number_samples_per_class["classes"].items():
                label_samples_generated = to_categorical([label_class] * number_instances,
                                                         num_classes=number_samples_per_class["number_classes"])
                label_samples_generated = torch.tensor(label_samples_generated, dtype=torch.float32,
                                                       device=self._device)

                latent_noise = torch.randn(number_instances, self._latent_dimension, device=self._device) * \
                               self._latent_standard_deviation + self._latent_mean_distribution

                generated_samples = self._generator([latent_noise, label_samples_generated])
                generated_data[label_class] = generated_samples.cpu().numpy()

        self._generator.train()
        return generated_data

    def save_model(self, directory, file_name):
        """Save the generator and discriminator models with shape information."""
        if not os.path.exists(directory):
            os.makedirs(directory)

        generator_file_name = os.path.join(directory, f"fold_{file_name}_generator")
        discriminator_file_name = os.path.join(directory, f"fold_{file_name}_discriminator")

        torch.save(self._generator.state_dict(), f"{generator_file_name}.pt")
        self._save_model_architecture(self._generator, f"{generator_file_name}.json")

        torch.save(self._discriminator.state_dict(), f"{discriminator_file_name}.pt")
        self._save_model_architecture(self._discriminator, f"{discriminator_file_name}.json")

        # Save shape information
        shape_info = {
            'input_shape': self._input_shape,
            'inferred_shape': self._inferred_shape,
            'latent_dimension': self._latent_dimension,
            'gradient_penalty_weight': self._gradient_penalty_weight
        }
        shape_file = os.path.join(directory, f"fold_{file_name}_shape_info.json")
        with open(shape_file, 'w') as f:
            json.dump(shape_info, f)

    @staticmethod
    def _save_model_architecture(model, file_path):
        """Save model architecture to a JSON file."""
        architecture = {
            'class': model.__class__.__name__,
            'state_dict_keys': list(model.state_dict().keys())
        }
        with open(file_path, "w") as json_file:
            json.dump(architecture, json_file, indent=2)

    def load_models(self, directory, file_name):
        """Load the generator and discriminator models with shape information."""
        generator_file_name = os.path.join(directory, f"{file_name}_generator.pt")
        discriminator_file_name = os.path.join(directory, f"{file_name}_discriminator.pt")

        self._generator.load_state_dict(torch.load(generator_file_name, map_location=self._device))
        self._discriminator.load_state_dict(torch.load(discriminator_file_name, map_location=self._device))

        # Load shape information if available
        shape_file = os.path.join(directory, f"{file_name}_shape_info.json")
        if os.path.exists(shape_file):
            with open(shape_file, 'r') as f:
                shape_info = json.load(f)
                self._input_shape = tuple(shape_info.get('input_shape', ()))
                self._inferred_shape = tuple(shape_info.get('inferred_shape', ()))
                self._latent_dimension = shape_info.get('latent_dimension', self._latent_dimension)
                self._gradient_penalty_weight = shape_info.get('gradient_penalty_weight', self._gradient_penalty_weight)

    # ========== PROPERTIES ==========

    @property
    def generator(self) -> Any:
        """Get the generator model instance."""
        return self._generator

    @generator.setter
    def generator(self, value: Any) -> None:
        """Set the generator model instance."""
        self._generator = value

    @property
    def discriminator(self) -> Any:
        """Get the discriminator model instance."""
        return self._discriminator

    @discriminator.setter
    def discriminator(self, value: Any) -> None:
        """Set the discriminator model instance."""
        self._discriminator = value

    @property
    def latent_dimension(self) -> int:
        """Get the dimension of the latent space."""
        return self._latent_dimension

    @latent_dimension.setter
    def latent_dimension(self, value: int) -> None:
        """Set the dimension of the latent space."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Latent dimension must be a positive integer")
        self._latent_dimension = value

    @property
    def discriminator_loss_fn(self) -> Callable:
        """Get the discriminator loss function."""
        return self._discriminator_loss_fn

    @discriminator_loss_fn.setter
    def discriminator_loss_fn(self, value: Callable) -> None:
        """Set the discriminator loss function."""
        self._discriminator_loss_fn = value

    @property
    def generator_loss_fn(self) -> Callable:
        """Get the generator loss function."""
        return self._generator_loss_fn

    @generator_loss_fn.setter
    def generator_loss_fn(self, value: Callable) -> None:
        """Set the generator loss function."""
        self._generator_loss_fn = value

    @property
    def gradient_penalty_weight(self) -> float:
        """Get the weight for gradient penalty in WGAN-GP."""
        return self._gradient_penalty_weight

    @gradient_penalty_weight.setter
    def gradient_penalty_weight(self, value: float) -> None:
        """Set the weight for gradient penalty in WGAN-GP."""
        if value < 0:
            raise ValueError("Gradient penalty weight cannot be negative")
        self._gradient_penalty_weight = value

    @property
    def smooth_rate(self) -> float:
        """Get the label smoothing rate."""
        return self._smooth_rate

    @smooth_rate.setter
    def smooth_rate(self, value: float) -> None:
        """Set the label smoothing rate."""
        if not 0 <= value <= 1:
            raise ValueError("Smoothing rate must be between 0 and 1")
        self._smooth_rate = value

    @property
    def latent_mean_distribution(self) -> float:
        """Get the mean of the latent space distribution."""
        return self._latent_mean_distribution

    @latent_mean_distribution.setter
    def latent_mean_distribution(self, value: float) -> None:
        """Set the mean of the latent space distribution."""
        self._latent_mean_distribution = value

    @property
    def latent_standard_deviation(self) -> float:
        """Get the standard deviation of the latent space distribution."""
        return self._latent_standard_deviation

    @latent_standard_deviation.setter
    def latent_standard_deviation(self, value: float) -> None:
        """Set the standard deviation of the latent space distribution."""
        if value <= 0:
            raise ValueError("Standard deviation must be positive")
        self._latent_standard_deviation = value

    @property
    def file_name_discriminator(self) -> str:
        """Get the discriminator model save filename."""
        return self._file_name_discriminator

    @file_name_discriminator.setter
    def file_name_discriminator(self, value: str) -> None:
        """Set the discriminator model save filename."""
        self._file_name_discriminator = value

    @property
    def file_name_generator(self) -> str:
        """Get the generator model save filename."""
        return self._file_name_generator

    @file_name_generator.setter
    def file_name_generator(self, value: str) -> None:
        """Set the generator model save filename."""
        self._file_name_generator = value

    @property
    def models_saved_path(self) -> str:
        """Get the path for saving models."""
        return self._models_saved_path

    @models_saved_path.setter
    def models_saved_path(self, value: str) -> None:
        """Set the path for saving models."""
        self._models_saved_path = value

    @property
    def discriminator_steps(self) -> int:
        """Get the number of discriminator steps per iteration."""
        return self._discriminator_steps

    @discriminator_steps.setter
    def discriminator_steps(self, value: int) -> None:
        """Set the number of discriminator steps per iteration."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Discriminator steps must be a positive integer")
        self._discriminator_steps = value

    @property
    def input_shape(self):
        """Get the current input shape."""
        return self.get_input_shape()

    @property
    def metrics(self):
        """Returns dictionary of metrics tracked during training."""
        return {
            'gradient_penalty_weight': self._gradient_penalty_weight
        }