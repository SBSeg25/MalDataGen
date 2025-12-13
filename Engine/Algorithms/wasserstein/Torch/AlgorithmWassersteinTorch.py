#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/07'
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
    import sys
    import os
    import torch
    import torch.nn as nn
    import numpy as np
    from typing import Optional, Callable, List, Any
    from pathlib import Path

except ImportError as error:
    print(error)
    sys.exit(-1)


class WassersteinAlgorithmTorch:
    """
    Training algorithm wrapper for wasserstein GAN with Gradient Penalty (WGAN-GP).

    This class manages the complete training process for a wasserstein GAN, including:
    - Alternating critic and generator training
    - Gradient penalty computation
    - Loss tracking and logging
    - Model checkpointing
    - Callback management

    The algorithm implements the WGAN-GP framework as described in:
    Gulrajani et al., "Improved Training of wasserstein GANs" (2017)

    Attributes:
        generator_model: The generator network
        discriminator_model: The critic/discriminator network
        latent_dimension: Dimensionality of the latent space
        generator_loss_fn: Loss function for generator (optional, uses wasserstein loss by default)
        discriminator_loss_fn: Loss function for discriminator (optional, uses wasserstein loss by default)
        file_name_discriminator: Filename for saving discriminator checkpoints
        file_name_generator: Filename for saving generator checkpoints
        models_saved_path: Directory path for saving model checkpoints
        latent_mean_distribution: Mean of the latent distribution (default: 0.0)
        latent_standard_deviation: Std dev of the latent distribution (default: 1.0)
        smoothing_rate: Label smoothing rate (not used in WGAN-GP)
        discriminator_steps: Number of critic updates per generator update
        clip_value: Gradient clipping value (for numerical stability)
    """

    def __init__(self,
                 generator_model: nn.Module,
                 discriminator_model: nn.Module,
                 latent_dimension: int,
                 generator_loss_fn: Optional[Callable] = None,
                 discriminator_loss_fn: Optional[Callable] = None,
                 file_name_discriminator: str = "discriminator",
                 file_name_generator: str = "generator",
                 models_saved_path: str = "./models",
                 latent_mean_distribution: float = 0.0,
                 latent_standard_deviation: float = 1.0,
                 smoothing_rate: float = 0.0,
                 discriminator_steps: int = 5,
                 clip_value: float = 0.01):
        """
        Initialize the wasserstein GAN training algorithm.

        Args:
            generator_model: Generator neural network
            discriminator_model: Discriminator/critic neural network
            latent_dimension: Size of the latent space vector
            generator_loss_fn: Custom generator loss function (optional)
            discriminator_loss_fn: Custom discriminator loss function (optional)
            file_name_discriminator: Base name for discriminator checkpoint files
            file_name_generator: Base name for generator checkpoint files
            models_saved_path: Directory to save model checkpoints
            latent_mean_distribution: Mean for sampling latent vectors
            latent_standard_deviation: Std dev for sampling latent vectors
            smoothing_rate: Label smoothing (not used in WGAN-GP)
            discriminator_steps: Critic updates per generator update
            clip_value: Gradient clipping value
        """
        self.generator = generator_model
        self.discriminator = discriminator_model
        self.latent_dimension = latent_dimension

        # Loss functions (use wasserstein loss if not provided)
        self.generator_loss_fn = generator_loss_fn if generator_loss_fn else self._wasserstein_generator_loss
        self.discriminator_loss_fn = discriminator_loss_fn if discriminator_loss_fn else self._wasserstein_discriminator_loss

        # Model saving configuration
        self.file_name_discriminator = file_name_discriminator
        self.file_name_generator = file_name_generator
        self.models_saved_path = Path(models_saved_path)
        self.models_saved_path.mkdir(parents=True, exist_ok=True)

        # Latent space configuration
        self.latent_mean = latent_mean_distribution
        self.latent_std = latent_standard_deviation

        # Training configuration
        self.smoothing_rate = smoothing_rate
        self.discriminator_steps = discriminator_steps
        self.clip_value = clip_value

        # Optimizers (to be set in compile)
        self.generator_optimizer = None
        self.discriminator_optimizer = None

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # Training history
        self.history = {
            'critic_loss': [],
            'generator_loss': [],
            'gradient_penalty': []
        }

    @staticmethod
    def _wasserstein_discriminator_loss(real_validity: torch.Tensor,
                                        fake_validity: torch.Tensor) -> torch.Tensor:
        """
        wasserstein discriminator/critic loss.
        Critic tries to maximize the difference between real and fake scores.

        Args:
            real_validity: Critic scores for real samples
            fake_validity: Critic scores for fake samples

        Returns:
            Critic loss (to be minimized, so we use -(real - fake))
        """
        return fake_validity.mean() - real_validity.mean()

    @staticmethod
    def _wasserstein_generator_loss(fake_validity: torch.Tensor) -> torch.Tensor:
        """
        wasserstein generator loss.
        Generator tries to maximize the critic score for fake samples.

        Args:
            fake_validity: Critic scores for generated samples

        Returns:
            Generator loss (to be minimized, so we use -fake_validity)
        """
        return -fake_validity.mean()

    def compute_gradient_penalty(self,
                                 real_samples: torch.Tensor,
                                 fake_samples: torch.Tensor,
                                 labels: torch.Tensor,
                                 lambda_gp: float = 10.0) -> torch.Tensor:
        """
        Compute gradient penalty for WGAN-GP.

        Args:
            real_samples: Real data samples
            fake_samples: Generated samples
            labels: Class labels (one-hot encoded)
            lambda_gp: Gradient penalty weight

        Returns:
            Gradient penalty value
        """
        batch_size = real_samples.size(0)
        device = real_samples.device

        # Random interpolation weight
        alpha = torch.rand(batch_size, 1, device=device)
        alpha = alpha.expand_as(real_samples)

        # Interpolated samples
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)

        # Critic scores for interpolated samples
        critic_interpolates = self.discriminator(interpolates, labels)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=critic_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(critic_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Flatten gradients
        gradients = gradients.view(batch_size, -1)

        # Compute gradient penalty
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()

        return gradient_penalty

    def compile(self,
                generator_optimizer: torch.optim.Optimizer,
                discriminator_optimizer: torch.optim.Optimizer,
                generator_loss_fn: Optional[Callable] = None,
                discriminator_loss_fn: Optional[Callable] = None):
        """
        Compile the WGAN with optimizers and loss functions.

        Args:
            generator_optimizer: Optimizer for generator
            discriminator_optimizer: Optimizer for discriminator
            generator_loss_fn: Custom generator loss (optional)
            discriminator_loss_fn: Custom discriminator loss (optional)
        """
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        if generator_loss_fn is not None:
            self.generator_loss_fn = generator_loss_fn
        if discriminator_loss_fn is not None:
            self.discriminator_loss_fn = discriminator_loss_fn

    def _sample_latent(self, batch_size: int) -> torch.Tensor:
        """
        Sample random vectors from the latent space.

        Args:
            batch_size: Number of samples to generate

        Returns:
            Latent vectors of shape (batch_size, latent_dimension)
        """
        return torch.randn(batch_size, self.latent_dimension, device=self.device) * self.latent_std + self.latent_mean

    def train_step(self,
                   real_samples: torch.Tensor,
                   labels: torch.Tensor,
                   lambda_gp: float = 10.0) -> dict:
        """
        Perform one training step (multiple critic updates + one generator update).

        Args:
            real_samples: Batch of real data
            labels: Batch of labels (one-hot encoded)
            lambda_gp: Gradient penalty weight

        Returns:
            Dictionary with loss values
        """
        batch_size = real_samples.size(0)

        # Train Critic/Discriminator for multiple steps
        critic_losses = []
        gradient_penalties = []

        for _ in range(self.discriminator_steps):
            self.discriminator_optimizer.zero_grad()

            # Generate fake samples
            z = self._sample_latent(batch_size)
            fake_samples = self.generator(z, labels)

            # Critic scores
            real_validity = self.discriminator(real_samples, labels)
            fake_validity = self.discriminator(fake_samples.detach(), labels)

            # wasserstein loss
            critic_loss = self.discriminator_loss_fn(real_validity, fake_validity)

            # Gradient penalty
            gp = self.compute_gradient_penalty(real_samples, fake_samples, labels, lambda_gp)

            # Total critic loss
            total_critic_loss = critic_loss + gp
            total_critic_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.clip_value)

            self.discriminator_optimizer.step()

            critic_losses.append(critic_loss.item())
            gradient_penalties.append(gp.item())

        # Train Generator
        self.generator_optimizer.zero_grad()

        # Generate fake samples
        z = self._sample_latent(batch_size)
        fake_samples = self.generator(z, labels)

        # Generator loss
        fake_validity = self.discriminator(fake_samples, labels)
        generator_loss = self.generator_loss_fn(fake_validity)

        generator_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.clip_value)

        self.generator_optimizer.step()

        return {
            'critic_loss': np.mean(critic_losses),
            'generator_loss': generator_loss.item(),
            'gradient_penalty': np.mean(gradient_penalties)
        }

    def fit(self, x=None, y=None, batch_size=32, epochs=1, verbose=1,
            callbacks=None, validation_data=None, shuffle=True,
            initial_epoch=0, steps_per_epoch=None, validation_steps=None,
            validation_freq=1, optimizer=None, learning_rate=0.001, lambda_gp=10.0, **kwargs):
        """
        Train the model with a simplified progress bar.

        Args:
            x: Input data (numpy array or torch.utils.data.DataLoader).
            y: Target data (labels, one-hot encoded).
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
            optimizer: Tuple of (generator_optimizer, discriminator_optimizer) or None.
            learning_rate: Learning rate for optimizers (only used if optimizer is None).
            lambda_gp: Gradient penalty weight for WGAN-GP.

        Returns:
            A History object with training metrics.
        """

        # Set optimizers if provided
        if optimizer is not None:
            if isinstance(optimizer, tuple) and len(optimizer) == 2:
                self.generator_optimizer = optimizer[0]
                self.discriminator_optimizer = optimizer[1]
            else:
                raise ValueError("optimizer must be a tuple of (generator_optimizer, discriminator_optimizer)")
        elif self.generator_optimizer is None or self.discriminator_optimizer is None:
            # Create default optimizers if none exist
            self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate,
                                                        betas=(0.5, 0.999))
            self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate,
                                                            betas=(0.5, 0.999))

        # Prepare the dataset
        if isinstance(x, torch.utils.data.DataLoader):
            train_loader = x
        else:
            if y is None:
                y = x

            # Convert to torch tensors if numpy arrays
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            if isinstance(y, np.ndarray):
                y = torch.tensor(y, dtype=torch.float32)

            train_dataset = torch.utils.data.TensorDataset(x, y)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle
            )

        # Calculate steps per epoch if not provided
        if steps_per_epoch is None:
            steps_per_epoch = len(train_loader)

        # History to store metrics
        history = {'loss': [], 'd_loss': [], 'g_loss': []}
        if validation_data is not None:
            history['val_loss'] = []

        # Training loop
        for epoch in range(initial_epoch, epochs):
            # Trackers for epoch metrics
            epoch_d_losses = []
            epoch_g_losses = []
            epoch_total_losses = []

            if verbose == 1:
                print(f'\nEpoch {epoch + 1}/{epochs}')

            # Progress tracking
            step = 0
            for batch_data in train_loader:
                step += 1

                # Unpack batch data
                if isinstance(batch_data, (tuple, list)):
                    batch_x, batch_y = batch_data
                else:
                    batch_x = batch_data
                    batch_y = batch_data

                # Move to device
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Perform training step
                metrics = self.train_step(batch_x, batch_y, lambda_gp)

                current_d_loss = float(metrics['critic_loss'])
                current_g_loss = float(metrics['generator_loss'])
                current_loss = current_d_loss + current_g_loss

                # Track losses for this epoch
                epoch_d_losses.append(current_d_loss)
                epoch_g_losses.append(current_g_loss)
                epoch_total_losses.append(current_loss)

                # Simple progress bar
                if verbose == 1:
                    progress = int(50 * step / steps_per_epoch)
                    bar = '=' * progress + '>' + '.' * (50 - progress - 1)
                    print(
                        f'\r[{bar}] {step}/{steps_per_epoch} - d_loss: {current_d_loss:.4f} - g_loss: {current_g_loss:.4f}',
                        end='', flush=True)

                if step >= steps_per_epoch:
                    break

            # Store epoch losses
            epoch_loss = np.mean(epoch_total_losses) if epoch_total_losses else 0.0
            epoch_d_loss = np.mean(epoch_d_losses) if epoch_d_losses else 0.0
            epoch_g_loss = np.mean(epoch_g_losses) if epoch_g_losses else 0.0

            history['loss'].append(epoch_loss)
            history['d_loss'].append(epoch_d_loss)
            history['g_loss'].append(epoch_g_loss)

            if verbose == 1:
                print(f' - d_loss: {epoch_d_loss:.4f} - g_loss: {epoch_g_loss:.4f} - total: {epoch_loss:.4f}')
            elif verbose == 2:
                print(f'Epoch {epoch + 1}/{epochs} - d_loss: {epoch_d_loss:.4f} - g_loss: {epoch_g_loss:.4f}')

            # Validation
            if validation_data is not None and (epoch + 1) % validation_freq == 0:
                val_loss = self._evaluate_validation(validation_data, validation_steps, lambda_gp)
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

    def _evaluate_validation(self, validation_data, validation_steps=None, lambda_gp=10.0):
        """
        Evaluate the model on validation data.

        Args:
            validation_data: Validation dataset (tuple of (x_val, y_val) or DataLoader).
            validation_steps: Number of validation steps.
            lambda_gp: Gradient penalty weight.

        Returns:
            Average validation loss.
        """
        self.generator.eval()
        self.discriminator.eval()

        val_losses = []
        step = 0

        # Handle different validation data formats
        if isinstance(validation_data, torch.utils.data.DataLoader):
            val_loader = validation_data
        elif isinstance(validation_data, tuple) and len(validation_data) == 2:
            val_x, val_y = validation_data

            # Convert to torch tensors if numpy arrays
            if isinstance(val_x, np.ndarray):
                val_x = torch.tensor(val_x, dtype=torch.float32)
            if isinstance(val_y, np.ndarray):
                val_y = torch.tensor(val_y, dtype=torch.float32)

            val_dataset = torch.utils.data.TensorDataset(val_x, val_y)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
        else:
            raise ValueError("validation_data must be a DataLoader or tuple of (x_val, y_val)")

        with torch.no_grad():
            for batch_data in val_loader:
                # Unpack batch
                if isinstance(batch_data, (tuple, list)):
                    batch_x, batch_y = batch_data
                else:
                    batch_x = batch_data
                    batch_y = batch_data

                # Move to device
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Generate fake samples
                z = self._sample_latent(batch_x.size(0))
                fake_samples = self.generator(z, batch_y)

                # Compute discriminator scores
                real_validity = self.discriminator(batch_x, batch_y)
                fake_validity = self.discriminator(fake_samples, batch_y)

                # Compute losses
                critic_loss = self.discriminator_loss_fn(real_validity, fake_validity)
                gp = self.compute_gradient_penalty(batch_x, fake_samples, batch_y, lambda_gp)

                total_loss = critic_loss + gp
                val_losses.append(float(total_loss))

                step += 1
                if validation_steps is not None and step >= validation_steps:
                    break

        self.generator.train()
        self.discriminator.train()

        return np.mean(val_losses) if val_losses else 0.0
    def save_models(self, epoch: int):
        """
        Save generator and discriminator models.

        Args:
            epoch: Current epoch number
        """
        gen_path = self.models_saved_path / f"{self.file_name_generator}_epoch_{epoch}.pt"
        disc_path = self.models_saved_path / f"{self.file_name_discriminator}_epoch_{epoch}.pt"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.generator_optimizer.state_dict(),
        }, gen_path)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.discriminator.state_dict(),
            'optimizer_state_dict': self.discriminator_optimizer.state_dict(),
        }, disc_path)

        print(f"Models saved at epoch {epoch}")

    def load_models(self, epoch: int):
        """
        Load generator and discriminator models.

        Args:
            epoch: Epoch number to load
        """
        gen_path = self.models_saved_path / f"{self.file_name_generator}_epoch_{epoch}.pt"
        disc_path = self.models_saved_path / f"{self.file_name_discriminator}_epoch_{epoch}.pt"

        if gen_path.exists():
            checkpoint = torch.load(gen_path)
            self.generator.load_state_dict(checkpoint['model_state_dict'])
            if self.generator_optimizer:
                self.generator_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Generator loaded from epoch {epoch}")

        if disc_path.exists():
            checkpoint = torch.load(disc_path)
            self.discriminator.load_state_dict(checkpoint['model_state_dict'])
            if self.discriminator_optimizer:
                self.discriminator_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Discriminator loaded from epoch {epoch}")

    def get_samples(self, num_samples, labels: Optional[torch.Tensor] = None) -> dict:
        """
        Generate samples using the trained generator.

        Args:
            num_samples: Number of samples to generate (int or dict mapping class to count)
            labels: Optional labels for conditional generation

        Returns:
            Dictionary mapping class indices to generated samples as numpy arrays
            Format: {class_0: np.array([samples]), class_1: np.array([samples]), ...}
        """
        self.generator.eval()

        # Handle dictionary input (samples per class)
        if isinstance(num_samples, dict):
            # Check if 'classes' key contains the actual sample counts
            if 'classes' in num_samples and isinstance(num_samples['classes'], dict):
                num_samples = num_samples['classes']

            # Filter out non-numeric keys and convert string keys to int
            filtered_samples = {}
            for key, value in num_samples.items():
                try:
                    # Try to convert key to integer (handles both int and string numeric keys)
                    numeric_key = int(key)

                    # Handle nested dictionary values or ensure value is an integer
                    if isinstance(value, dict):
                        actual_count = sum(value.values()) if value else 0
                    else:
                        actual_count = int(value)

                    if actual_count > 0:
                        filtered_samples[numeric_key] = actual_count
                except (ValueError, TypeError):
                    # Skip non-numeric keys like 'classes', 'number_classes'
                    continue

            if not filtered_samples:
                raise ValueError(f"No valid class samples found in num_samples dictionary.")

            # Determine number of classes
            num_classes = max(filtered_samples.keys()) + 1

            # Generate samples for each class and store in dictionary
            generated_samples_dict = {}

            for class_idx, count in sorted(filtered_samples.items()):
                with torch.no_grad():
                    z = self._sample_latent(count)

                    # Create one-hot labels for this class
                    class_labels = torch.zeros((count, num_classes), device=self.device)
                    class_labels[:, class_idx] = 1

                    generated = self.generator(z, class_labels)

                    # Store generated samples for this class
                    generated_samples_dict[class_idx] = generated.cpu().numpy()

            self.generator.train()
            return generated_samples_dict

        # Handle integer input (original behavior)
        # Split evenly across available classes
        with torch.no_grad():
            total_samples = int(num_samples)

            if labels is None:
                # Binary classification - split samples between 2 classes
                num_classes = 2
                samples_per_class = total_samples // num_classes

                generated_samples_dict = {}
                for class_idx in range(num_classes):
                    z = self._sample_latent(samples_per_class)
                    class_labels = torch.zeros((samples_per_class, num_classes), device=self.device)
                    class_labels[:, class_idx] = 1

                    generated = self.generator(z, class_labels)
                    generated_samples_dict[class_idx] = generated.cpu().numpy()
            else:
                # Use provided labels
                generated = self.generator(self._sample_latent(total_samples), labels)
                generated_samples_dict = {0: generated.cpu().numpy()}

        self.generator.train()
        return generated_samples_dict