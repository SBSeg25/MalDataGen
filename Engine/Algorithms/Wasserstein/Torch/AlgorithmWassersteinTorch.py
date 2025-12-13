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
    Training algorithm wrapper for Wasserstein GAN with Gradient Penalty (WGAN-GP).

    This class manages the complete training process for a Wasserstein GAN, including:
    - Alternating critic and generator training
    - Gradient penalty computation
    - Loss tracking and logging
    - Model checkpointing
    - Callback management

    The algorithm implements the WGAN-GP framework as described in:
    Gulrajani et al., "Improved Training of Wasserstein GANs" (2017)

    Attributes:
        generator_model: The generator network
        discriminator_model: The critic/discriminator network
        latent_dimension: Dimensionality of the latent space
        generator_loss_fn: Loss function for generator (optional, uses Wasserstein loss by default)
        discriminator_loss_fn: Loss function for discriminator (optional, uses Wasserstein loss by default)
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
        Initialize the Wasserstein GAN training algorithm.

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

        # Loss functions (use Wasserstein loss if not provided)
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
        Wasserstein discriminator/critic loss.
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
        Wasserstein generator loss.
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

            # Wasserstein loss
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

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            epochs: int,
            batch_size: int,
            lambda_gp: float = 10.0,
            callbacks: Optional[List[Any]] = None,
            verbose: int = 1):
        """
        Train the WGAN-GP model.

        Args:
            x_train: Training data
            y_train: Training labels (one-hot encoded)
            epochs: Number of training epochs
            batch_size: Batch size
            lambda_gp: Gradient penalty weight
            callbacks: List of callback objects
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
        """
        if self.generator_optimizer is None or self.discriminator_optimizer is None:
            raise ValueError("Model must be compiled before training. Call compile() first.")

        num_samples = len(x_train)
        num_batches = num_samples // batch_size

        # Initialize callbacks
        if callbacks:
            for callback in callbacks:
                # Initialize params if needed
                if hasattr(callback, 'params'):
                    if callback.params is None:
                        callback.params = {}
                    callback.params.update({
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'steps': num_batches,
                        'samples': num_samples
                    })

                # Call on_train_begin
                if hasattr(callback, 'on_train_begin'):
                    callback.on_train_begin()

        for epoch in range(epochs):
            epoch_critic_loss = 0.0
            epoch_gen_loss = 0.0
            epoch_gp = 0.0

            # Shuffle data
            indices = np.random.permutation(num_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            for batch_idx in range(num_batches):
                # Get batch
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size

                batch_x = torch.tensor(x_shuffled[start_idx:end_idx],
                                       dtype=torch.float32, device=self.device)
                batch_y = torch.tensor(y_shuffled[start_idx:end_idx],
                                       dtype=torch.float32, device=self.device)

                # Train step
                losses = self.train_step(batch_x, batch_y, lambda_gp)

                epoch_critic_loss += losses['critic_loss']
                epoch_gen_loss += losses['generator_loss']
                epoch_gp += losses['gradient_penalty']

            # Average losses
            avg_critic_loss = epoch_critic_loss / num_batches
            avg_gen_loss = epoch_gen_loss / num_batches
            avg_gp = epoch_gp / num_batches

            # Store history
            self.history['critic_loss'].append(avg_critic_loss)
            self.history['generator_loss'].append(avg_gen_loss)
            self.history['gradient_penalty'].append(avg_gp)

            # Verbose output
            if verbose > 0:
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Critic Loss: {avg_critic_loss:.4f}, "
                      f"Gen Loss: {avg_gen_loss:.4f}, "
                      f"GP: {avg_gp:.4f}")

            # Callbacks
            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        logs = {
                            'epoch': epoch,
                            'critic_loss': avg_critic_loss,
                            'generator_loss': avg_gen_loss,
                            'gradient_penalty': avg_gp,
                            'loss': avg_critic_loss + avg_gen_loss  # Total loss for compatibility
                        }
                        callback.on_epoch_end(epoch, logs)

            # Save checkpoints periodically
            if (epoch + 1) % 10 == 0:
                self.save_models(epoch + 1)

        # Call on_train_end for callbacks
        if callbacks:
            for callback in callbacks:
                if hasattr(callback, 'on_train_end'):
                    callback.on_train_end()

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

            self.generator.fit()
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

        self.generator.fit()
        return generated_samples_dict