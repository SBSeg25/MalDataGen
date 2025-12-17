#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch Implementation of Adversarial Training Algorithm for Generative Adversarial Networks (GANs)

This module implements a comprehensive GAN training framework using PyTorch,
supporting both conditional and unconditional generation with advanced training techniques.
It provides compatibility with TensorFlow/Keras-style APIs while leveraging PyTorch's dynamic computation graph.

Mathematical Overview:
----------------------
A GAN consists of two neural networks:
1. Generator G(z): Maps latent noise z ~ p_z to data space x' = G(z)
2. Discriminator D(x): Classifies real vs generated data: D(x) ∈ [0,1]

The minimax objective function:
min_G max_D V(D,G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]

In PyTorch, we typically use:
L_D = BCE(D(x), 1) + BCE(D(G(z)), 0)
L_G = BCE(D(G(z)), 1)  # Non-saturating loss

For conditional GANs (cGAN), both networks receive class labels y:
G(z|y) → x'
D(x|y) → [0,1]

References:
-----------
1. Goodfellow, I., et al. (2014). "Generative Adversarial Networks." NeurIPS.
2. Radford, A., et al. (2015). "Unsupervised Representation Learning with Deep Convolutional GANs." ICLR.
3. Mirza, M., & Osindero, S. (2014). "Conditional Generative Adversarial Nets." arXiv:1411.1784.
4. PyTorch Documentation: https://pytorch.org/docs/stable/index.html
"""

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{2}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/17'
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
    import numpy
    import logging

    from pathlib import Path
    from typing import Dict
    from typing import Optional
    from typing import Tuple
    from typing import Union
    from typing import List

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

except ImportError as error:
    logging.error(error)
    sys.exit(-1)


class AdversarialAlgorithmTorch(nn.Module):
    """
    Implements a complete adversarial training framework for Generative Adversarial Networks in PyTorch.

    This class provides a TensorFlow/Keras-compatible API while leveraging PyTorch's capabilities:
    - Conditional and unconditional GAN training
    - Label smoothing for discriminator stabilization
    - Dynamic computation graph with automatic differentiation
    - Model persistence and loading capabilities
    - Synthetic data generation utilities
    - Flexible optimizer configuration
    - Cross-framework compatibility (converts TensorFlow losses to PyTorch)

    Mathematical Components:
    -----------------------
    1. Latent Space: z ~ N(μ, σ²) where z ∈ ℝ^{latent_dim}
    2. Generator: G(z|y) → x̂ ∈ ℝ^{feature_dim}
    3. Discriminator: D(x|y) → [0,1]
    4. Loss Functions:
        - Standard: L_D = BCE(y_true, y_pred) in PyTorch format
        - Label Smoothing: y_smooth ~ U(α, β) where α,β ∈ [0,1]
    5. Optimization: Alternate gradient updates using PyTorch optimizers

    PyTorch-Specific Features:
    --------------------------
    - nn.Module inheritance for proper module management
    - Device-aware operations (CPU/GPU)
    - State dict for model serialization
    - Dynamic computation graph (no @tf.function needed)

    Attributes:
    -----------
    _generator : nn.Module
        Generator network that creates synthetic samples
    _discriminator : nn.Module
        Discriminator network that classifies real vs fake
    _latent_dimension : int
        Dimensionality of the latent space (z-vector)
    _optimizer_generator : torch.optim.Optimizer
        Optimizer for generator network (e.g., Adam)
    _optimizer_discriminator : torch.optim.Optimizer
        Optimizer for discriminator network
    _loss_generator : nn.Module
        Loss function for generator training (e.g., BCELoss)
    _loss_discriminator : nn.Module
        Loss function for discriminator training
    _smoothing_rate : float
        Degree of label smoothing (0.0 to 1.0)
    _latent_mean_distribution : float
        Mean of the latent noise distribution (μ)
    _latent_standard_deviation : float
        Standard deviation of latent noise (σ)
    _learning_rate_generator : float
        Learning rate for generator optimizer
    _learning_rate_discriminator : float
        Learning rate for discriminator optimizer
    device : torch.device
        Computational device (CPU or CUDA GPU)

    Example:
        >>> generator_model = build_generator(latent_dimension=100)
        >>> discriminator_model = build_discriminator()
        >>> adversarial_algorithm = AdversarialAlgorithmTorch(
        ...     generator_model=generator_model,
        ...     discriminator_model=discriminator_model,
        ...     latent_dimension=100,
        ...     loss_generator=nn.BCELoss(),
        ...     loss_discriminator=nn.BCELoss(),
        ...     file_name_discriminator="discriminator",
        ...     file_name_generator="generator",
        ...     models_saved_path="./models/",
        ...     latent_mean_distribution=0.0,
        ...     latent_standard_deviation=1.0,
        ...     smoothing_rate=0.1
        ... )
        >>> adversarial_algorithm.compile(
        ...     optimizer_generator=torch.optim.Adam(generator.parameters(), lr=0.0002),
        ...     optimizer_discriminator=torch.optim.Adam(discriminator.parameters(), lr=0.0002)
        ... )
        >>> history = adversarial_algorithm.fit(train_dataset, epochs=100)
    """

    def __init__(self, generator_model: nn.Module,
                 discriminator_model: nn.Module,
                 latent_dimension: int,
                 loss_generator: nn.Module,
                 loss_discriminator: nn.Module,
                 file_name_discriminator: str,
                 file_name_generator: str,
                 models_saved_path: str,
                 latent_mean_distribution: float,
                 latent_standard_deviation: float,
                 smoothing_rate: float,
                 optimizer_generator: Optional[torch.optim.Optimizer] = None,
                 optimizer_discriminator: Optional[torch.optim.Optimizer] = None,
                 learning_rate_generator: float = 0.0002,
                 learning_rate_discriminator: float = 0.0002):
        """
        Initialize the adversarial training algorithm with PyTorch components.

        Parameters:
        -----------
        generator_model : nn.Module
            Generator neural network architecture (must inherit from nn.Module)
        discriminator_model : nn.Module
            Discriminator neural network architecture (must inherit from nn.Module)
        latent_dimension : int
            Size of latent space vector, must be > 0
        loss_generator : nn.Module or str
            Loss function for generator optimization. Can be:
            - PyTorch nn.Module (e.g., nn.BCELoss())
            - String identifier (e.g., "bce", "mse")
            - TensorFlow/Keras loss (will be auto-converted)
        loss_discriminator : nn.Module or str
            Loss function for discriminator optimization
        file_name_discriminator : str
            Base filename for saving discriminator model (without extension)
        file_name_generator : str
            Base filename for saving generator model (without extension)
        models_saved_path : str
            Directory path for model persistence
        latent_mean_distribution : float
            Mean (μ) for latent noise sampling: z ~ N(μ, σ²)
        latent_standard_deviation : float
            Standard deviation (σ) for latent noise sampling, must be > 0
        smoothing_rate : float
            Label smoothing factor ∈ [0, 1]
            0 = no smoothing, 1 = maximum smoothing
        optimizer_generator : torch.optim.Optimizer, optional
            Optimizer for generator (default: Adam with learning_rate_generator)
        optimizer_discriminator : torch.optim.Optimizer, optional
            Optimizer for discriminator (default: Adam with learning_rate_discriminator)
        learning_rate_generator : float, optional
            Learning rate for default generator optimizer (default: 0.0002)
        learning_rate_discriminator : float, optional
            Learning rate for default discriminator optimizer (default: 0.0002)

        Raises:
        -------
        ValueError
            If input parameters fail validation checks
        TypeError
            If parameter types are incorrect
        """
        super().__init__()

        # Mathematical validation of parameters
        if latent_dimension <= 0:
            raise ValueError("Latent dimension must be greater than 0.")
        if not isinstance(file_name_discriminator, str) or not file_name_discriminator:
            raise ValueError("Discriminator file name must be a non-empty string.")
        if not isinstance(file_name_generator, str) or not file_name_generator:
            raise ValueError("Generator file name must be a non-empty string.")
        if not isinstance(models_saved_path, str) or not models_saved_path:
            raise ValueError("models saved path must be a non-empty string.")
        if not isinstance(latent_mean_distribution, (int, float)):
            raise TypeError("Latent mean distribution must be a number.")
        if not isinstance(latent_standard_deviation, (int, float)):
            raise TypeError("Latent standard deviation must be a number.")
        if latent_standard_deviation <= 0:
            raise ValueError("Latent standard deviation must be greater than 0.")
        if not (0.0 <= smoothing_rate <= 1.0):
            raise ValueError("Smoothing rate must be between 0 and 1.")
        if not isinstance(learning_rate_generator, (int, float)) or learning_rate_generator <= 0:
            raise ValueError("Learning rate for generator must be a positive number.")
        if not isinstance(learning_rate_discriminator, (int, float)) or learning_rate_discriminator <= 0:
            raise ValueError("Learning rate for discriminator must be a positive number.")

        # Core GAN components
        self._generator = generator_model
        self._discriminator = discriminator_model
        self._latent_dimension = latent_dimension

        # Convert loss functions to PyTorch format (handles TensorFlow/Keras compatibility)
        self._loss_generator = self._convert_loss_to_pytorch(loss_generator)
        self._loss_discriminator = self._convert_loss_to_pytorch(loss_discriminator)

        self._smoothing_rate = smoothing_rate
        self._latent_mean_distribution = latent_mean_distribution
        self._latent_standard_deviation = latent_standard_deviation
        self._file_name_discriminator = file_name_discriminator
        self._file_name_generator = file_name_generator
        self._models_saved_path = models_saved_path

        # Learning rates
        self._learning_rate_generator = learning_rate_generator
        self._learning_rate_discriminator = learning_rate_discriminator

        # Device configuration: auto-detect CUDA availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._generator.to(self.device)
        self._discriminator.to(self.device)

        # Initialize optimizers (use provided or create default)
        if optimizer_generator is not None:
            self._optimizer_generator = optimizer_generator
        else:
            self._optimizer_generator = torch.optim.Adam(
                self._generator.parameters(),
                lr=self._learning_rate_generator,
                betas=(0.5, 0.999)  # GAN-recommended beta parameters
            )

        if optimizer_discriminator is not None:
            self._optimizer_discriminator = optimizer_discriminator
        else:
            self._optimizer_discriminator = torch.optim.Adam(
                self._discriminator.parameters(),
                lr=self._learning_rate_discriminator,
                betas=(0.5, 0.999)  # GAN-recommended beta parameters
            )

    def _convert_loss_to_pytorch(self, loss: Union[nn.Module, str, object]) -> nn.Module:
        """
        Convert a loss function to PyTorch format with cross-framework compatibility.

        Supports:
        1. PyTorch nn.Module losses (returned as-is)
        2. String identifiers (mapped to PyTorch equivalents)
        3. TensorFlow/Keras losses (converted to PyTorch defaults)

        Mathematical Mapping:
        --------------------
        - 'binary_crossentropy'/'bce' → nn.BCELoss()
        - 'mean_squared_error'/'mse' → nn.MSELoss()
        - 'mean_absolute_error'/'mae' → nn.L1Loss()

        Parameters:
        -----------
        loss : nn.Module or str or object
            Loss function in any supported format

        Returns:
        --------
        nn.Module
            PyTorch loss function ready for training
        """
        # If it's already a PyTorch loss module, return it
        if isinstance(loss, nn.Module):
            return loss

        # If it's a string, map to PyTorch loss
        if isinstance(loss, str):
            loss_name = loss.lower()
            loss_map = {
                'binary_crossentropy': nn.BCELoss(),
                'bce': nn.BCELoss(),
                'mse': nn.MSELoss(),
                'mean_squared_error': nn.MSELoss(),
                'mae': nn.L1Loss(),
                'mean_absolute_error': nn.L1Loss(),
            }
            if loss_name in loss_map:
                return loss_map[loss_name]

        # If it's a TensorFlow/Keras loss or unknown type, default to BCELoss
        # This provides compatibility with TensorFlow-style code
        return nn.BCELoss()

    def compile(self, optimizer_generator: Optional[Union[torch.optim.Optimizer, float]] = None,
                optimizer_discriminator: Optional[Union[torch.optim.Optimizer, float]] = None,
                loss_generator: Optional[Union[nn.Module, str]] = None,
                loss_discriminator: Optional[Union[nn.Module, str]] = None,
                learning_rate_generator: Optional[float] = None,
                learning_rate_discriminator: Optional[float] = None):
        """
        Configure the training algorithm with optimizers and loss functions.

        Provides flexible input types for compatibility:
        - PyTorch optimizers (torch.optim.Optimizer)
        - Learning rates (float) - creates Adam optimizers automatically
        - TensorFlow/Keras optimizers - converts to PyTorch Adam

        Parameters:
        -----------
        optimizer_generator : torch.optim.Optimizer or float, optional
            Optimizer for generator network. If float, creates Adam with that learning rate.
            If None, keeps current optimizer
        optimizer_discriminator : torch.optim.Optimizer or float, optional
            Optimizer for discriminator network. If float, creates Adam with that learning rate.
            If None, keeps current optimizer
        loss_generator : nn.Module or str, optional
            Loss function for generator (overrides __init__ if provided)
        loss_discriminator : nn.Module or str, optional
            Loss function for discriminator (overrides __init__ if provided)
        learning_rate_generator : float, optional
            Learning rate for generator (only used if optimizer_generator is None)
        learning_rate_discriminator : float, optional
            Learning rate for discriminator (only used if optimizer_discriminator is None)
        """
        # Handle optimizer_generator with flexible input types
        if optimizer_generator is not None:
            if isinstance(optimizer_generator, (int, float)):
                # If a learning rate is passed, create Adam optimizer with GAN defaults
                self._learning_rate_generator = optimizer_generator
                self._optimizer_generator = torch.optim.Adam(
                    self._generator.parameters(),
                    lr=self._learning_rate_generator,
                    betas=(0.5, 0.999)  # GAN-recommended beta parameters
                )
            elif hasattr(optimizer_generator, 'zero_grad'):
                # It's a PyTorch optimizer (has zero_grad method)
                self._optimizer_generator = optimizer_generator
            else:
                # It's likely a TensorFlow/Keras optimizer, create PyTorch Adam with GAN defaults
                self._optimizer_generator = torch.optim.Adam(
                    self._generator.parameters(),
                    lr=self._learning_rate_generator,
                    betas=(0.5, 0.999)
                )
        elif learning_rate_generator is not None:
            # Update learning rate and create new optimizer
            self._learning_rate_generator = learning_rate_generator
            self._optimizer_generator = torch.optim.Adam(
                self._generator.parameters(),
                lr=self._learning_rate_generator,
                betas=(0.5, 0.999)
            )

        # Handle optimizer_discriminator similarly
        if optimizer_discriminator is not None:
            if isinstance(optimizer_discriminator, (int, float)):
                self._learning_rate_discriminator = optimizer_discriminator
                self._optimizer_discriminator = torch.optim.Adam(
                    self._discriminator.parameters(),
                    lr=self._learning_rate_discriminator,
                    betas=(0.5, 0.999)
                )
            elif hasattr(optimizer_discriminator, 'zero_grad'):
                self._optimizer_discriminator = optimizer_discriminator
            else:
                self._optimizer_discriminator = torch.optim.Adam(
                    self._discriminator.parameters(),
                    lr=self._learning_rate_discriminator,
                    betas=(0.5, 0.999)
                )
        elif learning_rate_discriminator is not None:
            # Update learning rate and create new optimizer
            self._learning_rate_discriminator = learning_rate_discriminator
            self._optimizer_discriminator = torch.optim.Adam(
                self._discriminator.parameters(),
                lr=self._learning_rate_discriminator,
                betas=(0.5, 0.999)
            )

        # Update loss functions if provided
        if loss_generator is not None:
            self._loss_generator = self._convert_loss_to_pytorch(loss_generator)
        if loss_discriminator is not None:
            self._loss_discriminator = self._convert_loss_to_pytorch(loss_discriminator)

    def fit(self, x=None, y=None, batch_size=32, epochs=1, verbose=1,
            callbacks=None, validation_data=None, shuffle=True,
            initial_epoch=0, steps_per_epoch=None, validation_steps=None,
            validation_freq=1, **kwargs):
        """
        Train the model with a TensorFlow/Keras-compatible interface.

        This method implements the complete training pipeline with:
        - Epoch-based training
        - Progress visualization
        - Validation monitoring
        - Callback support
        - Flexible input types (DataLoader, numpy arrays, tensors)

        PyTorch Implementation Details:
        ------------------------------
        - Uses DataLoader for batch iteration
        - Maintains training/eval modes properly
        - Tracks losses manually (no built-in metrics like TensorFlow)

        Parameters:
        -----------
        x : torch.utils.data.DataLoader or array-like
            Training data. Can be:
            - PyTorch DataLoader
            - Tuple of (features, labels) as numpy arrays or tensors
            - Single array for unconditional GANs
        y : array-like, optional
            Training labels for conditional GANs. If x is array and y is None, uses x as both features and labels.
        batch_size : int
            Number of samples per gradient update
        epochs : int
            Number of epochs to train
        verbose : int
            Verbosity mode: 0 = silent, 1 = progress bar, 2 = one line per epoch
        callbacks : list of callable
            List of callback functions with on_epoch_end method
        validation_data : DataLoader or tuple
            Validation data in same format as x
        shuffle : bool
            Whether to shuffle the training data each epoch
        initial_epoch : int
            Epoch at which to start training
        steps_per_epoch : int
            Number of steps (batches) per epoch
        validation_steps : int
            Number of validation steps
        validation_freq : int
            Frequency (in epochs) of validation

        Returns:
        --------
        History
            Training history with loss metrics in .history dictionary

        Raises:
        -------
        ValueError
            If data format is unsupported or parameters invalid
        """
        # Prepare the dataset - flexible input handling
        if isinstance(x, DataLoader):
            train_dataset = x
        else:
            # Convert numpy arrays to PyTorch tensors
            if y is None:
                y = x  # For unconditional GANs

            # Convert to tensors if not already
            x_tensor = torch.FloatTensor(x) if not isinstance(x, torch.Tensor) else x
            y_tensor = torch.FloatTensor(y) if not isinstance(y, torch.Tensor) else y

            # Create TensorDataset and DataLoader
            dataset = TensorDataset(x_tensor, y_tensor)
            train_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        # Calculate steps per epoch if not provided
        if steps_per_epoch is None:
            steps_per_epoch = len(train_dataset)

        # Prepare validation data if provided
        val_dataloader = None
        if validation_data is not None:
            if isinstance(validation_data, DataLoader):
                val_dataloader = validation_data
            elif isinstance(validation_data, tuple) and len(validation_data) == 2:
                val_x, val_y = validation_data
                val_x_tensor = torch.FloatTensor(val_x) if not isinstance(val_x, torch.Tensor) else val_x
                val_y_tensor = torch.FloatTensor(val_y) if not isinstance(val_y, torch.Tensor) else val_y
                val_dataset = TensorDataset(val_x_tensor, val_y_tensor)
                val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # History to store metrics (compatible with Keras format)
        history = {'loss': [], 'loss_d': [], 'loss_g': []}
        if validation_data is not None:
            history['val_loss'] = []

        # Training loop over epochs
        for epoch in range(initial_epoch, epochs):
            # Reset loss accumulators for this epoch
            epoch_loss_total = 0.0
            epoch_loss_d = 0.0
            epoch_loss_g = 0.0
            num_batches = 0

            if verbose == 1:
                print(f'\nEpoch {epoch + 1}/{epochs}')

            # Progress tracking
            step = 0
            for batch_data in train_dataset:
                step += 1

                # Perform training step (returns dict with 'loss_d' and 'loss_g')
                metrics = self.train_step(batch_data)

                # Calculate combined loss (average of discriminator and generator losses)
                # Mathematical: L_total = (L_D + L_G) / 2
                current_loss = (metrics['loss_d'] + metrics['loss_g']) / 2.0
                current_loss_d = metrics['loss_d']
                current_loss_g = metrics['loss_g']

                # Accumulate losses for epoch average
                epoch_loss_total += current_loss
                epoch_loss_d += current_loss_d
                epoch_loss_g += current_loss_g
                num_batches += 1

                # Simple progress bar visualization
                if verbose == 1:
                    progress = int(50 * step / steps_per_epoch)
                    bar = '=' * progress + '>' + '.' * (50 - progress - 1)
                    print(
                        f'\r[{bar}] {step}/{steps_per_epoch} - loss: {current_loss:.4f} - loss_d: {current_loss_d:.4f} - loss_g: {current_loss_g:.4f}',
                        end='', flush=True)

                if step >= steps_per_epoch:
                    break

            # Calculate average losses for the epoch
            avg_loss = epoch_loss_total / num_batches if num_batches > 0 else 0.0
            avg_loss_d = epoch_loss_d / num_batches if num_batches > 0 else 0.0
            avg_loss_g = epoch_loss_g / num_batches if num_batches > 0 else 0.0

            # Store epoch losses in history
            history['loss'].append(avg_loss)
            history['loss_d'].append(avg_loss_d)
            history['loss_g'].append(avg_loss_g)

            # Print epoch summary based on verbosity
            if verbose == 1:
                print(f' - loss: {avg_loss:.4f} - loss_d: {avg_loss_d:.4f} - loss_g: {avg_loss_g:.4f}', end='')
            elif verbose == 2:
                print(
                    f'Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f} - loss_d: {avg_loss_d:.4f} - loss_g: {avg_loss_g:.4f}',
                    end='')

            # Validation phase (evaluate without training)
            if val_dataloader is not None and (epoch + 1) % validation_freq == 0:
                val_loss = self._evaluate_validation(val_dataloader, validation_steps)
                history['val_loss'].append(val_loss)

                if verbose >= 1:
                    print(f' - val_loss: {val_loss:.4f}')
            else:
                if verbose >= 1:
                    print()  # New line after metrics

            # Execute callbacks (Keras-style compatibility)
            if callbacks is not None:
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, {
                            'loss': avg_loss,
                            'loss_d': avg_loss_d,
                            'loss_g': avg_loss_g
                        })

        # Return history object with Keras-compatible interface
        class History:
            def __init__(self, history_dict):
                self.history = history_dict

        return History(history)

    def _evaluate_validation(self, validation_data: DataLoader,
                             validation_steps: Optional[int] = None) -> float:
        """
        Evaluate the model on validation data WITHOUT updating weights.

        Mathematical Operation:
        ----------------------
        1. For each validation batch:
            x_val, y_val ~ p_val
            z ~ N(μ, σ²)
            x̂ = G(z|y_val)
        2. L_D_val = BCE(0, D(x_val|y_val)) + BCE(1, D(x̂|y_val))
        3. L_G_val = BCE(0, D(x̂|y_val))
        4. L_val = (L_D_val + L_G_val) / 2

        Note: No label smoothing during validation for accurate assessment.

        Parameters:
        -----------
        validation_data : DataLoader
            Validation dataset as PyTorch DataLoader
        validation_steps : int, optional
            Number of validation batches to process

        Returns:
        --------
        float
            Average validation loss across batches
        """
        # Set models to evaluation mode (disables dropout, batch norm stats)
        self._generator.eval()
        self._discriminator.eval()

        val_losses = []
        step = 0

        # Disable gradient computation for validation
        with torch.no_grad():
            for batch_data in validation_data:
                real_feature, real_samples_label = batch_data

                # Move data to appropriate device (GPU/CPU)
                real_feature = real_feature.to(self.device)
                real_samples_label = real_samples_label.to(self.device)

                # Get the current batch size
                batch_size = real_feature.shape[0]

                # Expand label dimension if needed (adds channel dimension)
                if len(real_samples_label.shape) == 1:
                    real_samples_label = real_samples_label.unsqueeze(-1)

                # Sample random noise vectors from normal distribution
                # Mathematical: z ~ N(μ, σ²)
                latent_space = torch.randn(batch_size, self._latent_dimension, device=self.device)
                latent_space = latent_space * self._latent_standard_deviation + self._latent_mean_distribution

                # Generate synthetic features (no gradient computation)
                synthetic_feature = self._generator([latent_space, real_samples_label])

                # Get discriminator predictions on real and synthetic data
                label_predicted_real = self._discriminator([real_feature, real_samples_label])
                label_predicted_synthetic = self._discriminator([synthetic_feature, real_samples_label])

                # Concatenate predictions for batch processing
                label_predicted_all_samples = torch.cat([label_predicted_real, label_predicted_synthetic], dim=0)

                # Validation labels: exact 0 for real, 1 for synthetic (no smoothing)
                # Mathematical: y_real = 0, y_fake = 1
                tensor_labels_predicted = torch.cat([
                    torch.zeros_like(label_predicted_real),
                    torch.ones_like(label_predicted_synthetic)
                ], dim=0)

                # Compute discriminator loss (PyTorch order: predictions first, then targets)
                loss_d = self._loss_discriminator(label_predicted_all_samples, tensor_labels_predicted)

                # Compute generator loss (want discriminator to output 0 for generated)
                predicted_labels = self._discriminator([synthetic_feature, real_samples_label])
                loss_g = self._loss_generator(predicted_labels, torch.zeros_like(predicted_labels))

                # Total validation loss (average of both)
                total_loss = (loss_d.item() + loss_g.item()) / 2.0
                val_losses.append(total_loss)

                step += 1
                if validation_steps is not None and step >= validation_steps:
                    break

        # Return mean validation loss
        return numpy.mean(val_losses) if val_losses else 0.0

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step for both generator and discriminator.

        Implements the core GAN training algorithm in PyTorch:
        1. Discriminator update: max_D V(D,G)
        2. Generator update: min_G V(D,G)

        Mathematical Steps:
        ------------------
        Phase 1 (Discriminator):
            1. Sample batch: (x, y) ~ p_data
            2. Sample noise: z ~ N(μ, σ²)
            3. Generate: x̂ = G(z|y)
            4. Compute: L_D = BCE(y_smooth, D(x|y)) + BCE(y_smooth, D(x̂|y))

        Phase 2 (Generator):
            1. Sample new noise: z' ~ N(μ, σ²)
            2. Generate: x̂' = G(z'|y)
            3. Compute: L_G = BCE(0, D(x̂'|y))  # Non-saturating loss

        PyTorch Implementation Details:
        ------------------------------
        - Uses torch.no_grad() for generator during discriminator training
        - Properly manages training/eval modes
        - Handles gradient zeroing and optimization steps

        Parameters:
        -----------
        batch : tuple
            Contains (real_features, real_labels) as PyTorch tensors

        Returns:
        --------
        dict
            Dictionary containing loss metrics:
            - 'loss_d': Discriminator loss (float)
            - 'loss_g': Generator loss (float)
        """
        # Unpack the batch into real features and real labels
        real_feature, real_samples_label = batch

        # Move to device (GPU/CPU)
        real_feature = real_feature.to(self.device)
        real_samples_label = real_samples_label.to(self.device)

        # Get the current batch size
        batch_size = real_feature.shape[0]

        # Expand label dimension if needed (for conditional GANs)
        if len(real_samples_label.shape) == 1:
            real_samples_label = real_samples_label.unsqueeze(-1)

        # ==================== PHASE 1: TRAIN DISCRIMINATOR ====================
        # Set discriminator to training mode, generator to eval mode
        self._discriminator.train()
        self._generator.eval()  # Generator not being trained in this phase

        # Zero out discriminator gradients
        self._optimizer_discriminator.zero_grad()

        # Sample random noise vectors (latent space) from normal distribution
        # Mathematical: z ~ N(μ, σ²)
        latent_space = torch.randn(batch_size, self._latent_dimension, device=self.device)
        latent_space = latent_space * self._latent_standard_deviation + self._latent_mean_distribution

        # Generate synthetic features WITHOUT computing gradients for generator
        # This is correct PyTorch practice: don't compute gradients for generator during discriminator training
        with torch.no_grad():
            synthetic_feature = self._generator([latent_space, real_samples_label])

        # Get discriminator predictions on real and synthetic samples
        label_predicted_real = self._discriminator([real_feature, real_samples_label])
        label_predicted_synthetic = self._discriminator([synthetic_feature, real_samples_label])

        # Concatenate predictions for batch loss computation
        label_predicted_all_samples = torch.cat([label_predicted_real, label_predicted_synthetic], dim=0)

        # Label smoothing technique to prevent discriminator from becoming overconfident
        # Real labels: random values in [0.0, 0.15] (close to 0 = real)
        # Fake labels: random values in [0.85, 1.0] (close to 1 = fake)
        smooth_real_labels = torch.rand_like(label_predicted_real) * 0.15
        smooth_synthetic_labels = 0.85 + torch.rand_like(label_predicted_synthetic) * 0.15

        # Concatenate smoothed labels
        tensor_labels_predicted = torch.cat([
            smooth_real_labels,
            smooth_synthetic_labels
        ], dim=0)

        # Compute discriminator loss using binary cross-entropy
        # PyTorch BCELoss order: predictions first, then targets
        loss_d = self._loss_discriminator(label_predicted_all_samples, tensor_labels_predicted)

        # Backward pass and optimization (computes gradients and updates weights)
        loss_d.backward()
        self._optimizer_discriminator.step()

        # ==================== PHASE 2: TRAIN GENERATOR ====================
        # Set generator to training mode, discriminator to eval mode
        self._generator.train()
        self._discriminator.eval()  # Discriminator in eval mode while training generator

        # Zero out generator gradients
        self._optimizer_generator.zero_grad()

        # Generate NEW latent space (don't reuse from discriminator phase)
        # This ensures fresh noise for generator training
        latent_space_new = torch.randn(batch_size, self._latent_dimension, device=self.device)
        latent_space_new = latent_space_new * self._latent_standard_deviation + self._latent_mean_distribution

        # Generate new synthetic samples (WITH gradients for generator)
        synthetic_feature = self._generator([latent_space_new, real_samples_label])

        # Get discriminator predictions for synthetic samples
        predicted_labels = self._discriminator([synthetic_feature, real_samples_label])

        # Generator loss: we want discriminator to classify synthetic data as real (target=0)
        # Using non-saturating loss: L_G = BCE(D(G(z)), 0)
        loss_g = self._loss_generator(predicted_labels, torch.zeros_like(predicted_labels))

        # Backward pass and optimization
        loss_g.backward()
        self._optimizer_generator.step()

        # Return losses as dictionary (converted to Python floats)
        return {"loss_d": loss_d.item(), "loss_g": loss_g.item()}

    def get_samples(self, number_samples_per_class: Dict) -> Dict:
        """
        Generate synthetic data samples for each specified class using the trained generator.

        Mathematical Operation:
        ----------------------
        For each class c:
            1. Create labels: y = one_hot(c)
            2. Sample noise: z ~ N(μ, σ²)
            3. Generate: x̂ = G(z|y)

        Parameters:
        -----------
        number_samples_per_class : dict
            Dictionary containing sampling specifications:
            - "classes": dict of {class_label: number_of_samples}
            - "number_classes": total number of classes (for one-hot encoding)

        Returns:
        --------
        dict
            Dictionary mapping class labels to generated samples as numpy arrays
        """
        # Set generator to evaluation mode (disables dropout, uses saved batch norm stats)
        self._generator.eval()
        generated_data = {}

        # Disable gradient computation for generation
        with torch.no_grad():
            for label_class, number_instances in number_samples_per_class["classes"].items():
                # Create one-hot encoded labels for conditional generation
                # Uses PyTorch's F.one_hot for efficient one-hot encoding
                label_samples_generated = F.one_hot(
                    torch.tensor([label_class] * number_instances, device=self.device),
                    num_classes=number_samples_per_class["number_classes"]
                ).float()

                # Generate random noise vectors from normal distribution
                # Mathematical: z ~ N(μ, σ²)
                latent_noise = torch.normal(
                    mean=self._latent_mean_distribution,
                    std=self._latent_standard_deviation,
                    size=(number_instances, self._latent_dimension)
                ).to(self.device)

                # Generate synthetic samples using the trained generator
                generated_samples = self._generator([latent_noise, label_samples_generated])

                # Convert to numpy array and store (detach from computation graph)
                generated_data[label_class] = generated_samples.cpu().numpy()

        return generated_data

    def save_model(self, path_output: str, k_fold: int):
        """
        Save generator and discriminator models to disk in PyTorch format.

        Saves both model architecture information and learned weights:
        - Model state_dict: Contains all learnable parameters
        - Model architecture: String representation for reference

        PyTorch Serialization Format:
        ----------------------------
        Uses torch.save() which can save to:
        - .pth files (recommended)
        - .pt files (alternative)
        - .tar files (for checkpoints with additional info)

        Parameters:
        -----------
        path_output : str
            Base output directory path
        k_fold : int
            Current fold number for cross-validation naming

        Raises:
        -------
        Exception
            If directory creation or file writing fails
        """
        try:
            logging.info(f"Starting to save adversarial Model for fold {k_fold}...")

            # Create directory if it doesn't exist
            path_directory = os.path.join(path_output, self._models_saved_path)
            Path(path_directory).mkdir(parents=True, exist_ok=True)

            # Create filenames with fold identifier
            discriminator_file_name = f"{self._file_name_discriminator}_{k_fold}"
            generator_file_name = f"{self._file_name_generator}_{k_fold}"

            # Full paths for model files (save directly in directory)
            discriminator_path = os.path.join(path_directory, discriminator_file_name)
            generator_path = os.path.join(path_directory, generator_file_name)

            # Save discriminator model
            logging.info("Saving discriminator model...")
            torch.save({
                'model_state_dict': self._discriminator.state_dict(),  # All learnable parameters
                'model_architecture': str(self._discriminator)  # String representation
            }, discriminator_path + ".pth")
            logging.info(f"Discriminator model saved at: {discriminator_path}.pth")

            # Save generator model
            logging.info("Saving generator model...")
            torch.save({
                'model_state_dict': self._generator.state_dict(),
                'model_architecture': str(self._generator)
            }, generator_path + ".pth")
            logging.info(f"Generator model saved at: {generator_path}.pth")

            logging.info(f"Models saved successfully for fold {k_fold}")

        except Exception as e:
            logging.error(f"An error occurred while saving the models: {e}")
            raise

    def load_models(self, path_output: str, k_fold: int):
        """
        Load generator and discriminator models from disk.

        PyTorch Loading Notes:
        ---------------------
        - Uses torch.load() with map_location to handle CPU/GPU compatibility
        - Loads state_dict which contains all learnable parameters
        - Model architecture must match between save and load

        Parameters:
        -----------
        path_output : str
            Base directory path containing saved models
        k_fold : int
            Fold number for model file naming

        Raises:
        -------
        FileNotFoundError
            If model files are not found
        Exception
            If model loading fails for other reasons
        """
        try:
            logging.info(f"Loading adversarial Model for fold {k_fold}...")

            # Construct directory path
            path_directory = os.path.join(path_output, self._models_saved_path)

            # Create filenames with fold identifier (must match save_model)
            discriminator_file_name = f"{self._file_name_discriminator}_{k_fold}"
            generator_file_name = f"{self._file_name_generator}_{k_fold}"

            # Full paths for model files
            discriminator_path = os.path.join(path_directory, discriminator_file_name)
            generator_path = os.path.join(path_directory, generator_file_name)

            # Load discriminator model
            logging.info(f"Loading discriminator model from: {discriminator_path}.pth")
            discriminator_checkpoint = torch.load(discriminator_path + ".pth", map_location=self.device)
            self._discriminator.load_state_dict(discriminator_checkpoint['model_state_dict'])
            logging.info("Loaded discriminator weights")

            # Load generator model
            logging.info(f"Loading generator model from: {generator_path}.pth")
            generator_checkpoint = torch.load(generator_path + ".pth", map_location=self.device)
            self._generator.load_state_dict(generator_checkpoint['model_state_dict'])
            logging.info("Loaded generator weights")

            # Set models to evaluation mode (disables dropout, uses saved batch norm stats)
            self._generator.eval()
            self._discriminator.eval()

            logging.info(f"Models loaded successfully for fold {k_fold}")

        except FileNotFoundError:
            logging.error("Model file not found. Please provide an existing and valid model.")
            raise
        except Exception as e:
            logging.error(f"An error occurred while loading the models: {e}")
            raise

    # ==================================================
    # SETTER METHODS FOR DYNAMIC CONFIGURATION
    # ==================================================

    def set_generator(self, generator: nn.Module):
        """
        Set or replace the generator model.

        Parameters:
        -----------
        generator : nn.Module
            New generator network
        """
        self._generator = generator
        self._generator.to(self.device)  # Move to appropriate device

    def set_discriminator(self, discriminator: nn.Module):
        """
        Set or replace the discriminator model.

        Parameters:
        -----------
        discriminator : nn.Module
            New discriminator network
        """
        self._discriminator = discriminator
        self._discriminator.to(self.device)

    def set_latent_dimension(self, latent_dimension: int):
        """
        Update the latent space dimension.

        Parameters:
        -----------
        latent_dimension : int
            New latent space dimension (must be > 0)

        Raises:
        -------
        ValueError
            If latent_dimension <= 0
        """
        if latent_dimension <= 0:
            raise ValueError("Latent dimension must be greater than 0.")
        self._latent_dimension = latent_dimension

    def set_optimizer_generator(self, optimizer_generator: torch.optim.Optimizer):
        """
        Update the generator optimizer.

        Parameters:
        -----------
        optimizer_generator : torch.optim.Optimizer
            New optimizer for generator
        """
        self._optimizer_generator = optimizer_generator

    def set_optimizer_discriminator(self, optimizer_discriminator: torch.optim.Optimizer):
        """
        Update the discriminator optimizer.

        Parameters:
        -----------
        optimizer_discriminator : torch.optim.Optimizer
            New optimizer for discriminator
        """
        self._optimizer_discriminator = optimizer_discriminator

    def set_loss_generator(self, loss_generator: Union[nn.Module, str]):
        """
        Update the generator loss function.

        Supports same formats as __init__:
        - PyTorch nn.Module
        - String identifier
        - TensorFlow/Keras loss (auto-converted)

        Parameters:
        -----------
        loss_generator : nn.Module or str
            New loss function for generator
        """
        self._loss_generator = self._convert_loss_to_pytorch(loss_generator)

    def set_loss_discriminator(self, loss_discriminator: Union[nn.Module, str]):
        """
        Update the discriminator loss function.

        Parameters:
        -----------
        loss_discriminator : nn.Module or str
            New loss function for discriminator
        """
        self._loss_discriminator = self._convert_loss_to_pytorch(loss_discriminator)

    def set_learning_rate_generator(self, learning_rate: float):
        """
        Update the learning rate for generator optimizer.

        Creates a new Adam optimizer with the updated learning rate.

        Parameters:
        -----------
        learning_rate : float
            New learning rate for generator (must be > 0)

        Raises:
        -------
        ValueError
            If learning_rate <= 0
        """
        if learning_rate <= 0:
            raise ValueError("Learning rate must be greater than 0.")
        self._learning_rate_generator = learning_rate
        self._optimizer_generator = torch.optim.Adam(
            self._generator.parameters(),
            lr=self._learning_rate_generator,
            betas=(0.5, 0.999)
        )

    def set_learning_rate_discriminator(self, learning_rate: float):
        """
        Update the learning rate for discriminator optimizer.

        Creates a new Adam optimizer with the updated learning rate.

        Parameters:
        -----------
        learning_rate : float
            New learning rate for discriminator (must be > 0)

        Raises:
        -------
        ValueError
            If learning_rate <= 0
        """
        if learning_rate <= 0:
            raise ValueError("Learning rate must be greater than 0.")
        self._learning_rate_discriminator = learning_rate
        self._optimizer_discriminator = torch.optim.Adam(
            self._discriminator.parameters(),
            lr=self._learning_rate_discriminator,
            betas=(0.5, 0.999)
        )

    # ==================================================
    # GETTER METHODS FOR ACCESSING CONFIGURATION
    # ==================================================

    def get_generator(self) -> nn.Module:
        """Get the generator model."""
        return self._generator

    def get_discriminator(self) -> nn.Module:
        """Get the discriminator model."""
        return self._discriminator

    def get_optimizer_generator(self) -> torch.optim.Optimizer:
        """Get the generator optimizer."""
        return self._optimizer_generator

    def get_optimizer_discriminator(self) -> torch.optim.Optimizer:
        """Get the discriminator optimizer."""
        return self._optimizer_discriminator

    def get_learning_rate_generator(self) -> float:
        """Get the current learning rate for generator."""
        return self._learning_rate_generator

    def get_learning_rate_discriminator(self) -> float:
        """Get the current learning rate for discriminator."""
        return self._learning_rate_discriminator

    # ==================================================
    # PROPERTIES FOR ACCESS TO INTERNAL COMPONENTS
    # ==================================================

    @property
    def generator(self) -> nn.Module:
        """
        Get the generator model.

        Returns:
        --------
        nn.Module
            Generator network
        """
        return self._generator

    @property
    def discriminator(self) -> nn.Module:
        """
        Get the discriminator model.

        Returns:
        --------
        nn.Module
            Discriminator network
        """
        return self._discriminator

    # ==================================================
    # HISTORY CLASS FOR KERAS COMPATIBILITY
    # ==================================================

    class History:
        """
        History object for TensorFlow/Keras compatibility.

        Stores training history and provides access via .history attribute.
        This enables code written for TensorFlow to work with PyTorch implementation.

        Attributes:
        -----------
        history : dict
            Dictionary containing loss metrics over training epochs
        """

        def __init__(self):
            """Initialize empty history dictionary."""
            self.history = {}

        def __getitem__(self, key: str):
            """
            Get history value by key.

            Parameters:
            -----------
            key : str
                History key (e.g., 'loss', 'loss_d', 'loss_g')

            Returns:
            --------
            list
                List of values for the given key
            """
            return self.history[key]

        def __setitem__(self, key: str, value: List[float]):
            """
            Set history value by key.

            Parameters:
            -----------
            key : str
                History key
            value : list
                List of values to store
            """
            self.history[key] = value

        def keys(self):
            """
            Get all history keys.

            Returns:
            --------
            dict_keys
                View of all history keys
            """
            return self.history.keys()