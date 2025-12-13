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

    from abc import ABC

    from typing import Any
    from typing import Callable

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
    A wassersteingp Generative adversarial Network (wassersteingp GAN) model.

    This class represents a wassersteingp GAN consisting of a generator and discriminator model.
    It implements the wassersteingp loss with gradient penalty to improve the training of the discriminator and generator.

    Reference:
        Arjovsky, M., Chintala, S., & Bottou, L. (2017). wassersteingp GAN.
        In Proceedings of the 34th International Conference on Machine Learning (ICML 2017) (Vol. 70, pp. 214-223).
        http://proceedings.mlr.press/v70/arjovsky17a.html

    Attributes:
        @_generator (nn.Module):
            The generator model responsible for generating synthetic data.
        @_discriminator (nn.Module):
            The discriminator model used to evaluate the authenticity of generated data.
        @_latent_dimension (int):
            The dimension of the latent space from which the generator takes input.
        @_generator_optimizer (Optimizer):
            Optimizer used for training the generator.
        @_discriminator_optimizer (Optimizer):
            Optimizer used for training the discriminator.
        @_generator_loss_fn (function):
            Loss function used for training the generator.
        @_discriminator_loss_fn (function):
            Loss function used for training the discriminator.
        @_latent_mean_distribution (float):
            Mean of the latent space distribution.
        @_latent_stander_deviation (float):
            Standard deviation of the latent space distribution.
        @_smoothing_rate (float):
            Rate for label smoothing applied to the discriminator's true labels.
        @_gradient_penalty_weight (float):
            Weight for the gradient penalty term in the wassersteingp loss.
        @_discriminator_steps (int):
            Number of discriminator updates per generator update.
        @_file_name_discriminator (str):
            File name for saving/loading the discriminator model.
        @_file_name_generator (str):
            File name for saving/loading the generator model.
        @_models_saved_path (str):
            Path where the models are saved.

    Raises:
        ValueError:
            Raised if:
            - The latent dimension is non-positive.
            - The gradient penalty weight is non-positive.
            - The smoothing rate is outside the valid range (0, 1).
            - The number of discriminator steps is non-positive.

    Example:
        >>> generator = build_generator_model()
        >>> discriminator = build_discriminator_model()
        >>> wgan = WassersteinGPAlgorithmTorch(
        ...     generator_model=generator,
        ...     discriminator_model=discriminator,
        ...     latent_dimension=100,
        ...     generator_loss_fn=generator_loss_fn,
        ...     discriminator_loss_fn=discriminator_loss_fn,
        ...     file_name_discriminator='discriminator_model.pt',
        ...     file_name_generator='generator_model.pt',
        ...     models_saved_path='./models/',
        ...     latent_mean_distribution=0.0,
        ...     latent_stander_deviation=1.0,
        ...     smoothing_rate=0.1,
        ...     gradient_penalty_weight=10.0,
        ...     discriminator_steps=5
        ... )
        >>> # number_samples_per_class is calculated automatically during fit()
        >>> wgan.fit(x_train, y_train, batch_size=64, epochs=10)
    """

    def __init__(self,
                 generator_model,
                 discriminator_model,
                 latent_dimension,
                 generator_loss_fn,
                 discriminator_loss_fn,
                 file_name_discriminator,
                 file_name_generator,
                 models_saved_path,
                 latent_mean_distribution,
                 latent_stander_deviation,
                 smoothing_rate,
                 gradient_penalty_weight,
                 discriminator_steps,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        # Initialize instance variables with provided or default values
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
        self._latent_stander_deviation = latent_stander_deviation
        self._file_name_discriminator = file_name_discriminator
        self._file_name_generator = file_name_generator
        self._models_saved_path = models_saved_path
        self._discriminator_steps = discriminator_steps
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compile(self, optimizer_generator, optimizer_discriminator,
                loss_generator, loss_discriminator, *args, **kwargs):
        """
        Compile the wassersteingp Generative adversarial Network (WGAN) with custom optimizers and loss functions.

        Args:
            optimizer_generator (torch.optim.Optimizer or keras.optimizers.Optimizer):
                The optimizer for the generator. Can be Keras or PyTorch optimizer.
            optimizer_discriminator (torch.optim.Optimizer or keras.optimizers.Optimizer):
                The optimizer for the discriminator. Can be Keras or PyTorch optimizer.
            loss_generator (Callable):
                The loss function for the generator.
            loss_discriminator (Callable):
                The loss function for the discriminator.
            *args:
                Additional positional arguments.
            **kwargs:
                Additional keyword arguments.

        This method compiles the GAN with custom optimizers and loss functions specified as arguments.
        It sets the optimizer and loss for both the generator and discriminator.
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

        # Wrap loss functions to handle TensorFlow-style calls
        self._discriminator_loss_fn = self._wrap_loss_function(loss_discriminator, is_discriminator=True)
        self._generator_loss_fn = self._wrap_loss_function(loss_generator, is_discriminator=False)

    def _wrap_loss_function(self, loss_fn, is_discriminator=True):
        """
        Wraps a loss function to ensure compatibility between TensorFlow and PyTorch.

        Args:
            loss_fn: Original loss function (may use TensorFlow operations)
            is_discriminator: Whether this is for discriminator (True) or generator (False)

        Returns:
            Wrapped loss function that works with PyTorch tensors
        """

        def wrapped_loss(*args, **kwargs):
            if is_discriminator:
                # Discriminator loss: expects real_img and fake_img
                real_img = kwargs.get('real_img', args[0] if len(args) > 0 else None)
                fake_img = kwargs.get('fake_img', args[1] if len(args) > 1 else None)

                # Convert to PyTorch operations if they're tensors
                if isinstance(real_img, torch.Tensor) and isinstance(fake_img, torch.Tensor):
                    # Standard wasserstein loss: E[D(fake)] - E[D(real)]
                    return torch.mean(fake_img) - torch.mean(real_img)
                else:
                    # Call original function
                    return loss_fn(*args, **kwargs)
            else:
                # Generator loss: expects fake_img
                fake_img = args[0] if len(args) > 0 else kwargs.get('fake_img')

                # Convert to PyTorch operations if it's a tensor
                if isinstance(fake_img, torch.Tensor):
                    # Standard wasserstein generator loss: -E[D(fake)]
                    return -torch.mean(fake_img)
                else:
                    # Call original function
                    return loss_fn(*args, **kwargs)

        return wrapped_loss

    def gradient_penalty(self, batch_size, real_feature, real_label, synthetic_feature):
        """
        Compute the gradient penalty for the wassersteingp GAN.

        The gradient penalty is used to enforce the Lipschitz constraint on the discriminator's output.

        Parameters:
            batch_size (int):
                The batch size of the input data.
            real_feature (torch.Tensor):
                Real data features.
            real_label (torch.Tensor):
                Real data labels.
            synthetic_feature (torch.Tensor):
                Synthetic (generated) data features.

        """
        # Generate random noise for smoothing.
        random_smooth = torch.randn(batch_size, 1, device=self._device) * 0.1

        # Calculate the linear distance between real and synthetic features.
        linear_distance = synthetic_feature - real_feature

        # Interpolate between real and synthetic features using the random noise.
        interpolated_feature = real_feature + random_smooth * linear_distance
        interpolated_feature.requires_grad_(True)

        # Get discriminator's output for the interpolated features.
        labels_predicted = self.discriminator([interpolated_feature, real_label])

        # Calculate the gradient of the discriminator's output with respect to the interpolated features.
        gradients = torch.autograd.grad(
            outputs=labels_predicted,
            inputs=interpolated_feature,
            grad_outputs=torch.ones_like(labels_predicted),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Compute the gradient magnitude and normalize it.
        gradient_normalized = torch.sqrt(torch.sum(gradients ** 2, dim=1))

        # Calculate the final gradient penalty as the mean squared difference from 1.0 and return.
        gradient_penalty_final = torch.mean((gradient_normalized - 1.0) ** 2)

        return gradient_penalty_final

    def calculate_samples_per_class(self, y_labels):
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
        Train the model for a fixed number of epochs with a simplified progress bar.

        This method mimics Keras's fit() API for compatibility with PyTorch.

        Args:
            x: Input data (array-like or torch.utils.data.Dataset).
            y: Target data (labels). Not needed if x is a Dataset.
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
            optimizer: Optimizer (dict with 'generator' and 'discriminator' keys or single optimizer).
            learning_rate: Learning rate for optimizers (if optimizer is None).
            **kwargs: Additional arguments for compatibility.

        Returns:
            History object containing training loss history.
        """
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader

        # Set optimizers if provided
        if optimizer is not None:
            if isinstance(optimizer, dict):
                self._generator_optimizer = optimizer.get('generator')
                self._discriminator_optimizer = optimizer.get('discriminator')
            else:
                self._generator_optimizer = optimizer
                self._discriminator_optimizer = optimizer
        elif self._generator_optimizer is None or self._discriminator_optimizer is None:
            # Create default optimizers if none exist
            self._generator_optimizer = optim.Adam(
                self._generator.parameters(),
                lr=learning_rate
            )
            self._discriminator_optimizer = optim.Adam(
                self._discriminator.parameters(),
                lr=learning_rate
            )

        # Prepare the training dataset
        if isinstance(x, torch.utils.data.Dataset):
            train_loader = DataLoader(x, batch_size=batch_size, shuffle=shuffle)
        else:
            # Convert numpy arrays to tensors
            if not isinstance(x, torch.Tensor):
                x_tensor = torch.tensor(x, dtype=torch.float32)
            else:
                x_tensor = x

            if not isinstance(y, torch.Tensor):
                y_tensor = torch.tensor(y, dtype=torch.long)
            else:
                y_tensor = y

            dataset = TensorDataset(x_tensor, y_tensor)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        # Calculate steps per epoch if not provided
        if steps_per_epoch is None:
            steps_per_epoch = len(train_loader)

        # Prepare validation dataset if provided
        val_loader = None
        if validation_data is not None:
            if isinstance(validation_data, torch.utils.data.Dataset):
                val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
            else:
                val_x, val_y = validation_data
                if not isinstance(val_x, torch.Tensor):
                    val_x = torch.tensor(val_x, dtype=torch.float32)
                if not isinstance(val_y, torch.Tensor):
                    val_y = torch.tensor(val_y, dtype=torch.long)
                val_dataset = TensorDataset(val_x, val_y)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Move models to device
        self._generator.to(self._device)
        self._discriminator.to(self._device)

        # History to store metrics
        history = {'d_loss': [], 'g_loss': []}

        # Training loop
        for epoch in range(initial_epoch, epochs):
            # Trackers for epoch metrics
            epoch_d_losses = []
            epoch_g_losses = []

            if verbose == 1:
                print(f'\nEpoch {epoch + 1}/{epochs}')

            # Progress tracking
            step = 0
            for batch_data in train_loader:
                step += 1

                # Perform training step
                losses = self.train_step(batch_data)
                current_d_loss = losses['d_loss']
                current_g_loss = losses['g_loss']

                # Track losses for this epoch
                epoch_d_losses.append(current_d_loss)
                epoch_g_losses.append(current_g_loss)

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
            epoch_d_loss = numpy.mean(epoch_d_losses) if epoch_d_losses else 0.0
            epoch_g_loss = numpy.mean(epoch_g_losses) if epoch_g_losses else 0.0

            history['d_loss'].append(epoch_d_loss)
            history['g_loss'].append(epoch_g_loss)

            if verbose == 1:
                print(f' - d_loss: {epoch_d_loss:.4f} - g_loss: {epoch_g_loss:.4f}')
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
        Evaluate the model on validation data.

        Args:
            val_loader: Validation DataLoader.
            validation_steps: Number of validation steps.

        Returns:
            Average validation loss.
        """
        val_d_losses = []
        val_g_losses = []
        step = 0

        # Set models to evaluation mode
        self._generator.eval()
        self._discriminator.eval()

        with torch.no_grad():
            for batch_data in val_loader:
                real_feature, real_samples_label = batch_data

                # Move data to device
                if not isinstance(real_feature, torch.Tensor):
                    real_feature = torch.tensor(real_feature, dtype=torch.float32, device=self._device)
                else:
                    real_feature = real_feature.to(self._device)

                if not isinstance(real_samples_label, torch.Tensor):
                    real_samples_label = torch.tensor(real_samples_label, dtype=torch.long, device=self._device)
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
                               self._latent_stander_deviation + self._latent_mean_distribution

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

        # Set models back to training mode
        self._generator.train()
        self._discriminator.train()

        # Return average of discriminator and generator losses
        avg_val_loss = (numpy.mean(val_d_losses) + numpy.mean(val_g_losses)) / 2
        return avg_val_loss if val_d_losses else 0.0
    def train_step(self, batch):
        """
        Executes one training step for the GAN model.

        This step updates both the discriminator and the generator.
        The discriminator is updated multiple times (controlled by self._discriminator_steps),
        while the generator is updated once.

        Args:
            batch (tuple): A tuple containing:
                - real_feature: A batch of real data samples (features).
                - real_samples_label: Corresponding class labels for each sample.

        Returns:
            dict: Dictionary containing the discriminator and generator loss for the current training step.
        """

        # Unpack batch into features and labels.
        real_feature, real_samples_label = batch

        # Convert to torch tensors if needed
        if not isinstance(real_feature, torch.Tensor):
            real_feature = torch.tensor(real_feature, dtype=torch.float32, device=self._device)
        else:
            real_feature = real_feature.to(self._device)

        if not isinstance(real_samples_label, torch.Tensor):
            # Check if it's one-hot encoded (2D with num_classes columns)
            if len(real_samples_label.shape) == 2 and real_samples_label.shape[1] > 1:
                # Already one-hot encoded
                real_samples_label = torch.tensor(real_samples_label, dtype=torch.float32, device=self._device)
            else:
                # Convert to long for indexing
                real_samples_label = torch.tensor(real_samples_label, dtype=torch.long, device=self._device)
        else:
            real_samples_label = real_samples_label.to(self._device)

        batch_size = real_feature.shape[0]

        # Handle label dimensions - check if already one-hot encoded
        if len(real_samples_label.shape) == 2 and real_samples_label.shape[1] > 1:
            # Already one-hot encoded, use as is
            labels_for_model = real_samples_label
        else:
            # Expand label dimensions to match input expectations (e.g., (batch_size, 1)).
            if len(real_samples_label.shape) == 1:
                real_samples_label = real_samples_label.unsqueeze(-1)
            labels_for_model = real_samples_label

        # === Discriminator Training Loop ===
        for _ in range(self._discriminator_steps):
            # Generate random noise vectors for the latent space.
            latent_space = torch.randn(batch_size, self._latent_dimension, device=self._device) * \
                           self._latent_stander_deviation + self._latent_mean_distribution

            # Zero discriminator gradients
            self._discriminator_optimizer.zero_grad()

            # Generate synthetic samples from the generator using noise and labels.
            with torch.no_grad():
                synthetic_feature = self._generator([latent_space, labels_for_model])

            # Predict "real/fake" labels using the discriminator for real and synthetic samples.
            label_predicted_real = self._discriminator([real_feature, labels_for_model])
            label_predicted_synthetic = self._discriminator([synthetic_feature, labels_for_model])

            # Compute discriminator loss (real vs fake).
            discriminator_loss_result = self._discriminator_loss_fn(
                real_img=label_predicted_real, fake_img=label_predicted_synthetic)

            # Compute gradient penalty for improved stability (WGAN-GP, etc.).
            gradient_penalty = self.gradient_penalty(batch_size,
                                                     real_feature,
                                                     labels_for_model,
                                                     synthetic_feature)

            # Combine loss with gradient penalty.
            all_discriminator_loss = discriminator_loss_result + gradient_penalty * self._gradient_penalty_weight

            # Backpropagate and update discriminator
            all_discriminator_loss.backward()
            self._discriminator_optimizer.step()

        # === Generator Training Step ===
        # Generate fresh random noise vectors for the latent space.
        latent_space = torch.randn(batch_size, self._latent_dimension, device=self._device) * \
                       self._latent_stander_deviation + self._latent_mean_distribution

        # Zero generator gradients
        self._generator_optimizer.zero_grad()

        # Generate synthetic samples from the generator.
        synthetic_feature = self._generator([latent_space, labels_for_model])

        # Predict "real/fake" labels for synthetic samples using the discriminator.
        predicted_labels = self._discriminator([synthetic_feature, labels_for_model])

        # Compute generator loss (how well generator fools the discriminator).
        all_generator_loss = self._generator_loss_fn(predicted_labels)

        # Backpropagate and update generator
        all_generator_loss.backward()
        self._generator_optimizer.step()

        # Return the loss values for monitoring/tracking.
        return {"d_loss": all_discriminator_loss.item(), "g_loss": all_generator_loss.item()}

    def get_samples(self, number_samples_per_class):
        """
        Generates synthetic samples for each specified class using the trained generator.

        This method creates samples conditioned on class labels, using random noise vectors
        and the generator to produce the samples.

        Args:
            number_samples_per_class (dict): A dictionary containing:
                - "classes" (dict): Mapping of class labels to the number of samples to generate for each class.
                - "number_classes" (int): Total number of classes (used for one-hot encoding).

        Returns:
            dict: A dictionary where each key is a class label and the value is an array of generated samples.

        Raises:
            ValueError: If number_samples_per_class is not provided.
        """
        if number_samples_per_class is None:
            raise ValueError("number_samples_per_class is required for generating samples")

        # Dictionary to store generated samples for each class.
        generated_data = {}

        # Set generator to evaluation mode
        self._generator.eval()

        with torch.no_grad():
            # Loop through each class and the desired number of samples for that class.
            for label_class, number_instances in number_samples_per_class["classes"].items():
                # Create one-hot encoded labels for all samples of the current class.
                label_samples_generated = to_categorical([label_class] * number_instances,
                                                         num_classes=number_samples_per_class["number_classes"])
                label_samples_generated = torch.tensor(label_samples_generated, dtype=torch.float32,
                                                       device=self._device)

                # Sample random noise vectors from a normal distribution.
                latent_noise = torch.randn(number_instances, self._latent_dimension, device=self._device) * \
                               self._latent_stander_deviation + self._latent_mean_distribution

                # Generate synthetic samples using the generator.
                generated_samples = self._generator([latent_noise, label_samples_generated])

                # Convert to numpy and round to integer values
                generated_samples = generated_samples.cpu().numpy()
                generated_samples = numpy.rint(generated_samples)

                # Store generated samples for the current class.
                generated_data[label_class] = generated_samples

        # Set generator back to training mode
        self._generator.train()

        # Return the dictionary containing generated samples for all requested classes.
        return generated_data

    def save_model(self, directory, file_name):
        """
        Save the encoder and decoder models in both JSON and PyTorch formats.

        Args:
            directory (str): Directory where models will be saved.
            file_name (str): Base file name for saving models.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Construct file names for encoder and decoder models
        generator_file_name = os.path.join(directory, f"fold_{file_name}_generator")
        discriminator_file_name = os.path.join(directory, f"fold_{file_name}_discriminator")

        # Save generator model
        torch.save(self._generator.state_dict(), f"{generator_file_name}.pt")
        self._save_model_architecture(self._generator, f"{generator_file_name}.json")

        # Save discriminator model
        torch.save(self._discriminator.state_dict(), f"{discriminator_file_name}.pt")
        self._save_model_architecture(self._discriminator, f"{discriminator_file_name}.json")

    @staticmethod
    def _save_model_architecture(model, file_path):
        """
        Save model architecture to a JSON file.

        Args:
            model (nn.Module): Model to save.
            file_path (str): Path to the JSON file.
        """
        architecture = {
            'class': model.__class__.__name__,
            'state_dict_keys': list(model.state_dict().keys())
        }
        with open(file_path, "w") as json_file:
            json.dump(architecture, json_file, indent=2)

    def load_models(self, directory, file_name):
        """
        Load the generator and discriminator models from a directory.

        Args:
            directory (str): Directory where models are stored.
            file_name (str): Base file name for loading models.
        """

        # Construct file names for generator and discriminator models
        generator_file_name = os.path.join(directory, f"{file_name}_generator.pt")
        discriminator_file_name = os.path.join(directory, f"{file_name}_discriminator.pt")

        # Load the generator and discriminator state dicts
        self._generator.load_state_dict(torch.load(generator_file_name, map_location=self._device))
        self._discriminator.load_state_dict(torch.load(discriminator_file_name, map_location=self._device))

    @property
    def discriminator(self) -> Any:
        """Get the discriminator model instance.

        Returns:
            The discriminator model used in GAN training.
        """
        return self._discriminator

    @discriminator.setter
    def discriminator(self, value: Any) -> None:
        """Set the discriminator model instance.

        Args:
            value: The discriminator model to set.
        """
        self._discriminator = value

    @property
    def latent_dimension(self) -> int:
        """Get the dimension of the latent space.

        Returns:
            The size of the latent space dimension (positive integer).
        """
        return self._latent_dimension

    @latent_dimension.setter
    def latent_dimension(self, value: int) -> None:
        """Set the dimension of the latent space.

        Args:
            value: The latent dimension size (must be positive).

        Raises:
            ValueError: If value is not a positive integer.
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Latent dimension must be a positive integer")
        self._latent_dimension = value

    @property
    def discriminator_loss_fn(self) -> Callable:
        """Get the discriminator loss function.

        Returns:
            The loss function used for discriminator training.
        """
        return self._discriminator_loss_fn

    @discriminator_loss_fn.setter
    def discriminator_loss_fn(self, value: Callable) -> None:
        """Set the discriminator loss function.

        Args:
            value: The loss function to use for discriminator training.
        """
        self._discriminator_loss_fn = value

    @property
    def generator_loss_fn(self) -> Callable:
        """Get the generator loss function.

        Returns:
            The loss function used for generator training.
        """
        return self._generator_loss_fn

    @generator_loss_fn.setter
    def generator_loss_fn(self, value: Callable) -> None:
        """Set the generator loss function.

        Args:
            value: The loss function to use for generator training.
        """
        self._generator_loss_fn = value

    @property
    def gradient_penalty_weight(self) -> float:
        """Get the weight for gradient penalty in WGAN-GP.

        Returns:
            The weight factor for gradient penalty term.
        """
        return self._gradient_penalty_weight

    @gradient_penalty_weight.setter
    def gradient_penalty_weight(self, value: float) -> None:
        """Set the weight for gradient penalty in WGAN-GP.

        Args:
            value: The penalty weight (must be non-negative).

        Raises:
            ValueError: If value is negative.
        """
        if value < 0:
            raise ValueError("Gradient penalty weight cannot be negative")
        self._gradient_penalty_weight = value

    @property
    def smooth_rate(self) -> float:
        """Get the label smoothing rate.

        Returns:
            The rate used for one-sided label smoothing.
        """
        return self._smooth_rate

    @smooth_rate.setter
    def smooth_rate(self, value: float) -> None:
        """Set the label smoothing rate.

        Args:
            value: The smoothing rate (typically between 0 and 0.3).

        Raises:
            ValueError: If value is not between 0 and 1.
        """
        if not 0 <= value <= 1:
            raise ValueError("Smoothing rate must be between 0 and 1")
        self._smooth_rate = value

    @property
    def latent_mean_distribution(self) -> float:
        """Get the mean of the latent space distribution.

        Returns:
            The mean value used for latent space sampling.
        """
        return self._latent_mean_distribution

    @latent_mean_distribution.setter
    def latent_mean_distribution(self, value: float) -> None:
        """Set the mean of the latent space distribution.

        Args:
            value: The mean value for latent distribution.
        """
        self._latent_mean_distribution = value

    @property
    def latent_stander_deviation(self) -> float:
        """Get the standard deviation of the latent space distribution.

        Returns:
            The standard deviation used for latent space sampling.
        """
        return self._latent_stander_deviation

    @latent_stander_deviation.setter
    def latent_stander_deviation(self, value: float) -> None:
        """Set the standard deviation of the latent space distribution.

        Args:
            value: The standard deviation (must be positive).

        Raises:
            ValueError: If value is not positive.
        """
        if value <= 0:
            raise ValueError("Standard deviation must be positive")
        self._latent_stander_deviation = value

    @property
    def file_name_discriminator(self) -> str:
        """Get the discriminator model save filename.

        Returns:
            The filename pattern for saving discriminator models.
        """
        return self._file_name_discriminator

    @file_name_discriminator.setter
    def file_name_discriminator(self, value: str) -> None:
        """Set the discriminator model save filename.

        Args:
            value: The filename pattern to use.
        """
        self._file_name_discriminator = value

    @property
    def file_name_generator(self) -> str:
        """Get the generator model save filename.

        Returns:
            The filename pattern for saving generator models.
        """
        return self._file_name_generator

    @file_name_generator.setter
    def file_name_generator(self, value: str) -> None:
        """Set the generator model save filename.

        Args:
            value: The filename pattern to use.
        """
        self._file_name_generator = value

    @property
    def models_saved_path(self) -> str:
        """Get the path for saving models.

        Returns:
            The directory path where models are saved.
        """
        return self._models_saved_path

    @models_saved_path.setter
    def models_saved_path(self, value: str) -> None:
        """Set the path for saving models.

        Args:
            value: The directory path to use for saving models.
        """
        self._models_saved_path = value

    @property
    def discriminator_steps(self) -> int:
        """Get the number of discriminator steps per iteration.

        Returns:
            The number of discriminator training steps per GAN iteration.
        """
        return self._discriminator_steps

    @discriminator_steps.setter
    def discriminator_steps(self, value: int) -> None:
        """Set the number of discriminator steps per iteration.

        Args:
            value: The number of steps (must be positive integer).

        Raises:
            ValueError: If value is not a positive integer.
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Discriminator steps must be a positive integer")
        self._discriminator_steps = value