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

try:
    import os
    import sys
    import json
    import numpy
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

except ImportError as error:
    print(error)
    sys.exit(-1)


class VariationalAutoencoderAlgorithmTorch(nn.Module):
    """
    Implements an adaptive Variational AutoEncoder (VAE) model for generating synthetic data.

    The model includes an encoder and a decoder for encoding input data and reconstructing
    it from a learned latent space. During training, it computes both the reconstruction loss
    and the KL divergence loss. The trained decoder can be used to generate synthetic data.

    This class supports customizable latent space parameters and loss functions, making it
    adaptable for different generative tasks. It automatically adapts to any input shape:
    (x), (x, y), (x, y, z), etc.

    Example:
        >>> # Example with 1D data
        >>> vae_1d = VariationalAutoencoderAlgorithmTorch(
        ...     encoder_model=encoder_1d,
        ...     decoder_model=decoder_1d,
        ...     loss_function='mse',
        ...     latent_dimension=64,
        ...     input_shape=(100,)
        ... )
        >>> vae_1d.fit(data_1d, epochs=50)

        >>> # Example with 2D data (images)
        >>> vae_2d = VariationalAutoencoderAlgorithmTorch(
        ...     encoder_model=encoder_2d,
        ...     decoder_model=decoder_2d,
        ...     loss_function='bce',
        ...     latent_dimension=128,
        ...     input_shape=(28, 28)
        ... )
        >>> vae_2d.fit(data_2d, epochs=100)
    """

    def __init__(self,
                 encoder_model,
                 decoder_model,
                 loss_function,
                 latent_dimension,
                 decoder_latent_dimension=None,
                 latent_mean_distribution=0.0,
                 latent_standard_deviation=1.0,
                 file_name_encoder=None,
                 file_name_decoder=None,
                 models_saved_path=None,
                 input_shape=None,
                 auto_adapt_shape=True):
        """
        Initializes the VariationalAutoencoderAlgorithmTorch model.

        Args:
            @encoder_model (nn.Module): The encoder model.
            @decoder_model (nn.Module): The decoder model.
            @loss_function (callable or str): The loss function (auto-compiled if string).
            @latent_dimension (int): The dimensionality of the latent space.
            @decoder_latent_dimension (int): The dimensionality for decoder (defaults to latent_dimension).
            @latent_mean_distribution (float): The mean of the latent distribution.
            @latent_standard_deviation (float): The standard deviation of the latent distribution.
            @file_name_encoder (str): The filename for saving the encoder model.
            @file_name_decoder (str): The filename for saving the decoder model.
            @models_saved_path (str): The directory where models will be saved.
            @input_shape (tuple): Expected input shape (without batch dimension).
            @auto_adapt_shape (bool): Whether to automatically adapt to input data shape.
        """
        # Call parent __init__ first
        super(VariationalAutoencoderAlgorithmTorch, self).__init__()

        # Direct assignment to register as submodules
        self._encoder = encoder_model
        self._decoder = decoder_model

        # Loss function - will be auto-compiled if string
        self._loss_function_string = loss_function if isinstance(loss_function, str) else None
        self._loss_function = self._convert_loss_to_function(loss_function)

        # Metrics for tracking losses
        self._total_loss_tracker = 0.0
        self._reconstruction_loss_tracker = 0.0
        self._kl_loss_tracker = 0.0

        self._latent_mean_distribution = latent_mean_distribution
        self._latent_standard_deviation = latent_standard_deviation
        self._latent_dimension = latent_dimension
        self._decoder_latent_dimension = decoder_latent_dimension if decoder_latent_dimension is not None else latent_dimension

        # File names for saving models
        self._file_name_encoder = file_name_encoder
        self._file_name_decoder = file_name_decoder

        # Path for saving models
        self._models_saved_path = models_saved_path

        # Shape adaptation
        self._input_shape = input_shape
        self._auto_adapt_shape = auto_adapt_shape
        self._inferred_shape = None

        # Optimizer will be configured later
        self.optimizer = None

        # Compiled flag
        self._is_compiled = True  # Auto-compiled

    @staticmethod
    def _convert_loss_to_function(loss):
        """
        Convert string loss names to PyTorch loss functions.

        Args:
            loss: Loss function (string, callable, or loss object)

        Returns:
            Loss function or callable
        """
        if loss is None:
            return F.binary_cross_entropy

        if isinstance(loss, str):
            loss_map = {
                'mse': F.mse_loss,
                'mean_squared_error': F.mse_loss,
                'mae': F.l1_loss,
                'mean_absolute_error': F.l1_loss,
                'l1': F.l1_loss,
                'bce': F.binary_cross_entropy,
                'binary_crossentropy': F.binary_cross_entropy,
                'crossentropy': F.cross_entropy,
                'cross_entropy': F.cross_entropy,
                'nll': F.nll_loss,
                'nll_loss': F.nll_loss,
                'kld': F.kl_div,
                'kl_div': F.kl_div,
                'huber': F.smooth_l1_loss,
                'smooth_l1': F.smooth_l1_loss,
            }
            loss_lower = loss.lower()
            if loss_lower in loss_map:
                return loss_map[loss_lower]
            else:
                raise ValueError(f"Unknown loss function: {loss}. Available: {list(loss_map.keys())}")

        return loss

    def compile(self, loss=None, optimizer=None, **kwargs):
        """
        Configure the model for training (PyTorch compatibility method).

        Args:
            loss: Loss function (can be string name, function, or loss object)
            optimizer: PyTorch optimizer
            **kwargs: Additional arguments (ignored)

        Returns:
            self: Returns self for method chaining
        """
        if loss is not None:
            self._loss_function = self._convert_loss_to_function(loss)
            self._loss_function_string = loss if isinstance(loss, str) else None

        if optimizer is not None:
            self.optimizer = optimizer

        self._is_compiled = True
        return self

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
            tuple: (batch_x, batch_y, batch_labels) where batch_y is the reconstruction target.
        """
        if isinstance(batch, (tuple, list)):
            if len(batch) == 1:
                # Single input, use as both input and target
                batch_x = batch[0]
                batch_y = batch[0]
                batch_labels = None
            elif len(batch) == 2:
                # Input and target provided
                batch_x, batch_y = batch
                # Check if batch_y is labels (different shape) or target (same shape)
                if batch_x.shape != batch_y.shape:
                    # batch_y are labels, use batch_x as target
                    batch_labels = batch_y
                    batch_y = batch_x
                else:
                    batch_labels = None
            elif len(batch) == 3:
                # Input, target, and labels provided
                batch_x, batch_y, batch_labels = batch
            else:
                # Multiple inputs, use first as input, second as target
                batch_x = batch[0]
                batch_y = batch[1]
                batch_labels = None
        else:
            # Single tensor, use as both input and target
            batch_x = batch
            batch_y = batch
            batch_labels = None

        return batch_x, batch_y, batch_labels

    def train_step(self, batch):
        """
        Perform a training step for the Variational AutoEncoder (VAE).
        Automatically adapts to different batch formats.

        Args:
            batch: Input data batch.

        Returns:
            dict: Dictionary containing the loss values (total loss, reconstruction loss, KL divergence loss).
        """
        # Prepare batch data
        batch_x, batch_y, batch_labels = self._prepare_batch(batch)

        # Move to device
        device = next(self.parameters()).device
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        if batch_labels is not None:
            batch_labels = batch_labels.to(device)

        # Zero gradients
        self.optimizer.zero_grad()

        try:
            # Forward pass through encoder
            if batch_labels is not None:
                encoder_output = self._encoder(batch_x, batch_labels)
            else:
                encoder_output = self._encoder(batch_x)

            # Extract encoder outputs
            if isinstance(encoder_output, tuple) and len(encoder_output) >= 3:
                z_mean, z_log_var, latent, label_output = encoder_output[:4] if len(encoder_output) >= 4 else (
                    *encoder_output, batch_labels)
            else:
                # If encoder returns only latent
                latent = encoder_output
                label_output = batch_labels
                z_mean = z_log_var = None

            # Create dummy labels if needed
            if label_output is None:
                batch_size = latent.shape[0]
                label_output = torch.zeros((batch_size, 2)).to(device)

            # Forward pass through decoder
            reconstruction_data = self._decoder(latent, label_output)

        except Exception as e:
            print(f"ERROR in forward pass: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Calculate reconstruction loss
        reconstruction_loss = self._loss_function(reconstruction_data, batch_y, reduction='mean')

        # Calculate KL divergence if available
        if z_mean is not None and z_log_var is not None:
            kl_loss = -0.5 * torch.mean(torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1))
        else:
            kl_loss = torch.tensor(0.0).to(device)

        # Total loss
        total_loss = reconstruction_loss + kl_loss

        # Backward pass
        total_loss.backward()

        # Update weights
        self.optimizer.step()

        # Update loss metrics
        self._total_loss_tracker = total_loss.item()
        self._reconstruction_loss_tracker = reconstruction_loss.item()
        self._kl_loss_tracker = kl_loss.item()

        return {
            "loss": self._total_loss_tracker,
            "reconstruction_loss": self._reconstruction_loss_tracker,
            "kl_loss": self._kl_loss_tracker
        }

    def fit(self, x=None, y=None, batch_size=32, epochs=1, verbose=1,
            callbacks=None, validation_data=None, shuffle=True,
            initial_epoch=0, steps_per_epoch=None, validation_steps=None,
            validation_freq=1, optimizer=None, learning_rate=0.001, **kwargs):
        """
        Train the model with automatic shape adaptation.

        Args:
            x: Input data (any shape).
            y: Target data (if None, x is used as target).
            batch_size: Number of samples per gradient update.
            epochs: Number of epochs to train.
            verbose: 0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks: List of callbacks to apply during training.
            validation_data: Data for validation.
            shuffle: Whether to shuffle data before each epoch.
            initial_epoch: Epoch at which to start training.
            steps_per_epoch: Number of steps per epoch.
            validation_steps: Number of validation steps.
            validation_freq: Validation frequency.
            optimizer: PyTorch optimizer (if None, uses already compiled optimizer).
            learning_rate: Learning rate for optimizer (only used if optimizer is None).

        Returns:
            A History object with training metrics.
        """
        device = next(self.parameters()).device

        # Set optimizer if provided
        if optimizer is not None:
            self.optimizer = optimizer
        elif self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Prepare the dataset
        if isinstance(x, DataLoader):
            train_dataloader = x
            # Try to infer shape from dataloader
            for batch in train_dataloader:
                self._validate_and_adapt_shape(batch)
                break
        else:
            # Validate and adapt shape
            self._validate_and_adapt_shape(x)

            # Handle different input formats
            if isinstance(x, tuple):
                # x is tuple: (data, target) or (data, target, labels)
                if len(x) == 3:
                    x_data, y_data, labels = x
                elif len(x) == 2:
                    x_data, y_data = x
                    labels = None
                else:
                    x_data = x[0]
                    y_data = x[0]  # autoencoder: reconstruct input
                    labels = None
            else:
                x_data = x
                if y is None:
                    y_data = x  # autoencoder: reconstruct input
                    labels = None
                elif isinstance(y, tuple):
                    if len(y) == 2:
                        y_data, labels = y
                    else:
                        y_data = y[0]
                        labels = None
                else:
                    # Check if y has different shape (labels) or same shape (target)
                    y_shape = self._infer_data_shape(y)
                    x_shape = self._infer_data_shape(x)

                    if y_shape != x_shape:
                        # y are labels, x is both input and target
                        y_data = x
                        labels = y
                    else:
                        y_data = y
                        labels = None

            # Convert to tensors
            if isinstance(x_data, numpy.ndarray):
                x_data = torch.from_numpy(x_data).float()
            if isinstance(y_data, numpy.ndarray):
                y_data = torch.from_numpy(y_data).float()
            if labels is not None and isinstance(labels, numpy.ndarray):
                labels = torch.from_numpy(labels).float()

            # Create TensorDataset
            if labels is not None:
                dataset = TensorDataset(x_data, y_data, labels)
            else:
                dataset = TensorDataset(x_data, y_data)

            train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        # Calculate steps per epoch
        if steps_per_epoch is None:
            steps_per_epoch = len(train_dataloader)

        # History to store metrics
        history = {'loss': [], 'reconstruction_loss': [], 'kl_loss': []}

        # Training loop
        for epoch in range(initial_epoch, epochs):
            self.train()
            self._total_loss_tracker = 0.0
            self._reconstruction_loss_tracker = 0.0
            self._kl_loss_tracker = 0.0

            epoch_losses = []
            epoch_recon_losses = []
            epoch_kl_losses = []

            if verbose == 1:
                print(f'\nEpoch {epoch + 1}/{epochs}')

            step = 0
            for batch_data in train_dataloader:
                step += 1

                # Perform training step
                metrics = self.train_step(batch_data)
                current_loss = float(metrics['loss'])
                current_recon_loss = float(metrics['reconstruction_loss'])
                current_kl_loss = float(metrics['kl_loss'])

                epoch_losses.append(current_loss)
                epoch_recon_losses.append(current_recon_loss)
                epoch_kl_losses.append(current_kl_loss)

                # Progress bar
                if verbose == 1:
                    progress = int(50 * step / steps_per_epoch)
                    bar = '=' * progress + '>' + '.' * (50 - progress - 1)
                    print(
                        f'\r[{bar}] {step}/{steps_per_epoch} - loss: {current_loss:.4f} - recon_loss: {current_recon_loss:.4f} - kl_loss: {current_kl_loss:.4f}',
                        end='', flush=True)

                if step >= steps_per_epoch:
                    break

            # Store epoch metrics
            epoch_loss = self._total_loss_tracker
            epoch_recon_loss = numpy.mean(epoch_recon_losses) if epoch_recon_losses else 0.0
            epoch_kl_loss = numpy.mean(epoch_kl_losses) if epoch_kl_losses else 0.0

            history['loss'].append(epoch_loss)
            history['reconstruction_loss'].append(epoch_recon_loss)
            history['kl_loss'].append(epoch_kl_loss)

            if verbose == 1:
                print(f' - loss: {epoch_loss:.4f} - recon_loss: {epoch_recon_loss:.4f} - kl_loss: {epoch_kl_loss:.4f}')
            elif verbose == 2:
                print(
                    f'Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.4f} - recon_loss: {epoch_recon_loss:.4f} - kl_loss: {epoch_kl_loss:.4f}')

            # Validation
            if validation_data is not None and (epoch + 1) % validation_freq == 0:
                val_loss = self._evaluate_validation(validation_data, validation_steps)
                if 'val_loss' not in history:
                    history['val_loss'] = []
                history['val_loss'].append(val_loss)

                if verbose >= 1:
                    print(f' - val_loss: {val_loss:.4f}')

            # callbacks
            if callbacks is not None:
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, {
                            'loss': epoch_loss,
                            'reconstruction_loss': epoch_recon_loss,
                            'kl_loss': epoch_kl_loss
                        })

        # Return history
        class History:
            def __init__(self, history_dict):
                self.history = history_dict

        return History(history)

    def _evaluate_validation(self, validation_data, validation_steps=None):
        """
        Evaluate the model on validation data with automatic shape handling.

        Args:
            validation_data: Validation dataset (DataLoader or tuple).
            validation_steps: Number of validation steps.

        Returns:
            Average validation loss.
        """
        self.eval()
        device = next(self.parameters()).device

        val_losses = []
        val_recon_losses = []
        val_kl_losses = []
        step = 0

        # Prepare validation dataset
        if isinstance(validation_data, DataLoader):
            val_dataloader = validation_data
        else:
            val_x, val_y = validation_data
            if isinstance(val_x, numpy.ndarray):
                val_x = torch.from_numpy(val_x).float()
            if isinstance(val_y, numpy.ndarray):
                val_y = torch.from_numpy(val_y).float()
            val_dataset = TensorDataset(val_x, val_y)
            val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        with torch.no_grad():
            for batch_data in val_dataloader:
                batch_x, batch_y, batch_labels = self._prepare_batch(batch_data)
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                if batch_labels is not None:
                    batch_labels = batch_labels.to(device)

                # Forward pass through encoder
                if batch_labels is not None:
                    encoder_output = self._encoder(batch_x, batch_labels)
                else:
                    encoder_output = self._encoder(batch_x)

                # Extract encoder outputs
                if isinstance(encoder_output, tuple) and len(encoder_output) >= 3:
                    latent_mean, latent_log_variation, latent, label = encoder_output[:4] if len(
                        encoder_output) >= 4 else (*encoder_output, batch_labels)
                else:
                    latent = encoder_output
                    latent_mean = latent_log_variation = None
                    label = batch_labels if batch_labels is not None else torch.zeros((latent.shape[0], 2)).to(device)

                # Forward pass through decoder
                reconstruction_data = self._decoder(latent, label)

                # Calculate reconstruction loss
                reconstruction_loss = self._loss_function(reconstruction_data, batch_y, reduction='mean')

                # Calculate KL divergence
                if latent_mean is not None and latent_log_variation is not None:
                    kl_loss = -0.5 * torch.sum(
                        1 + latent_log_variation - torch.square(latent_mean) - torch.exp(latent_log_variation),
                        dim=1
                    )
                    kl_divergence_loss = torch.mean(kl_loss)
                else:
                    kl_divergence_loss = torch.tensor(0.0).to(device)

                # Total loss
                total_loss = reconstruction_loss + kl_divergence_loss

                val_losses.append(float(total_loss))
                val_recon_losses.append(float(reconstruction_loss))
                val_kl_losses.append(float(kl_divergence_loss))

                step += 1
                if validation_steps is not None and step >= validation_steps:
                    break

        self.train()
        return numpy.mean(val_losses) if val_losses else 0.0

    def configure_optimizer(self,
                            learning_rate=0.001,
                            beta_1=0.9,
                            beta_2=0.999,
                            epsilon=1e-7,
                            amsgrad=False,
                            weight_decay=1e-5):
        """
        Configure the Adam optimizer with custom parameters.
        """
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            betas=(beta_1, beta_2),
            eps=epsilon,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )

    def get_decoder_trained(self):
        return self._decoder

    def get_encoder_trained(self):
        return self._encoder

    def create_embedding(self, data, labels=None):
        """
        Generates latent space embeddings using the trained encoder.

        Args:
            data: Input data
            labels: Optional labels (one-hot encoded or indices)
        """
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            if isinstance(data, numpy.ndarray):
                data = torch.from_numpy(data).float().to(device)

            # Accept optional labels
            if labels is not None:
                if isinstance(labels, numpy.ndarray):
                    labels = torch.from_numpy(labels).float().to(device)
                encoder_output = self._encoder(data, labels)
            else:
                encoder_output = self._encoder(data)

            # Extract z_mean
            if isinstance(encoder_output, tuple) and len(encoder_output) >= 1:
                latent_mean = encoder_output[0]
            else:
                latent_mean = encoder_output

        return latent_mean.cpu().numpy()

    def get_samples(self, number_samples_per_class):
        """
        Generate synthetic samples for each specified class using the trained decoder.

        Args:
            number_samples_per_class (dict):
                Dictionary specifying the number of samples to generate for each class.
                Expected structure:
                {
                    "classes": {class_label: number_of_samples, ...},
                    "number_classes": total_number_of_classes
                }

        Returns:
            dict:
                A dictionary where each key is a class label and the value is an array of generated samples.
        """
        self.eval()
        device = next(self.parameters()).device
        generated_data = {}

        with torch.no_grad():
            for label_class, number_instances in number_samples_per_class["classes"].items():
                # Create one-hot encoded label array
                label_samples_generated = torch.zeros(number_instances, number_samples_per_class["number_classes"])
                label_samples_generated[:, label_class] = 1
                label_samples_generated = label_samples_generated.to(device)

                # Sample random latent vectors
                latent_noise = torch.randn(number_instances, self._decoder_latent_dimension).to(device)

                # Use the decoder to generate samples
                generated_samples = self._decoder(latent_noise, label_samples_generated)

                # Store the generated samples
                generated_data[label_class] = generated_samples.cpu().numpy()

        return generated_data

    def generate_synthetic_data(self, number_samples_generate, label_class, num_classes, latent_dimension=None):
        """
        Generate synthetic data using the Variational AutoEncoder (VAE).

        Args:
            number_samples_generate: Number of samples to generate
            label_class: Class label (integer) to generate
            num_classes: Total number of classes
            latent_dimension: Dimension of latent space (uses self._latent_dimension if None)
        """
        self.eval()
        device = next(self.parameters()).device

        if latent_dimension is None:
            latent_dimension = self._latent_dimension

        with torch.no_grad():
            # Generate random noise samples in the latent space
            random_noise_generate = torch.randn(
                number_samples_generate,
                latent_dimension,
                device=device
            ) * self._latent_standard_deviation + self._latent_mean_distribution

            # Create one-hot encoded labels
            label_list = torch.zeros(number_samples_generate, num_classes, device=device)
            label_list[:, label_class] = 1.0

            # Generate synthetic data
            synthetic_data = self._decoder(random_noise_generate, label_list)

        return synthetic_data

    def get_input_shape(self):
        """
        Get the current input shape.

        Returns:
            tuple: Current input shape (excluding batch dimension).
        """
        return self._inferred_shape if self._inferred_shape is not None else self._input_shape

    @staticmethod
    def reshape_data(data, target_shape):
        """
        Reshape data to target shape if needed.

        Args:
            data: Input data.
            target_shape: Desired shape (excluding batch dimension).

        Returns:
            Reshaped data.
        """
        if isinstance(data, numpy.ndarray):
            if data.shape[1:] != target_shape:
                batch_size = data.shape[0]
                return data.reshape((batch_size,) + target_shape)
        elif torch.is_tensor(data):
            if tuple(data.shape[1:]) != target_shape:
                batch_size = data.shape[0]
                return data.reshape((batch_size,) + target_shape)
        return data

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
        if torch.is_tensor(y_labels):
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

    @property
    def metrics(self):
        """
        Returns:
            dict: Dictionary of metrics tracked during training.
        """
        return {
            "loss": self._total_loss_tracker,
            "reconstruction_loss": self._reconstruction_loss_tracker,
            "kl_loss": self._kl_loss_tracker
        }

    @property
    def input_shape(self):
        return self.get_input_shape()

    @property
    def decoder(self):
        return self._decoder

    @property
    def encoder(self):
        return self._encoder

    def save_model(self, directory, file_name):
        """
        Save the encoder and decoder models.

        Args:
            directory (str): Directory where models will be saved.
            file_name (str): Base file name for saving models.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Construct file names for encoder and decoder models
        encoder_file_name = os.path.join(directory, f"fold_{file_name}_encoder.pth")
        decoder_file_name = os.path.join(directory, f"fold_{file_name}_decoder.pth")

        # Save encoder model
        torch.save(self._encoder.state_dict(), encoder_file_name)

        # Save decoder model
        torch.save(self._decoder.state_dict(), decoder_file_name)

        # Save shape information
        shape_info = {
            'input_shape': self._input_shape,
            'inferred_shape': self._inferred_shape,
            'latent_dimension': self._latent_dimension,
            'decoder_latent_dimension': self._decoder_latent_dimension
        }
        shape_file = os.path.join(directory, f"fold_{file_name}_shape_info.json")
        with open(shape_file, 'w') as f:
            json.dump(shape_info, f)

    def load_models(self, directory, file_name):
        """
        Load the encoder and decoder models from a directory.

        Args:
            directory (str): Directory where models are stored.
            file_name (str): Base file name for loading models.
        """
        device = next(self.parameters()).device

        # Construct file names for encoder and decoder models
        encoder_file_name = os.path.join(directory, f"fold_{file_name}_encoder.pth")
        decoder_file_name = os.path.join(directory, f"fold_{file_name}_decoder.pth")

        # Load the encoder and decoder models
        self._encoder.load_state_dict(torch.load(encoder_file_name, map_location=device))
        self._decoder.load_state_dict(torch.load(decoder_file_name, map_location=device))

        # Load shape information if available
        shape_file = os.path.join(directory, f"fold_{file_name}_shape_info.json")
        if os.path.exists(shape_file):
            with open(shape_file, 'r') as f:
                shape_info = json.load(f)
                self._input_shape = tuple(shape_info.get('input_shape', ()))
                self._inferred_shape = tuple(shape_info.get('inferred_shape', ()))
                self._latent_dimension = shape_info.get('latent_dimension', self._latent_dimension)
                self._decoder_latent_dimension = shape_info.get('decoder_latent_dimension',
                                                                self._decoder_latent_dimension)