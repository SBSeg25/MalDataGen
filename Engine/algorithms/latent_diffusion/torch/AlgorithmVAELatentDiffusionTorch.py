#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/14'
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
    from torch.utils.data import DataLoader, TensorDataset

except ImportError as error:
    print(error)
    sys.exit(-1)


class VAELatentDiffusionAlgorithmTorch(nn.Module):
    """
    Implements a Variational AutoEncoder (VAE) model for generating synthetic data.

    The model includes an encoder and a decoder for encoding input data and reconstructing
    it from a learned latent space. During training, it computes both the reconstruction loss
    and the KL divergence loss. The trained decoder can be used to generate synthetic data.

    This class supports customizable latent space parameters and loss functions, making it
    adaptable for different generative tasks.

    Attributes:
        _encoder (nn.Module):
            Encoder model that encodes input data into the latent space.
        _decoder (nn.Module):
            Decoder model that reconstructs data from the latent representation.
        _loss_function (callable):
            Function used to compute the total loss during training.
        _total_loss (float):
            Tracks the overall loss during training.
        _reconstruction_loss (float):
            Tracks the reconstruction loss during training.
        _kl_loss (float):
            Tracks the KL divergence loss during training.
        _latent_mean_distribution (float):
            Mean of the latent distribution.
        _latent_standard_deviation (float):
            Standard deviation of the latent distribution.
        _latent_dimension (int):
            Dimensionality of the latent space.
        _decoder_latent_dimension (int):
            Dimensionality of the latent space used by the decoder.
        _file_name_encoder (str):
            File name for saving the encoder model.
        _file_name_decoder (str):
            File name for saving the decoder model.
        _models_saved_path (str):
            Directory path where the encoder and decoder models are saved.
    """

    def __init__(self,
                 encoder_model,
                 decoder_model,
                 loss_function,
                 latent_dimension,
                 decoder_latent_dimension,
                 latent_mean_distribution,
                 latent_standard_deviation,
                 file_name_encoder,
                 file_name_decoder,
                 models_saved_path,
                 *args,
                 **kwargs):
        """
        Initializes the VAELatentDiffusionAlgorithm model.

        Args:
            encoder_model (nn.Module): The encoder model.
            decoder_model (nn.Module): The decoder model.
            loss_function (callable): The loss function.
            latent_dimension (int): The dimensionality of the latent space.
            decoder_latent_dimension (int): Decoder latent dimension.
            latent_mean_distribution (float): Mean of the latent distribution.
            latent_standard_deviation (float): Standard deviation of the latent distribution.
            file_name_encoder (str): Filename for saving the encoder.
            file_name_decoder (str): Filename for saving the decoder.
            models_saved_path (str): Directory where models will be saved.
        """
        super().__init__(*args, **kwargs)

        # Initialize the encoder and decoder models
        self._encoder = encoder_model
        self._decoder = decoder_model

        # loss function
        self._loss_function = loss_function

        # Initialize loss trackers
        self._total_loss = 0.0
        self._reconstruction_loss = 0.0
        self._kl_loss = 0.0
        self._loss_count = 0

        self._latent_mean_distribution = latent_mean_distribution
        self._latent_standard_deviation = latent_standard_deviation
        self._latent_dimension = latent_dimension
        self._decoder_latent_dimension = decoder_latent_dimension

        # File names for saving models
        self._file_name_encoder = file_name_encoder
        self._file_name_decoder = file_name_decoder

        # Path for saving models
        self._models_saved_path = models_saved_path

        # Optimizer (will be set in fit() if not provided)
        self._optimizer = None

    def train_step(self, batch, labels=None, optimizer=None):
        """
        Perform a training step for the Variational AutoEncoder (VAE).

        Args:
            batch: Tuple of (batch_x, batch_target) or just batch_x - input data and reconstruction target.
            labels: One-hot encoded class labels for conditioning (optional, can be part of batch).
            optimizer: PyTorch optimizer for updating model parameters (optional if set in fit()).

        Returns:
            dict: Dictionary containing the loss values.
        """
        # Handle different batch formats
        if isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                batch_x, batch_target = batch
                # If labels not provided separately, use batch_target as labels
                if labels is None:
                    labels = batch_target
            elif len(batch) == 3:
                batch_x, batch_target, labels = batch
            else:
                raise ValueError(f"Batch must be a tuple of 2 or 3 elements, got {len(batch)}")
        else:
            batch_x = batch
            batch_target = batch
            if labels is None:
                raise ValueError("Labels must be provided when batch is not a tuple")

        # Use internal optimizer if not provided
        if optimizer is None:
            optimizer = self._optimizer
            if optimizer is None:
                raise ValueError("No optimizer available. Provide optimizer or call fit() first.")

        # Ensure tensors are on the same device as the model
        device = next(self.parameters()).device
        batch_x = batch_x.to(device)
        batch_target = batch_target.to(device)
        labels = labels.to(device)

        # Flatten inputs to 2D if necessary (batch_size, features)
        if len(batch_x.shape) > 2:
            batch_x = batch_x.view(batch_x.shape[0], -1)
        if len(batch_target.shape) > 2:
            batch_target = batch_target.view(batch_target.shape[0], -1)
        if len(labels.shape) > 2:
            labels = labels.view(labels.shape[0], -1)

        # Set model to training mode
        self.train()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass: Encode input data and sample from the latent space
        latent_mean, latent_log_variation, latent, _ = self._encoder(batch_x, labels)

        # Decode the sampled latent space and generate reconstructed data
        reconstruction_data = self._decoder(latent, labels)

        # Calculate binary cross-entropy loss for reconstruction
        binary_cross_entropy_loss = F.binary_cross_entropy(
            reconstruction_data, batch_target, reduction='none'
        )
        reconstruction_loss = binary_cross_entropy_loss.mean()

        # Calculate KL divergence loss
        encoder_output = (1 + latent_log_variation - latent_mean.pow(2))
        kl_divergence_loss = -0.5 * (encoder_output - torch.exp(latent_log_variation))
        kl_divergence_loss = kl_divergence_loss.sum(dim=1).mean()

        # Total loss is the sum of reconstruction loss and KL divergence loss
        loss_model_in_reconstruction = reconstruction_loss + kl_divergence_loss

        # Backward pass
        loss_model_in_reconstruction.backward()

        # Update weights
        optimizer.step()

        # Update loss trackers
        self._total_loss += loss_model_in_reconstruction.item()
        self._reconstruction_loss += reconstruction_loss.item()
        self._kl_loss += kl_divergence_loss.item()
        self._loss_count += 1

        # Return a dictionary containing the current loss values
        return {
            "loss": loss_model_in_reconstruction.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "kl_loss": kl_divergence_loss.item()
        }

    def fit(self, x=None, y=None, batch_size=32, epochs=1, verbose=1,
            callbacks=None, validation_data=None, shuffle=True,
            initial_epoch=0, steps_per_epoch=None, validation_steps=None,
            validation_freq=1, optimizer=None, learning_rate=0.001, **kwargs):
        """
        Train the model with a simplified progress bar.

        Args:
            x: Input data (numpy array or torch tensor).
            y: Target data (labels, numpy array or torch tensor). If None, uses x as target.
            batch_size: Number of samples per gradient update.
            epochs: Number of epochs to train.
            verbose: 0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks: List of callbacks to apply during training.
            validation_data: Data for validation (tuple of (x_val, y_val) or (x_val, y_val, labels_val)).
            shuffle: Whether to shuffle data before each epoch.
            initial_epoch: Epoch at which to start training.
            steps_per_epoch: Number of steps per epoch.
            validation_steps: Number of validation steps.
            validation_freq: Validation frequency.
            optimizer: PyTorch optimizer (if provided, replaces current optimizer).
            learning_rate: Learning rate for optimizer (only used if optimizer is None).

        Returns:
            A History object with training metrics.
        """

        # Set optimizer
        if optimizer is not None:
            self._optimizer = optimizer
        elif self._optimizer is None:
            # Create default optimizer if none exists
            self._optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

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
            if len(validation_data) == 2:
                x_val, y_val = validation_data
                labels_val = None
            elif len(validation_data) == 3:
                x_val, y_val, labels_val = validation_data
            else:
                raise ValueError("validation_data must be a tuple of 2 or 3 elements")

            if not isinstance(x_val, torch.Tensor):
                x_val = torch.from_numpy(x_val).float()
            if not isinstance(y_val, torch.Tensor):
                y_val = torch.from_numpy(y_val).float()

            if labels_val is not None:
                if not isinstance(labels_val, torch.Tensor):
                    labels_val = torch.from_numpy(labels_val).float()
                val_dataset = DataLoader(
                    TensorDataset(x_val, y_val, labels_val),
                    batch_size=batch_size,
                    shuffle=False
                )
            else:
                val_dataset = DataLoader(
                    TensorDataset(x_val, y_val),
                    batch_size=batch_size,
                    shuffle=False
                )

        # History to store metrics
        history = {
            'loss': [],
            'reconstruction_loss': [],
            'kl_loss': []
        }

        # Training loop
        for epoch in range(initial_epoch, epochs):
            # Reset loss trackers
            self.reset_metrics()

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

            # Get epoch metrics
            epoch_metrics = self.get_metrics()
            epoch_loss = epoch_metrics['loss']
            epoch_recon_loss = epoch_metrics['reconstruction_loss']
            epoch_kl_loss = epoch_metrics['kl_loss']

            # Store epoch losses
            history['loss'].append(epoch_loss)
            history['reconstruction_loss'].append(epoch_recon_loss)
            history['kl_loss'].append(epoch_kl_loss)

            if verbose == 1:
                print(f' - loss: {epoch_loss:.4f} - recon_loss: {epoch_recon_loss:.4f} - kl_loss: {epoch_kl_loss:.4f}')
            elif verbose == 2:
                print(
                    f'Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.4f} - recon_loss: {epoch_recon_loss:.4f} - kl_loss: {epoch_kl_loss:.4f}')

            # Validation
            if val_dataset is not None and (epoch + 1) % validation_freq == 0:
                val_metrics = self._evaluate_validation(val_dataset, validation_steps)

                # Add validation metrics to history
                for key, value in val_metrics.items():
                    val_key = f'val_{key}'
                    if val_key not in history:
                        history[val_key] = []
                    history[val_key].append(value)

                if verbose >= 1:
                    print(
                        f' - val_loss: {val_metrics["loss"]:.4f} - val_recon_loss: {val_metrics["reconstruction_loss"]:.4f} - val_kl_loss: {val_metrics["kl_loss"]:.4f}')

            # callbacks
            if callbacks is not None:
                for callback in callbacks:
                    callback.on_epoch_end(epoch, {
                        'loss': epoch_loss,
                        'reconstruction_loss': epoch_recon_loss,
                        'kl_loss': epoch_kl_loss
                    })

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
            dict: Dictionary with averaged validation metrics.
        """
        self.eval()
        device = next(self.parameters()).device

        val_total_loss = []
        val_recon_loss = []
        val_kl_loss = []
        step = 0

        with torch.no_grad():
            for batch_data in validation_data:
                # Handle different batch formats
                if len(batch_data) == 2:
                    batch_x, batch_y = batch_data
                    labels = batch_y
                elif len(batch_data) == 3:
                    batch_x, batch_y, labels = batch_data
                else:
                    raise ValueError(f"Batch must have 2 or 3 elements, got {len(batch_data)}")

                # Move to device
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                labels = labels.to(device)

                # Flatten inputs if necessary
                if len(batch_x.shape) > 2:
                    batch_x = batch_x.view(batch_x.shape[0], -1)
                if len(batch_y.shape) > 2:
                    batch_y = batch_y.view(batch_y.shape[0], -1)
                if len(labels.shape) > 2:
                    labels = labels.view(labels.shape[0], -1)

                # Forward pass
                latent_mean, latent_log_variation, latent, _ = self._encoder(batch_x, labels)
                reconstruction_data = self._decoder(latent, labels)

                # Calculate losses
                binary_cross_entropy_loss = F.binary_cross_entropy(
                    reconstruction_data, batch_y, reduction='none'
                )
                reconstruction_loss = binary_cross_entropy_loss.mean()

                encoder_output = (1 + latent_log_variation - latent_mean.pow(2))
                kl_divergence_loss = -0.5 * (encoder_output - torch.exp(latent_log_variation))
                kl_divergence_loss = kl_divergence_loss.sum(dim=1).mean()

                total_loss = reconstruction_loss + kl_divergence_loss

                val_total_loss.append(float(total_loss))
                val_recon_loss.append(float(reconstruction_loss))
                val_kl_loss.append(float(kl_divergence_loss))

                step += 1
                if validation_steps is not None and step >= validation_steps:
                    break

        self.train()

        return {
            'loss': numpy.mean(val_total_loss) if val_total_loss else 0.0,
            'reconstruction_loss': numpy.mean(val_recon_loss) if val_recon_loss else 0.0,
            'kl_loss': numpy.mean(val_kl_loss) if val_kl_loss else 0.0
        }

    def get_decoder_trained(self):
        """Returns the trained decoder model."""
        return self._decoder

    def get_encoder_trained(self):
        """Returns the trained encoder model."""
        return self._encoder

    def create_embedding(self, data, batch_size=32):
        """
        Generates latent space embeddings using the trained encoder.

        Args:
            data (torch.Tensor or ndarray): Input data to encode.
            batch_size (int): Batch size for processing.

        Returns:
            ndarray: Latent space representations.
        """
        self.eval()

        # Convert numpy array to tensor if needed
        if isinstance(data, numpy.ndarray):
            data = torch.from_numpy(data).float()

        device = next(self.parameters()).device
        data = data.to(device)

        embeddings = []
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                num_samples = batch.shape[0]

                # Get number of classes from encoder
                num_classes = self._encoder._encoder_number_samples_per_class["number_classes"]
                dummy_labels = torch.zeros(num_samples, num_classes).to(device)

                latent_mean, _, _, _ = self._encoder(batch, dummy_labels)
                embeddings.append(latent_mean.cpu().numpy())

        return numpy.concatenate(embeddings, axis=0)

    def get_samples(self, number_samples_per_class):
        """
        Generate synthetic samples for each specified class using the trained decoder.

        Args:
            number_samples_per_class (dict): Dictionary specifying the number of samples.
                Expected format: {
                    "classes": {class_label: number_of_samples, ...},
                    "number_classes": total_number_of_classes
                }

        Returns:
            dict: Generated samples organized by class.
        """
        self.eval()
        device = next(self.parameters()).device

        generated_data = {}

        for label_class, number_instances in number_samples_per_class["classes"].items():
            # Create a one-hot encoded label array
            label_samples_generated = F.one_hot(
                torch.tensor([label_class] * number_instances, dtype=torch.long),
                num_classes=number_samples_per_class["number_classes"]
            ).float().to(device)

            # Sample random latent vectors
            latent_noise = torch.randn(number_instances, self._decoder_latent_dimension).to(device)

            # Generate samples
            with torch.no_grad():
                generated_samples = self._decoder(latent_noise, label_samples_generated)

            # Round the generated samples
            generated_samples = torch.round(generated_samples).cpu().numpy()

            generated_data[label_class] = generated_samples

        return generated_data

    def generate_synthetic_data(self, number_samples_generate, labels, latent_dimension):
        """
        Generate synthetic data using the VAE.

        Args:
            number_samples_generate (int): Number of synthetic samples to generate.
            labels: Labels for the generated data.
            latent_dimension (int): Dimension of the latent space.

        Returns:
            torch.Tensor: Synthetic data generated by the decoder.
        """
        self.eval()
        device = next(self.parameters()).device

        # Generate random noise samples in the latent space
        random_noise_generate = torch.randn(
            number_samples_generate,
            latent_dimension,
            device=device
        ) * self._latent_standard_deviation + self._latent_mean_distribution

        # Create label vectors
        label_list = torch.full(
            (number_samples_generate, 1),
            labels,
            dtype=torch.float32,
            device=device
        )

        # Generate synthetic data
        with torch.no_grad():
            synthetic_data = self._decoder(random_noise_generate, label_list)

        return synthetic_data

    def get_metrics(self):
        """
        Returns averaged metrics tracked during training.

        Returns:
            dict: Dictionary of averaged metrics.
        """
        if self._loss_count == 0:
            return {
                "loss": 0.0,
                "reconstruction_loss": 0.0,
                "kl_loss": 0.0
            }

        return {
            "loss": self._total_loss / self._loss_count,
            "reconstruction_loss": self._reconstruction_loss / self._loss_count,
            "kl_loss": self._kl_loss / self._loss_count
        }

    def reset_metrics(self):
        """Reset all loss trackers."""
        self._total_loss = 0.0
        self._reconstruction_loss = 0.0
        self._kl_loss = 0.0
        self._loss_count = 0

    def save_model(self, directory, file_name):
        """
        Save the encoder and decoder models.

        Args:
            directory (str): Directory where models will be saved.
            file_name (str): Base file name for saving models.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Construct file names
        encoder_file_name = os.path.join(directory, f"fold_{file_name}_encoder.pth")
        decoder_file_name = os.path.join(directory, f"fold_{file_name}_decoder.pth")

        # Save encoder model
        torch.save({
            'model_state_dict': self._encoder.state_dict(),
            'model_architecture': str(self._encoder)
        }, encoder_file_name)

        # Save decoder model
        torch.save({
            'model_state_dict': self._decoder.state_dict(),
            'model_architecture': str(self._decoder)
        }, decoder_file_name)

        print(f"Models saved to {directory}")

    def load_models(self, directory, file_name):
        """
        Load the encoder and decoder models from a directory.

        Args:
            directory (str): Directory where models are stored.
            file_name (str): Base file name for loading models.
        """
        # Construct file names
        encoder_file_name = os.path.join(directory, f"fold_{file_name}_encoder.pth")
        decoder_file_name = os.path.join(directory, f"fold_{file_name}_decoder.pth")

        # Load encoder
        encoder_checkpoint = torch.load(encoder_file_name)
        self._encoder.load_state_dict(encoder_checkpoint['model_state_dict'])

        # Load decoder
        decoder_checkpoint = torch.load(decoder_file_name)
        self._decoder.load_state_dict(decoder_checkpoint['model_state_dict'])

        print(f"Models loaded from {directory}")

    # Properties
    @property
    def decoder(self):
        return self._decoder

    @property
    def encoder(self):
        return self._encoder

    @decoder.setter
    def decoder(self, decoder):
        self._decoder = decoder

    @encoder.setter
    def encoder(self, encoder):
        self._encoder = encoder

    @property
    def latent_mean_distribution(self):
        return self._latent_mean_distribution

    @property
    def latent_deviation(self):
        return self._latent_standard_deviation

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value