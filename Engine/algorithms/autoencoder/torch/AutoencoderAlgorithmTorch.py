#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/07'
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
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader, TensorDataset

except ImportError as error:
    print(error)
    sys.exit(-1)

DEFAULT_LATENT_MEAN_DISTRIBUTION=0.0
DEFAULT_latent_standard_deviation=1.0
DEFAULT_LATENT_DIMENSION=64
DEFAULT_NUMBER_CLASSES=2



class AutoencoderAlgorithmTorch(nn.Module):
    """
    An abstract class for AutoEncoder models with Keras-style API compatibility.

    This class provides a foundation for AutoEncoder models with methods for training,
    generating synthetic data, saving and loading models. Designed to work with conditional
    autoencoders where both encoder and decoder expect [data, labels] as input.

    Args:
        @encoder_model (nn.Module):
            The encoder part of the AutoEncoder.
        @decoder_model (nn.Module):
            The decoder part of the AutoEncoder.
        @loss_function (nn.Module, optional):
            The loss function for training.
        @file_name_encoder (str, optional):
            The file name for saving the encoder model.
        @file_name_decoder (str, optional):
            The file name for saving the decoder model.
        @models_saved_path (str, optional):
            The path to save the models.
        @latent_mean_distribution (float, optional):
            Mean of the latent space distribution (default: 0.0).
        @latent_standard_deviation (float, optional):
            Standard deviation of the latent space distribution (default: 1.0).
        @latent_dimension (int, optional):
            The dimensionality of the latent space (default: 64).
        @number_classes (int, optional):
            The number of classes for conditional generation (default: 2).

    Example:
        >>> encoder_model = build_encoder(...)
        >>> decoder_model = build_decoder(...)
        >>> autoencoder = AutoencoderAlgorithmTorch(
        ...     encoder_model=encoder_model,
        ...     decoder_model=decoder_model,
        ...     loss_function=nn.MSELoss(),
        ...     number_classes=2,
        ...     latent_dimension=64
        ... )
        >>> autoencoder.compile(loss='mse')
        >>> history = autoencoder.fit(x_train, y_train, epochs=50, batch_size=32)
    """

    def __init__(self,
                 encoder_model,
                 decoder_model,
                 loss_function=None,
                 file_name_encoder=None,
                 file_name_decoder=None,
                 models_saved_path=None,
                 latent_mean_distribution=DEFAULT_LATENT_MEAN_DISTRIBUTION,
                 latent_standard_deviation=DEFAULT_latent_standard_deviation,
                 latent_dimension=DEFAULT_LATENT_DIMENSION):

        super().__init__()

        if not isinstance(encoder_model, nn.Module):
            raise TypeError("encoder_model must be a nn.Module instance.")

        if not isinstance(decoder_model, nn.Module):
            raise TypeError("decoder_model must be a nn.Module instance.")

        if file_name_encoder is not None and (not isinstance(file_name_encoder, str) or not file_name_encoder):
            raise ValueError("file_name_encoder must be a non-empty string.")

        if file_name_decoder is not None and (not isinstance(file_name_decoder, str) or not file_name_decoder):
            raise ValueError("file_name_decoder must be a non-empty string.")

        if models_saved_path is not None and (not isinstance(models_saved_path, str) or not models_saved_path):
            raise ValueError("models_saved_path must be a non-empty string.")

        if not isinstance(latent_mean_distribution, (int, float)):
            raise TypeError("latent_mean_distribution must be a number.")

        if not isinstance(latent_standard_deviation, (int, float)):
            raise TypeError("latent_standard_deviation must be a number.")

        if latent_standard_deviation <= 0:
            raise ValueError("latent_standard_deviation must be greater than 0.")

        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")

        # Initialize the encoder and decoder models
        self._encoder = encoder_model
        self._decoder = decoder_model

        # loss function and metric for tracking total loss
        self._loss_function = loss_function
        self._total_loss_tracker = 0.0
        self._latent_mean_distribution = latent_mean_distribution
        self._latent_standard_deviation = latent_standard_deviation
        self._latent_dimension = latent_dimension


        # File names for saving models
        self._file_name_encoder = file_name_encoder
        self._file_name_decoder = file_name_decoder

        # Path for saving models
        self._models_saved_path = models_saved_path

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._encoder.to(self.device)
        self._decoder.to(self.device)

    def compile(self, loss=None, optimizer=None, metrics=None, **kwargs):
        """
        Compile the autoencoder model (PyTorch-compatible version).

        This method mimics Keras' compile() for compatibility with existing code,
        but PyTorch doesn't require compilation. It just stores the loss function.

        Args:
            loss: loss function (can be string name or nn.Module instance)
            optimizer: Optimizer (not used in PyTorch compile, pass to fit instead)
            metrics: metrics to track (not implemented yet)
            **kwargs: Additional arguments (ignored)
        """
        if loss is not None:
            if isinstance(loss, str):
                # Convert string loss names to PyTorch loss functions
                loss_map = {
                    'mse': nn.MSELoss(),
                    'mean_squared_error': nn.MSELoss(),
                    'mae': nn.L1Loss(),
                    'mean_absolute_error': nn.L1Loss(),
                    'bce': nn.BCELoss(),
                    'binary_crossentropy': nn.BCELoss(),
                    'crossentropy': nn.CrossEntropyLoss(),
                    'categorical_crossentropy': nn.CrossEntropyLoss(),
                }
                loss_lower = loss.lower()
                if loss_lower in loss_map:
                    self._loss_function = loss_map[loss_lower]
                else:
                    raise ValueError(f"Unknown loss function: {loss}")
            elif isinstance(loss, nn.Module):
                self._loss_function = loss
            else:
                raise TypeError("loss must be a string or nn.Module instance")

        return self

    def forward(self, x, labels=None):
        """
        Forward pass through the encoder-decoder model.

        Args:
            x: Input tensor or tuple of (data, labels).
            labels: Label tensor (optional, for conditional models).

        Returns:
            Reconstructed output tensor.
        """
        # Handle both tuple input and separate data/labels
        if isinstance(x, (tuple, list)) and len(x) == 2:
            data_input, label_input = x
        elif labels is not None:
            data_input = x
            label_input = labels
        else:
            # Non-conditional case: just pass data through
            data_input = x
            label_input = None

        # Encoder forward pass
        if label_input is not None:
            latent, _ = self._encoder([data_input, label_input])
            # Decoder forward pass
            reconstructed = self._decoder([latent, label_input])
        else:
            latent = self._encoder(data_input)
            reconstructed = self._decoder(latent)

        return reconstructed

    def train_step(self, batch, optimizer):
        """
        Perform a training step for the conditional AutoEncoder.

        This implementation assumes the encoder and decoder are ALWAYS conditional,
        meaning they expect [data, labels] as input.

        Args:
            batch: Input data batch (tuple of batch_x, batch_y).
                   - batch_x: Input data features
                   - batch_y: Target reconstruction concatenated with one-hot labels [data | labels]
            optimizer: PyTorch optimizer.

        Returns:
            dict: Dictionary containing the loss value.
        """
        batch_x, batch_y = batch

        # Move data to device
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)

        # Set model to training mode
        self.train()

        # Zero the gradients
        optimizer.zero_grad()

        # The encoder/decoder in this implementation are ALWAYS conditional
        # They always expect [data, labels] as input

        # Check if batch_y is larger than batch_x (indicates concatenated data + labels)
        # Check if batch_y is larger than batch_x (indicates concatenated data + labels)
        if batch_y.shape[1] > batch_x.shape[1]:
            # Extract target data (first part, same size as input)
            target_data = batch_y[:, :batch_x.shape[1]]
            # Extract one-hot labels (remaining part)
            one_hot_labels = batch_y[:, batch_x.shape[1]:]
        else:
            # batch_y contains only labels, use batch_x as reconstruction target
            target_data = batch_x  #  Use input as target for autoencoder
            # Assume batch_y is already one-hot encoded labels
            one_hot_labels = batch_y

        # Forward pass: encoder expects [data, labels], returns (latent, labels)
        latent_representation, labels_passthrough = self._encoder([batch_x, one_hot_labels])

        # Decoder expects [latent, labels]
        reconstructed_data = self._decoder([latent_representation, one_hot_labels])

        # Calculate the loss
        if self._loss_function is not None:
            update_gradient_loss = self._loss_function(reconstructed_data, target_data)
        else:
            # Default to MSE if no loss function is set
            update_gradient_loss = torch.mean(torch.square(target_data - reconstructed_data))

        # Backward pass
        update_gradient_loss.backward()

        # Update weights
        optimizer.step()

        # Update the total loss tracker
        self._total_loss_tracker = update_gradient_loss.item()

        # Return a dictionary containing the current loss value
        return {"loss": self._total_loss_tracker}


    def fit(self, x=None, y=None, batch_size=32, epochs=1, verbose=1,
            callbacks=None, validation_data=None, shuffle=True,
            initial_epoch=0, steps_per_epoch=None, validation_steps=None,
            validation_freq=1, optimizer=None, learning_rate=0.001, **kwargs):
        """
        Train the model with a simplified progress bar.

        Args:
            x: Input data (array, tensor, tuple of (x,y), or DataLoader).
            y: Target data.
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
            optimizer: PyTorch optimizer (if None, Adam is used).
            learning_rate: Learning rate for optimizer.

        Returns:
            A History object with training metrics.
        """

        # Create default optimizer if none provided
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Handle different input formats
        if isinstance(x, DataLoader):
            dataloader = x

        elif isinstance(x, tuple) and len(x) == 2:
            x_data, y_data = x
            x_tensor = torch.FloatTensor(x_data) if not isinstance(x_data, torch.Tensor) else x_data
            y_tensor = torch.FloatTensor(y_data) if not isinstance(y_data, torch.Tensor) else y_data
            dataset = TensorDataset(x_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        elif x is not None and y is not None:
            x_tensor = torch.FloatTensor(x) if not isinstance(x, torch.Tensor) else x
            y_tensor = torch.FloatTensor(y) if not isinstance(y, torch.Tensor) else y
            dataset = TensorDataset(x_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        else:
            raise ValueError("Invalid input format. Provide either a DataLoader, (x, y) tuple, or x and y separately.")

        # Calculate steps per epoch if not provided
        if steps_per_epoch is None:
            steps_per_epoch = len(dataloader)

        # History to store metrics
        history = self.History()
        history['loss'] = []

        # Training loop
        for epoch in range(initial_epoch, epochs):
            self._total_loss_tracker = 0.0

            if verbose == 1:
                print(f'\nEpoch {epoch + 1}/{epochs}')

            # Progress tracking
            step = 0
            epoch_loss = 0.0

            for batch_data in dataloader:
                step += 1

                # Perform training step
                metrics = self.train_step(batch_data, optimizer)
                current_loss = metrics['loss']
                epoch_loss += current_loss

                # Simple progress bar
                if verbose == 1:
                    progress = int(50 * step / steps_per_epoch)
                    bar = '=' * progress + '>' + '.' * (50 - progress - 1)
                    print(f'\r[{bar}] {step}/{steps_per_epoch} - loss: {current_loss:.4f}',
                          end='', flush=True)

                if step >= steps_per_epoch:
                    break

            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / step if step > 0 else 0.0
            history['loss'].append(avg_epoch_loss)

            if verbose == 1:
                print(f' - loss: {avg_epoch_loss:.4f}')

            elif verbose == 2:
                print(f'Epoch {epoch + 1}/{epochs} - loss: {avg_epoch_loss:.4f}')

            # Validation
            if validation_data is not None and (epoch + 1) % validation_freq == 0:

                val_loss = self._evaluate_validation(validation_data, validation_steps)

                if 'val_loss' not in history.history:
                    history['val_loss'] = []

                history['val_loss'].append(val_loss)

                if verbose >= 1:
                    print(f' - val_loss: {val_loss:.4f}')

            # callbacks
            if callbacks is not None:
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, {'loss': avg_epoch_loss})

        return history

    def _evaluate_validation(self, validation_data, validation_steps=None):
        """
        Evaluate the model on validation data.

        Args:
            validation_data: Validation dataset (DataLoader or tuple).
            validation_steps: Number of validation steps.

        Returns:
            Average validation loss.
        """

        # Set model to evaluation mode
        self.eval()

        # Handle different validation data formats
        if isinstance(validation_data, DataLoader):
            val_dataloader = validation_data

        elif isinstance(validation_data, tuple) and len(validation_data) == 2:
            x_val, y_val = validation_data
            x_tensor = torch.FloatTensor(x_val) if not isinstance(x_val, torch.Tensor) else x_val
            y_tensor = torch.FloatTensor(y_val) if not isinstance(y_val, torch.Tensor) else y_val
            dataset = TensorDataset(x_tensor, y_tensor)
            val_dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        else:
            return 0.0

        val_losses = []
        step = 0

        with torch.no_grad():
            for batch_data in val_dataloader:
                batch_x, batch_y = batch_data
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Extract target data and labels (same logic as train_step)
                if batch_y.shape[1] > batch_x.shape[1]:
                    target_data = batch_y[:, :batch_x.shape[1]]
                    one_hot_labels = batch_y[:, batch_x.shape[1]:]
                else:
                    target_data = batch_x
                    one_hot_labels = batch_y

                # Forward pass
                latent_representation, _ = self._encoder([batch_x, one_hot_labels])
                reconstructed = self._decoder([latent_representation, one_hot_labels])

                # Calculate loss
                if self._loss_function is not None:
                    loss = self._loss_function(reconstructed, target_data)
                else:
                    loss = torch.mean(torch.square(target_data - reconstructed))

                val_losses.append(loss.item())

                step += 1
                if validation_steps is not None and step >= validation_steps:
                    break

        # Set back to training mode
        self.train()

        return numpy.mean(val_losses) if val_losses else 0.0

    def get_samples(self, number_samples_per_class):
        """
        Generates synthetic data samples for each specified class using the trained decoder.
        This function creates synthetic samples conditioned on class labels, typically used
        when working with conditional generative models (like conditional VAEs or conditional GANs).

        Args:
            number_samples_per_class (dict):
                A dictionary specifying how many synthetic samples should be generated per class.
                Expected structure:
                {
                    "classes": {class_label: number_of_samples, ...},
                    "number_classes": total_number_of_classes
                }

        Returns:
            dict:
                A dictionary where each key is a class label and the value is an array of generated samples.
                Each array contains the synthetic samples generated for the corresponding class.
        """

        # Set model to evaluation mode
        self.eval()

        # Initialize an empty dictionary to store generated samples grouped by class label
        generated_data = {}

        with torch.no_grad():
            # Loop through each class label and the corresponding number of samples to generate
            for label_class, number_instances in number_samples_per_class["classes"].items():
                # Create a batch of one-hot encoded class labels, all set to the current class
                label_samples_generated = F.one_hot(
                    torch.tensor([label_class] * number_instances),
                    num_classes=number_samples_per_class["number_classes"]
                ).float().to(self.device)

                # Generate random noise vectors (latent space vectors) for each sample
                latent_noise = torch.normal(
                    mean=self._latent_mean_distribution,
                    std=self._latent_standard_deviation,
                    size=(number_instances, self._latent_dimension)
                ).to(self.device)

                # Use the decoder to generate synthetic samples from the latent space and class labels
                generated_samples = self._decoder([latent_noise, label_samples_generated])

                # Convert to numpy and store
                generated_data[label_class] = generated_samples.cpu().numpy()

        # Return the dictionary containing all generated samples, organized by class
        return generated_data

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
        torch.save({
            'model_state_dict': self._encoder.state_dict(),
            'model_architecture': str(self._encoder)
        }, encoder_file_name)

        # Save decoder model
        torch.save({
            'model_state_dict': self._decoder.state_dict(),
            'model_architecture': str(self._decoder)
        }, decoder_file_name)

    def load_models(self, directory, file_name):
        """
        Load the encoder and decoder models from a directory.

        Args:
            directory (str): Directory where models are stored.
            file_name (str): Base file name for loading models.
        """

        # Construct file names for encoder and decoder models
        encoder_file_name = os.path.join(directory, f"{file_name}_encoder.pth")
        decoder_file_name = os.path.join(directory, f"{file_name}_decoder.pth")

        # Load encoder
        encoder_checkpoint = torch.load(encoder_file_name, map_location=self.device)
        self._encoder.load_state_dict(encoder_checkpoint['model_state_dict'])

        # Load decoder
        decoder_checkpoint = torch.load(decoder_file_name, map_location=self.device)
        self._decoder.load_state_dict(decoder_checkpoint['model_state_dict'])

        # Set to evaluation mode
        self._encoder.eval()
        self._decoder.eval()

    @property
    def decoder(self):
        return self._decoder

    @property
    def encoder(self):
        return self._encoder

    @decoder.setter
    def decoder(self, decoder):
        self._decoder = decoder
        self._decoder.to(self.device)

    @encoder.setter
    def encoder(self, encoder):
        self._encoder = encoder
        self._encoder.to(self.device)

    class History:
        """
        History object for Keras compatibility.
        Stores training history and provides access via .history attribute.
        """

        def __init__(self):
            self.history = {}

        def __getitem__(self, key):
            return self.history[key]

        def __setitem__(self, key, value):
            self.history[key] = value

        def keys(self):
            return self.history.keys()