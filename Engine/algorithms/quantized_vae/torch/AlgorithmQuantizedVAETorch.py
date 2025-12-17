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
    import time
    import numpy
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

except ImportError as error:
    print(error)
    sys.exit(-1)


class QuantizedVAEAlgorithmTorch:
    """
    Implements a Vector Quantized Variational Autoencoder (VQ-VAE) with adaptive input handling.

    This model combines an encoder, decoder, and vector quantization layer to learn compressed
    representations of input data. It automatically adapts to any input shape: (x), (x, y), (x, y, z), etc.

    The algorithm supports training with reconstruction loss and commitment loss for
    the quantization layer, enabling stable training of discrete latent variables.

    References:
        - van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). "Neural Discrete
        Representation Learning." Advances in Neural Information Processing Systems (NeurIPS).
        Available at: https://arxiv.org/abs/1711.00937

    Example:
        >>> # Example with 1D data
        >>> vq_vae_1d = QuantizedVAEAlgorithmTorch(
        ...     encoder_model=encoder_1d,
        ...     decoder_model=decoder_1d,
        ...     quantized_vae_model=vq_vae_1d,
        ...     train_variance=0.1,
        ...     latent_dimension=64,
        ...     number_embeddings=512,
        ...     input_shape=(100,)
        ... )
        >>> vq_vae_1d.fit(data_1d, epochs=50)

        >>> # Example with 2D data (images)
        >>> vq_vae_2d = QuantizedVAEAlgorithmTorch(
        ...     encoder_model=encoder_2d,
        ...     decoder_model=decoder_2d,
        ...     quantized_vae_model=vq_vae_2d,
        ...     train_variance=0.1,
        ...     latent_dimension=128,
        ...     number_embeddings=512,
        ...     input_shape=(28, 28)
        ... )
        >>> vq_vae_2d.fit(data_2d, labels_2d, epochs=100)
    """

    def __init__(self,
                 encoder_model,
                 decoder_model,
                 quantized_vae_model,
                 train_variance,
                 latent_dimension,
                 number_embeddings,
                 file_name_encoder=None,
                 file_name_decoder=None,
                 models_saved_path=None,
                 input_shape=None,
                 auto_adapt_shape=True,
                 **kwargs):
        """
        Initializes the QuantizedVAEAlgorithm with encoder, decoder, and VQ-VAE components.

        Args:
            encoder_model: Encoder network that compresses input data to latent space
            decoder_model: Decoder network that reconstructs data from quantized latent codes
            quantized_vae_model: Complete VQ-VAE model including quantization layer
            train_variance: Data variance used to scale reconstruction loss
            latent_dimension: Dimensionality of latent space before quantization
            number_embeddings: Size of quantization codebook
            file_name_encoder: Filename for saving encoder weights
            file_name_decoder: Filename for saving decoder weights
            models_saved_path: Directory path for model weight storage
            input_shape: Expected input shape (without batch dimension)
            auto_adapt_shape: Whether to automatically adapt to input data shape
        """
        self._train_variance = train_variance
        self._latent_dimension = latent_dimension
        self._number_embeddings = number_embeddings

        self._encoder = encoder_model
        self._decoder = decoder_model
        self._quantized_vae_model = quantized_vae_model

        self._file_name_encoder = file_name_encoder
        self._file_name_decoder = file_name_decoder
        self._models_saved_path = models_saved_path

        # Shape adaptation
        self._input_shape = input_shape
        self._auto_adapt_shape = auto_adapt_shape
        self._inferred_shape = None

        # Loss trackers
        self.total_loss_tracker = 0.0
        self.reconstruction_loss_tracker = 0.0
        self.vq_loss_tracker = 0.0

        # Training state
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._quantized_vae_model.to(self.device)

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

        if isinstance(data, torch.Tensor):
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
            tuple: (batch_x, batch_y) formatted for VQ-VAE training.
        """
        if isinstance(batch, (tuple, list)):
            if len(batch) == 1:
                # Single input
                batch_x = batch[0]
                batch_y = batch[0]  # Autoencoder: reconstruct input
            elif len(batch) == 2:
                # Check if it's (input, target) or (input, labels)
                x, y = batch
                # VQ-VAE typically uses input as both x and y for reconstruction
                # If y has different shape than x, treat it as labels
                if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
                    if x.shape == y.shape:
                        batch_x = [x, y]
                        batch_y = y
                    else:
                        # y is labels, x is both input and target
                        batch_x = [x, y]
                        batch_y = x
                else:
                    batch_x = [x, y]
                    batch_y = x
            elif len(batch) == 3:
                # (input, labels, target)
                batch_x = [batch[0], batch[1]]
                batch_y = batch[2]
            else:
                # Multiple inputs
                batch_x = list(batch)
                batch_y = batch[0]
        else:
            # Single tensor
            batch_x = [batch, batch]
            batch_y = batch

        return batch_x, batch_y

    @property
    def metrics(self):
        """
        Returns the list of metrics tracked during training.

        Returns:
            dict: Dictionary with loss metrics.
        """
        return {
            "total_loss": self.total_loss_tracker,
            "reconstruction_loss": self.reconstruction_loss_tracker,
            "vq_loss": self.vq_loss_tracker
        }

    def compile(self, optimizer):
        """
        Configures the model for training.

        Args:
            optimizer: PyTorch optimizer instance
        """
        self.optimizer = optimizer

    def train_step(self, data):
        """
        Performs a single training step on a batch of data with automatic batch handling.

        Args:
            data: Input data batch.

        Returns:
            dict: Dictionary with loss metrics for the current step.
        """
        # Prepare batch
        x, y = self._prepare_batch(data)

        # Extract output tensor (for VQ-VAE, it's typically the first element)
        if isinstance(x, (tuple, list)):
            output_tensor = x[0]
        else:
            output_tensor = x

        # Move to device
        if isinstance(x, list):
            x = [t.to(self.device) if isinstance(t, torch.Tensor) else t for t in x]
        else:
            x = x.to(self.device)

        output_tensor = output_tensor.to(self.device)
        y = y.to(self.device)

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass through VQ-VAE
        reconstructions = self._quantized_vae_model(x)

        # Compute reconstruction loss (MSE normalized by data variance)
        reconstruction_loss = F.mse_loss(reconstructions, output_tensor) / self._train_variance

        # Total loss is reconstruction loss plus VQ losses (commitment + codebook)
        vq_losses = sum(self._quantized_vae_model.losses) if hasattr(self._quantized_vae_model, 'losses') else 0
        total_loss = reconstruction_loss + vq_losses

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        # Update metric trackers
        self.total_loss_tracker = total_loss.item()
        self.reconstruction_loss_tracker = reconstruction_loss.item()
        self.vq_loss_tracker = vq_losses.item() if isinstance(vq_losses, torch.Tensor) else vq_losses

        # Return results
        return {
            "loss": self.total_loss_tracker,
            "reconstruction_loss": self.reconstruction_loss_tracker,
            "vqvae_loss": self.vq_loss_tracker,
        }

    def fit(self, x=None, y=None, batch_size=32, epochs=1, verbose=1,
            callbacks=None, validation_data=None, shuffle=True,
            initial_epoch=0, steps_per_epoch=None, validation_steps=None,
            validation_freq=1, optimizer=None, learning_rate=0.001, **kwargs):
        """
        Train the model with automatic shape adaptation.

        Args:
            x: Input data (any shape) or DataLoader.
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

        # Set optimizer if provided
        if optimizer is not None:
            self.optimizer = optimizer
        elif not hasattr(self, 'optimizer') or self.optimizer is None:
            # Create default optimizer if none exists
            self.optimizer = torch.optim.Adam(self._quantized_vae_model.parameters(),
                                              lr=learning_rate)

        # Prepare the dataset
        if isinstance(x, DataLoader):
            train_dataloader = x
            # Try to infer shape from dataset
            for batch in train_dataloader:
                self._validate_and_adapt_shape(batch)
                break
        else:
            # Validate and adapt shape
            self._validate_and_adapt_shape(x)

            # Handle different input formats
            if isinstance(x, tuple):
                # x is tuple: (data, labels) or (data, target)
                if len(x) == 2:
                    x_data, target_or_labels = x
                else:
                    x_data = x[0]
                    target_or_labels = None
            else:
                x_data = x
                target_or_labels = y

            # Convert to tensors
            if isinstance(x_data, numpy.ndarray):
                x_data = torch.FloatTensor(x_data)

            if target_or_labels is not None:
                if isinstance(target_or_labels, numpy.ndarray):
                    target_or_labels = torch.FloatTensor(target_or_labels)
                # Create dataset with both
                dataset = TensorDataset(x_data, target_or_labels)
            else:
                # Use x_data as both input and target (autoencoder behavior)
                dataset = TensorDataset(x_data, x_data)

            train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        # Calculate steps per epoch if not provided
        if steps_per_epoch is None:
            try:
                steps_per_epoch = len(train_dataloader)
            except:
                steps_per_epoch = 100  # Default fallback

        # Initialize callbacks
        if callbacks:
            for callback in callbacks:
                if not hasattr(callback, 'params') or callback.params is None:
                    callback.params = {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "steps": steps_per_epoch,
                        "samples": len(train_dataloader.dataset) if hasattr(train_dataloader, 'dataset') else 0,
                        "verbose": verbose,
                        "metrics": ["loss", "reconstruction_loss", "vqvae_loss"]
                    }

                if not hasattr(callback, 'data') or callback.data is None:
                    callback.data = {"start_time": time.time()}
                elif "start_time" not in callback.data:
                    callback.data["start_time"] = time.time()

                if hasattr(callback, 'on_train_begin'):
                    callback.on_train_begin()

        # History to store metrics
        history = {'loss': [], 'reconstruction_loss': [], 'vqvae_loss': []}

        # Training loop
        self._quantized_vae_model.train()

        for epoch in range(initial_epoch, epochs):
            # Reset loss trackers
            self.total_loss_tracker = 0.0
            self.reconstruction_loss_tracker = 0.0
            self.vq_loss_tracker = 0.0

            if verbose == 1:
                print(f'\nEpoch {epoch + 1}/{epochs}')

            # Progress tracking
            step = 0
            for batch_data in train_dataloader:
                step += 1

                # Perform training step
                metrics = self.train_step(batch_data)
                current_loss = metrics['loss']
                current_recon_loss = metrics['reconstruction_loss']
                current_vq_loss = metrics['vqvae_loss']

                # Simple progress bar
                if verbose == 1:
                    progress = int(50 * step / steps_per_epoch)
                    bar = '=' * progress + '>' + '.' * (50 - progress - 1)
                    print(
                        f'\r[{bar}] {step}/{steps_per_epoch} - loss: {current_loss:.4f} - recon: {current_recon_loss:.4f} - vq: {current_vq_loss:.4f}',
                        end='', flush=True)

                if step >= steps_per_epoch:
                    break

            # Store epoch losses
            epoch_loss = self.total_loss_tracker
            epoch_reconstruction_loss = self.reconstruction_loss_tracker
            epoch_vq_loss = self.vq_loss_tracker

            history['loss'].append(epoch_loss)
            history['reconstruction_loss'].append(epoch_reconstruction_loss)
            history['vqvae_loss'].append(epoch_vq_loss)

            if verbose == 1:
                print(f' - loss: {epoch_loss:.4f} - recon: {epoch_reconstruction_loss:.4f} - vq: {epoch_vq_loss:.4f}')
            elif verbose == 2:
                print(f'Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.4f}')

            # Validation
            if validation_data is not None and (epoch + 1) % validation_freq == 0:
                val_loss = self._evaluate_validation(validation_data, validation_steps)
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
                            'reconstruction_loss': epoch_reconstruction_loss,
                            'vqvae_loss': epoch_vq_loss
                        })

        # Call on_train_end
        if callbacks:
            for callback in callbacks:
                if hasattr(callback, 'on_train_end'):
                    callback.on_train_end()

        # Return history object
        class History:
            def __init__(self, history_dict):
                self.history = history_dict

        return History(history)

    def _evaluate_validation(self, validation_data, validation_steps=None):
        """
        Evaluate the model on validation data with automatic shape handling.

        Args:
            validation_data: Validation dataset.
            validation_steps: Number of validation steps.

        Returns:
            Average validation loss.
        """
        self._quantized_vae_model.eval()
        val_losses = []
        step = 0

        # Prepare validation dataset
        if isinstance(validation_data, DataLoader):
            val_dataloader = validation_data
        elif isinstance(validation_data, tuple) and len(validation_data) == 2:
            val_x, val_y = validation_data
            if isinstance(val_x, numpy.ndarray):
                val_x = torch.FloatTensor(val_x)
            if isinstance(val_y, numpy.ndarray):
                val_y = torch.FloatTensor(val_y)
            val_dataset = TensorDataset(val_x, val_y)
            val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        else:
            raise ValueError("validation_data must be a DataLoader or tuple of (x_val, y_val)")

        with torch.no_grad():
            for batch_data in val_dataloader:
                batch_x, batch_y = self._prepare_batch(batch_data)

                # Get output tensor
                if isinstance(batch_x, (tuple, list)):
                    output_tensor = batch_x[0]
                else:
                    output_tensor = batch_x

                # Move to device
                if isinstance(batch_x, list):
                    batch_x = [t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch_x]
                else:
                    batch_x = batch_x.to(self.device)

                output_tensor = output_tensor.to(self.device)

                reconstructed = self._quantized_vae_model(batch_x)
                loss = F.mse_loss(reconstructed, output_tensor)
                val_losses.append(loss.item())

                step += 1
                if validation_steps is not None and step >= validation_steps:
                    break

        self._quantized_vae_model.train()
        return numpy.mean(val_losses) if val_losses else 0.0

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
        elif isinstance(data, torch.Tensor):
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

    def get_samples(self, number_samples_per_class):
        """
        Generates samples from the latent space using the decoder.

        Samples are generated by:
        1. Randomly selecting indices from the codebook
        2. Gathering corresponding latent vectors
        3. Decoding these vectors conditioned on class labels

        Args:
            number_samples_per_class (dict): Dictionary with 'classes' and 'number_classes' keys

        Returns:
            dict: Generated samples keyed by class label.
        """
        self._quantized_vae_model.eval()
        generated_data = {}

        # Get codebook embeddings from the quantization layer
        codebook = self._quantized_vae_model.get_layer("vector_quantizer").embeddings.weight.data
        number_embeddings = self._number_embeddings

        with torch.no_grad():
            for label_class, number_instances in number_samples_per_class["classes"].items():
                # Create one-hot encoded labels for the samples
                label_samples_generated = torch.zeros(number_instances,
                                                      number_samples_per_class["number_classes"])
                label_samples_generated[:, label_class] = 1
                label_samples_generated = label_samples_generated.to(self.device)

                # Randomly sample from codebook
                sampled_indices = numpy.random.choice(number_embeddings, size=number_instances)

                # Get corresponding latent vectors
                quantized_vectors = codebook[sampled_indices].to(self.device)

                # Decode the latent vectors
                generated_samples = self._decoder([quantized_vectors, label_samples_generated])

                generated_data[label_class] = generated_samples.cpu().numpy()

        return generated_data

    def save_model(self, directory, file_name):
        """
        Save the encoder and decoder models with shape information.

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
            'number_embeddings': self._number_embeddings,
            'train_variance': self._train_variance
        }
        shape_file = os.path.join(directory, f"fold_{file_name}_shape_info.json")
        with open(shape_file, 'w') as f:
            json.dump(shape_info, f)

    def load_models(self, directory, file_name):
        """
        Load the encoder and decoder models with shape information.

        Args:
            directory (str): Directory where models are stored.
            file_name (str): Base file name for loading models.
        """
        # Construct file names for encoder and decoder models
        encoder_file_name = os.path.join(directory, f"fold_{file_name}_encoder.pth")
        decoder_file_name = os.path.join(directory, f"fold_{file_name}_decoder.pth")

        # Load the encoder and decoder models
        self._encoder.load_state_dict(torch.load(encoder_file_name))
        self._decoder.load_state_dict(torch.load(decoder_file_name))

        self._encoder.to(self.device)
        self._decoder.to(self.device)

        # Load shape information if available
        shape_file = os.path.join(directory, f"fold_{file_name}_shape_info.json")
        if os.path.exists(shape_file):
            with open(shape_file, 'r') as f:
                shape_info = json.load(f)
                self._input_shape = tuple(shape_info.get('input_shape', ()))
                self._inferred_shape = tuple(shape_info.get('inferred_shape', ()))
                self._latent_dimension = shape_info.get('latent_dimension', self._latent_dimension)
                self._number_embeddings = shape_info.get('number_embeddings', self._number_embeddings)
                self._train_variance = shape_info.get('train_variance', self._train_variance)

    # ========== PROPERTIES ==========

    @property
    def encoder(self):
        """Get the encoder model."""
        return self._encoder

    @encoder.setter
    def encoder(self, value):
        """Set the encoder model."""
        self._encoder = value

    @property
    def decoder(self):
        """Get the decoder model."""
        return self._decoder

    @decoder.setter
    def decoder(self, value):
        """Set the decoder model."""
        self._decoder = value

    @property
    def quantized_vae_model(self):
        """Get the complete VQ-VAE model."""
        return self._quantized_vae_model

    @quantized_vae_model.setter
    def quantized_vae_model(self, value):
        """Set the complete VQ-VAE model."""
        self._quantized_vae_model = value

    @property
    def train_variance(self):
        """Get the training variance."""
        return self._train_variance

    @train_variance.setter
    def train_variance(self, value):
        """Set the training variance."""
        if value <= 0:
            raise ValueError("Train variance must be positive")
        self._train_variance = value

    @property
    def latent_dimension(self):
        """Get the latent dimension."""
        return self._latent_dimension

    @latent_dimension.setter
    def latent_dimension(self, value):
        """Set the latent dimension."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Latent dimension must be a positive integer")
        self._latent_dimension = value

    @property
    def number_embeddings(self):
        """Get the number of embeddings."""
        return self._number_embeddings

    @number_embeddings.setter
    def number_embeddings(self, value):
        """Set the number of embeddings."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Number of embeddings must be a positive integer")
        self._number_embeddings = value

    @property
    def input_shape(self):
        """Get the current input shape."""
        return self.get_input_shape()

    @property
    def file_name_encoder(self):
        """Get the encoder filename."""
        return self._file_name_encoder

    @file_name_encoder.setter
    def file_name_encoder(self, value):
        """Set the encoder filename."""
        self._file_name_encoder = value

    @property
    def file_name_decoder(self):
        """Get the decoder filename."""
        return self._file_name_decoder

    @file_name_decoder.setter
    def file_name_decoder(self, value):
        """Set the decoder filename."""
        self._file_name_decoder = value

    @property
    def models_saved_path(self):
        """Get the models save path."""
        return self._models_saved_path

    @models_saved_path.setter
    def models_saved_path(self, value):
        """Set the models save path."""
        self._models_saved_path = value