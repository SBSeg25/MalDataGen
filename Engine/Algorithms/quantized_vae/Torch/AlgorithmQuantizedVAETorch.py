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
    Implements a Vector Quantized Variational autoencoder (VQ-VAE) for discrete latent
    representation learning and generation. This model combines an encoder, decoder,
    and vector quantization layer to learn compressed representations of input data.

    The algorithm supports training with reconstruction loss and commitment loss for
    the quantization layer, enabling stable training of discrete latent variables.

    Attributes:
        @train_variance (float):
            Variance of the training data used to normalize the reconstruction loss.
        @latent_dimension (int):
            Dimensionality of the latent space before quantization.
        @number_embeddings (int):
            Number of embeddings in the vector quantization codebook.
        @encoder (nn.Module):
            Encoder network that maps input data to latent representations.
        @decoder (nn.Module):
            Decoder network that reconstructs data from quantized latent codes.
        @quantized_vae_model (nn.Module):
            Complete VQ-VAE model combining encoder, quantization, and decoder.
        @file_name_encoder (str):
            Filename for saving the encoder weights.
        @file_name_decoder (str):
            Filename for saving the decoder weights.
        @models_saved_path (str):
            Directory path for saving model weights.
        @total_loss_tracker (float):
            Metric tracker for total training loss (reconstruction + VQ losses).
        @reconstruction_loss_tracker (float):
            Metric tracker for reconstruction loss component.
        @vq_loss_tracker (float):
            Metric tracker for vector quantization loss components.

    References:
        - van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). "Neural Discrete
        Representation Learning." Advances in Neural Information Processing Systems (NeurIPS).
        Available at: https://arxiv.org/abs/1711.00937

    Example:
        >>> vae = QuantizedVAEAlgorithm(
        ...     encoder_model=encoder,
        ...     decoder_model=decoder,
        ...     quantized_vae_model=vq_vae,
        ...     train_variance=0.1,
        ...     latent_dimension=64,
        ...     number_embeddings=512,
        ...     file_name_encoder="encoder_weights.pth",
        ...     file_name_decoder="decoder_weights.pth",
        ...     models_saved_path="./saved_models/"
        ... )
        >>> vae.compile(optimizer=torch.optim.Adam(vae.parameters()))
        >>> vae.fit(train_dataset, epochs=10)
        >>> generated_samples = vae.get_samples({
        ...     "classes": {0: 5, 1: 5},
        ...     "number_classes": 2
        ... })
    """

    def __init__(self,
                 encoder_model,
                 decoder_model,
                 quantized_vae_model,
                 train_variance,
                 latent_dimension,
                 number_embeddings,
                 file_name_encoder,
                 file_name_decoder,
                 models_saved_path,
                 **kwargs):
        """
        Initializes the QuantizedVAEAlgorithm with encoder, decoder, and VQ-VAE components.

        Args:
            @encoder_model (nn.Module):
                Encoder network that compresses input data to latent space.
            @decoder_model (nn.Module):
                Decoder network that reconstructs data from quantized latent codes.
            @quantized_vae_model (nn.Module):
                Complete VQ-VAE model including quantization layer.
            @train_variance (float):
                Data variance used to scale reconstruction loss.
            @latent_dimension (int):
                Dimensionality of latent space before quantization.
            @number_embeddings (int):
                Size of quantization codebook (number of discrete latent codes).
            @file_name_encoder (str):
                Filename for saving encoder weights.
            @file_name_decoder (str):
                Filename for saving decoder weights.
            @models_saved_path (str):
                Directory path for model weight storage.
            @**kwargs:
                Additional keyword arguments.

        Raises:
            ValueError:
                If latent_dimension <= 0.
                If number_embeddings <= 0.
                If train_variance <= 0.
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

        # Loss trackers
        self.total_loss_tracker = 0.0
        self.reconstruction_loss_tracker = 0.0
        self.vq_loss_tracker = 0.0

        # Training state
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._quantized_vae_model.to(self.device)

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
        Performs a single training step on a batch of data.

        The training step includes:
        1. Forward pass through the VQ-VAE
        2. Loss computation (reconstruction + VQ losses)
        3. Gradient computation and weight updates

        Args:
            data (tuple): Input data tuple containing (input_tensor, labels).

        Returns:
            dict: Dictionary with loss metrics for the current step.
        """
        x, y = data
        output_tensor, labels = x

        # Move to device
        output_tensor = output_tensor.to(self.device)
        labels = labels.to(self.device)

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass
        reconstructions = self._quantized_vae_model([output_tensor, labels])

        # Compute reconstruction loss (MSE normalized by data variance)
        reconstruction_loss = F.mse_loss(reconstructions, output_tensor) / self._train_variance

        # Total loss includes VQ losses
        vq_losses = sum(self._quantized_vae_model.losses)
        total_loss = reconstruction_loss + vq_losses

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        # Update trackers
        self.total_loss_tracker = total_loss.item()
        self.reconstruction_loss_tracker = reconstruction_loss.item()
        self.vq_loss_tracker = vq_losses.item()

        return {
            "loss": self.total_loss_tracker,
            "reconstruction_loss": self.reconstruction_loss_tracker,
            "vqvae_loss": self.vq_loss_tracker
        }

    def fit(self, x=None, y=None, batch_size=32, epochs=1, verbose=1,
            callbacks=None, validation_data=None, shuffle=True,
            initial_epoch=0, steps_per_epoch=None, validation_steps=None,
            validation_freq=1, optimizer=None, learning_rate=0.001, **kwargs):
        """
        Train the model with a simplified progress bar.

        Args:
            x: Input data (tuple of (input_tensor, labels)) or DataLoader.
            y: Target data.
            batch_size: Number of samples per gradient update.
            epochs: Number of epochs to train.
            verbose: 0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks: List of callbacks to apply during training.
            validation_data: Data for validation (tuple or DataLoader).
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
        else:
            if y is None:
                y = x[0] if isinstance(x, tuple) else x

            # Extract input data and labels
            if isinstance(x, tuple):
                input_data, labels = x
            else:
                input_data = x
                labels = torch.zeros(len(input_data), 1)  # Dummy labels

            # Convert to tensors if needed
            if not isinstance(input_data, torch.Tensor):
                input_data = torch.FloatTensor(input_data)
            if not isinstance(labels, torch.Tensor):
                labels = torch.FloatTensor(labels)
            if not isinstance(y, torch.Tensor):
                y = torch.FloatTensor(y)

            # Create dataset
            dataset = TensorDataset(input_data, labels, y)
            train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        # Calculate steps per epoch if not provided
        if steps_per_epoch is None:
            steps_per_epoch = len(train_dataloader)

        # Prepare validation data if provided
        val_dataloader = None
        if validation_data is not None:
            if isinstance(validation_data, DataLoader):
                val_dataloader = validation_data
            else:
                val_x, val_y = validation_data
                if isinstance(val_x, tuple):
                    val_input, val_labels = val_x
                else:
                    val_input = val_x
                    val_labels = torch.zeros(len(val_input), 1)

                if not isinstance(val_input, torch.Tensor):
                    val_input = torch.FloatTensor(val_input)
                if not isinstance(val_labels, torch.Tensor):
                    val_labels = torch.FloatTensor(val_labels)
                if not isinstance(val_y, torch.Tensor):
                    val_y = torch.FloatTensor(val_y)

                val_dataset = TensorDataset(val_input, val_labels, val_y)
                val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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

            epoch_losses = {"loss": [], "reconstruction_loss": [], "vqvae_loss": []}

            if verbose == 1:
                print(f'\nEpoch {epoch + 1}/{epochs}')

            # Progress tracking
            step = 0
            for batch_data in train_dataloader:
                step += 1

                # Unpack batch data
                if len(batch_data) == 3:
                    batch_input, batch_labels, batch_target = batch_data
                    data = ((batch_input, batch_labels), batch_target)
                else:
                    data = batch_data

                # Perform training step
                metrics = self.train_step(data)
                current_loss = metrics['loss']

                # Collect metrics
                epoch_losses['loss'].append(current_loss)
                epoch_losses['reconstruction_loss'].append(metrics['reconstruction_loss'])
                epoch_losses['vqvae_loss'].append(metrics['vqvae_loss'])

                # Simple progress bar
                if verbose == 1:
                    progress = int(50 * step / steps_per_epoch)
                    bar = '=' * progress + '>' + '.' * (50 - progress - 1)
                    print(f'\r[{bar}] {step}/{steps_per_epoch} - loss: {current_loss:.4f}',
                          end='', flush=True)

                if step >= steps_per_epoch:
                    break

            # Store epoch loss (average)
            epoch_loss = numpy.mean(epoch_losses['loss'])
            epoch_reconstruction_loss = numpy.mean(epoch_losses['reconstruction_loss'])
            epoch_vq_loss = numpy.mean(epoch_losses['vqvae_loss'])

            history['loss'].append(epoch_loss)
            history['reconstruction_loss'].append(epoch_reconstruction_loss)
            history['vqvae_loss'].append(epoch_vq_loss)

            if verbose == 1:
                print(f' - loss: {epoch_loss:.4f}')
            elif verbose == 2:
                print(f'Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.4f}')

            # Validation
            if val_dataloader is not None and (epoch + 1) % validation_freq == 0:
                val_loss = self._evaluate_validation(val_dataloader, validation_steps)
                if 'val_loss' not in history:
                    history['val_loss'] = []
                history['val_loss'].append(val_loss)

                if verbose >= 1:
                    print(f' - val_loss: {val_loss:.4f}')

            # Callbacks
            if callbacks is not None:
                for callback in callbacks:
                    # Ensure callback params are initialized
                    if not hasattr(callback, 'params') or callback.params is None:
                        callback.params = {
                            "epochs": epochs,
                            "batch_size": batch_size,
                            "steps": steps_per_epoch
                        }

                    if not hasattr(callback, 'data') or callback.data is None:
                        callback.data = {"start_time": time.time()}

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
        Evaluate the model on validation data.

        Args:
            validation_data: Validation DataLoader.
            validation_steps: Number of validation steps.

        Returns:
            Average validation loss.
        """
        self._quantized_vae_model.eval()
        val_losses = []
        step = 0

        with torch.no_grad():
            for batch_data in validation_data:
                # Unpack batch data
                if len(batch_data) == 3:
                    batch_input, batch_labels, batch_target = batch_data
                else:
                    batch_input, batch_labels = batch_data[:2]
                    batch_target = batch_input

                # Move to device
                batch_input = batch_input.to(self.device)
                batch_labels = batch_labels.to(self.device)
                batch_target = batch_target.to(self.device)

                # Forward pass
                reconstructed = self._quantized_vae_model([batch_input, batch_labels])

                # Compute loss
                loss = F.mse_loss(reconstructed, batch_target)
                val_losses.append(loss.item())

                step += 1
                if validation_steps is not None and step >= validation_steps:
                    break

        self._quantized_vae_model.train()
        return numpy.mean(val_losses) if val_losses else 0.0

    def get_samples(self, number_samples_per_class):
        """
        Generates samples from the latent space using the decoder.

        Samples are generated by:
        1. Randomly selecting indices from the codebook
        2. Gathering corresponding latent vectors
        3. Decoding these vectors conditioned on class labels

        Args:
            number_samples_per_class (dict):
                Dictionary specifying samples to generate per class with structure:
                {
                    "classes": {class_label: num_samples, ...},
                    "number_classes": total_num_classes
                }

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

                # Sample random indices from codebook
                sampled_indices = numpy.random.choice(number_embeddings, size=number_instances)

                # Get corresponding latent vectors
                quantized_vectors = codebook[sampled_indices].to(self.device)

                # Decode the latent vectors
                generated_samples = self._decoder([quantized_vectors, label_samples_generated])

                # Convert to numpy and round
                generated_samples = generated_samples.cpu().numpy()
                generated_samples = numpy.rint(generated_samples)

                generated_data[label_class] = generated_samples

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
        torch.save(self._encoder.state_dict(), encoder_file_name)

        # Save decoder model
        torch.save(self._decoder.state_dict(), decoder_file_name)

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

        # Load the encoder and decoder models
        self._encoder.load_state_dict(torch.load(encoder_file_name))
        self._decoder.load_state_dict(torch.load(decoder_file_name))

        self._encoder.to(self.device)
        self._decoder.to(self.device)