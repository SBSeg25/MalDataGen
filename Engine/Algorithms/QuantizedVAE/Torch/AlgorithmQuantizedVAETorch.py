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

    def fit(self, x, y, epochs, batch_size, callbacks=None):
        """
        Trains the VQ-VAE model.

        Args:
            x: Tuple of (input_data, labels)
            y: Target data (same as input for autoencoder)
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            callbacks (list): List of callback functions
        """
        input_data, labels = x

        # Convert to tensors
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.FloatTensor(input_data)
        if not isinstance(labels, torch.Tensor):
            labels = torch.FloatTensor(labels)
        if not isinstance(y, torch.Tensor):
            y = torch.FloatTensor(y)

        # Create dataset and dataloader
        dataset = TensorDataset(input_data, labels, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize callbacks before training
        if callbacks:
            for callback in callbacks:
                # Initialize params attribute with training configuration
                if not hasattr(callback, 'params') or callback.params is None:
                    callback.params = {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "steps": len(dataloader),
                        "samples": len(dataset),
                        "verbose": 1,
                        "metrics": ["loss", "reconstruction_loss", "vqvae_loss"]
                    }

                # Initialize data attribute
                if not hasattr(callback, 'data') or callback.data is None:
                    callback.data = {"start_time": time.time()}
                elif "start_time" not in callback.data:
                    callback.data["start_time"] = time.time()

                # Call on_train_begin if it exists
                if hasattr(callback, 'on_train_begin'):
                    callback.on_train_begin()

        # Training loop
        self._quantized_vae_model.fit()

        for epoch in range(epochs):
            epoch_losses = {"loss": [], "reconstruction_loss": [], "vqvae_loss": []}

            for batch_idx, (batch_input, batch_labels, batch_target) in enumerate(dataloader):
                # Prepare data
                data = ((batch_input, batch_labels), batch_target)

                # Training step
                metrics = self.train_step(data)

                # Collect metrics
                for key in epoch_losses:
                    epoch_losses[key].append(metrics[key])

            # Average epoch metrics
            avg_loss = numpy.mean(epoch_losses["loss"])
            avg_recon = numpy.mean(epoch_losses["reconstruction_loss"])
            avg_vq = numpy.mean(epoch_losses["vqvae_loss"])

            print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f} - "
                  f"reconstruction_loss: {avg_recon:.4f} - vqvae_loss: {avg_vq:.4f}")

            # Execute callbacks
            if callbacks:
                for callback in callbacks:
                    # Ensure callback params and data are still initialized (defensive)
                    if not hasattr(callback, 'params') or callback.params is None:
                        callback.params = {
                            "epochs": epochs,
                            "batch_size": batch_size,
                            "steps": len(dataloader),
                            "samples": len(dataset)
                        }

                    if not hasattr(callback, 'data') or callback.data is None:
                        callback.data = {"start_time": time.time()}

                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, {
                            "loss": avg_loss,
                            "reconstruction_loss": avg_recon,
                            "vqvae_loss": avg_vq
                        })

        # Call on_train_end if it exists
        if callbacks:
            for callback in callbacks:
                if hasattr(callback, 'on_train_end'):
                    callback.on_train_end()
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