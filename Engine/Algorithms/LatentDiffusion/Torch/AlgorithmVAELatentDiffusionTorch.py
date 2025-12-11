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
        @_encoder (nn.Module):
            Encoder model that encodes input data into the latent space.
        @_decoder (nn.Module):
            Decoder model that reconstructs data from the latent representation.
        @_loss_function (callable):
            Function used to compute the total loss during training.
        @_total_loss (float):
            Tracks the overall loss during training.
        @_reconstruction_loss (float):
            Tracks the reconstruction loss during training.
        @_kl_loss (float):
            Tracks the KL divergence loss during training.
        @_latent_mean_distribution (float):
            Mean of the latent distribution.
        @_latent_stander_deviation (float):
            Standard deviation of the latent distribution.
        @_latent_dimension (int):
            Dimensionality of the latent space.
        @_decoder_latent_dimension (int):
            Dimensionality of the latent space used by the decoder.
        @_file_name_encoder (str):
            File name for saving the encoder model.
        @_file_name_decoder (str):
            File name for saving the decoder model.
        @_models_saved_path (str):
            Directory path where the encoder and decoder models are saved.

    Raises:
        ValueError:
            Raised in cases where:
            - The latent dimension is non-positive.
            - The standard deviation of the latent space is non-positive.
            - The file paths are invalid.

    Example:
        >>> vae_model = VAELatentDiffusionAlgorithmPyTorch(
        ...     encoder_model=encoder,
        ...     decoder_model=decoder,
        ...     loss_function=custom_loss_function,
        ...     latent_dimension=128,
        ...     latent_mean_distribution=0.0,
        ...     latent_stander_deviation=1.0,
        ...     file_name_encoder="encoder_model.pth",
        ...     file_name_decoder="decoder_model.pth",
        ...     models_saved_path="models/"
        ... )
        >>> optimizer = torch.optim.Adam(vae_model.parameters(), lr=0.001)
        >>> loss_dict = vae_model.train_step(data, optimizer)
    """

    def __init__(self,
                 encoder_model,
                 decoder_model,
                 loss_function,
                 latent_dimension,
                 decoder_latent_dimension,
                 latent_mean_distribution,
                 latent_stander_deviation,
                 file_name_encoder,
                 file_name_decoder,
                 models_saved_path,
                 *args,
                 **kwargs):
        """
        Initializes the VariationalAlgorithm model with provided encoder and decoder models,
        loss function, and latent space parameters.

        This constructor sets up the architecture, metrics, and paths for saving the models.

        Args:
            @encoder_model (nn.Module):
                The encoder model responsible for encoding input data into latent variables.
            @decoder_model (nn.Module):
                The decoder model responsible for reconstructing data from the latent space.
            @loss_function (callable):
                The loss function used to compute the training loss.
            @latent_dimension (int):
                The dimensionality of the latent space.
            @decoder_latent_dimension (int):
                The dimensionality of the latent space used by the decoder.
            @latent_mean_distribution (float):
                The mean of the latent distribution (usually 0).
            @latent_stander_deviation (float):
                The standard deviation of the latent distribution (usually 1).
            @file_name_encoder (str):
                The filename for saving the encoder model.
            @file_name_decoder (str):
                The filename for saving the decoder model.
            @models_saved_path (str):
                The directory where the models will be saved.
            @*args:
                Additional arguments for the parent class.
            @**kwargs:
                Additional keyword arguments for the parent class.

        Raises:
            ValueError:
                If latent_dimension <= 0.
                If latent_stander_deviation <= 0.
                If file paths are invalid.
        """
        super().__init__(*args, **kwargs)

        # Initialize the encoder and decoder models
        self._encoder = encoder_model
        self._decoder = decoder_model

        # Loss function and metrics for tracking losses
        self._loss_function = loss_function

        # Initialize loss trackers
        self._total_loss = 0.0
        self._reconstruction_loss = 0.0
        self._kl_loss = 0.0
        self._loss_count = 0

        self._latent_mean_distribution = latent_mean_distribution
        self._latent_stander_deviation = latent_stander_deviation
        self._latent_dimension = latent_dimension
        self._decoder_latent_dimension = decoder_latent_dimension

        # File names for saving models
        self._file_name_encoder = file_name_encoder
        self._file_name_decoder = file_name_decoder

        # Path for saving models
        self._models_saved_path = models_saved_path

    def train_step(self, batch, labels, optimizer):
        """
        Perform a training step for the Variational AutoEncoder (VAE).

        Args:
            batch: Tuple of (batch_x, batch_target) - input data and reconstruction target.
            labels: One-hot encoded class labels for conditioning.
            optimizer: PyTorch optimizer for updating model parameters.

        Returns:
            dict: Dictionary containing the loss values (total loss, reconstruction loss, KL divergence loss).
        """
        batch_x, batch_target = batch

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
        binary_cross_entropy_loss = F.binary_cross_entropy(reconstruction_data, batch_target, reduction='none')
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
                dummy_labels = torch.zeros(num_samples,
                                           self._encoder._encoder_number_samples_per_class["number_classes"]).to(device)
                latent_mean, _, _, _ = self._encoder(batch, dummy_labels)
                embeddings.append(latent_mean.cpu().numpy())

        return numpy.concatenate(embeddings, axis=0)

    def get_samples(self, number_samples_per_class):
        """
        Generate synthetic samples for each specified class using the trained decoder.

        This function generates samples by sampling from a normal distribution in the latent space
        and conditioning the generation process on class labels.

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
                Each array contains the synthetic samples generated for the corresponding class.
        """
        self.eval()
        device = next(self.parameters()).device

        # Initialize a dictionary to store the generated samples for each class
        generated_data = {}

        # Iterate over each class and the corresponding number of samples to generate
        for label_class, number_instances in number_samples_per_class["classes"].items():
            # Create a one-hot encoded label array for all samples in the current class
            label_samples_generated = F.one_hot(
                torch.tensor([label_class] * number_instances, dtype=torch.long),
                num_classes=number_samples_per_class["number_classes"]
            ).float().to(device)

            # Sample random latent vectors from a standard normal distribution
            latent_noise = torch.randn(number_instances, self._decoder_latent_dimension).to(device)

            # Use the decoder to generate samples conditioned on the latent vectors and class labels
            with torch.no_grad():
                generated_samples = self._decoder(latent_noise, label_samples_generated)

            # Round the generated samples to the nearest integer
            generated_samples = torch.round(generated_samples).cpu().numpy()

            # Store the generated samples in the dictionary under the corresponding class label
            generated_data[label_class] = generated_samples

        # Return the dictionary with all generated samples, organized by class
        return generated_data

    def generate_synthetic_data(self, number_samples_generate, labels, latent_dimension):
        """
        Generate synthetic data using the Variational AutoEncoder (VAE).

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
        ) * self._latent_stander_deviation + self._latent_mean_distribution

        # Create label vectors for the generated data
        label_list = torch.full(
            (number_samples_generate, 1),
            labels,
            dtype=torch.float32,
            device=device
        )

        # Generate synthetic data by passing random noise and labels through the decoder
        with torch.no_grad():
            synthetic_data = self._decoder(random_noise_generate, label_list)

        # Return the generated synthetic data
        return synthetic_data

    def get_metrics(self):
        """
        Returns:
            dict: Dictionary of averaged metrics tracked during training.
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
        Save the encoder and decoder models in PyTorch format.

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

        print(f"Models saved to {directory}")

    def load_models(self, directory, file_name):
        """
        Load the encoder and decoder models from a directory.

        Args:
            directory (str): Directory where models are stored.
            file_name (str): Base file name for loading models.
        """
        # Construct file names for encoder and decoder models
        encoder_file_name = os.path.join(directory, f"fold_{file_name}_encoder.pth")
        decoder_file_name = os.path.join(directory, f"fold_{file_name}_decoder.pth")

        # Load encoder
        encoder_checkpoint = torch.load(encoder_file_name)
        self._encoder.load_state_dict(encoder_checkpoint['model_state_dict'])

        # Load decoder
        decoder_checkpoint = torch.load(decoder_file_name)
        self._decoder.load_state_dict(decoder_checkpoint['model_state_dict'])

        print(f"Models loaded from {directory}")

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
        return self._latent_stander_deviation