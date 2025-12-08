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
    from torch.utils.data import DataLoader, TensorDataset

except ImportError as error:
    print(error)
    sys.exit(-1)


class VariationalAlgorithmTorch(nn.Module):
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
        @_total_loss_tracker (float):
            Tracks the overall loss during training.
        @_reconstruction_loss_tracker (float):
            Tracks the reconstruction loss during training.
        @_kl_loss_tracker (float):
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
        >>> vae_model = VariationalAlgorithm(
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
        >>> vae_model.train_step(data)
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

        super().__init__(*args, **kwargs)
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
        # Initialize the encoder and decoder models
        self._encoder = encoder_model
        self._decoder = decoder_model

        # Loss function and metrics for tracking losses
        self._loss_function = loss_function
        self._total_loss_tracker = 0.0
        self._reconstruction_loss_tracker = 0.0
        self._kl_loss_tracker = 0.0
        self._latent_mean_distribution = latent_mean_distribution
        self._latent_stander_deviation = latent_stander_deviation
        self._latent_dimension = latent_dimension
        self._decoder_latent_dimension = decoder_latent_dimension

        # File names for saving models
        self._file_name_encoder = file_name_encoder
        self._file_name_decoder = file_name_decoder

        # Path for saving models
        self._models_saved_path = models_saved_path

        # Optimizer will be configured later
        self.optimizer = None
        self.configure_optimizer()

    def train_step(self, batch):
        """
        Perform a training step for the Variational AutoEncoder (VAE).

        Args:
            batch: Input data batch.

        Returns:
            dict: Dictionary containing the loss values (total loss, reconstruction loss, KL divergence loss).
        """
        batch_x, batch_y = batch

        # Move to device
        device = next(self.parameters()).device
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass: Encode input data and sample from the latent space
        latent_mean, latent_log_variation, latent, label = self._encoder(batch_x)

        # Decode the sampled latent space and generate reconstructed data
        reconstruction_data = self._decoder(latent, label)

        # Calculate binary cross-entropy loss for reconstruction
        binary_cross_entropy_loss = F.binary_cross_entropy(reconstruction_data, batch_y, reduction='none')
        sum_reduced = binary_cross_entropy_loss
        reconstruction_loss = torch.mean(sum_reduced)

        # Calculate KL divergence loss
        encoder_output = (1 + latent_log_variation - torch.square(latent_mean))
        kl_divergence_loss = -0.5 * (encoder_output - torch.exp(latent_log_variation))
        kl_divergence_loss = torch.mean(torch.sum(kl_divergence_loss, dim=1))

        # Total loss is the sum of reconstruction loss and KL divergence loss
        loss_model_in_reconstruction = reconstruction_loss + kl_divergence_loss

        # Backward pass
        loss_model_in_reconstruction.backward()

        # Update weights
        self.optimizer.step()

        # Update loss metrics
        self._total_loss_tracker = loss_model_in_reconstruction.item()
        self._reconstruction_loss_tracker = reconstruction_loss.item()
        self._kl_loss_tracker = kl_divergence_loss.item()

        # Return a dictionary containing the current loss values
        return {"loss": self._total_loss_tracker,
                "reconstruction_loss": self._reconstruction_loss_tracker,
                "kl_loss": self._kl_loss_tracker}

    def configure_optimizer(self,
                            learning_rate=0.001,
                            beta_1=0.9,
                            beta_2=0.999,
                            epsilon=1e-7,
                            amsgrad=False,
                            weight_decay=1e-5):
        """
        Configure the Adam optimizer with custom parameters.

        Args:
            learning_rate: Learning rate
            beta_1: Adam beta1 parameter
            beta_2: Adam beta2 parameter
            epsilon: Adam epsilon
            amsgrad: Whether to use AMSGrad variant
            weight_decay: L2 regularization factor
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

    def create_embedding(self, data):
        """
        Generates latent space embeddings using the trained encoder.

        Args:
            data (ndarray): Input data to encode.

        Returns:
            ndarray: Latent space representations.
        """
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            if isinstance(data, numpy.ndarray):
                data = torch.from_numpy(data).float().to(device)

            latent_mean, _, _, _ = self._encoder(data)

        return latent_mean.cpu().numpy()

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
        generated_data = {}

        with torch.no_grad():
            for label_class, number_instances in number_samples_per_class["classes"].items():
                # Create one-hot encoded label array
                label_samples_generated = torch.zeros(number_instances, number_samples_per_class["number_classes"])
                label_samples_generated[:, label_class] = 1
                label_samples_generated = label_samples_generated.to(device)

                # Sample random latent vectors from a standard normal distribution
                latent_noise = torch.randn(number_instances, self._decoder_latent_dimension).to(device)

                # Use the decoder to generate samples
                generated_samples = self._decoder(latent_noise, label_samples_generated)

                # Round the generated samples to the nearest integer
                generated_samples = torch.round(generated_samples)

                # Store the generated samples
                generated_data[label_class] = generated_samples.cpu().numpy()

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

        with torch.no_grad():
            # Generate random noise samples in the latent space
            random_noise_generate = torch.randn(
                number_samples_generate,
                latent_dimension,
                device=device
            ) * self._latent_stander_deviation + self._latent_mean_distribution

            # Create label vectors for the generated data
            label_list = torch.full((number_samples_generate, 1), labels, dtype=torch.float32, device=device)

            # Generate synthetic data by passing random noise and labels through the decoder
            synthetic_data = self._decoder(random_noise_generate, label_list)

        return synthetic_data

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
        device = next(self.parameters()).device

        # Construct file names for encoder and decoder models
        encoder_file_name = os.path.join(directory, f"fold_{file_name}_encoder.pth")
        decoder_file_name = os.path.join(directory, f"fold_{file_name}_decoder.pth")

        # Load the encoder and decoder models
        self._encoder.load_state_dict(torch.load(encoder_file_name, map_location=device))
        self._decoder.load_state_dict(torch.load(decoder_file_name, map_location=device))

    def compile(self, loss, optimizer):
        """
        Configure the model for training (PyTorch compatibility method).

        Args:
            loss: Loss function (ignored, using built-in loss)
            optimizer: Optimizer to use for training
        """
        self.optimizer = optimizer

    def fit(self, train_data, y_data, epochs, batch_size, callbacks=None):
        """
        Train the VAE model.

        Args:
            train_data: Tuple of (x_train, y_labels) or x_train
            y_data: Target data (reconstruction target)
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: List of callbacks (optional)
        """
        device = next(self.parameters()).device
        self.train()

        # Unpack training data
        if isinstance(train_data, tuple):
            x_train, y_labels = train_data
        else:
            x_train = train_data
            y_labels = None

        # Convert to tensors
        if isinstance(x_train, numpy.ndarray):
            x_train = torch.from_numpy(x_train).float()
        if isinstance(y_data, numpy.ndarray):
            y_data = torch.from_numpy(y_data).float()
        if y_labels is not None and isinstance(y_labels, numpy.ndarray):
            y_labels = torch.from_numpy(y_labels).float()

        # Create dataset and dataloader
        if y_labels is not None:
            dataset = TensorDataset(x_train, y_labels, y_data)
        else:
            dataset = TensorDataset(x_train, y_data)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        for epoch in range(epochs):
            epoch_losses = []

            for batch in dataloader:
                if y_labels is not None:
                    batch_x, batch_y_labels, batch_y = batch
                    batch_x = torch.cat([batch_x, batch_y_labels], dim=1) if batch_y_labels.dim() > 1 else batch_x
                else:
                    batch_x, batch_y = batch

                loss_dict = self.train_step((batch_x, batch_y))
                epoch_losses.append(loss_dict)

            # Calculate average losses for epoch
            avg_loss = sum(d['loss'] for d in epoch_losses) / len(epoch_losses)
            avg_recon = sum(d['reconstruction_loss'] for d in epoch_losses) / len(epoch_losses)
            avg_kl = sum(d['kl_loss'] for d in epoch_losses) / len(epoch_losses)

            print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f} - "
                  f"reconstruction_loss: {avg_recon:.4f} - kl_loss: {avg_kl:.4f}")

            # Execute callbacks
            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, {'loss': avg_loss})

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