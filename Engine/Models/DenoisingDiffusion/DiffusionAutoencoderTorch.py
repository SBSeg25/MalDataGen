#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/07'
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
    import sys
    import logging
    import torch
    import torch.nn as nn
    import json
    from typing import Tuple, List, Optional
    from Engine.Activations.ActivationsTorch import ActivationsTorch

except ImportError as error:
    logging.error(error)
    sys.exit(-1)

DEFAULT_DIFFUSION_AUTOENCODER_LOSS = 'mse'
DEFAULT_DIFFUSION_AUTOENCODER_ENCODER_FILTERS = [320, 160]
DEFAULT_DIFFUSION_AUTOENCODER_DECODER_FILTERS = [160, 320]
DEFAULT_DIFFUSION_AUTOENCODER_LAST_LAYER_ACTIVATION = 'sigmoid'
DEFAULT_DIFFUSION_AUTOENCODER_LATENT_DIMENSION = 64
DEFAULT_DIFFUSION_AUTOENCODER_BATCH_SIZE_CREATE_EMBEDDING = 128
DEFAULT_DIFFUSION_AUTOENCODER_BATCH_SIZE_TRAINING = 64
DEFAULT_DIFFUSION_AUTOENCODER_INTERMEDIARY_ACTIVATION = 'swish'
DEFAULT_DIFFUSION_AUTOENCODER_INTERMEDIARY_ACTIVATION_ALPHA = 0.05
DEFAULT_DIFFUSION_AUTOENCODER_ACTIVATION_OUTPUT_ENCODER = 'sigmoid'


class EncoderModule(nn.Module):
    """
    Encoder module for the Diffusion Autoencoder.
    """

    def __init__(self, input_shape: int, latent_dimension: int,
                 encoder_neurons: List[int], intermediary_activation: str,
                 activation_output_encoder: str, intermediary_activation_alpha: float,
                 activation_helper):
        super(EncoderModule, self).__init__()

        self.input_shape = input_shape
        self.latent_dimension = latent_dimension
        self.activation_helper = activation_helper

        # Build encoder layers
        layers = []
        current_dim = input_shape

        for neurons in encoder_neurons:
            layers.append(nn.Linear(current_dim, neurons))
            layers.append(activation_helper.get_activation(intermediary_activation,
                                                           intermediary_activation_alpha))
            current_dim = neurons

        # Latent layer
        layers.append(nn.Linear(current_dim, latent_dimension))
        layers.append(activation_helper.get_activation(activation_output_encoder,
                                                       intermediary_activation_alpha))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)

        # Pass through encoder
        encoded = self.encoder(x)

        # Reshape to add spatial dimension (batch, latent_dim, 1)
        return encoded.unsqueeze(-1)


class DecoderModule(nn.Module):
    """
    Decoder module for the Diffusion Autoencoder.
    """

    def __init__(self, latent_dimension: int, output_shape: int,
                 decoder_neurons: List[int], intermediary_activation: str,
                 last_activation: str, intermediary_activation_alpha: float,
                 activation_helper):
        super(DecoderModule, self).__init__()

        self.output_shape = output_shape
        self.latent_dimension = latent_dimension
        self.activation_helper = activation_helper

        # Build decoder layers
        layers = []

        # Initial layer from latent space
        layers.append(nn.Linear(latent_dimension, latent_dimension))
        layers.append(activation_helper.get_activation(intermediary_activation,
                                                       intermediary_activation_alpha))

        current_dim = latent_dimension

        # Hidden layers
        for neurons in decoder_neurons:
            layers.append(nn.Linear(current_dim, neurons))
            layers.append(activation_helper.get_activation(intermediary_activation,
                                                           intermediary_activation_alpha))
            current_dim = neurons

        # Output layer
        layers.append(nn.Linear(current_dim, output_shape))
        layers.append(activation_helper.get_activation(last_activation,
                                                       intermediary_activation_alpha))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten input (batch, latent_dim, 1) -> (batch, latent_dim)
        x = x.view(x.size(0), -1)

        # Pass through decoder
        return self.decoder(x)


class DiffusionAutoencoderModelTorch(ActivationsTorch):
    """
    A LatentDiffusion Autoencoder model (PyTorch version) that combines an
    encoder-decoder architecture with diffusion-based latent space learning.
    This model enables flexible configuration of activation functions, layer
    structures, and loss functions, making it suitable for various generative
    and representation learning tasks.

    The autoencoder follows a variational approach, where the latent space
    undergoes a diffusion process to enhance feature disentanglement. This
    technique has been explored in deep generative models to improve the
    quality of generated data while preserving meaningful representations.

    References:
        - Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models."
          Advances in Neural Information Processing Systems (NeurIPS).
          Available at: https://arxiv.org/abs/2006.11239

    Example:
        >>> model = DiffusionAutoencoderModelTorch(
        ...     input_shape=128,
        ...     latent_dimension=32,
        ...     number_encoder_neurons=[256, 128, 64],
        ...     number_decoder_neurons=[64, 128, 256],
        ...     loss_function="mse",
        ...     last_activation="sigmoid",
        ...     activation_output_encoder="relu",
        ...     batch_size_create_embedding=32,
        ...     intermediary_activation_autoencoder="leaky_relu",
        ...     intermediary_activation_alpha=0.2
        ... )
    """

    def __init__(self, input_shape: int,
                 latent_dimension: int = DEFAULT_DIFFUSION_AUTOENCODER_LATENT_DIMENSION,
                 number_encoder_neurons: Optional[List[int]] = None,
                 number_decoder_neurons: Optional[List[int]] = None,
                 loss_function: str = DEFAULT_DIFFUSION_AUTOENCODER_LOSS,
                 last_activation: str = DEFAULT_DIFFUSION_AUTOENCODER_LAST_LAYER_ACTIVATION,
                 activation_output_encoder: str = DEFAULT_DIFFUSION_AUTOENCODER_ACTIVATION_OUTPUT_ENCODER,
                 batch_size_create_embedding: int = DEFAULT_DIFFUSION_AUTOENCODER_BATCH_SIZE_TRAINING,
                 intermediary_activation_autoencoder: str = DEFAULT_DIFFUSION_AUTOENCODER_INTERMEDIARY_ACTIVATION,
                 intermediary_activation_alpha: float = DEFAULT_DIFFUSION_AUTOENCODER_INTERMEDIARY_ACTIVATION_ALPHA):
        """
        Initializes the DiffusionAutoencoderModelTorch with user-defined architecture
        and training parameters.

        Args:
            input_shape (int): Number of features in the input data.
            latent_dimension (int): Size of the latent space (bottleneck layer).
            number_encoder_neurons (List[int]): Number of neurons per encoder layer.
            number_decoder_neurons (List[int]): Number of neurons per decoder layer.
            loss_function (str): Loss function used during training.
            last_activation (str): Activation function for the final decoder layer.
            activation_output_encoder (str): Activation function for the encoder output.
            batch_size_create_embedding (int): Batch size used for embedding generation.
            intermediary_activation_autoencoder (str): Activation function for intermediate layers.
            intermediary_activation_alpha (float): Parameter for activations requiring an alpha value.
        """
        super(DiffusionAutoencoderModelTorch, self).__init__()

        if number_decoder_neurons is None:
            number_decoder_neurons = DEFAULT_DIFFUSION_AUTOENCODER_DECODER_FILTERS

        if number_encoder_neurons is None:
            number_encoder_neurons = DEFAULT_DIFFUSION_AUTOENCODER_ENCODER_FILTERS

        self._decoder_model_loaded = None
        self._encoder_model_loaded = None
        self._input_shape = input_shape
        self._latent_dimension = latent_dimension
        self._list_number_neurons_per_layer_encoder = number_encoder_neurons
        self._list_number_neurons_per_layer_decoder = number_decoder_neurons
        self._intermediary_activation_function = intermediary_activation_autoencoder
        self._intermediary_activation_alpha = intermediary_activation_alpha
        self._activation_output_encoder = activation_output_encoder
        self._last_activation = last_activation
        self._loss_function = loss_function
        self._batch_size_create_embedding = batch_size_create_embedding
        self._neural_model = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build_autoencoder(self):
        """
        Builds the full autoencoder model by combining the encoder and decoder.
        """
        self._encoder_model_loaded = EncoderModule(
            self._input_shape,
            self._latent_dimension,
            self._list_number_neurons_per_layer_encoder,
            self._intermediary_activation_function,
            self._activation_output_encoder,
            self._intermediary_activation_alpha,
            self
        ).to(self._device)

        self._decoder_model_loaded = DecoderModule(
            self._latent_dimension,
            self._input_shape,
            self._list_number_neurons_per_layer_decoder,
            self._intermediary_activation_function,
            self._last_activation,
            self._intermediary_activation_alpha,
            self
        ).to(self._device)

        # Create full autoencoder as sequential model
        class FullAutoencoder(nn.Module):
            def __init__(self, encoder, decoder):
                super(FullAutoencoder, self).__init__()
                self.encoder = encoder
                self.decoder = decoder

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded

        self._neural_model = FullAutoencoder(
            self._encoder_model_loaded,
            self._decoder_model_loaded
        ).to(self._device)

    def load_model(self, model_path: str):
        """
        Loads the autoencoder model from a saved state dict.

        Args:
            model_path (str): Path to the saved model file (.pth or .pt).
        """
        if self._neural_model is None:
            self.build_autoencoder()

        self._neural_model.load_state_dict(torch.load(model_path, map_location=self._device))
        self._neural_model.eval()

    def save_model(self, model_path: str):
        """
        Saves the autoencoder model state dict.

        Args:
            model_path (str): Path where to save the model.
        """
        if self._neural_model is None:
            raise ValueError("Model has not been built yet.")

        torch.save(self._neural_model.state_dict(), model_path)

    def get_encoder_and_decoder(self) -> Tuple[nn.Module, nn.Module]:
        """
        Builds and retrieves the encoder and decoder models from the autoencoder.

        Returns:
            Tuple[nn.Module, nn.Module]: (Encoder model, Decoder model)

        Raises:
            ValueError: If the autoencoder model has not been built.
        """
        if self._neural_model is None:
            self.build_autoencoder()

        return self._encoder_model_loaded, self._decoder_model_loaded

    def training(self, x_train_data, epochs: int, batch_size: int,
                 learning_rate: float = 0.001, validation_split: float = 0.2):
        """
        Compiles and trains the autoencoder model.

        Args:
            x_train_data: Training data (torch.Tensor or numpy array).
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for optimizer.
            validation_split (float): Fraction of data to use for validation.
        """
        if self._neural_model is None:
            self.build_autoencoder()

        # Convert to tensor if needed
        if not isinstance(x_train_data, torch.Tensor):
            x_train_data = torch.FloatTensor(x_train_data)

        x_train_data = x_train_data.to(self._device)

        # Split into train and validation
        n_samples = len(x_train_data)
        n_val = int(n_samples * validation_split)
        indices = torch.randperm(n_samples)

        train_indices = indices[n_val:]
        val_indices = indices[:n_val]

        x_train = x_train_data[train_indices]
        x_val = x_train_data[val_indices]

        # Setup loss and optimizer
        if self._loss_function.lower() == 'mse':
            criterion = nn.MSELoss()
        elif self._loss_function.lower() == 'bce':
            criterion = nn.BCELoss()
        else:
            criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(self._neural_model.parameters(), lr=learning_rate)

        # Training loop
        self._neural_model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            # Mini-batch training
            for i in range(0, len(x_train), batch_size):
                batch = x_train[i:i + batch_size]

                optimizer.zero_grad()
                outputs = self._neural_model(batch)
                loss = criterion(outputs, batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Validation
            self._neural_model.eval()
            with torch.no_grad():
                val_outputs = self._neural_model(x_val)
                val_loss = criterion(val_outputs, x_val)
            self._neural_model.train()

            avg_train_loss = epoch_loss / n_batches
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss.item():.6f}")

    def create_embedding(self, data):
        """
        Generates latent space embeddings using the trained encoder.

        Args:
            data: Input data to encode (torch.Tensor or numpy array).

        Returns:
            numpy.ndarray: Latent space representations.
        """
        if self._encoder_model_loaded is None:
            raise ValueError("Encoder has not been built yet.")

        # Convert to tensor if needed
        if not isinstance(data, torch.Tensor):
            data = torch.FloatTensor(data)

        data = data.to(self._device)

        self._encoder_model_loaded.eval()
        with torch.no_grad():
            embeddings = []
            for i in range(0, len(data), self._batch_size_create_embedding):
                batch = data[i:i + self._batch_size_create_embedding]
                batch_embedding = self._encoder_model_loaded(batch)
                embeddings.append(batch_embedding.cpu().numpy())

        return torch.cat([torch.from_numpy(e) for e in embeddings], dim=0).numpy()

    # Properties with getters and setters
    @property
    def input_shape(self) -> int:
        """Get the input shape of the model."""
        return self._input_shape

    @input_shape.setter
    def input_shape(self, value: int):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Input shape must be a positive integer")
        self._input_shape = value

    @property
    def latent_dimension(self) -> int:
        """Get the latent dimension size."""
        return self._latent_dimension

    @latent_dimension.setter
    def latent_dimension(self, value: int):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Latent dimension must be a positive integer")
        self._latent_dimension = value

    @property
    def encoder_neurons(self) -> List[int]:
        """Get the list of neurons per encoder layer."""
        return self._list_number_neurons_per_layer_encoder

    @encoder_neurons.setter
    def encoder_neurons(self, value: List[int]):
        if not isinstance(value, list) or not all(isinstance(x, int) and x > 0 for x in value):
            raise ValueError("Encoder neurons must be a list of positive integers")
        self._list_number_neurons_per_layer_encoder = value

    @property
    def decoder_neurons(self) -> List[int]:
        """Get the list of neurons per decoder layer."""
        return self._list_number_neurons_per_layer_decoder

    @decoder_neurons.setter
    def decoder_neurons(self, value: List[int]):
        if not isinstance(value, list) or not all(isinstance(x, int) and x > 0 for x in value):
            raise ValueError("Decoder neurons must be a list of positive integers")
        self._list_number_neurons_per_layer_decoder = value

    @property
    def intermediary_activation(self) -> str:
        """Get the intermediary activation function."""
        return self._intermediary_activation_function

    @intermediary_activation.setter
    def intermediary_activation(self, value: str):
        if not isinstance(value, str):
            raise ValueError("Intermediary activation must be a string")
        self._intermediary_activation_function = value

    @property
    def intermediary_activation_alpha(self) -> float:
        """Get the alpha parameter for intermediary activation."""
        return self._intermediary_activation_alpha

    @intermediary_activation_alpha.setter
    def intermediary_activation_alpha(self, value: float):
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError("Intermediary activation alpha must be a non-negative number")
        self._intermediary_activation_alpha = value

    @property
    def encoder_output_activation(self) -> str:
        """Get the encoder output activation function."""
        return self._activation_output_encoder

    @encoder_output_activation.setter
    def encoder_output_activation(self, value: str):
        if not isinstance(value, str):
            raise ValueError("Encoder output activation must be a string")
        self._activation_output_encoder = value

    @property
    def last_activation(self) -> str:
        """Get the last layer activation function."""
        return self._last_activation

    @last_activation.setter
    def last_activation(self, value: str):
        if not isinstance(value, str):
            raise ValueError("Last activation must be a string")
        self._last_activation = value

    @property
    def loss_function(self) -> str:
        """Get the loss function."""
        return self._loss_function

    @loss_function.setter
    def loss_function(self, value: str):
        if not isinstance(value, str):
            raise ValueError("Loss function must be a string")
        self._loss_function = value

    @property
    def batch_size_create_embedding(self) -> int:
        """Get the batch size for embedding creation."""
        return self._batch_size_create_embedding

    @batch_size_create_embedding.setter
    def batch_size_create_embedding(self, value: int):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Batch size must be a positive integer")
        self._batch_size_create_embedding = value

    @property
    def neural_model(self):
        """Get the neural model."""
        return self._neural_model

    @neural_model.setter
    def neural_model(self, value):
        self._neural_model = value

    @property
    def encoder_model_loaded(self):
        """Get the loaded encoder model."""
        return self._encoder_model_loaded

    @encoder_model_loaded.setter
    def encoder_model_loaded(self, value):
        self._encoder_model_loaded = value

    @property
    def decoder_model_loaded(self):
        """Get the loaded decoder model."""
        return self._decoder_model_loaded

    @decoder_model_loaded.setter
    def decoder_model_loaded(self, value):
        self._decoder_model_loaded = value

    @property
    def device(self) -> torch.device:
        """Get the device (CPU or CUDA) being used."""
        return self._device

    @device.setter
    def device(self, value: torch.device):
        """Set the device and move models to it."""
        self._device = value
        if self._neural_model is not None:
            self._neural_model.to(self._device)