#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
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
    import torch.nn.functional as F

    from Engine.Layers.Torch.Activations import Activations

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


class DiffusionAutoencoderModelTorch(nn.Module, Activations):
    """
    A LatentDiffusion Autoencoder model that combines an encoder-decoder architecture
    with diffusion-based latent space learning. This model enables flexible
    configuration of activation functions, layer structures, and loss functions,
    making it suitable for various generative and representation learning tasks.

    The autoencoder follows a variational approach, where the latent space
    undergoes a diffusion process to enhance feature disentanglement. This
    technique has been explored in deep generative models to improve the
    quality of generated data while preserving meaningful representations.

    References:
        - Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising LatentDiffusion Probabilistic Architectures."
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

    def __init__(self, input_shape,
                 latent_dimension: int = DEFAULT_DIFFUSION_AUTOENCODER_LATENT_DIMENSION,
                 number_encoder_neurons=None,
                 number_decoder_neurons=None,
                 loss_function: str = DEFAULT_DIFFUSION_AUTOENCODER_LOSS,
                 last_activation: str = DEFAULT_DIFFUSION_AUTOENCODER_LAST_LAYER_ACTIVATION,
                 activation_output_encoder: str = DEFAULT_DIFFUSION_AUTOENCODER_ACTIVATION_OUTPUT_ENCODER,
                 batch_size_create_embedding: int = DEFAULT_DIFFUSION_AUTOENCODER_BATCH_SIZE_TRAINING,
                 intermediary_activation_autoencoder: str = DEFAULT_DIFFUSION_AUTOENCODER_INTERMEDIARY_ACTIVATION,
                 intermediary_activation_alpha: float = DEFAULT_DIFFUSION_AUTOENCODER_INTERMEDIARY_ACTIVATION_ALPHA):
        """
        Initializes the DiffusionAutoencoderModel with user-defined architecture
        and training parameters.

        Args:
            input_shape (int): Number of features in the input data.
            latent_dimension (int): Size of the latent space (bottleneck layer).
            number_encoder_neurons (list[int]): Number of neurons per encoder layer.
            number_decoder_neurons (list[int]): Number of neurons per decoder layer.
            loss_function (str): Loss function used during training.
            last_activation (str): Activation function for the final decoder layer.
            activation_output_encoder (str): Activation function for the encoder output.
            batch_size_create_embedding (int): Batch size used for embedding generation.
            intermediary_activation_autoencoder (str): Activation function for intermediate layers.
            intermediary_activation_alpha (float): Parameter for activations requiring an alpha value (e.g., LeakyReLU).
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

        # Build the autoencoder
        self.build_autoencoder()

    def build_autoencoder(self):
        """
        Builds the full autoencoder model by combining the encoder and decoder.
        """
        # Build encoder layers
        encoder_layers = []

        # Add encoder hidden layers
        prev_size = self._input_shape
        for num_neurons in self._list_number_neurons_per_layer_encoder:
            encoder_layers.append(nn.Linear(prev_size, num_neurons))
            prev_size = num_neurons

        # Latent space layer
        encoder_layers.append(nn.Linear(prev_size, self._latent_dimension))

        self.encoder_layers = nn.ModuleList(encoder_layers)

        # Build decoder layers
        decoder_layers = []

        # Initial dense layer in decoder
        decoder_layers.append(nn.Linear(self._latent_dimension, self._latent_dimension))

        # Add decoder hidden layers
        prev_size = self._latent_dimension
        for num_neurons in self._list_number_neurons_per_layer_decoder:
            decoder_layers.append(nn.Linear(prev_size, num_neurons))
            prev_size = num_neurons

        # Output layer
        decoder_layers.append(nn.Linear(prev_size, self._input_shape))

        self.decoder_layers = nn.ModuleList(decoder_layers)

    def _build_encoder(self, x):
        """
        Builds the encoder part of the autoencoder.

        Args:
            x: Input tensor.

        Returns:
            Encoder output tensor.
        """
        # Flatten if needed
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)

        # Pass through encoder layers
        for i, layer in enumerate(self.encoder_layers[:-1]):
            x = layer(x)
            x = self._get_activation(self._intermediary_activation_function)(x)

        # Latent space layer
        x = self.encoder_layers[-1](x)
        x = self._get_activation(self._activation_output_encoder)(x)

        # Reshape to add spatial dimension
        x = x.unsqueeze(-1)

        return x

    def _build_decoder(self, x):
        """
        Builds the decoder part of the autoencoder.

        Args:
            x: Input tensor from encoder.

        Returns:
            Decoder output tensor.
        """
        # Flatten
        x = x.view(x.shape[0], -1)

        # Pass through decoder layers
        for i, layer in enumerate(self.decoder_layers[:-1]):
            x = layer(x)
            x = self._get_activation(self._intermediary_activation_function)(x)

        # Output layer
        x = self.decoder_layers[-1](x)
        x = self._get_activation(self._last_activation)(x)

        return x

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x: Input tensor.

        Returns:
            Reconstructed output tensor.
        """
        encoder_output = self._build_encoder(x)
        decoder_output = self._build_decoder(encoder_output)
        return decoder_output

    def load_model(self, model_path):
        """
        Loads the autoencoder model weights from a file.

        Args:
            model_path (str): Path to the model weights file.
        """
        self.load_state_dict(torch.load(model_path))

    def save_model(self, model_path):
        """
        Saves the autoencoder model weights to a file.

        Args:
            model_path (str): Path to save the model weights.
        """
        torch.save(self.state_dict(), model_path)

    def get_encoder_and_decoder(self):
        """
        Builds and retrieves the encoder and decoder models from the trained autoencoder.

        Returns:
            tuple: (Encoder model, Decoder model)
        """

        # Create encoder model wrapper
        class EncoderModel(nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.parent = parent

            def forward(self, x):
                return self.parent._build_encoder(x)

        # Create decoder model wrapper
        class DecoderModel(nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.parent = parent

            def forward(self, x):
                return self.parent._build_decoder(x)

        self._encoder_model_loaded = EncoderModel(self)
        self._decoder_model_loaded = DecoderModel(self)

        return self._encoder_model_loaded, self._decoder_model_loaded

    def training(self, x_train_data, epochs, batch_size, optimizer=None, device='cpu'):
        """
        Trains the autoencoder model.

        Args:
            x_train_data (Tensor): Training data (inputs and targets are the same).
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            optimizer: PyTorch optimizer (if None, uses Adam).
            device (str): Device to train on ('cpu' or 'cuda').
        """
        self.to(device)
        self.train()

        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters())

        # Get loss function
        if self._loss_function == 'mse':
            criterion = nn.MSELoss()
        elif self._loss_function == 'bce':
            criterion = nn.BCELoss()
        else:
            criterion = nn.MSELoss()

        # Training loop
        num_batches = len(x_train_data) // batch_size

        for epoch in range(epochs):
            total_loss = 0

            for i in range(num_batches):
                batch_start = i * batch_size
                batch_end = batch_start + batch_size
                batch_data = x_train_data[batch_start:batch_end].to(device)

                optimizer.zero_grad()
                output = self.forward(batch_data)
                loss = criterion(output, batch_data)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / num_batches
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

    def create_embedding(self, data, device='cpu'):
        """
        Generates latent space embeddings using the trained encoder.

        Args:
            data (Tensor): Input data to encode.
            device (str): Device to use for inference.

        Returns:
            Tensor: Latent space representations.
        """
        self.eval()
        self.to(device)

        embeddings = []
        num_batches = (len(data) + self._batch_size_create_embedding - 1) // self._batch_size_create_embedding

        with torch.no_grad():
            for i in range(num_batches):
                batch_start = i * self._batch_size_create_embedding
                batch_end = min(batch_start + self._batch_size_create_embedding, len(data))
                batch_data = data[batch_start:batch_end].to(device)

                encoder_output = self._build_encoder(batch_data)
                embeddings.append(encoder_output.cpu())

        return torch.cat(embeddings, dim=0)

    def _get_activation(self, name):
        """Get activation function by name."""
        if callable(name):
            return name

        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(self._intermediary_activation_alpha),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'swish': nn.SiLU(),
            'silu': nn.SiLU(),
            'linear': nn.Identity(),
            'elu': nn.ELU(self._intermediary_activation_alpha),
        }
        return activations.get(name.lower(), nn.Identity())

    # Properties
    @property
    def input_shape(self):
        """Get the input shape of the model."""
        return self._input_shape

    @input_shape.setter
    def input_shape(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Input shape must be a positive integer")
        self._input_shape = value

    @property
    def latent_dimension(self):
        """Get the latent dimension size."""
        return self._latent_dimension

    @latent_dimension.setter
    def latent_dimension(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Latent dimension must be a positive integer")
        self._latent_dimension = value

    @property
    def encoder_neurons(self):
        """Get the list of neurons per encoder layer."""
        return self._list_number_neurons_per_layer_encoder

    @encoder_neurons.setter
    def encoder_neurons(self, value):
        if not isinstance(value, list) or not all(isinstance(x, int) and x > 0 for x in value):
            raise ValueError("Encoder neurons must be a list of positive integers")
        self._list_number_neurons_per_layer_encoder = value

    @property
    def decoder_neurons(self):
        """Get the list of neurons per decoder layer."""
        return self._list_number_neurons_per_layer_decoder

    @decoder_neurons.setter
    def decoder_neurons(self, value):
        if not isinstance(value, list) or not all(isinstance(x, int) and x > 0 for x in value):
            raise ValueError("Decoder neurons must be a list of positive integers")
        self._list_number_neurons_per_layer_decoder = value

    @property
    def intermediary_activation(self):
        """Get the intermediary activation function."""
        return self._intermediary_activation_function

    @intermediary_activation.setter
    def intermediary_activation(self, value):
        if not isinstance(value, str):
            raise ValueError("Intermediary activation must be a string")
        self._intermediary_activation_function = value

    @property
    def intermediary_activation_alpha(self):
        """Get the alpha parameter for intermediary activation."""
        return self._intermediary_activation_alpha

    @intermediary_activation_alpha.setter
    def intermediary_activation_alpha(self, value):
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError("Intermediary activation alpha must be a non-negative number")
        self._intermediary_activation_alpha = value

    @property
    def encoder_output_activation(self):
        """Get the encoder output activation function."""
        return self._activation_output_encoder

    @encoder_output_activation.setter
    def encoder_output_activation(self, value):
        if not isinstance(value, str):
            raise ValueError("Encoder output activation must be a string")
        self._activation_output_encoder = value

    @property
    def last_activation(self):
        """Get the last layer activation function."""
        return self._last_activation

    @last_activation.setter
    def last_activation(self, value):
        if not isinstance(value, str):
            raise ValueError("Last activation must be a string")
        self._last_activation = value

    @property
    def loss_function(self):
        """Get the loss function."""
        return self._loss_function

    @loss_function.setter
    def loss_function(self, value):
        if not isinstance(value, str):
            raise ValueError("Loss function must be a string")
        self._loss_function = value

    @property
    def batch_size_create_embedding(self):
        """Get the batch size for embedding creation."""
        return self._batch_size_create_embedding

    @batch_size_create_embedding.setter
    def batch_size_create_embedding(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Batch size must be a positive integer")
        self._batch_size_create_embedding = value

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