#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

from Engine.Algorithms.VariationalAutoencoder.VariationalAutoencoderAlgorithm import VariationalAutoencoderAlgorithm
from Engine.Architectures.VariationalAutoencoder.VariationalAutoencoderModel import VariationalAutoencoderModel

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
    import numpy as np
    import logging
    import torch

except ImportError as error:
    logging.error(error)
    sys.exit(-1)

# Default values from your constants file
DEFAULT_VARIATIONAL_AUTOENCODER_LATENT_DIMENSION = 32
DEFAULT_VARIATIONAL_AUTOENCODER_TRAINING_ALGORITHM = "Adam"
DEFAULT_VARIATIONAL_AUTOENCODER_ACTIVATION_INTERMEDIARY = "swish"
DEFAULT_VARIATIONAL_AUTOENCODER_DROPOUT_DECAY_RATE_ENCODER = 0.25
DEFAULT_VARIATIONAL_AUTOENCODER_DROPOUT_DECAY_RATE_DECODER = 0.25
DEFAULT_VARIATIONAL_AUTOENCODER_BATCH_SIZE = 64
DEFAULT_VARIATIONAL_AUTOENCODER_NUMBER_EPOCHS = 300
DEFAULT_VARIATIONAL_AUTOENCODER_NUMBER_CLASSES = 2
DEFAULT_VARIATIONAL_AUTOENCODER_DENSE_LAYERS_SETTINGS_ENCODER = [320, 160]
DEFAULT_VARIATIONAL_AUTOENCODER_DENSE_LAYERS_SETTINGS_DECODER = [160, 320]
DEFAULT_VARIATIONAL_AUTOENCODER_LOSS = "binary_crossentropy"
DEFAULT_VARIATIONAL_AUTOENCODER_MOMENTUM = 0.8
DEFAULT_VARIATIONAL_AUTOENCODER_LAST_ACTIVATION_LAYER = "sigmoid"
DEFAULT_VARIATIONAL_AUTOENCODER_INITIALIZER_MEAN = 0
DEFAULT_VARIATIONAL_AUTOENCODER_INITIALIZER_DEVIATION = 0.125
DEFAULT_VARIATIONAL_AUTOENCODER_LOSS_FUNCTION = 'binary_crossentropy'
DEFAULT_VARIATIONAL_AUTOENCODER_FILE_NAME_ENCODER = "encoder_model"
DEFAULT_VARIATIONAL_AUTOENCODER_FILE_NAME_DECODER = "decoder_model"
DEFAULT_VARIATIONAL_AUTOENCODER_PATH_OUTPUT_MODELS = "models_saved/"
DEFAULT_VARIATIONAL_AUTOENCODER_MEAN_DISTRIBUTION = 0.5
DEFAULT_VARIATIONAL_AUTOENCODER_STANDER_DEVIATION = 0.125


class VariationalAutoencoder:
    """
    A class that instantiates and manages a Variational Autoencoder (VAE) model.
    This implementation provides complete configuration, training, and management capabilities
    for variational autoencoder-based learning tasks within the Synthetic Ocean ecosystem.

    Attributes:
        _variational_algorithm (VariationalAutoencoderAlgorithm): Manages the VAE training process
        _variational_model (VariationalAutoencoderModel): Contains encoder and decoder components

    Configuration Parameters (with getters/setters):
        _variational_latent_dimension (int): Size of the latent space
        _variational_training_algorithm (str): Training algorithm specification
        _variational_activation_function (str): Activation function for hidden layers
        _variational_dropout_decay_rate_encoder (float): Encoder dropout rate
        _variational_dropout_decay_rate_decoder (float): Decoder dropout rate
        _variational_dense_layer_sizes_encoder (list[int]): Encoder layer sizes
        _variational_dense_layer_sizes_decoder (list[int]): Decoder layer sizes
        _variational_batch_size (int): Size of training batches
        _variational_number_epochs (int): Number of training epochs
        _variational_number_classes (int): Number of output classes
        _variational_loss_function (str): Loss function for reconstruction
        _variational_momentum (float): Momentum parameter for optimization
        _variational_last_activation_layer (str): Last layer activation function
        _variational_initializer_mean (float): Mean for weight initialization
        _variational_initializer_deviation (float): Std dev for weight initialization
        _variational_latent_mean_distribution (float): Latent space mean
        _variational_latent_stander_deviation (float): Latent space std dev
        _variational_file_name_encoder (str): Encoder model filename
        _variational_file_name_decoder (str): Decoder model filename
        _variational_path_output_models (str): Path for saving models
        _variational_learning_rate (float): Learning rate for optimizer
        _variational_beta_1 (float): Beta1 parameter for optimizer
        _variational_beta_2 (float): Beta2 parameter for optimizer
    """

    def __init__(
            self,
            latent_dimension: int = DEFAULT_VARIATIONAL_AUTOENCODER_LATENT_DIMENSION,
            training_algorithm: str = DEFAULT_VARIATIONAL_AUTOENCODER_TRAINING_ALGORITHM,
            activation_function: str = DEFAULT_VARIATIONAL_AUTOENCODER_ACTIVATION_INTERMEDIARY,
            dropout_decay_rate_encoder: float = DEFAULT_VARIATIONAL_AUTOENCODER_DROPOUT_DECAY_RATE_ENCODER,
            dropout_decay_rate_decoder: float = DEFAULT_VARIATIONAL_AUTOENCODER_DROPOUT_DECAY_RATE_DECODER,
            dense_layer_sizes_encoder: list[int] = None,
            dense_layer_sizes_decoder: list[int] = None,
            batch_size: int = DEFAULT_VARIATIONAL_AUTOENCODER_BATCH_SIZE,
            number_epochs: int = DEFAULT_VARIATIONAL_AUTOENCODER_NUMBER_EPOCHS,
            number_classes: int = DEFAULT_VARIATIONAL_AUTOENCODER_NUMBER_CLASSES,
            loss_function: str = DEFAULT_VARIATIONAL_AUTOENCODER_LOSS,
            momentum: float = DEFAULT_VARIATIONAL_AUTOENCODER_MOMENTUM,
            last_activation_layer: str = DEFAULT_VARIATIONAL_AUTOENCODER_LAST_ACTIVATION_LAYER,
            initializer_mean: float = DEFAULT_VARIATIONAL_AUTOENCODER_INITIALIZER_MEAN,
            initializer_deviation: float = DEFAULT_VARIATIONAL_AUTOENCODER_INITIALIZER_DEVIATION,
            latent_mean_distribution: float = DEFAULT_VARIATIONAL_AUTOENCODER_MEAN_DISTRIBUTION,
            latent_stander_deviation: float = DEFAULT_VARIATIONAL_AUTOENCODER_STANDER_DEVIATION,
            file_name_encoder: str = DEFAULT_VARIATIONAL_AUTOENCODER_FILE_NAME_ENCODER,
            file_name_decoder: str = DEFAULT_VARIATIONAL_AUTOENCODER_FILE_NAME_DECODER,
            path_output_models: str = DEFAULT_VARIATIONAL_AUTOENCODER_PATH_OUTPUT_MODELS,
            learning_rate: float = 0.001,
            beta_1: float = 0.9,
            beta_2: float = 0.999,
            algorithm: VariationalAutoencoderAlgorithm | None = None,
            model: VariationalAutoencoderModel | None = None,
    ) -> None:
        """
        Initializes the variational autoencoder instance with configuration parameters.

        Args:
            latent_dimension: Latent space dimension (default: 32)
            training_algorithm: Training algorithm (default: "Adam")
            activation_function: Activation function (default: "swish")
            dropout_decay_rate_encoder: Encoder dropout rate (default: 0.25)
            dropout_decay_rate_decoder: Decoder dropout rate (default: 0.25)
            dense_layer_sizes_encoder: Encoder layer sizes (default: [320, 160])
            dense_layer_sizes_decoder: Decoder layer sizes (default: [160, 320])
            batch_size: Batch size (default: 64)
            number_epochs: Training epochs (default: 300)
            number_classes: Number of classes (default: 2)
            loss_function: Loss function (default: "binary_crossentropy")
            momentum: Momentum parameter (default: 0.8)
            last_activation_layer: Last layer activation (default: "sigmoid")
            initializer_mean: Weight init mean (default: 0)
            initializer_deviation: Weight init std dev (default: 0.125)
            latent_mean_distribution: Latent distribution mean (default: 0.5)
            latent_stander_deviation: Latent distribution std dev (default: 0.125)
            file_name_encoder: Encoder filename (default: "encoder_model")
            file_name_decoder: Decoder filename (default: "decoder_model")
            path_output_models: Output models path (default: "models_saved/")
            learning_rate: Learning rate (default: 0.001)
            beta_1: Beta1 optimizer parameter (default: 0.9)
            beta_2: Beta2 optimizer parameter (default: 0.999)
            algorithm: Optional pre-initialized VariationalAutoencoderAlgorithm (default: None)
            model: Optional pre-initialized VariationalAutoencoderModel (default: None)
            number_samples_per_class: Optional class distribution information (default: None)
        """
        # Store pre-initialized instances if provided
        self._variational_algorithm: VariationalAutoencoderAlgorithm | None = algorithm
        self._variational_model: VariationalAutoencoderModel | None = model

        # Store class distribution information
        self._number_samples_per_class: dict | None = number_classes

        # ** Variational Autoencoder Model Configuration Parameters **
        self._variational_latent_dimension: int = latent_dimension
        self._variational_training_algorithm: str = training_algorithm
        self._variational_activation_function: str = activation_function
        self._variational_dropout_decay_rate_encoder: float = dropout_decay_rate_encoder
        self._variational_dropout_decay_rate_decoder: float = dropout_decay_rate_decoder

        # Handle mutable default values safely
        self._variational_dense_layer_sizes_encoder: list[int] = (
            dense_layer_sizes_encoder
            if dense_layer_sizes_encoder is not None
            else DEFAULT_VARIATIONAL_AUTOENCODER_DENSE_LAYERS_SETTINGS_ENCODER.copy()
        )
        self._variational_dense_layer_sizes_decoder: list[int] = (
            dense_layer_sizes_decoder
            if dense_layer_sizes_decoder is not None
            else DEFAULT_VARIATIONAL_AUTOENCODER_DENSE_LAYERS_SETTINGS_DECODER.copy()
        )

        self._variational_batch_size: int = batch_size
        self._variational_number_epochs: int = number_epochs
        self._variational_number_classes: int = number_classes
        self._variational_loss_function: str = loss_function
        self._variational_momentum: float = momentum
        self._variational_last_activation_layer: str = last_activation_layer
        self._variational_initializer_mean: float = initializer_mean
        self._variational_initializer_deviation: float = initializer_deviation
        self._variational_latent_mean_distribution: float = latent_mean_distribution
        self._variational_latent_stander_deviation: float = latent_stander_deviation
        self._variational_file_name_encoder: str = file_name_encoder
        self._variational_file_name_decoder: str = file_name_decoder
        self._variational_path_output_models: str = path_output_models
        self._variational_learning_rate: float = learning_rate
        self._variational_beta_1: float = beta_1
        self._variational_beta_2: float = beta_2

        # Flags to indicate if instances were provided
        self._has_external_algorithm: bool = algorithm is not None
        self._has_external_model: bool = model is not None

    def _get_variational(self, input_shape: tuple[int, ...]) -> None:
        """
        Initialize and configure the Variational Autoencoder model.

        This method sets up a Variational Autoencoder model by configuring the encoder and decoder
        using the `VariationalAutoencoderModel` class and links them with the `VariationalAutoencoderAlgorithm` class.
        The model is initialized with specified configurations such as latent dimension, activation functions,
        dropout rates, and layer sizes for both the encoder and decoder.

        If pre-initialized instances were provided in the constructor, they are used instead of creating new ones.

        Args:
            input_shape: The shape of the input data, which determines the output shape for the models.

        Initializes:
            self._variational_model:
                An instance of the `VariationalAutoencoderModel` class, including the encoder and decoder setup,
                with configurations for activation functions, layer sizes, dropout rates, and more.
            self._variational_algorithm:
                An instance of the `VariationalAutoencoderAlgorithm` class, managing the VAE training process,
                including encoder, decoder, loss function, latent dimension, and model file paths.
        """
        # Only create new model if none was provided
        if not self._has_external_model:
            if self._number_samples_per_class is None:
                raise ValueError(
                    "number_samples_per_class is required when creating a new VariationalAutoencoderModel.")

            # Variational Model setup for Encoder and Decoder
            self._variational_model = VariationalAutoencoderModel(
                latent_dimension=self._variational_latent_dimension,
                output_shape=input_shape[0] if isinstance(input_shape, tuple) else input_shape,
                activation_function=self._variational_activation_function,
                initializer_mean=self._variational_initializer_mean,
                initializer_deviation=self._variational_initializer_deviation,
                dropout_decay_encoder=self._variational_dropout_decay_rate_encoder,
                dropout_decay_decoder=self._variational_dropout_decay_rate_decoder,
                last_layer_activation=self._variational_last_activation_layer,
                number_neurons_encoder=self._variational_dense_layer_sizes_encoder,
                number_neurons_decoder=self._variational_dense_layer_sizes_decoder,
                dataset_type=np.float32,
                number_samples_per_class=self._number_samples_per_class
            )

        # Only create new algorithm if none was provided
        if not self._has_external_algorithm:
            # Ensure we have a model to get components from
            if self._variational_model is None:
                raise ValueError("VariationalAutoencoderModel instance is required but was not provided.")

            # Variational Algorithm setup for training and model operations
            self._variational_algorithm = VariationalAutoencoderAlgorithm(
                encoder_model=self._variational_model.get_encoder(input_shape),
                decoder_model=self._variational_model.get_decoder(input_shape),
                loss_function=self._variational_loss_function,
                latent_dimension=self._variational_latent_dimension,
                decoder_latent_dimension=self._variational_latent_dimension,
                latent_mean_distribution=self._variational_latent_mean_distribution,
                latent_stander_deviation=self._variational_latent_stander_deviation,
                file_name_encoder=self._variational_file_name_encoder,
                file_name_decoder=self._variational_file_name_decoder,
                models_saved_path=self._variational_path_output_models
            )
        else:
            # If algorithm was provided externally, update its configuration if needed
            if hasattr(self._variational_algorithm, 'loss_function'):
                self._variational_algorithm.loss_function = self._variational_loss_function
            if hasattr(self._variational_algorithm, 'latent_dimension'):
                self._variational_algorithm.latent_dimension = self._variational_latent_dimension
            if hasattr(self._variational_algorithm, 'decoder_latent_dimension'):
                self._variational_algorithm.decoder_latent_dimension = self._variational_latent_dimension
            if hasattr(self._variational_algorithm, 'latent_mean_distribution'):
                self._variational_algorithm.latent_mean_distribution = self._variational_latent_mean_distribution
            if hasattr(self._variational_algorithm, 'latent_stander_deviation'):
                self._variational_algorithm.latent_stander_deviation = self._variational_latent_stander_deviation
            if hasattr(self._variational_algorithm, 'file_name_encoder'):
                self._variational_algorithm.file_name_encoder = self._variational_file_name_encoder
            if hasattr(self._variational_algorithm, 'file_name_decoder'):
                self._variational_algorithm.file_name_decoder = self._variational_file_name_decoder
            if hasattr(self._variational_algorithm, 'models_saved_path'):
                self._variational_algorithm.models_saved_path = self._variational_path_output_models

    def fit_model(
            self,
            input_shape: tuple[int, ...],
            arguments: 'argparse.Namespace',
            x_real_samples: np.ndarray,
            y_real_samples: np.ndarray
    ) -> None:
        """
        Executes the complete variational autoencoder training process.

        Args:
            input_shape: Shape of input data samples
            arguments: Configuration parameters namespace
            x_real_samples: Training data samples
            y_real_samples: Corresponding class labels

        Process:
            1. Initializes model architecture (or uses provided)
            2. Configures optimizer with specified parameters
            3. Sets up training callbacks
            4. Prepares one-hot encoded labels
            5. Executes VAE training with reconstruction target
            6. Manages model saving and monitoring
        """
        # Initialize the variational autoencoder model (or use provided)
        self._get_variational(input_shape)

        # Print the model summaries for the encoder and decoder if available
        if self._variational_model is not None:
            print("\nEncoder Model:")
            print(self._variational_model.get_encoder(input_shape))
            print("\nDecoder Model:")
            print(self._variational_model.get_decoder(input_shape))

        # Ensure we have an algorithm
        if self._variational_algorithm is None:
            raise ValueError("VariationalAutoencoderAlgorithm instance is required but was not provided or created.")

        # Configure optimizer with specified parameters or defaults
        learning_rate = getattr(arguments, 'variational_learning_rate', self._variational_learning_rate)
        beta_1 = getattr(arguments, 'variational_beta_1', self._variational_beta_1)
        beta_2 = getattr(arguments, 'variational_beta_2', self._variational_beta_2)

        self._variational_algorithm.configure_optimizer(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2
        )

        # Prepare callbacks list
        callbacks_list = []

        if arguments.use_early_stop:
            callbacks_list.append(self._callback_early_stop)

        # One-hot encode labels for conditional VAE
        y_labels_one_hot = self._one_hot_encode(
            y_real_samples,
            self._number_samples_per_class["number_classes"]
        )

        # The target (y_data) should be x_real_samples for reconstruction
        y_data = x_real_samples

        # Fit the variational autoencoder model
        self._variational_algorithm.fit(
            (x_real_samples, y_labels_one_hot),  # x and labels
            y_data,  # y (target for reconstruction)
            epochs=self._variational_number_epochs,
            batch_size=self._variational_batch_size,
            callbacks=callbacks_list
        )

    def _one_hot_encode(self, labels: np.ndarray | torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        One-hot encode labels for PyTorch compatibility.

        Args:
            labels: Integer labels with shape [batch_size] or [batch_size, 1]
            num_classes: Number of classes

        Returns:
            One-hot encoded labels with shape [batch_size, num_classes]
        """
        if isinstance(labels, np.ndarray):
            # Remove extra dimension if it's (n, 1) instead of (n,)
            if len(labels.shape) == 2 and labels.shape[1] == 1:
                labels = labels.flatten()  # Convert from (n, 1) to (n,)

            # Now labels should be 1D
            if len(labels.shape) == 1:
                # Convert to tensor and do one-hot
                labels_tensor = torch.from_numpy(labels).long()
                one_hot = torch.nn.functional.one_hot(labels_tensor, num_classes).float()
                return one_hot
            else:
                raise ValueError(f"Labels should be 1D after flattening, but got shape {labels.shape}")

        elif isinstance(labels, torch.Tensor):
            # Same logic for tensors
            if len(labels.shape) == 2 and labels.shape[1] == 1:
                labels = labels.squeeze(1)

            if len(labels.shape) == 1:
                return torch.nn.functional.one_hot(labels.long(), num_classes).float()
            else:
                raise ValueError(f"Tensor labels should be 1D, but got shape {labels.shape}")

        else:
            raise ValueError(f"Unsupported labels type: {type(labels)}")

    # Additional getters for the algorithm and model
    @property
    def variational_algorithm(self) -> VariationalAutoencoderAlgorithm | None:
        """Get the variational autoencoder algorithm instance."""
        return self._variational_algorithm

    @property
    def variational_model(self) -> VariationalAutoencoderModel | None:
        """Get the variational autoencoder model instance."""
        return self._variational_model

    @property
    def number_samples_per_class(self) -> dict | None:
        """Get the number samples per class information."""
        return self._number_samples_per_class

    @number_samples_per_class.setter
    def number_samples_per_class(self, value: dict) -> None:
        """Set the number samples per class information."""
        self._number_samples_per_class = value

    # Property getters and setters
    @property
    def variational_latent_dimension(self) -> int:
        """Get the latent dimension."""
        return self._variational_latent_dimension

    @variational_latent_dimension.setter
    def variational_latent_dimension(self, value: int) -> None:
        """Set the latent dimension."""
        self._variational_latent_dimension = value

    @property
    def variational_training_algorithm(self) -> str:
        """Get the training algorithm."""
        return self._variational_training_algorithm

    @variational_training_algorithm.setter
    def variational_training_algorithm(self, value: str) -> None:
        """Set the training algorithm."""
        self._variational_training_algorithm = value

    @property
    def variational_activation_function(self) -> str:
        """Get the activation function."""
        return self._variational_activation_function

    @variational_activation_function.setter
    def variational_activation_function(self, value: str) -> None:
        """Set the activation function."""
        self._variational_activation_function = value

    @property
    def variational_dropout_decay_rate_encoder(self) -> float:
        """Get the encoder dropout decay rate."""
        return self._variational_dropout_decay_rate_encoder

    @variational_dropout_decay_rate_encoder.setter
    def variational_dropout_decay_rate_encoder(self, value: float) -> None:
        """Set the encoder dropout decay rate."""
        self._variational_dropout_decay_rate_encoder = value

    @property
    def variational_dropout_decay_rate_decoder(self) -> float:
        """Get the decoder dropout decay rate."""
        return self._variational_dropout_decay_rate_decoder

    @variational_dropout_decay_rate_decoder.setter
    def variational_dropout_decay_rate_decoder(self, value: float) -> None:
        """Set the decoder dropout decay rate."""
        self._variational_dropout_decay_rate_decoder = value

    @property
    def variational_dense_layer_sizes_encoder(self) -> list[int]:
        """Get the encoder dense layer sizes."""
        return self._variational_dense_layer_sizes_encoder

    @variational_dense_layer_sizes_encoder.setter
    def variational_dense_layer_sizes_encoder(self, value: list[int]) -> None:
        """Set the encoder dense layer sizes."""
        self._variational_dense_layer_sizes_encoder = value

    @property
    def variational_dense_layer_sizes_decoder(self) -> list[int]:
        """Get the decoder dense layer sizes."""
        return self._variational_dense_layer_sizes_decoder

    @variational_dense_layer_sizes_decoder.setter
    def variational_dense_layer_sizes_decoder(self, value: list[int]) -> None:
        """Set the decoder dense layer sizes."""
        self._variational_dense_layer_sizes_decoder = value

    @property
    def variational_batch_size(self) -> int:
        """Get the batch size."""
        return self._variational_batch_size

    @variational_batch_size.setter
    def variational_batch_size(self, value: int) -> None:
        """Set the batch size."""
        self._variational_batch_size = value

    @property
    def variational_number_epochs(self) -> int:
        """Get the number of epochs."""
        return self._variational_number_epochs

    @variational_number_epochs.setter
    def variational_number_epochs(self, value: int) -> None:
        """Set the number of epochs."""
        self._variational_number_epochs = value

    @property
    def variational_number_classes(self) -> int:
        """Get the number of classes."""
        return self._variational_number_classes

    @variational_number_classes.setter
    def variational_number_classes(self, value: int) -> None:
        """Set the number of classes."""
        self._variational_number_classes = value

    @property
    def variational_loss_function(self) -> str:
        """Get the loss function."""
        return self._variational_loss_function

    @variational_loss_function.setter
    def variational_loss_function(self, value: str) -> None:
        """Set the loss function."""
        self._variational_loss_function = value

    @property
    def variational_momentum(self) -> float:
        """Get the momentum."""
        return self._variational_momentum

    @variational_momentum.setter
    def variational_momentum(self, value: float) -> None:
        """Set the momentum."""
        self._variational_momentum = value

    @property
    def variational_last_activation_layer(self) -> str:
        """Get the last activation layer."""
        return self._variational_last_activation_layer

    @variational_last_activation_layer.setter
    def variational_last_activation_layer(self, value: str) -> None:
        """Set the last activation layer."""
        self._variational_last_activation_layer = value

    @property
    def variational_initializer_mean(self) -> float:
        """Get the initializer mean."""
        return self._variational_initializer_mean

    @variational_initializer_mean.setter
    def variational_initializer_mean(self, value: float) -> None:
        """Set the initializer mean."""
        self._variational_initializer_mean = value

    @property
    def variational_initializer_deviation(self) -> float:
        """Get the initializer deviation."""
        return self._variational_initializer_deviation

    @variational_initializer_deviation.setter
    def variational_initializer_deviation(self, value: float) -> None:
        """Set the initializer deviation."""
        self._variational_initializer_deviation = value

    @property
    def variational_latent_mean_distribution(self) -> float:
        """Get the latent mean distribution."""
        return self._variational_latent_mean_distribution

    @variational_latent_mean_distribution.setter
    def variational_latent_mean_distribution(self, value: float) -> None:
        """Set the latent mean distribution."""
        self._variational_latent_mean_distribution = value

    @property
    def variational_latent_stander_deviation(self) -> float:
        """Get the latent stander deviation."""
        return self._variational_latent_stander_deviation

    @variational_latent_stander_deviation.setter
    def variational_latent_stander_deviation(self, value: float) -> None:
        """Set the latent stander deviation."""
        self._variational_latent_stander_deviation = value

    @property
    def variational_file_name_encoder(self) -> str:
        """Get the encoder file name."""
        return self._variational_file_name_encoder

    @variational_file_name_encoder.setter
    def variational_file_name_encoder(self, value: str) -> None:
        """Set the encoder file name."""
        self._variational_file_name_encoder = value

    @property
    def variational_file_name_decoder(self) -> str:
        """Get the decoder file name."""
        return self._variational_file_name_decoder

    @variational_file_name_decoder.setter
    def variational_file_name_decoder(self, value: str) -> None:
        """Set the decoder file name."""
        self._variational_file_name_decoder = value

    @property
    def variational_path_output_models(self) -> str:
        """Get the output models path."""
        return self._variational_path_output_models

    @variational_path_output_models.setter
    def variational_path_output_models(self, value: str) -> None:
        """Set the output models path."""
        self._variational_path_output_models = value

    @property
    def variational_learning_rate(self) -> float:
        """Get the learning rate."""
        return self._variational_learning_rate

    @variational_learning_rate.setter
    def variational_learning_rate(self, value: float) -> None:
        """Set the learning rate."""
        self._variational_learning_rate = value

    @property
    def variational_beta_1(self) -> float:
        """Get the beta 1 parameter."""
        return self._variational_beta_1

    @variational_beta_1.setter
    def variational_beta_1(self, value: float) -> None:
        """Set the beta 1 parameter."""
        self._variational_beta_1 = value

    @property
    def variational_beta_2(self) -> float:
        """Get the beta 2 parameter."""
        return self._variational_beta_2

    @variational_beta_2.setter
    def variational_beta_2(self, value: float) -> None:
        """Set the beta 2 parameter."""
        self._variational_beta_2 = value