#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/13'
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
    import keras
    import numpy as np
    import logging

    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    from tensorflow.python.keras.losses import MeanSquaredError
    from tensorflow.python.keras.losses import BinaryCrossentropy
    from Engine.Callbacks.CallbackEarlyStop import EarlyStopping
    from Engine.Algorithms.autoencoder.AutoencoderAlgorithm import AutoencoderAlgorithm
    from Engine.Architectures.Autoencoder.AutoencoderModel import AutoencoderModel

except ImportError as error:
    logging.error(error)
    sys.exit(-1)

# Default values from your constants file
DEFAULT_AUTOENCODER_LATENT_DIMENSION = 64
DEFAULT_AUTOENCODER_TRAINING_ALGORITHM = "Adam"
DEFAULT_AUTOENCODER_MODEL_ACTIVATION = "swish"
DEFAULT_AUTOENCODER_DROPOUT_DECAY_RATE_ENCODER = 0.25
DEFAULT_AUTOENCODER_DROPOUT_DECAY_RATE_DECODER = 0.25
DEFAULT_AUTOENCODER_BATCH_SIZE = 128
DEFAULT_AUTOENCODER_NUMBER_CLASSES = 2
DEFAULT_AUTOENCODER_DENSE_LAYERS_SETTINGS_ENCODER = [320, 160]
DEFAULT_AUTOENCODER_DENSE_LAYERS_SETTINGS_DECODER = [160, 320]
DEFAULT_AUTOENCODER_MOMENTUM = 0.8
DEFAULT_AUTOENCODER_LAST_ACTIVATION_LAYER = "sigmoid"
DEFAULT_AUTOENCODER_INITIALIZER_MEAN = 0.0
DEFAULT_AUTOENCODER_INITIALIZER_DEVIATION = 0.125
DEFAULT_AUTOENCODER_NUMBER_EPOCHS = 350
DEFAULT_AUTOENCODER_LOSS_FUNCTION = "mse"
DEFAULT_AUTOENCODER_FILE_NAME_ENCODER = "encoder_model"
DEFAULT_AUTOENCODER_FILE_NAME_DECODER = "decoder_model"
DEFAULT_AUTOENCODER_PATH_OUTPUT_MODELS = "models_saved/"
DEFAULT_AUTOENCODER_LATENT_MEAN_DISTRIBUTION = 0.5
DEFAULT_AUTOENCODER_STANDER_DEVIATION = 0.125


class Autoencoder:
    """
    A class that instantiates and manages an autoencoder model.
    This implementation provides complete configuration, training, and management capabilities
    for autoencoder-based learning tasks within the Synthetic Ocean ecosystem.

    NOW SUPPORTS MULTI-DIMENSIONAL DATA: Automatically handles 1D, 2D, 3D, and N-D inputs
    by flattening them during training and reshaping them during generation.

    Attributes:
        _autoencoder_model (AutoencoderModel): Contains encoder and decoder components
        _autoencoder_algorithm (AutoencoderAlgorithm): Manages the autoencoder training process
        _original_input_shape (tuple): Stores the original shape of input data for reconstruction

    Configuration Parameters (with getters/setters):
        _autoencoder_latent_dimension (int): Size of the latent space
        _autoencoder_training_algorithm (str): Training algorithm specification
        _autoencoder_activation_function (str): Activation function for hidden layers
        _autoencoder_dropout_decay_rate_encoder (float): Encoder dropout rate
        _autoencoder_dropout_decay_rate_decoder (float): Decoder dropout rate
        _autoencoder_dense_layer_sizes_encoder (list[int]): Encoder layer sizes
        _autoencoder_dense_layer_sizes_decoder (list[int]): Decoder layer sizes
        _autoencoder_batch_size (int): Size of training batches
        _autoencoder_number_epochs (int): Number of training epochs
        _autoencoder_number_classes (int): Number of output classes
        _autoencoder_loss_function (str): Loss function for reconstruction
        _autoencoder_momentum (float): Momentum parameter for optimization
        _autoencoder_last_activation_layer (str): Last layer activation function
        _autoencoder_initializer_mean (float): Mean for weight initialization
        _autoencoder_initializer_deviation (float): Std dev for weight initialization
        _autoencoder_latent_mean_distribution (float): Latent space mean
        _autoencoder_latent_stander_deviation (float): Latent space std dev
        _autoencoder_file_name_encoder (str): Encoder model filename
        _autoencoder_file_name_decoder (str): Decoder model filename
        _autoencoder_path_output_models (str): Path for saving models
    """

    def __init__(self,
                 latent_dimension: int = DEFAULT_AUTOENCODER_LATENT_DIMENSION,
                 training_algorithm: str = DEFAULT_AUTOENCODER_TRAINING_ALGORITHM,
                 activation_function: str = DEFAULT_AUTOENCODER_MODEL_ACTIVATION,
                 dropout_decay_rate_encoder: float = DEFAULT_AUTOENCODER_DROPOUT_DECAY_RATE_ENCODER,
                 dropout_decay_rate_decoder: float = DEFAULT_AUTOENCODER_DROPOUT_DECAY_RATE_DECODER,
                 dense_layer_sizes_encoder: list[int] = None,
                 dense_layer_sizes_decoder: list[int] = None,
                 batch_size: int = DEFAULT_AUTOENCODER_BATCH_SIZE,
                 number_epochs: int = DEFAULT_AUTOENCODER_NUMBER_EPOCHS,
                 number_classes: int = DEFAULT_AUTOENCODER_NUMBER_CLASSES,
                 loss_function: str = DEFAULT_AUTOENCODER_LOSS_FUNCTION,
                 momentum: float = DEFAULT_AUTOENCODER_MOMENTUM,
                 last_activation_layer: str = DEFAULT_AUTOENCODER_LAST_ACTIVATION_LAYER,
                 initializer_mean: float = DEFAULT_AUTOENCODER_INITIALIZER_MEAN,
                 initializer_deviation: float = DEFAULT_AUTOENCODER_INITIALIZER_DEVIATION,
                 latent_mean_distribution: float = DEFAULT_AUTOENCODER_LATENT_MEAN_DISTRIBUTION,
                 latent_stander_deviation: float = DEFAULT_AUTOENCODER_STANDER_DEVIATION,
                 file_name_encoder: str = DEFAULT_AUTOENCODER_FILE_NAME_ENCODER,
                 file_name_decoder: str = DEFAULT_AUTOENCODER_FILE_NAME_DECODER,
                 path_output_models: str = DEFAULT_AUTOENCODER_PATH_OUTPUT_MODELS,
                 number_samples_per_class: dict = None,
                 callback_model_monitor=None,
                 callback_early_stop=None,
                 algorithm: AutoencoderAlgorithm | None = None,
                 model: AutoencoderModel | None = None
                 ) -> None:
        """
        Initializes the autoencoder instance with configuration parameters.

        Args:
            latent_dimension: Latent space size (default: 64)
            training_algorithm: Training algorithm (default: "Adam")
            activation_function: Activation function (default: "swish")
            dropout_decay_rate_encoder: Encoder dropout rate (default: 0.25)
            dropout_decay_rate_decoder: Decoder dropout rate (default: 0.25)
            dense_layer_sizes_encoder: Encoder layer sizes (default: [320, 160])
            dense_layer_sizes_decoder: Decoder layer sizes (default: [160, 320])
            batch_size: Size of training batches (default: 128)
            number_epochs: Number of training epochs (default: 350)
            number_classes: Number of output classes (default: 2)
            loss_function: Loss function for reconstruction (default: "mse")
            momentum: Momentum parameter for optimization (default: 0.8)
            last_activation_layer: Last layer activation function (default: "sigmoid")
            initializer_mean: Mean for weight initialization (default: 0.0)
            initializer_deviation: Std dev for weight initialization (default: 0.125)
            latent_mean_distribution: Latent space mean (default: 0.5)
            latent_stander_deviation: Latent space std dev (default: 0.125)
            file_name_encoder: Encoder model filename (default: "encoder_model")
            file_name_decoder: Decoder model filename (default: "decoder_model")
            path_output_models: Path for saving models (default: "models_saved/")
            number_samples_per_class: Dictionary with class distribution info (default: None)
            callback_model_monitor: Callback for monitoring model training (default: None)
            callback_early_stop: Callback for early stopping (default: None)
            algorithm: Optional pre-initialized AutoencoderAlgorithm instance (default: None)
            model: Optional pre-initialized AutoencoderModel instance (default: None)
        """
        # Store pre-initialized instances if provided
        self._autoencoder_algorithm: AutoencoderAlgorithm | None = algorithm
        self._autoencoder_model: AutoencoderModel | None = model

        # ** autoencoder Model Configuration Parameters **
        self._autoencoder_latent_dimension: int = latent_dimension
        self._autoencoder_training_algorithm: str = training_algorithm
        self._autoencoder_activation_function: str = activation_function
        self._autoencoder_dropout_decay_rate_encoder: float = dropout_decay_rate_encoder
        self._autoencoder_dropout_decay_rate_decoder: float = dropout_decay_rate_decoder

        # Handle mutable default values safely
        self._autoencoder_dense_layer_sizes_encoder: list[int] = (
            dense_layer_sizes_encoder
            if dense_layer_sizes_encoder is not None
            else DEFAULT_AUTOENCODER_DENSE_LAYERS_SETTINGS_ENCODER.copy()
        )
        self._autoencoder_dense_layer_sizes_decoder: list[int] = (
            dense_layer_sizes_decoder
            if dense_layer_sizes_decoder is not None
            else DEFAULT_AUTOENCODER_DENSE_LAYERS_SETTINGS_DECODER.copy()
        )

        self._autoencoder_batch_size: int = batch_size
        self._autoencoder_number_epochs: int = number_epochs
        self._autoencoder_number_classes: int = number_classes
        self._autoencoder_loss_function: str = loss_function
        self._autoencoder_momentum: float = momentum
        self._autoencoder_last_activation_layer: str = last_activation_layer
        self._autoencoder_initializer_mean: float = initializer_mean
        self._autoencoder_initializer_deviation: float = initializer_deviation
        self._autoencoder_latent_mean_distribution: float = latent_mean_distribution
        self._autoencoder_latent_stander_deviation: float = latent_stander_deviation
        self._autoencoder_file_name_encoder: str = file_name_encoder
        self._autoencoder_file_name_decoder: str = file_name_decoder
        self._autoencoder_path_output_models: str = path_output_models

        # Initialize number_samples_per_class with default
        self._number_samples_per_class: dict = (
            number_samples_per_class
            if number_samples_per_class is not None
            else {"number_classes": number_classes}
        )

        # Initialize callbacks
        self._callback_model_monitor = callback_model_monitor
        self._callback_early_stop = callback_early_stop

        # Flag to indicate if instances were provided
        self._has_external_algorithm: bool = algorithm is not None
        self._has_external_model: bool = model is not None

        # Storage for original input shape (for multi-dimensional data)
        self._original_input_shape = None

    def _get_autoencoder(self, input_shape: tuple[int, ...]) -> None:
        """
        Initialize and configure the autoencoder model, including encoder and decoder components.

        This method sets up an autoencoder model by configuring both the encoder and decoder using the `AutoencoderModel`
        class and links them with the `AutoencoderAlgorithm` class. The model is initialized with specified configurations
        such as latent dimension, activation functions, dropout rates, and layer sizes for both the encoder and decoder.

        If pre-initialized instances were provided in the constructor, they are used instead of creating new ones.

        Args:
            input_shape: The shape of the input data, which determines the output shape for the models.

        Initializes:
            self._autoencoder_model:
                An instance of the `AutoencoderModel` class, including the encoder and decoder setup, with
                configurations for activation functions, layer sizes, dropout rates, and more.
            self._autoencoder_algorithm:
                An instance of the `AutoencoderAlgorithm` class, managing the autoencoder training process, including
                the encoder and decoder models, loss function, latent distributions, and model file paths.
        """
        # Only create new model if none was provided
        if not self._has_external_model:
            # autoencoder Model setup for Encoder and Decoder
            self._autoencoder_model = AutoencoderModel(
                latent_dimension=self._autoencoder_latent_dimension,
                output_shape=input_shape,
                activation_function=self._autoencoder_activation_function,
                initializer_mean=self._autoencoder_initializer_mean,
                initializer_deviation=self._autoencoder_initializer_deviation,
                dropout_decay_encoder=self._autoencoder_dropout_decay_rate_encoder,
                dropout_decay_decoder=self._autoencoder_dropout_decay_rate_decoder,
                last_layer_activation=self._autoencoder_last_activation_layer,
                number_neurons_encoder=self._autoencoder_dense_layer_sizes_encoder,
                number_neurons_decoder=self._autoencoder_dense_layer_sizes_decoder,
                dataset_type=np.float32,
                number_samples_per_class=self._number_samples_per_class
            )

        # Only create new algorithm if none was provided
        if not self._has_external_algorithm:
            # Ensure we have a model to get encoder and decoder from
            if self._autoencoder_model is None:
                raise ValueError("AutoencoderModel instance is required but was not provided.")

            # autoencoder Algorithm setup for training and model operations
            self._autoencoder_algorithm = AutoencoderAlgorithm(
                encoder_model=self._autoencoder_model.get_encoder(input_shape),
                decoder_model=self._autoencoder_model.get_decoder(input_shape),
                loss_function=self._autoencoder_loss_function,
                file_name_encoder=self._autoencoder_file_name_encoder,
                file_name_decoder=self._autoencoder_file_name_decoder,
                models_saved_path=self._autoencoder_path_output_models,
                latent_mean_distribution=self._autoencoder_latent_mean_distribution,
                latent_stander_deviation=self._autoencoder_latent_stander_deviation,
                latent_dimension=self._autoencoder_latent_dimension
            )
        else:
            # If algorithm was provided externally, update its configuration if needed
            if hasattr(self._autoencoder_algorithm, 'loss_function'):
                self._autoencoder_algorithm.loss_function = self._autoencoder_loss_function
            if hasattr(self._autoencoder_algorithm, 'file_name_encoder'):
                self._autoencoder_algorithm.file_name_encoder = self._autoencoder_file_name_encoder
            if hasattr(self._autoencoder_algorithm, 'file_name_decoder'):
                self._autoencoder_algorithm.file_name_decoder = self._autoencoder_file_name_decoder
            if hasattr(self._autoencoder_algorithm, 'models_saved_path'):
                self._autoencoder_algorithm.models_saved_path = self._autoencoder_path_output_models
            if hasattr(self._autoencoder_algorithm, 'latent_mean_distribution'):
                self._autoencoder_algorithm.latent_mean_distribution = self._autoencoder_latent_mean_distribution
            if hasattr(self._autoencoder_algorithm, 'latent_stander_deviation'):
                self._autoencoder_algorithm.latent_stander_deviation = self._autoencoder_latent_stander_deviation
            if hasattr(self._autoencoder_algorithm, 'latent_dimension'):
                self._autoencoder_algorithm.latent_dimension = self._autoencoder_latent_dimension

    def fit_model(
            self,
            input_shape: tuple[int, ...],
            x_real_samples: np.ndarray,
            y_real_samples: np.ndarray
    ) -> None:
        """
        Executes the complete autoencoder training process with automatic data flattening.

        NOW SUPPORTS MULTI-DIMENSIONAL DATA:
        - Automatically flattens N-D input data (1D, 2D, 3D, etc.)
        - Converts labels to categorical format
        - Passes data and labels as separate inputs (model concatenates internally)
        - Stores original shape for reconstruction during generation

        Args:
            input_shape: Shape of input data samples (e.g., (784,) for 1D, (28, 28, 1) for 2D, (16, 16, 16) for 3D)
            x_real_samples: Training data samples (can be N-dimensional)
            y_real_samples: Corresponding class labels (1D array of class indices)

        Process:
            1. Flattens multi-dimensional input data
            2. Converts labels to categorical format
            3. Passes data and labels as separate inputs to model
            4. Initializes model architecture (or uses provided)
            5. Configures loss function
            6. Sets up training callbacks
            7. Executes autoencoder training
            8. Manages model saving and monitoring
        """

        # Store original input shape for later reconstruction
        self._original_input_shape = input_shape

        # Calculate total flattened dimension
        flattened_dim = int(np.prod(input_shape))

        # Prepare data
        print(f"\nPreparing data for Conditional Autoencoder...")
        print(f"  - Original input shape: {input_shape}")
        print(f"  - Input data shape: {x_real_samples.shape}")

        # Flatten the input data if it has more than 2 dimensions
        # (batch_size, ...) -> (batch_size, flattened_features)
        if len(x_real_samples.shape) > 2:
            x_real_samples_flat = x_real_samples.reshape(x_real_samples.shape[0], -1)
            print(f"  - Flattened data shape: {x_real_samples_flat.shape}")
        else:
            x_real_samples_flat = x_real_samples
            print(f"  - Data already flat: {x_real_samples_flat.shape}")

        # Convert labels to categorical format
        y_categorical = to_categorical(y_real_samples, num_classes=self._autoencoder_number_classes)
        print(f"  - Categorical labels shape: {y_categorical.shape}")

        # Initialize the autoencoder model with flattened dimension
        # The model expects 2 separate inputs: [data, labels]
        self._get_autoencoder(flattened_dim)

        # Print the model summaries for the encoder and decoder if available
        if self._autoencoder_model is not None:
            print(f"\nEncoder Model:")
            self._autoencoder_model.get_encoder(flattened_dim)
            print(f"\nDecoder Model:")
            self._autoencoder_model.get_decoder(flattened_dim)

        # Ensure we have an algorithm
        if self._autoencoder_algorithm is None:
            raise ValueError("AutoencoderAlgorithm instance is required but was not provided or created.")

        # Compile the autoencoder algorithm with the specified loss function
        self._autoencoder_algorithm.compile(loss='mse')

        # Build callbacks list - only include callbacks that exist
        callbacks_list = []

        if self._callback_model_monitor is not None:
            callbacks_list.append(self._callback_model_monitor)

        if self._callback_early_stop is not None:
            callbacks_list.append(self._callback_early_stop)

        print(f"\nStarting training...")
        print(f"  - Epochs: {self._autoencoder_number_epochs}")
        print(f"  - Batch size: {self._autoencoder_batch_size}")
        print(f"  - Latent dimension: {self._autoencoder_latent_dimension}")

        # Fit the autoencoder model
        # IMPORTANT: Pass as tuple (data, labels) - model concatenates internally!
        # Input: [flattened_data, labels] as separate inputs
        # Target: flattened data only (reconstruction target)
        self._autoencoder_algorithm.fit(
            (x_real_samples_flat, y_categorical),  # Input: tuple of (data, labels)
            x_real_samples_flat,  # Target: only the data (no labels)
            epochs=self._autoencoder_number_epochs,
            batch_size=self._autoencoder_batch_size,
            callbacks=callbacks_list if callbacks_list else None
        )

        print(f"\n✓ Training completed successfully!")

    # Additional getters for the algorithm and model
    @property
    def autoencoder_algorithm(self) -> AutoencoderAlgorithm | None:
        """Get the autoencoder algorithm instance."""
        return self._autoencoder_algorithm

    @property
    def autoencoder_model(self) -> AutoencoderModel | None:
        """Get the autoencoder model instance."""
        return self._autoencoder_model

    # Getter and setter for autoencoder_latent_dimension
    @property
    def autoencoder_latent_dimension(self) -> int:
        """Get the latent space dimension."""
        return self._autoencoder_latent_dimension

    @autoencoder_latent_dimension.setter
    def autoencoder_latent_dimension(self, value: int) -> None:
        """Set the latent space dimension."""
        self._autoencoder_latent_dimension = value

    # Getter and setter for autoencoder_training_algorithm
    @property
    def autoencoder_training_algorithm(self) -> str:
        """Get the training algorithm."""
        return self._autoencoder_training_algorithm

    @autoencoder_training_algorithm.setter
    def autoencoder_training_algorithm(self, value: str) -> None:
        """Set the training algorithm."""
        self._autoencoder_training_algorithm = value

    # Getter and setter for autoencoder_activation_function
    @property
    def autoencoder_activation_function(self) -> str:
        """Get the activation function."""
        return self._autoencoder_activation_function

    @autoencoder_activation_function.setter
    def autoencoder_activation_function(self, value: str) -> None:
        """Set the activation function."""
        self._autoencoder_activation_function = value

    # Getter and setter for autoencoder_dropout_decay_rate_encoder
    @property
    def autoencoder_dropout_decay_rate_encoder(self) -> float:
        """Get the encoder dropout rate."""
        return self._autoencoder_dropout_decay_rate_encoder

    @autoencoder_dropout_decay_rate_encoder.setter
    def autoencoder_dropout_decay_rate_encoder(self, value: float) -> None:
        """Set the encoder dropout rate."""
        self._autoencoder_dropout_decay_rate_encoder = value

    # Getter and setter for autoencoder_dropout_decay_rate_decoder
    @property
    def autoencoder_dropout_decay_rate_decoder(self) -> float:
        """Get the decoder dropout rate."""
        return self._autoencoder_dropout_decay_rate_decoder

    @autoencoder_dropout_decay_rate_decoder.setter
    def autoencoder_dropout_decay_rate_decoder(self, value: float) -> None:
        """Set the decoder dropout rate."""
        self._autoencoder_dropout_decay_rate_decoder = value

    # Getter and setter for autoencoder_dense_layer_sizes_encoder
    @property
    def autoencoder_dense_layer_sizes_encoder(self) -> list[int]:
        """Get the encoder layer sizes."""
        return self._autoencoder_dense_layer_sizes_encoder

    @autoencoder_dense_layer_sizes_encoder.setter
    def autoencoder_dense_layer_sizes_encoder(self, value: list[int]) -> None:
        """Set the encoder layer sizes."""
        self._autoencoder_dense_layer_sizes_encoder = value

    # Getter and setter for autoencoder_dense_layer_sizes_decoder
    @property
    def autoencoder_dense_layer_sizes_decoder(self) -> list[int]:
        """Get the decoder layer sizes."""
        return self._autoencoder_dense_layer_sizes_decoder

    @autoencoder_dense_layer_sizes_decoder.setter
    def autoencoder_dense_layer_sizes_decoder(self, value: list[int]) -> None:
        """Set the decoder layer sizes."""
        self._autoencoder_dense_layer_sizes_decoder = value

    # Getter and setter for autoencoder_batch_size
    @property
    def autoencoder_batch_size(self) -> int:
        """Get the batch size."""
        return self._autoencoder_batch_size

    @autoencoder_batch_size.setter
    def autoencoder_batch_size(self, value: int) -> None:
        """Set the batch size."""
        self._autoencoder_batch_size = value

    # Getter and setter for autoencoder_number_classes
    @property
    def autoencoder_number_classes(self) -> int:
        """Get the number of classes."""
        return self._autoencoder_number_classes

    @autoencoder_number_classes.setter
    def autoencoder_number_classes(self, value: int) -> None:
        """Set the number of classes."""
        self._autoencoder_number_classes = value

    # Getter and setter for autoencoder_loss_function
    @property
    def autoencoder_loss_function(self) -> str:
        """Get the loss function."""
        return self._autoencoder_loss_function

    @autoencoder_loss_function.setter
    def autoencoder_loss_function(self, value: str) -> None:
        """Set the loss function."""
        self._autoencoder_loss_function = value

    # Getter and setter for autoencoder_momentum
    @property
    def autoencoder_momentum(self) -> float:
        """Get the momentum parameter."""
        return self._autoencoder_momentum

    @autoencoder_momentum.setter
    def autoencoder_momentum(self, value: float) -> None:
        """Set the momentum parameter."""
        self._autoencoder_momentum = value

    # Getter and setter for autoencoder_last_activation_layer
    @property
    def autoencoder_last_activation_layer(self) -> str:
        """Get the last layer activation function."""
        return self._autoencoder_last_activation_layer

    @autoencoder_last_activation_layer.setter
    def autoencoder_last_activation_layer(self, value: str) -> None:
        """Set the last layer activation function."""
        self._autoencoder_last_activation_layer = value

    # Getter and setter for autoencoder_initializer_mean
    @property
    def autoencoder_initializer_mean(self) -> float:
        """Get the weight initialization mean."""
        return self._autoencoder_initializer_mean

    @autoencoder_initializer_mean.setter
    def autoencoder_initializer_mean(self, value: float) -> None:
        """Set the weight initialization mean."""
        self._autoencoder_initializer_mean = value

    # Getter and setter for autoencoder_initializer_deviation
    @property
    def autoencoder_initializer_deviation(self) -> float:
        """Get the weight initialization standard deviation."""
        return self._autoencoder_initializer_deviation

    @autoencoder_initializer_deviation.setter
    def autoencoder_initializer_deviation(self, value: float) -> None:
        """Set the weight initialization standard deviation."""
        self._autoencoder_initializer_deviation = value

    # Getter and setter for autoencoder_latent_mean_distribution
    @property
    def autoencoder_latent_mean_distribution(self) -> float:
        """Get the latent space mean distribution."""
        return self._autoencoder_latent_mean_distribution

    @autoencoder_latent_mean_distribution.setter
    def autoencoder_latent_mean_distribution(self, value: float) -> None:
        """Set the latent space mean distribution."""
        self._autoencoder_latent_mean_distribution = value

    # Getter and setter for autoencoder_latent_stander_deviation
    @property
    def autoencoder_latent_stander_deviation(self) -> float:
        """Get the latent space standard deviation."""
        return self._autoencoder_latent_stander_deviation

    @autoencoder_latent_stander_deviation.setter
    def autoencoder_latent_stander_deviation(self, value: float) -> None:
        """Set the latent space standard deviation."""
        self._autoencoder_latent_stander_deviation = value

    # Getter and setter for autoencoder_file_name_encoder
    @property
    def autoencoder_file_name_encoder(self) -> str:
        """Get the encoder model filename."""
        return self._autoencoder_file_name_encoder

    @autoencoder_file_name_encoder.setter
    def autoencoder_file_name_encoder(self, value: str) -> None:
        """Set the encoder model filename."""
        self._autoencoder_file_name_encoder = value

    # Getter and setter for autoencoder_file_name_decoder
    @property
    def autoencoder_file_name_decoder(self) -> str:
        """Get the decoder model filename."""
        return self._autoencoder_file_name_decoder

    @autoencoder_file_name_decoder.setter
    def autoencoder_file_name_decoder(self, value: str) -> None:
        """Set the decoder model filename."""
        self._autoencoder_file_name_decoder = value

    # Getter and setter for autoencoder_path_output_models
    @property
    def autoencoder_path_output_models(self) -> str:
        """Get the path for saving models."""
        return self._autoencoder_path_output_models

    @autoencoder_path_output_models.setter
    def autoencoder_path_output_models(self, value: str) -> None:
        """Set the path for saving models."""
        self._autoencoder_path_output_models = value

    def get_samples(self, number_samples_per_class):
        """
        Generate synthetic samples and reshape them to original input shape.

        NOW SUPPORTS MULTI-DIMENSIONAL OUTPUT:
        - Automatically reshapes generated samples to original input dimensions
        - Works with 1D, 2D, 3D, and N-D data

        Args:
            number_samples_per_class: Dictionary specifying number of samples per class
                Format 1: {"number_classes": N, "classes": {0: n0, 1: n1, ...}}
                Format 2 (simplified): {"number_classes": N, 0: n0, 1: n1, ...}

        Returns:
            np.ndarray: Generated samples reshaped to original input dimensions
                - Shape: (total_samples, *original_input_shape)
                - Example: For 3D input (16, 16, 16) with 100 total samples -> (100, 16, 16, 16)
        """
        # Generate flattened samples from algorithm
        generated_data = self._autoencoder_algorithm.get_samples(number_samples_per_class)

        # Check if we have stored original input shape
        if not hasattr(self, '_original_input_shape') or self._original_input_shape is None:
            # If no original shape stored, return flattened data
            # Convert dict to array if needed
            if isinstance(generated_data, dict):
                all_samples = []
                for label in sorted(generated_data.keys()):
                    all_samples.append(generated_data[label])
                return np.concatenate(all_samples, axis=0)
            return generated_data

        # Reshape generated samples to original input shape
        reshaped_samples = []

        if isinstance(generated_data, dict):
            # If returned as dict, reshape each class
            for label in sorted(generated_data.keys()):
                samples = generated_data[label]
                # Reshape from (n_samples, flattened_features) to (n_samples, *original_shape)
                reshaped = samples.reshape(-1, *self._original_input_shape)
                reshaped_samples.append(reshaped)

            # Concatenate all classes
            return np.concatenate(reshaped_samples, axis=0)
        else:
            # If returned as array, reshape directly
            return generated_data.reshape(-1, *self._original_input_shape)

    @property
    def callback_model_monitor(self):
        """Get the model monitor callback."""
        return self._callback_model_monitor

    @callback_model_monitor.setter
    def callback_model_monitor(self, value) -> None:
        """Set the model monitor callback."""
        self._callback_model_monitor = value

    @property
    def callback_early_stop(self):
        """Get the early stop callback."""
        return self._callback_early_stop

    @callback_early_stop.setter
    def callback_early_stop(self, value) -> None:
        """Set the early stop callback."""
        self._callback_early_stop = value

    @property
    def number_samples_per_class(self) -> dict:
        """Get the number of samples per class dictionary."""
        return self._number_samples_per_class

    @number_samples_per_class.setter
    def number_samples_per_class(self, value: dict) -> None:
        """Set the number of samples per class dictionary."""
        self._number_samples_per_class = value

    @property
    def original_input_shape(self) -> tuple:
        """Get the original input shape (before flattening)."""
        return self._original_input_shape