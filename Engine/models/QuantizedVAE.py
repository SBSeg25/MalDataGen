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
    import numpy as np
    import logging
    import torch
    from tensorflow.keras.optimizers import Adam
    import torch.nn.functional as F
    from Engine.architectures.quantized_vae.QuantizedVAEModel import QuantizedVAEModel
    from Engine.algorithms.quantized_vae.QuantizedVAEAlgorithm import QuantizedVAEAlgorithm

except ImportError as error:
    logging.error(error)
    sys.exit(-1)

# Default values from your constants file
DEFAULT_QUANTIZED_VAE_LATENT_DIMENSION = 32
DEFAULT_QUANTIZED_VAE_NUMBER_EMBEDDING = 32
DEFAULT_QUANTIZED_VAE_TRAINING_ALGORITHM = "Adam"
DEFAULT_QUANTIZED_VAE_ACTIVATION_INTERMEDIARY = "swish"
DEFAULT_QUANTIZED_VAE_DROPOUT_DECAY_RATE_ENCODER = 0.25
DEFAULT_QUANTIZED_VAE_DROPOUT_DECAY_RATE_DECODER = 0.25
DEFAULT_QUANTIZED_VAE_BATCH_SIZE = 64
DEFAULT_QUANTIZED_VAE_NUMBER_EPOCHS = 1000
DEFAULT_QUANTIZED_VAE_NUMBER_CLASSES = 2
DEFAULT_QUANTIZED_VAE_DENSE_LAYERS_SETTINGS_ENCODER = [256, 256]
DEFAULT_QUANTIZED_VAE_DENSE_LAYERS_SETTINGS_DECODER = [256, 256]
DEFAULT_QUANTIZED_VAE_LOSS = "binary_crossentropy"
DEFAULT_QUANTIZED_VAE_MOMENTUM = 0.8
DEFAULT_QUANTIZED_VAE_LAST_ACTIVATION_LAYER = "sigmoid"
DEFAULT_QUANTIZED_VAE_INITIALIZER_MEAN = 0
DEFAULT_QUANTIZED_VAE_INITIALIZER_DEVIATION = 0.125
DEFAULT_QUANTIZED_VAE_LOSS_FUNCTION = 'binary_crossentropy'
DEFAULT_QUANTIZED_VAE_FILE_NAME_ENCODER = "encoder_model"
DEFAULT_QUANTIZED_VAE_FILE_NAME_DECODER = "decoder_model"
DEFAULT_QUANTIZED_VAE_PATH_OUTPUT_MODELS = "models_saved/"
DEFAULT_QUANTIZED_VAE_MEAN_DISTRIBUTION = 0.5
DEFAULT_QUANTIZED_VAE_TRAIN_VARIANCE = 0.5
DEFAULT_QUANTIZED_VAE_STANDER_DEVIATION = 0.125


class QuantizedVAE:
    """
    A class that instantiates and manages a Vector Quantized Variational autoencoder (VQ-VAE) model.
    This implementation provides complete configuration, training, and management capabilities
    for quantized latent space learning tasks within the Synthetic Ocean ecosystem.

    NOW SUPPORTS MULTI-DIMENSIONAL DATA: Automatically handles 1D, 2D, 3D, and N-D inputs
    by flattening them during training and reshaping them during generation.

    Attributes:
        _quantizedVAE_algorithm (QuantizedVAEAlgorithm): Manages the VQ-VAE training process
        _quantizedVAE_model (QuantizedVAEModel): Contains encoder, decoder and quantization components
        _original_input_shape (tuple): Stores the original shape of input data for reconstruction

    Configuration Parameters (with getters/setters):
        _quantized_vae_number_epochs (int): Number of training epochs
        _quantized_vae_batch_size (int): Size of training batches
        _quantized_vae_latent_dimension (int): Size of the latent space
        _quantized_vae_number_embeddings (int): Number of embeddings in the codebook
        _quantized_vae_activation_function (str): Activation function for hidden layers
        _quantized_vae_initializer_mean (float): Mean for weight initialization
        _quantized_vae_initializer_deviation (float): Std dev for weight initialization
        _quantized_vae_dropout_decay_encoder (float): Encoder dropout rate
        _quantized_vae_dropout_decay_decoder (float): Decoder dropout rate
        _quantized_vae_last_layer_activation (str): Last layer activation function
        _quantized_vae_number_neurons_encoder (list[int]): Encoder layer sizes
        _quantized_vae_number_neurons_decoder (list[int]): Decoder layer sizes
        _quantized_vae_train_variance (float): Training variance parameter
        _quantized_vae_file_name_encoder (str): Encoder model filename
        _quantized_vae_file_name_decoder (str): Decoder model filename
        _quantized_vae_path_output_models (str): Path for saving models
    """

    def __init__(self,
                 number_epochs: int = DEFAULT_QUANTIZED_VAE_NUMBER_EPOCHS,
                 batch_size: int = DEFAULT_QUANTIZED_VAE_BATCH_SIZE,
                 latent_dimension: int = DEFAULT_QUANTIZED_VAE_LATENT_DIMENSION,
                 number_embeddings: int = DEFAULT_QUANTIZED_VAE_NUMBER_EMBEDDING,
                 activation_function: str = DEFAULT_QUANTIZED_VAE_ACTIVATION_INTERMEDIARY,
                 initializer_mean: float = DEFAULT_QUANTIZED_VAE_INITIALIZER_MEAN,
                 initializer_deviation: float = DEFAULT_QUANTIZED_VAE_MEAN_DISTRIBUTION,
                 dropout_decay_encoder: float = DEFAULT_QUANTIZED_VAE_DROPOUT_DECAY_RATE_ENCODER,
                 dropout_decay_decoder: float = DEFAULT_QUANTIZED_VAE_DROPOUT_DECAY_RATE_DECODER,
                 last_layer_activation: str = DEFAULT_QUANTIZED_VAE_LAST_ACTIVATION_LAYER,
                 number_neurons_encoder: list[int] = None,
                 number_neurons_decoder: list[int] = None,
                 train_variance: float = DEFAULT_QUANTIZED_VAE_TRAIN_VARIANCE,
                 file_name_encoder: str = DEFAULT_QUANTIZED_VAE_FILE_NAME_ENCODER,
                 file_name_decoder: str = DEFAULT_QUANTIZED_VAE_FILE_NAME_DECODER,
                 path_output_models: str = DEFAULT_QUANTIZED_VAE_PATH_OUTPUT_MODELS,
                 number_classes: int = DEFAULT_QUANTIZED_VAE_NUMBER_CLASSES,
                 algorithm: QuantizedVAEAlgorithm | None = None,
                 model: QuantizedVAEModel | None = None) -> None:
        """
        Initializes the quantized VAE instance with configuration parameters.

        Args:
            number_epochs: Training epochs (default: 30)
            batch_size: Batch size (default: 64)
            latent_dimension: Latent space size (default: 16)
            number_embeddings: Codebook size (default: 16)
            activation_function: Activation function (default: "swish")
            initializer_mean: Weight init mean (default: 0)
            initializer_deviation: Weight init std dev (default: 0.125)
            dropout_decay_encoder: Encoder dropout rate (default: 0.25)
            dropout_decay_decoder: Decoder dropout rate (default: 0.25)
            last_layer_activation: Last layer activation (default: "sigmoid")
            number_neurons_encoder: Encoder layer sizes (default: [320, 160])
            number_neurons_decoder: Decoder layer sizes (default: [160, 320])
            train_variance: Training variance parameter (default: 0.5)
            file_name_encoder: Encoder model filename (default: "encoder_model")
            file_name_decoder: Decoder model filename (default: "decoder_model")
            path_output_models: Path for saving models (default: "models_saved/")
            number_classes: Number of classes (default: 2)
            algorithm: Optional pre-initialized QuantizedVAEAlgorithm (default: None)
            model: Optional pre-initialized QuantizedVAEModel (default: None)
        """
        # Store pre-initialized instances if provided
        self._quantized_vae_algorithm: QuantizedVAEAlgorithm | None = algorithm
        self._quantized_vae_model: QuantizedVAEModel | None = model

        # ** Vector Quantized Variational autoencoder (VQ-VAE) Configuration Parameters **
        self._quantized_vae_number_epochs: int = number_epochs
        self._quantized_vae_batch_size: int = batch_size
        self._quantized_vae_latent_dimension: int = latent_dimension
        self._quantized_vae_number_embeddings: int = number_embeddings
        self._quantized_vae_activation_function: str = activation_function
        self._quantized_vae_initializer_mean: float = initializer_mean
        self._quantized_vae_initializer_deviation: float = initializer_deviation
        self._quantized_vae_dropout_decay_encoder: float = dropout_decay_encoder
        self._quantized_vae_dropout_decay_decoder: float = dropout_decay_decoder
        self._quantized_vae_last_layer_activation: str = last_layer_activation

        # Handle mutable default values safely
        self._quantized_vae_number_neurons_encoder: list[int] = (
            number_neurons_encoder
            if number_neurons_encoder is not None
            else DEFAULT_QUANTIZED_VAE_DENSE_LAYERS_SETTINGS_ENCODER.copy()
        )
        self._quantized_vae_number_neurons_decoder: list[int] = (
            number_neurons_decoder
            if number_neurons_decoder is not None
            else DEFAULT_QUANTIZED_VAE_DENSE_LAYERS_SETTINGS_DECODER.copy()
        )

        self._quantized_vae_train_variance: float = train_variance
        self._quantized_vae_file_name_encoder: str = file_name_encoder
        self._quantized_vae_file_name_decoder: str = file_name_decoder
        self._quantized_vae_path_output_models: str = path_output_models

        # Number of classes configuration
        self._number_samples_per_class = {"number_classes": number_classes}

        # Flags to indicate if instances were provided
        self._has_external_algorithm: bool = algorithm is not None
        self._has_external_model: bool = model is not None

        # Storage for original input shape (for multi-dimensional data)
        self._original_input_shape = None

    def _get_quantized_vae(self, input_shape: tuple[int, ...]) -> None:
        """
        Initialize and configure the Quantized Variational autoencoder (VQ-VAE) model, including encoder, decoder,
        and quantization components.

        This method sets up a Quantized VAE model by configuring the encoder, decoder, and quantization layers using the
        `QuantizedVAEModel` class and links them with the `QuantizedVAEAlgorithm` class. The model is initialized with
        specified configurations such as latent dimension, number of embeddings, activation functions, dropout rates,
        and layer sizes for both the encoder and decoder.

        If pre-initialized instances were provided in the constructor, they are used instead of creating new ones.

        Args:
            input_shape: The shape of the input data (flattened dimension), which determines the output shape for the models.

        Initializes:
            self._quantized_vae_model:
                An instance of the `QuantizedVAEModel` class, including the encoder, decoder, and quantization setup,
                with configurations for activation functions, layer sizes, dropout rates, and more.
            self._quantized_vae_algorithm:
                An instance of the `QuantizedVAEAlgorithm` class, managing the quantized VAE training process, including
                the encoder, decoder, and quantized models, training variance, latent dimension, number of embeddings,
                and model file paths.
        """
        # Only create new model if none was provided
        if not self._has_external_model:
            # Quantized VAE Model setup for Encoder, Decoder, and Quantization
            self._quantized_vae_model = QuantizedVAEModel(
                latent_dimension=self._quantized_vae_latent_dimension,
                number_embeddings=self._quantized_vae_number_embeddings,
                output_shape=input_shape,
                activation_function=self._quantized_vae_activation_function,
                initializer_mean=self._quantized_vae_initializer_mean,
                initializer_deviation=self._quantized_vae_initializer_deviation,
                dropout_decay_encoder=self._quantized_vae_dropout_decay_encoder,
                dropout_decay_decoder=self._quantized_vae_dropout_decay_decoder,
                last_layer_activation=self._quantized_vae_last_layer_activation,
                number_neurons_encoder=self._quantized_vae_number_neurons_encoder,
                number_neurons_decoder=self._quantized_vae_number_neurons_decoder,
                dataset_type=np.float32,
                number_samples_per_class=self._number_samples_per_class
            )

        # Only create new algorithm if none was provided
        if not self._has_external_algorithm:
            # Ensure we have a model to get components from
            if self._quantized_vae_model is None:
                raise ValueError("QuantizedVAEModel instance is required but was not provided.")

            quantized_model = self._quantized_vae_model.get_quantized_model()

            # Quantized VAE Algorithm setup for training and model operations
            self._quantized_vae_algorithm = QuantizedVAEAlgorithm(
                encoder_model=self._quantized_vae_model.get_encoder(),
                decoder_model=self._quantized_vae_model.get_decoder(),
                quantized_vae_model=quantized_model,
                train_variance=self._quantized_vae_train_variance,
                latent_dimension=self._quantized_vae_latent_dimension,
                number_embeddings=self._quantized_vae_number_embeddings,
                file_name_encoder=self._quantized_vae_file_name_encoder,
                file_name_decoder=self._quantized_vae_file_name_decoder,
                models_saved_path=self._quantized_vae_path_output_models
            )
        else:
            # If algorithm was provided externally, update its configuration if needed
            if hasattr(self._quantized_vae_algorithm, 'train_variance'):
                self._quantized_vae_algorithm.train_variance = self._quantized_vae_train_variance

            if hasattr(self._quantized_vae_algorithm, 'latent_dimension'):
                self._quantized_vae_algorithm.latent_dimension = self._quantized_vae_latent_dimension

            if hasattr(self._quantized_vae_algorithm, 'number_embeddings'):
                self._quantized_vae_algorithm.number_embeddings = self._quantized_vae_number_embeddings

            if hasattr(self._quantized_vae_algorithm, 'file_name_encoder'):
                self._quantized_vae_algorithm.file_name_encoder = self._quantized_vae_file_name_encoder

            if hasattr(self._quantized_vae_algorithm, 'file_name_decoder'):
                self._quantized_vae_algorithm.file_name_decoder = self._quantized_vae_file_name_decoder

            if hasattr(self._quantized_vae_algorithm, 'models_saved_path'):
                self._quantized_vae_algorithm.models_saved_path = self._quantized_vae_path_output_models

    def fit_model(
            self,
            input_shape: tuple[int, ...],
            x_real_samples: np.ndarray,
            y_real_samples: np.ndarray,
            batch_size: int = None,
            epochs: int = None,
            verbose: int = 1,
            callbacks: list = None
    ) -> None:
        """
        Executes the complete quantized VAE training process with automatic data flattening.

        NOW SUPPORTS MULTI-DIMENSIONAL DATA:
        - Automatically flattens N-D input data (1D, 2D, 3D, etc.)
        - Converts labels to one-hot format
        - Passes data and labels as separate inputs (model concatenates internally)
        - Stores original shape for reconstruction during generation

        Args:
            input_shape: Shape of input data samples (e.g., (784,) for 1D, (28, 28, 1) for 2D, (16, 16, 16) for 3D)
            x_real_samples: Training data samples (can be N-dimensional)
            y_real_samples: Corresponding class labels (1D array of class indices)
            batch_size: Batch size for training. If None, uses the instance's default batch size.
            epochs: Number of training epochs. If None, uses the instance's default number of epochs.
            verbose: Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch)
            callbacks: List of callback instances to apply during training.
                       These will be merged with internally defined callbacks.

        Process:
            1. Stores original input shape for later reconstruction
            2. Flattens multi-dimensional input data
            3. Converts labels to one-hot encoding format
            4. Initializes model architecture with flattened dimension
            5. Compiles model with optimizer
            6. Executes quantized VAE training
            7. Manages model saving and monitoring
        """
        # Use provided parameters or fall back to instance defaults
        effective_batch_size = batch_size if batch_size is not None else self._quantized_vae_batch_size
        effective_epochs = epochs if epochs is not None else self._quantized_vae_number_epochs

        # Store original input shape for later reconstruction
        self._original_input_shape = input_shape

        # Calculate total flattened dimension
        flattened_dim = int(np.prod(input_shape))

        # Flatten the input data if it has more than 2 dimensions
        # (batch_size, ...) -> (batch_size, flattened_features)
        if len(x_real_samples.shape) > 2:
            x_real_samples_flat = x_real_samples.reshape(x_real_samples.shape[0], -1)
        else:
            x_real_samples_flat = x_real_samples

        # Convert labels to one-hot encoding
        num_classes = self._number_samples_per_class["number_classes"]
        y_one_hot = np.zeros((len(y_real_samples), num_classes))
        y_one_hot[np.arange(len(y_real_samples)), y_real_samples.astype(int)] = 1

        # Initialize the quantized VAE model with flattened dimension
        self._get_quantized_vae(flattened_dim)

        # Ensure we have an algorithm
        if self._quantized_vae_algorithm is None:
            raise ValueError("QuantizedVAEAlgorithm instance is required but was not provided or created.")

        # Determine framework and compile with appropriate optimizer
        framework = getattr(self._quantized_vae_algorithm, '_framework', 'tensorflow').lower()

        if framework == 'pytorch':
            import torch
            optimizer = torch.optim.Adam(
                self._quantized_vae_algorithm._quantized_vae_model.parameters(),
                lr=0.0001
            )
        else:
            optimizer = Adam(learning_rate=0.0001)

        self._quantized_vae_algorithm.compile(optimizer=optimizer)

        # Setup callbacks list
        callbacks_list = []

        # Add internally defined callbacks if they exist
        if hasattr(self, '_callback_resources_monitor') and self._callback_resources_monitor is not None:
            callbacks_list.append(self._callback_resources_monitor)

        if hasattr(self, '_callback_model_monitor') and self._callback_model_monitor is not None:
            callbacks_list.append(self._callback_model_monitor)

        if hasattr(self, '_callback_early_stop') and self._callback_early_stop is not None:
            callbacks_list.append(self._callback_early_stop)

        # Merge with user-provided callbacks
        if callbacks is not None:
            if isinstance(callbacks, list):
                callbacks_list.extend(callbacks)
            else:
                callbacks_list.append(callbacks)

        # Fit the quantized VAE model
        # IMPORTANT: Pass as tuple (data, labels) - model concatenates internally!
        # Input: [flattened_data, labels] as separate inputs
        # Target: flattened data only (reconstruction target)
        self._quantized_vae_algorithm.fit(
            (x_real_samples_flat, y_one_hot),  # Input: tuple of (data, labels)
            x_real_samples_flat,  # Target: only the data (no labels)
            epochs=effective_epochs,
            batch_size=effective_batch_size,
            callbacks=callbacks_list if callbacks_list else None,
            verbose=verbose
        )


    # Additional getters for the algorithm and model
    @property
    def quantized_vae_algorithm(self) -> QuantizedVAEAlgorithm | None:
        """Get the quantized VAE algorithm instance."""
        return self._quantized_vae_algorithm

    @property
    def quantized_vae_model(self) -> QuantizedVAEModel | None:
        """Get the quantized VAE model instance."""
        return self._quantized_vae_model

    # Property getters and setters
    @property
    def quantized_vae_number_epochs(self) -> int:
        """Get the number of training epochs."""
        return self._quantized_vae_number_epochs

    @quantized_vae_number_epochs.setter
    def quantized_vae_number_epochs(self, value: int) -> None:
        """Set the number of training epochs."""
        self._quantized_vae_number_epochs = value

    @property
    def quantized_vae_batch_size(self) -> int:
        """Get the batch size."""
        return self._quantized_vae_batch_size

    @quantized_vae_batch_size.setter
    def quantized_vae_batch_size(self, value: int) -> None:
        """Set the batch size."""
        self._quantized_vae_batch_size = value

    @property
    def quantized_vae_latent_dimension(self) -> int:
        """Get the latent dimension."""
        return self._quantized_vae_latent_dimension

    @quantized_vae_latent_dimension.setter
    def quantized_vae_latent_dimension(self, value: int) -> None:
        """Set the latent dimension."""
        self._quantized_vae_latent_dimension = value

    @property
    def quantized_vae_number_embeddings(self) -> int:
        """Get the number of embeddings."""
        return self._quantized_vae_number_embeddings

    @quantized_vae_number_embeddings.setter
    def quantized_vae_number_embeddings(self, value: int) -> None:
        """Set the number of embeddings."""
        self._quantized_vae_number_embeddings = value

    @property
    def quantized_vae_activation_function(self) -> str:
        """Get the activation function."""
        return self._quantized_vae_activation_function

    @quantized_vae_activation_function.setter
    def quantized_vae_activation_function(self, value: str) -> None:
        """Set the activation function."""
        self._quantized_vae_activation_function = value

    @property
    def quantized_vae_initializer_mean(self) -> float:
        """Get the initializer mean."""
        return self._quantized_vae_initializer_mean

    @quantized_vae_initializer_mean.setter
    def quantized_vae_initializer_mean(self, value: float) -> None:
        """Set the initializer mean."""
        self._quantized_vae_initializer_mean = value

    @property
    def quantized_vae_initializer_deviation(self) -> float:
        """Get the initializer deviation."""
        return self._quantized_vae_initializer_deviation

    @quantized_vae_initializer_deviation.setter
    def quantized_vae_initializer_deviation(self, value: float) -> None:
        """Set the initializer deviation."""
        self._quantized_vae_initializer_deviation = value

    @property
    def quantized_vae_dropout_decay_encoder(self) -> float:
        """Get the encoder dropout decay rate."""
        return self._quantized_vae_dropout_decay_encoder

    @quantized_vae_dropout_decay_encoder.setter
    def quantized_vae_dropout_decay_encoder(self, value: float) -> None:
        """Set the encoder dropout decay rate."""
        self._quantized_vae_dropout_decay_encoder = value

    @property
    def quantized_vae_dropout_decay_decoder(self) -> float:
        """Get the decoder dropout decay rate."""
        return self._quantized_vae_dropout_decay_decoder

    @quantized_vae_dropout_decay_decoder.setter
    def quantized_vae_dropout_decay_decoder(self, value: float) -> None:
        """Set the decoder dropout decay rate."""
        self._quantized_vae_dropout_decay_decoder = value

    @property
    def quantized_vae_last_layer_activation(self) -> str:
        """Get the last layer activation."""
        return self._quantized_vae_last_layer_activation

    @quantized_vae_last_layer_activation.setter
    def quantized_vae_last_layer_activation(self, value: str) -> None:
        """Set the last layer activation."""
        self._quantized_vae_last_layer_activation = value

    @property
    def quantized_vae_number_neurons_encoder(self) -> list[int]:
        """Get the encoder neuron sizes."""
        return self._quantized_vae_number_neurons_encoder

    @quantized_vae_number_neurons_encoder.setter
    def quantized_vae_number_neurons_encoder(self, value: list[int]) -> None:
        """Set the encoder neuron sizes."""
        self._quantized_vae_number_neurons_encoder = value

    @property
    def quantized_vae_number_neurons_decoder(self) -> list[int]:
        """Get the decoder neuron sizes."""
        return self._quantized_vae_number_neurons_decoder

    @quantized_vae_number_neurons_decoder.setter
    def quantized_vae_number_neurons_decoder(self, value: list[int]) -> None:
        """Set the decoder neuron sizes."""
        self._quantized_vae_number_neurons_decoder = value

    @property
    def quantized_vae_train_variance(self) -> float:
        """Get the training variance."""
        return self._quantized_vae_train_variance

    @quantized_vae_train_variance.setter
    def quantized_vae_train_variance(self, value: float) -> None:
        """Set the training variance."""
        self._quantized_vae_train_variance = value

    @property
    def quantized_vae_file_name_encoder(self) -> str:
        """Get the encoder filename."""
        return self._quantized_vae_file_name_encoder

    @quantized_vae_file_name_encoder.setter
    def quantized_vae_file_name_encoder(self, value: str) -> None:
        """Set the encoder filename."""
        self._quantized_vae_file_name_encoder = value

    @property
    def quantized_vae_file_name_decoder(self) -> str:
        """Get the decoder filename."""
        return self._quantized_vae_file_name_decoder

    @quantized_vae_file_name_decoder.setter
    def quantized_vae_file_name_decoder(self, value: str) -> None:
        """Set the decoder filename."""
        self._quantized_vae_file_name_decoder = value

    @property
    def quantized_vae_path_output_models(self) -> str:
        """Get the output models path."""
        return self._quantized_vae_path_output_models

    @quantized_vae_path_output_models.setter
    def quantized_vae_path_output_models(self, value: str) -> None:
        """Set the output models path."""
        self._quantized_vae_path_output_models = value

    @property
    def original_input_shape(self) -> tuple:
        """Get the original input shape (before flattening)."""
        return self._original_input_shape

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
        generated_data = self._quantized_vae_algorithm.get_samples(number_samples_per_class)

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