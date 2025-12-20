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

    import numpy as np

    import logging
    import tensorflow as tf

    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    from tensorflow.python.keras.losses import MeanSquaredError, BinaryCrossentropy
    from Engine.architectures.adversarial.AdversarialModel import AdversarialModel
    from Engine.algorithms.adversarial.AdversarialAlgorithm import AdversarialAlgorithm

except ImportError as error:
    logging.error(error)
    sys.exit(-1)

# Default values from your constants file
DEFAULT_ADVERSARIAL_NUMBER_EPOCHS = 200
DEFAULT_ADVERSARIAL_LATENT_DIMENSION = 64
DEFAULT_ADVERSARIAL_TRAINING_ALGORITHM = "Adam"
DEFAULT_ADVERSARIAL_INTERMEDIARY_ACTIVATION = "swish"
DEFAULT_ADVERSARIAL_LAST_ACTIVATION_LAYER = "linear"
DEFAULT_ADVERSARIAL_DROPOUT_DECAY_RATE_G = 0.0
DEFAULT_ADVERSARIAL_DROPOUT_DECAY_RATE_D = 0.1
DEFAULT_ADVERSARIAL_INITIALIZER_MEAN = 0.0
DEFAULT_ADVERSARIAL_INITIALIZER_DEVIATION = 0.15
DEFAULT_ADVERSARIAL_BATCH_SIZE = 512
DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_G = [4096, 2048]
DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_D = [256]
DEFAULT_ADVERSARIAL_RANDOM_latent_standard_deviation = 0.50
DEFAULT_ADVERSARIAL_LOSS_GENERATOR = 'binary_crossentropy'
DEFAULT_ADVERSARIAL_LOSS_DISCRIMINATOR = 'binary_crossentropy'
DEFAULT_ADVERSARIAL_SMOOTHING_RATE = 0.15
DEFAULT_ADVERSARIAL_LATENT_MEAN_DISTRIBUTION = 0.0
DEFAULT_ADVERSARIAL_latent_standard_deviation = 0.50
DEFAULT_ADVERSARIAL_FILE_NAME_DISCRIMINATOR = "discriminator_model"
DEFAULT_ADVERSARIAL_FILE_NAME_GENERATOR = "generator_model"
DEFAULT_ADVERSARIAL_PATH_OUTPUT_MODELS = "models_saved/"
DEFAULT_VARIATIONAL_AUTOENCODER_NUMBER_EPOCHS = 10


class Adversarial:
    """
    A class that instantiates and manages a Conditional Generative adversarial Network (CGAN) model.
    This implementation provides complete configuration, training, and management capabilities
    for adversarial learning tasks within the Synthetic Ocean ecosystem.

    NOW SUPPORTS MULTI-DIMENSIONAL DATA: Automatically handles 1D, 2D, 3D, and N-D inputs
    with OPTIONAL flattening during training and reshaping during generation.

    Attributes:
        _adversarial_algorithm (AdversarialAlgorithm): Manages the adversarial training process
        _adversarial_model (AdversarialModel): Contains generator and discriminator components
        _original_input_shape (tuple): Stores the original shape of input data for reconstruction
        _data_was_flattened (bool): Flag indicating if data was flattened during training

    Configuration Parameters (with getters/setters):
        _adversarial_number_epochs (int): Number of training epochs
        _adversarial_batch_size (int): Size of training batches
        _adversarial_initializer_mean (float): Mean for weight initialization
        _adversarial_initializer_deviation (float): Std dev for weight initialization
        _adversarial_latent_dimension (int): Size of the latent space
        _adversarial_training_algorithm (str): Training algorithm specification
        _adversarial_activation_function (str): Activation function for hidden layers
        _adversarial_dropout_decay_rate_g (float): Generator dropout rate
        _adversarial_dropout_decay_rate_d (float): Discriminator dropout rate
        _adversarial_dense_layer_sizes_g (list[int]): Generator layer sizes
        _adversarial_dense_layer_sizes_d (list[int]): Discriminator layer sizes
        _adversarial_loss_generator (str): Generator loss function
        _adversarial_loss_discriminator (str): Discriminator loss function
        _adversarial_smoothing_rate (float): Label smoothing rate
        _adversarial_latent_mean_distribution (float): Latent space mean
        _adversarial_latent_standard_deviation (float): Latent space std dev
        _adversarial_file_name_discriminator (str): Discriminator model filename
        _adversarial_file_name_generator (str): Generator model filename
        _adversarial_path_output_models (str): Path for saving models
        _adversarial_last_layer_activation (str): Last layer activation function
        _variational_autoencoder_number_epochs (int): Epochs for VAE pre-training
        _number_samples_per_class (dict): Number of samples per class configuration
        _callback_model_monitor: Callback for monitoring model performance
        _callback_early_stop: Callback for early stopping
        _callback_resources_monitor: Callback for monitoring resources
    """

    def __init__(self,
                 number_epochs: int = DEFAULT_ADVERSARIAL_NUMBER_EPOCHS,
                 batch_size: int = DEFAULT_ADVERSARIAL_BATCH_SIZE,
                 initializer_mean: float = DEFAULT_ADVERSARIAL_INITIALIZER_MEAN,
                 initializer_deviation: float = DEFAULT_ADVERSARIAL_INITIALIZER_DEVIATION,
                 latent_dimension: int = DEFAULT_ADVERSARIAL_LATENT_DIMENSION,
                 training_algorithm: str = DEFAULT_ADVERSARIAL_TRAINING_ALGORITHM,
                 activation_function: str = DEFAULT_ADVERSARIAL_INTERMEDIARY_ACTIVATION,
                 dropout_decay_rate_g: float = DEFAULT_ADVERSARIAL_DROPOUT_DECAY_RATE_G,
                 dropout_decay_rate_d: float = DEFAULT_ADVERSARIAL_DROPOUT_DECAY_RATE_D,
                 dense_layer_sizes_g: list[int] = None,
                 dense_layer_sizes_d: list[int] = None,
                 number_classes: int = 2,
                 loss_generator: str = DEFAULT_ADVERSARIAL_LOSS_GENERATOR,
                 loss_discriminator: str = DEFAULT_ADVERSARIAL_LOSS_DISCRIMINATOR,
                 smoothing_rate: float = DEFAULT_ADVERSARIAL_SMOOTHING_RATE,
                 latent_mean_distribution: float = DEFAULT_ADVERSARIAL_LATENT_MEAN_DISTRIBUTION,
                 latent_standard_deviation: float = DEFAULT_ADVERSARIAL_latent_standard_deviation,
                 file_name_discriminator: str = DEFAULT_ADVERSARIAL_FILE_NAME_DISCRIMINATOR,
                 file_name_generator: str = DEFAULT_ADVERSARIAL_FILE_NAME_GENERATOR,
                 path_output_models: str = DEFAULT_ADVERSARIAL_PATH_OUTPUT_MODELS,
                 last_layer_activation: str = DEFAULT_ADVERSARIAL_LAST_ACTIVATION_LAYER,
                 autoencoder_number_epochs: int = DEFAULT_VARIATIONAL_AUTOENCODER_NUMBER_EPOCHS,
                 number_samples_per_class: dict = None,
                 callback_model_monitor=None,
                 callback_early_stop=None,
                 callback_resources_monitor=None,
                 algorithm: AdversarialAlgorithm | None = None,
                 model: AdversarialModel | None = None) -> None:
        """
        Initializes the adversarial instance with configuration parameters.

        Args:
            number_epochs: Training epochs (default: 20)
            batch_size: Batch size (default: 32)
            initializer_mean: Weight init mean (default: 0.0)
            initializer_deviation: Weight init std dev (default: 0.5)
            latent_dimension: Size of the latent space (default: 128)
            training_algorithm: Training algorithm specification (default: "Adam")
            activation_function: Activation function for hidden layers (default: "LeakyReLU")
            dropout_decay_rate_g: Generator dropout rate (default: 0.2)
            dropout_decay_rate_d: Discriminator dropout rate (default: 0.4)
            dense_layer_sizes_g: Generator layer sizes (default: [128])
            dense_layer_sizes_d: Discriminator layer sizes (default: [128])
            loss_generator: Generator loss function (default: 'binary_crossentropy')
            loss_discriminator: Discriminator loss function (default: 'binary_crossentropy')
            smoothing_rate: Label smoothing rate (default: 0.15)
            latent_mean_distribution: Latent space mean (default: 0.0)
            latent_standard_deviation: Latent space std dev (default: 1.0)
            file_name_discriminator: Discriminator model filename (default: "discriminator_model")
            file_name_generator: Generator model filename (default: "generator_model")
            path_output_models: Path for saving models (default: "models_saved/")
            last_layer_activation: Last layer activation function (default: "Sigmoid")
            autoencoder_number_epochs: Epochs for VAE pre-training (default: 10)
            number_samples_per_class: Dict with class sample information (default: None)
            callback_model_monitor: Optional callback for monitoring model performance (default: None)
            callback_early_stop: Optional callback for early stopping (default: None)
            callback_resources_monitor: Optional callback for monitoring resources (default: None)
            algorithm: Optional pre-initialized AdversarialAlgorithm instance (default: None)
            model: Optional pre-initialized AdversarialModel instance (default: None)
        """
        # Store pre-initialized instances if provided
        self._adversarial_algorithm: AdversarialAlgorithm | None = algorithm
        self._adversarial_model: AdversarialModel | None = model

        # ** adversarial Model (GAN) Configuration Parameters **
        self._adversarial_number_epochs: int = number_epochs
        self._adversarial_batch_size: int = batch_size
        self._adversarial_initializer_mean: float = initializer_mean
        self._adversarial_initializer_deviation: float = initializer_deviation
        self._adversarial_latent_dimension: int = latent_dimension
        self._adversarial_training_algorithm: str = training_algorithm
        self._adversarial_activation_function: str = activation_function
        self._adversarial_dropout_decay_rate_g: float = dropout_decay_rate_g
        self._adversarial_dropout_decay_rate_d: float = dropout_decay_rate_d

        # Handle mutable default values safely
        self._adversarial_dense_layer_sizes_g: list[int] = (
            dense_layer_sizes_g
            if dense_layer_sizes_g is not None
            else DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_G.copy()
        )
        self._adversarial_dense_layer_sizes_d: list[int] = (
            dense_layer_sizes_d
            if dense_layer_sizes_d is not None
            else DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_D.copy()
        )

        self._adversarial_loss_generator: str = loss_generator
        self._adversarial_loss_discriminator: str = loss_discriminator
        self._adversarial_smoothing_rate: float = smoothing_rate
        self._adversarial_latent_mean_distribution: float = latent_mean_distribution
        self._adversarial_latent_standard_deviation: float = latent_standard_deviation
        self._adversarial_file_name_discriminator: str = file_name_discriminator
        self._adversarial_file_name_generator: str = file_name_generator
        self._adversarial_path_output_models: str = path_output_models
        self._adversarial_last_layer_activation: str = last_layer_activation
        self._variational_autoencoder_number_epochs: int = autoencoder_number_epochs

        # Initialize the missing attribute
        self._number_samples_per_class: dict = (
            number_samples_per_class if number_samples_per_class is not None else {}
        )

        # Initialize callback attributes
        self._callback_model_monitor = callback_model_monitor
        self._callback_early_stop = callback_early_stop
        self._callback_resources_monitor = callback_resources_monitor

        # Flag to indicate if instances were provided
        self._has_external_algorithm: bool = algorithm is not None
        self._has_external_model: bool = model is not None

        # Storage for original input shape (for multi-dimensional data)
        self._original_input_shape = None
        # Flag to indicate if data was flattened during training
        self._data_was_flattened: bool = False

    def _get_adversarial_model(self, input_shape: tuple[int, ...]) -> None:
        """
        Initialize and configure the adversarial model, including both the generator and discriminator components.
        """
        # If external model was provided, ensure it has the number_samples_per_class configured
        if self._has_external_model:
            # Update the model's number_samples_per_class attribute
            if hasattr(self._adversarial_model, '_generator_number_samples_per_class'):
                self._adversarial_model._generator_number_samples_per_class = self._number_samples_per_class
            if hasattr(self._adversarial_model, '_discriminator_number_samples_per_class'):
                self._adversarial_model._discriminator_number_samples_per_class = self._number_samples_per_class

        # Only create new model if none was provided
        if not self._has_external_model:
            # Adversarial Model setup for Generator and Discriminator
            self._adversarial_model = AdversarialModel(
                latent_dimension=self._adversarial_latent_dimension,
                output_shape=input_shape,
                activation_function=self._adversarial_activation_function,
                initializer_mean=self._adversarial_initializer_mean,
                initializer_deviation=self._adversarial_initializer_deviation,
                dropout_decay_rate_g=self._adversarial_dropout_decay_rate_g,
                dropout_decay_rate_d=self._adversarial_dropout_decay_rate_d,
                last_layer_activation=self._adversarial_last_layer_activation,
                dense_layer_sizes_g=self._adversarial_dense_layer_sizes_g,
                dense_layer_sizes_d=self._adversarial_dense_layer_sizes_d,
                dataset_type=np.float32,
                number_samples_per_class=self._number_samples_per_class
            )

        # Only create new algorithm if none was provided
        if not self._has_external_algorithm:
            # Ensure we have a model to get generator and discriminator from
            if self._adversarial_model is None:
                raise ValueError("AdversarialModel instance is required but was not provided.")

            # Adversarial Algorithm setup for training and model operations
            self._adversarial_algorithm = AdversarialAlgorithm(
                generator_model=self._adversarial_model.get_generator(),
                discriminator_model=self._adversarial_model.get_discriminator(),
                latent_dimension=self._adversarial_latent_dimension,
                loss_generator=self._adversarial_loss_generator,
                loss_discriminator=self._adversarial_loss_discriminator,
                file_name_discriminator=self._adversarial_file_name_discriminator,
                file_name_generator=self._adversarial_file_name_generator,
                models_saved_path=self._adversarial_path_output_models,
                latent_mean_distribution=self._adversarial_latent_mean_distribution,
                latent_standard_deviation=self._adversarial_latent_standard_deviation,
                smoothing_rate=self._adversarial_smoothing_rate
            )
        else:
            # If algorithm was provided externally, update its configuration if needed
            if hasattr(self._adversarial_algorithm, 'latent_dimension'):
                self._adversarial_algorithm.latent_dimension = self._adversarial_latent_dimension

            if hasattr(self._adversarial_algorithm, 'loss_generator'):
                self._adversarial_algorithm.loss_generator = self._adversarial_loss_generator

            if hasattr(self._adversarial_algorithm, 'loss_discriminator'):
                self._adversarial_algorithm.loss_discriminator = self._adversarial_loss_discriminator

            if hasattr(self._adversarial_algorithm, 'file_name_discriminator'):
                self._adversarial_algorithm.file_name_discriminator = self._adversarial_file_name_discriminator

            if hasattr(self._adversarial_algorithm, 'file_name_generator'):
                self._adversarial_algorithm.file_name_generator = self._adversarial_file_name_generator

            if hasattr(self._adversarial_algorithm, 'models_saved_path'):
                self._adversarial_algorithm.models_saved_path = self._adversarial_path_output_models

            if hasattr(self._adversarial_algorithm, 'latent_mean_distribution'):
                self._adversarial_algorithm.latent_mean_distribution = self._adversarial_latent_mean_distribution

            if hasattr(self._adversarial_algorithm, 'latent_standard_deviation'):
                self._adversarial_algorithm.latent_standard_deviation = self._adversarial_latent_standard_deviation

            if hasattr(self._adversarial_algorithm, 'smoothing_rate'):
                self._adversarial_algorithm.smoothing_rate = self._adversarial_smoothing_rate
    @staticmethod
    def _validate_callbacks(callbacks) -> list:
        """
        Validates and sanitizes the callbacks parameter.

        Args:
            callbacks: Can be None, a single callback, or a list of callbacks

        Returns:
            list: A validated list of callbacks

        Raises:
            TypeError: If callbacks parameter has invalid type
            Warning: If any callback in the list doesn't have required methods
        """
        if callbacks is None:
            return []

        # Convert single callback to list
        if not isinstance(callbacks, list):
            callbacks = [callbacks]

        validated_callbacks = []
        for i, callback in enumerate(callbacks):
            try:
                # Check if callback has basic Keras callback interface
                # Most Keras callbacks should have on_epoch_end method
                if not (hasattr(callback, 'on_epoch_end') or
                        hasattr(callback, 'on_batch_end') or
                        hasattr(callback, 'on_train_begin')):
                    logging.warning(
                        f"Callback at index {i} ({type(callback).__name__}) "
                        "may not be a valid Keras callback. It will be included but may cause issues."
                    )

                validated_callbacks.append(callback)

            except Exception as e:
                logging.error(
                    f"Error validating callback at index {i}: {str(e)}. "
                    "This callback will be skipped."
                )
                continue

        return validated_callbacks

    def fit_model(
            self,
            input_shape: tuple[int, ...],
            x_real_samples: np.ndarray,
            y_real_samples: np.ndarray,
            batch_size: int = None,
            epochs: int = None,
            verbose: int = 1,
            callbacks: list = None,
            flatten: bool = True
    ) -> None:
        """
        Executes the complete adversarial training process with optional data flattening.

        NOW SUPPORTS OPTIONAL FLATTENING:
        - flatten=True (default): Automatically flattens N-D input data (1D, 2D, 3D, etc.)
        - flatten=False: Uses data as-is, without flattening
        - Converts labels to categorical format
        - Stores original shape and flatten flag for reconstruction during generation

        Args:
            input_shape: Shape of input data samples (e.g., (784,) for 1D, (28, 28, 1) for 2D, (16, 16, 16) for 3D)
            x_real_samples: Training data samples (can be N-dimensional)
            y_real_samples: Corresponding class labels (1D array of class indices)
            batch_size: Batch size for training (default: uses value from __init__ or DEFAULT_ADVERSARIAL_BATCH_SIZE)
            epochs: Number of training epochs (default: uses value from __init__ or DEFAULT_ADVERSARIAL_NUMBER_EPOCHS)
            verbose: Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch) (default: 1)
            callbacks: List of Keras callbacks to apply during training (default: uses callbacks from __init__)
                      Can be a single callback or a list of callbacks. Invalid callbacks will be filtered out with warnings.
            flatten: If True, flattens multi-dimensional input data. If False, uses data as-is (default: True)

        Raises:
            TypeError: If callbacks parameter has invalid type
            ValueError: If batch_size or epochs are invalid (< 1)

        Process:
            1. Validates input parameters (batch_size, epochs, callbacks)
            2. Optionally flattens multi-dimensional input data (if flatten=True)
            3. Converts labels to categorical format
            4. Auto-detects number of classes if not set
            5. Initializes model architecture (or uses provided)
            6. Configures optimizers and loss functions
            7. Merges default callbacks with provided callbacks
            8. Executes adversarial training
            9. Manages model saving and monitoring
        """
        # Parameter validation and defaults
        if batch_size is None:
            batch_size = self._adversarial_batch_size
        elif batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        if epochs is None:
            epochs = self._adversarial_number_epochs
        elif epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")

        if not isinstance(verbose, int) or verbose not in [0, 1, 2]:
            logging.warning(
                f"verbose should be 0, 1, or 2. Got {verbose}. Using default value 1."
            )
            verbose = 1

        # Store original input shape for later reconstruction
        self._original_input_shape = input_shape
        self._data_was_flattened = flatten

        # Determine the dimension to use for model initialization
        if flatten:
            # Calculate total flattened dimension
            flattened_dim = int(np.prod(input_shape))

            # Flatten the input data if it has more than 2 dimensions
            # (batch_size, ...) -> (batch_size, flattened_features)
            if len(x_real_samples.shape) > 2:
                x_real_samples_processed = x_real_samples.reshape(x_real_samples.shape[0], -1)
            else:
                x_real_samples_processed = x_real_samples

            model_input_dim = flattened_dim
        else:
            # Use data as-is without flattening
            x_real_samples_processed = x_real_samples
            model_input_dim = input_shape

        # Auto-detect number_samples_per_class if not set
        if not self._number_samples_per_class:
            unique_classes = np.unique(y_real_samples)
            self._number_samples_per_class = {
                "number_classes": len(unique_classes),
                "classes": {int(cls): np.sum(y_real_samples == cls) for cls in unique_classes}
            }

        # Convert labels to categorical format
        y_categorical = to_categorical(y_real_samples, num_classes=self._number_samples_per_class["number_classes"])

        # Initialize the adversarial model with appropriate dimension
        self._get_adversarial_model(model_input_dim)

        # Print the model summaries for the generator and discriminator if available
        if self._adversarial_model is not None:
            self._adversarial_model.get_generator()
            self._adversarial_model.get_discriminator()

        # Ensure we have an algorithm
        if self._adversarial_algorithm is None:
            raise ValueError("AdversarialAlgorithm instance is required but was not provided or created.")

        # Set up optimizers for the generator and discriminator
        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.5, beta_2=0.9)

        # Compile the adversarial algorithm with binary cross-entropy loss
        self._adversarial_algorithm.compile(
            generator_optimizer,
            discriminator_optimizer,
            BinaryCrossentropy(),
            BinaryCrossentropy()
        )

        # Build callbacks list starting with default callbacks from __init__
        callbacks_list = []

        if self._callback_model_monitor is not None:
            callbacks_list.append(self._callback_model_monitor)

        if self._callback_resources_monitor is not None:
            callbacks_list.append(self._callback_resources_monitor)

        if self._callback_early_stop is not None:
            callbacks_list.append(self._callback_early_stop)

        # Validate and add user-provided callbacks
        if callbacks is not None:
            try:
                validated_callbacks = self._validate_callbacks(callbacks)
                callbacks_list.extend(validated_callbacks)
            except TypeError as e:
                logging.error(
                    f"Invalid callbacks parameter: {str(e)}. "
                    "Using only default callbacks from initialization."
                )
            except Exception as e:
                logging.error(
                    f"Unexpected error processing callbacks: {str(e)}. "
                    "Using only default callbacks from initialization."
                )

        # Fit the model with processed samples and the corresponding labels
        self._adversarial_algorithm.fit(
            x_real_samples_processed,
            y_categorical,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks_list if callbacks_list else None
        )

    def get_samples(self, number_samples_per_class):
        """
        Generate synthetic samples and optionally reshape them to original input shape.

        NOW SUPPORTS OPTIONAL RESHAPING:
        - If data was flattened during training: Automatically reshapes to original dimensions
        - If data was not flattened: Returns data as-is from generator
        - Works with 1D, 2D, 3D, and N-D data

        Args:
            number_samples_per_class: Dictionary specifying number of samples per class
                Format 1: {"number_classes": N, "classes": {0: n0, 1: n1, ...}}
                Format 2 (simplified): {"number_classes": N, 0: n0, 1: n1, ...}

        Returns:
            np.ndarray: Generated samples in appropriate shape
                - If flattened during training: (total_samples, *original_input_shape)
                - If not flattened: (total_samples, *model_output_shape)
                - Example: For 3D input (16, 16, 16) with 100 total samples -> (100, 16, 16, 16)
        """
        # Generate samples from algorithm
        generated_data = self._adversarial_algorithm.get_samples(number_samples_per_class)

        # If data was not flattened during training, return as-is
        if not self._data_was_flattened:
            # Convert dict to array if needed
            if isinstance(generated_data, dict):
                all_samples = []
                for label in sorted(generated_data.keys()):
                    all_samples.append(generated_data[label])
                return np.concatenate(all_samples, axis=0)
            return generated_data

        # Data was flattened - need to reshape back to original shape
        # Check if we have stored original input shape
        if not hasattr(self, '_original_input_shape') or self._original_input_shape is None:
            # If no original shape stored, return as generated
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

    # Additional getters for the algorithm and model
    @property
    def adversarial_algorithm(self) -> AdversarialAlgorithm | None:
        """Get the adversarial algorithm instance."""
        return self._adversarial_algorithm

    @property
    def adversarial_model(self) -> AdversarialModel | None:
        """Get the adversarial model instance."""
        return self._adversarial_model

    @property
    def data_was_flattened(self) -> bool:
        """Get flag indicating if data was flattened during training."""
        return self._data_was_flattened

    # Getter and setter for number_samples_per_class
    @property
    def number_samples_per_class(self) -> dict:
        """Get the number of samples per class configuration."""
        return self._number_samples_per_class

    @number_samples_per_class.setter
    def number_samples_per_class(self, value: dict) -> None:
        """Set the number of samples per class configuration."""
        self._number_samples_per_class = value

    # Getter and setter for callback_model_monitor
    @property
    def callback_model_monitor(self):
        """Get the model monitoring callback."""
        return self._callback_model_monitor

    @callback_model_monitor.setter
    def callback_model_monitor(self, value) -> None:
        """Set the model monitoring callback."""
        self._callback_model_monitor = value

    # Getter and setter for callback_early_stop
    @property
    def callback_early_stop(self):
        """Get the early stopping callback."""
        return self._callback_early_stop

    @callback_early_stop.setter
    def callback_early_stop(self, value) -> None:
        """Set the early stopping callback."""
        self._callback_early_stop = value

    # Getter and setter for callback_resources_monitor
    @property
    def callback_resources_monitor(self):
        """Get the resources monitoring callback."""
        return self._callback_resources_monitor

    @callback_resources_monitor.setter
    def callback_resources_monitor(self, value) -> None:
        """Set the resources monitoring callback."""
        self._callback_resources_monitor = value

    # Getter and setter for adversarial_number_epochs
    @property
    def adversarial_number_epochs(self) -> int:
        """Get the number of training epochs."""
        return self._adversarial_number_epochs

    @adversarial_number_epochs.setter
    def adversarial_number_epochs(self, value: int) -> None:
        """Set the number of training epochs."""
        self._adversarial_number_epochs = value

    # Getter and setter for adversarial_batch_size
    @property
    def adversarial_batch_size(self) -> int:
        """Get the batch size."""
        return self._adversarial_batch_size

    @adversarial_batch_size.setter
    def adversarial_batch_size(self, value: int) -> None:
        """Set the batch size."""
        self._adversarial_batch_size = value

    # Getter and setter for adversarial_initializer_mean
    @property
    def adversarial_initializer_mean(self) -> float:
        """Get the mean for weight initialization."""
        return self._adversarial_initializer_mean

    @adversarial_initializer_mean.setter
    def adversarial_initializer_mean(self, value: float) -> None:
        """Set the mean for weight initialization."""
        self._adversarial_initializer_mean = value

    # Getter and setter for adversarial_initializer_deviation
    @property
    def adversarial_initializer_deviation(self) -> float:
        """Get the standard deviation for weight initialization."""
        return self._adversarial_initializer_deviation

    @adversarial_initializer_deviation.setter
    def adversarial_initializer_deviation(self, value: float) -> None:
        """Set the standard deviation for weight initialization."""
        self._adversarial_initializer_deviation = value

    # Getter and setter for adversarial_latent_dimension
    @property
    def adversarial_latent_dimension(self) -> int:
        """Get the size of the latent space."""
        return self._adversarial_latent_dimension

    @adversarial_latent_dimension.setter
    def adversarial_latent_dimension(self, value: int) -> None:
        """Set the size of the latent space."""
        self._adversarial_latent_dimension = value

    # Getter and setter for adversarial_training_algorithm
    @property
    def adversarial_training_algorithm(self) -> str:
        """Get the training algorithm specification."""
        return self._adversarial_training_algorithm

    @adversarial_training_algorithm.setter
    def adversarial_training_algorithm(self, value: str) -> None:
        """Set the training algorithm specification."""
        self._adversarial_training_algorithm = value

    # Getter and setter for adversarial_activation_function
    @property
    def adversarial_activation_function(self) -> str:
        """Get the activation function for hidden layers."""
        return self._adversarial_activation_function

    @adversarial_activation_function.setter
    def adversarial_activation_function(self, value: str) -> None:
        """Set the activation function for hidden layers."""
        self._adversarial_activation_function = value

    # Getter and setter for adversarial_dropout_decay_rate_g
    @property
    def adversarial_dropout_decay_rate_g(self) -> float:
        """Get the generator dropout rate."""
        return self._adversarial_dropout_decay_rate_g

    @adversarial_dropout_decay_rate_g.setter
    def adversarial_dropout_decay_rate_g(self, value: float) -> None:
        """Set the generator dropout rate."""
        self._adversarial_dropout_decay_rate_g = value

    # Getter and setter for adversarial_dropout_decay_rate_d
    @property
    def adversarial_dropout_decay_rate_d(self) -> float:
        """Get the discriminator dropout rate."""
        return self._adversarial_dropout_decay_rate_d

    @adversarial_dropout_decay_rate_d.setter
    def adversarial_dropout_decay_rate_d(self, value: float) -> None:
        """Set the discriminator dropout rate."""
        self._adversarial_dropout_decay_rate_d = value

    # Getter and setter for adversarial_dense_layer_sizes_g
    @property
    def adversarial_dense_layer_sizes_g(self) -> list[int]:
        """Get the generator layer sizes."""
        return self._adversarial_dense_layer_sizes_g

    @adversarial_dense_layer_sizes_g.setter
    def adversarial_dense_layer_sizes_g(self, value: list[int]) -> None:
        """Set the generator layer sizes."""
        self._adversarial_dense_layer_sizes_g = value

    # Getter and setter for adversarial_dense_layer_sizes_d
    @property
    def adversarial_dense_layer_sizes_d(self) -> list[int]:
        """Get the discriminator layer sizes."""
        return self._adversarial_dense_layer_sizes_d

    @adversarial_dense_layer_sizes_d.setter
    def adversarial_dense_layer_sizes_d(self, value: list[int]) -> None:
        """Set the discriminator layer sizes."""
        self._adversarial_dense_layer_sizes_d = value

    # Getter and setter for adversarial_loss_generator
    @property
    def adversarial_loss_generator(self) -> str:
        """Get the generator loss function."""
        return self._adversarial_loss_generator

    @adversarial_loss_generator.setter
    def adversarial_loss_generator(self, value: str) -> None:
        """Set the generator loss function."""
        self._adversarial_loss_generator = value

    # Getter and setter for adversarial_loss_discriminator
    @property
    def adversarial_loss_discriminator(self) -> str:
        """Get the discriminator loss function."""
        return self._adversarial_loss_discriminator

    @adversarial_loss_discriminator.setter
    def adversarial_loss_discriminator(self, value: str) -> None:
        """Set the discriminator loss function."""
        self._adversarial_loss_discriminator = value

    # Getter and setter for adversarial_smoothing_rate
    @property
    def adversarial_smoothing_rate(self) -> float:
        """Get the label smoothing rate."""
        return self._adversarial_smoothing_rate

    @adversarial_smoothing_rate.setter
    def adversarial_smoothing_rate(self, value: float) -> None:
        """Set the label smoothing rate."""
        self._adversarial_smoothing_rate = value

    # Getter and setter for adversarial_latent_mean_distribution
    @property
    def adversarial_latent_mean_distribution(self) -> float:
        """Get the latent space mean distribution."""
        return self._adversarial_latent_mean_distribution

    @adversarial_latent_mean_distribution.setter
    def adversarial_latent_mean_distribution(self, value: float) -> None:
        """Set the latent space mean distribution."""
        self._adversarial_latent_mean_distribution = value

    # Getter and setter for adversarial_latent_standard_deviation
    @property
    def adversarial_latent_standard_deviation(self) -> float:
        """Get the latent space standard deviation."""
        return self._adversarial_latent_standard_deviation

    @adversarial_latent_standard_deviation.setter
    def adversarial_latent_standard_deviation(self, value: float) -> None:
        """Set the latent space standard deviation."""
        self._adversarial_latent_standard_deviation = value

    # Getter and setter for adversarial_file_name_discriminator
    @property
    def adversarial_file_name_discriminator(self) -> str:
        """Get the discriminator model filename."""
        return self._adversarial_file_name_discriminator

    @adversarial_file_name_discriminator.setter
    def adversarial_file_name_discriminator(self, value: str) -> None:
        """Set the discriminator model filename."""
        self._adversarial_file_name_discriminator = value

    # Getter and setter for adversarial_file_name_generator
    @property
    def adversarial_file_name_generator(self) -> str:
        """Get the generator model filename."""
        return self._adversarial_file_name_generator

    @adversarial_file_name_generator.setter
    def adversarial_file_name_generator(self, value: str) -> None:
        """Set the generator model filename."""
        self._adversarial_file_name_generator = value

    # Getter and setter for adversarial_path_output_models
    @property
    def adversarial_path_output_models(self) -> str:
        """Get the path for saving models."""
        return self._adversarial_path_output_models

    @adversarial_path_output_models.setter
    def adversarial_path_output_models(self, value: str) -> None:
        """Set the path for saving models."""
        self._adversarial_path_output_models = value

    @property
    def original_input_shape(self) -> tuple:
        """Get the original input shape (before flattening)."""
        return self._original_input_shape