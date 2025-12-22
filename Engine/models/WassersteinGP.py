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
    import torch
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    from Engine.callbacks.CallbackEarlyStop import EarlyStopping
    from Engine.callbacks.CallbackModel import ModelMonitorCallback
    from Engine.architectures.wasserstein_gp.WassersteinGPModel import WassersteinGPModel
    from Engine.algorithms.wasserstein_gp.WassersteinGPAlgorithm import WassersteinGPAlgorithm

except ImportError as error:
    logging.error(error)
    sys.exit(-1)

# Default values from your constants file
DEFAULT_WASSERSTEIN_GAN_GP_LATENT_DIMENSION = 64
DEFAULT_WASSERSTEIN_GAN_GP_TRAINING_ALGORITHM = "Adam"
DEFAULT_WASSERSTEIN_GAN_GP_ACTIVATION = "LeakyReLU"
DEFAULT_WASSERSTEIN_GAN_GP_DROPOUT_DECAY_RATE_G = 0.0
DEFAULT_WASSERSTEIN_GAN_GP_DROPOUT_DECAY_RATE_D = 0.1
DEFAULT_WASSERSTEIN_GAN_GP_BATCH_SIZE = 512
DEFAULT_WASSERSTEIN_GAN_GP_NUMBER_CLASSES = 2
DEFAULT_WASSERSTEIN_GAN_GP_NUMBER_EPOCHS = 100
DEFAULT_WASSERSTEIN_GAN_GP_DENSE_LAYERS_SETTINGS_GENERATOR = [512]
DEFAULT_WASSERSTEIN_GAN_GP_DENSE_LAYERS_SETTINGS_DISCRIMINATOR = [256]
DEFAULT_WASSERSTEIN_GAN_GP_LOSS = "wasserstein"
DEFAULT_WASSERSTEIN_GAN_GP_MOMENTUM = 0.8
DEFAULT_WASSERSTEIN_GAN_GP_LAST_ACTIVATION_LAYER = "linear"
DEFAULT_WASSERSTEIN_GAN_GP_INITIALIZER_MEAN = 0.0
DEFAULT_WASSERSTEIN_GAN_GP_INITIALIZER_DEVIATION = 0.125
DEFAULT_WASSERSTEIN_GAN_GP_OPTIMIZER_GENERATOR_LEARNING = 0.0001
DEFAULT_WASSERSTEIN_GAN_GP_OPTIMIZER_DISCRIMINATOR_LEARNING = 0.0002
DEFAULT_WASSERSTEIN_GAN_GP_OPTIMIZER_GENERATOR_BETA = 0.5
DEFAULT_WASSERSTEIN_GAN_GP_OPTIMIZER_DISCRIMINATOR_BETA = 0.5
DEFAULT_WASSERSTEIN_GAN_GP_ADAM_LEARNING_RATE = 0.0001
DEFAULT_WASSERSTEIN_GAN_GP_ADAM_BETA = 0.5
DEFAULT_WASSERSTEIN_GAN_GP_DISCRIMINATOR_STEPS = 3
DEFAULT_WASSERSTEIN_GAN_GP_SMOOTHING_RATE = 0.15
DEFAULT_WASSERSTEIN_GAN_GP_LATENT_MEAN_DISTRIBUTION = 0.5
DEFAULT_WASSERSTEIN_GAN_GP_latent_standard_deviation = 0.5
DEFAULT_WASSERSTEIN_GAN_GP_GRADIENT_PENALTY = 10.0
DEFAULT_WASSERSTEIN_GAN_GP_FILE_NAME_DISCRIMINATOR = "discriminator_model"
DEFAULT_WASSERSTEIN_GAN_GP_FILE_NAME_GENERATOR = "generator_model"
DEFAULT_WASSERSTEIN_GAN_GP_PATH_OUTPUT_MODELS = "models_saved/"


class WassersteinGP:
    """
    A class that implements a wasserstein Generative adversarial Network with Gradient Penalty (WGAN-GP).
    This version improves upon standard WGAN by using gradient penalty instead of weight clipping
    to enforce the Lipschitz constraint, leading to more stable training and higher quality results.

    NOW SUPPORTS MULTI-DIMENSIONAL DATA: Automatically handles 1D, 2D, 3D, and N-D inputs
    with OPTIONAL flattening during training and reshaping during generation.

    Attributes:
        _wasserstein_gp_algorithm (WassersteinGPAlgorithm): Orchestrates the WGAN-GP training process
        _wasserstein_gp_model (WassersteinGPModel): Stores the generator and critic models
        _original_input_shape (tuple): Stores the original shape of input data for reconstruction
        _data_was_flattened (bool): Flag indicating if data was flattened during training

    Configuration Parameters (with getters/setters):
        _wasserstein_gp_latent_dimension (int): Dimensionality of the latent space
        _wasserstein_gp_training_algorithm (str): Type of training algorithm used
        _wasserstein_gp_activation_function (str): Activation function for hidden layers
        _wasserstein_gp_dropout_decay_rate_g (float): Dropout rate decay for generator
        _wasserstein_gp_dropout_decay_rate_d (float): Dropout rate decay for critic
        _wasserstein_gp_dense_layer_sizes_generator (list[int]): Layer sizes for generator
        _wasserstein_gp_dense_layer_sizes_discriminator (list[int]): Layer sizes for critic
        _wasserstein_gp_batch_size (int): Batch size for training
        _wasserstein_gp_number_epochs (int): Number of training epochs
        _wasserstein_gp_number_classes (int): Number of output classes
        _wasserstein_gp_loss_function (str): Base loss function used
        _wasserstein_gp_momentum (float): Momentum parameter for optimizers
        _wasserstein_gp_last_activation_layer (str): Activation for final layer
        _wasserstein_gp_initializer_mean (float): Mean for weight initialization
        _wasserstein_gp_initializer_deviation (float): Std dev for weight initialization
        _wasserstein_gp_optimizer_generator_learning_rate (float): Generator learning rate
        _wasserstein_gp_optimizer_discriminator_learning_rate (float): Critic learning rate
        _wasserstein_gp_optimizer_generator_beta (float): Beta1 for generator optimizer
        _wasserstein_gp_optimizer_discriminator_beta (float): Beta1 for critic optimizer
        _wasserstein_gp_discriminator_steps (int): Number of critic steps per generator step
        _wasserstein_gp_smoothing_rate (float): Label smoothing rate
        _wasserstein_gp_latent_mean_distribution (float): Distribution type for latent space
        _wasserstein_gp_latent_standard_deviation (float): Std dev for latent distribution
        _wasserstein_gp_gradient_penalty (float): Weight for gradient penalty term
        _wasserstein_gp_file_name_discriminator (str): Filename for saving critic
        _wasserstein_gp_file_name_generator (str): Filename for saving generator
        _wasserstein_gp_path_output_models (str): Path for saving models
    """

    def __init__(
            self,
            latent_dimension: int = DEFAULT_WASSERSTEIN_GAN_GP_LATENT_DIMENSION,
            training_algorithm: str = DEFAULT_WASSERSTEIN_GAN_GP_TRAINING_ALGORITHM,
            activation_function: str = DEFAULT_WASSERSTEIN_GAN_GP_ACTIVATION,
            dropout_decay_rate_g: float = DEFAULT_WASSERSTEIN_GAN_GP_DROPOUT_DECAY_RATE_G,
            dropout_decay_rate_d: float = DEFAULT_WASSERSTEIN_GAN_GP_DROPOUT_DECAY_RATE_D,
            dense_layer_sizes_generator: list[int] = None,
            dense_layer_sizes_discriminator: list[int] = None,
            batch_size: int = DEFAULT_WASSERSTEIN_GAN_GP_BATCH_SIZE,
            number_epochs: int = DEFAULT_WASSERSTEIN_GAN_GP_NUMBER_EPOCHS,
            number_classes: int = DEFAULT_WASSERSTEIN_GAN_GP_NUMBER_CLASSES,
            loss_function: str = DEFAULT_WASSERSTEIN_GAN_GP_LOSS,
            momentum: float = DEFAULT_WASSERSTEIN_GAN_GP_MOMENTUM,
            last_activation_layer: str = DEFAULT_WASSERSTEIN_GAN_GP_LAST_ACTIVATION_LAYER,
            initializer_mean: float = DEFAULT_WASSERSTEIN_GAN_GP_INITIALIZER_MEAN,
            initializer_deviation: float = DEFAULT_WASSERSTEIN_GAN_GP_INITIALIZER_DEVIATION,
            optimizer_generator_learning_rate: float = DEFAULT_WASSERSTEIN_GAN_GP_OPTIMIZER_GENERATOR_LEARNING,
            optimizer_discriminator_learning_rate: float = DEFAULT_WASSERSTEIN_GAN_GP_OPTIMIZER_DISCRIMINATOR_LEARNING,
            optimizer_generator_beta: float = DEFAULT_WASSERSTEIN_GAN_GP_OPTIMIZER_GENERATOR_BETA,
            optimizer_discriminator_beta: float = DEFAULT_WASSERSTEIN_GAN_GP_OPTIMIZER_DISCRIMINATOR_BETA,
            discriminator_steps: int = DEFAULT_WASSERSTEIN_GAN_GP_DISCRIMINATOR_STEPS,
            smoothing_rate: float = DEFAULT_WASSERSTEIN_GAN_GP_SMOOTHING_RATE,
            latent_mean_distribution: float = DEFAULT_WASSERSTEIN_GAN_GP_LATENT_MEAN_DISTRIBUTION,
            latent_standard_deviation: float = DEFAULT_WASSERSTEIN_GAN_GP_latent_standard_deviation,
            gradient_penalty: float = DEFAULT_WASSERSTEIN_GAN_GP_GRADIENT_PENALTY,
            file_name_discriminator: str = DEFAULT_WASSERSTEIN_GAN_GP_FILE_NAME_DISCRIMINATOR,
            file_name_generator: str = DEFAULT_WASSERSTEIN_GAN_GP_FILE_NAME_GENERATOR,
            path_output_models: str = DEFAULT_WASSERSTEIN_GAN_GP_PATH_OUTPUT_MODELS,
            algorithm: WassersteinGPAlgorithm | None = None,
            model: WassersteinGPModel | None = None
    ) -> None:
        """
        Initializes the WGAN-GP instance with configuration parameters.

        Args:
            latent_dimension: Latent space dimension (default: 128)
            training_algorithm: Training algorithm (default: "Adam")
            activation_function: Activation function (default: "LeakyReLU")
            dropout_decay_rate_g: Generator dropout rate (default: 0.2)
            dropout_decay_rate_d: Critic dropout rate (default: 0.4)
            dense_layer_sizes_generator: Generator layer sizes (default: [128])
            dense_layer_sizes_discriminator: Critic layer sizes (default: [128])
            batch_size: Batch size (default: 32)
            number_epochs: Training epochs (default: 20)
            number_classes: Number of classes (default: 2)
            loss_function: loss function (default: "binary_crossentropy")
            momentum: Momentum parameter (default: 0.8)
            last_activation_layer: Last layer activation (default: "sigmoid")
            initializer_mean: Weight init mean (default: 0.0)
            initializer_deviation: Weight init std dev (default: 0.125)
            optimizer_generator_learning_rate: Generator learning rate (default: 0.0001)
            optimizer_discriminator_learning_rate: Critic learning rate (default: 0.0001)
            optimizer_generator_beta: Generator optimizer beta (default: 0.5)
            optimizer_discriminator_beta: Critic optimizer beta (default: 0.5)
            discriminator_steps: Critic steps per generator step (default: 3)
            smoothing_rate: Label smoothing rate (default: 0.15)
            latent_mean_distribution: Latent distribution mean (default: 0.0)
            latent_standard_deviation: Latent distribution std dev (default: 0.125)
            gradient_penalty: Gradient penalty weight (default: 10.0)
            file_name_discriminator: Critic filename (default: "discriminator_model")
            file_name_generator: Generator filename (default: "generator_model")
            path_output_models: Output models path (default: "models_saved/")
            algorithm: Optional pre-initialized WassersteinGPAlgorithm (default: None)
            model: Optional pre-initialized WassersteinGPModel (default: None)
        """
        # Store pre-initialized instances if provided
        self._wasserstein_gp_algorithm: WassersteinGPAlgorithm | None = algorithm
        self._wasserstein_gp_model: WassersteinGPModel | None = model

        # ** wasserstein_gp GAN with Gradient Penalty (WGAN-GP) Configuration Parameters **
        self._wasserstein_gp_latent_dimension: int = latent_dimension
        self._wasserstein_gp_training_algorithm: str = training_algorithm
        self._wasserstein_gp_activation_function: str = activation_function
        self._wasserstein_gp_dropout_decay_rate_g: float = dropout_decay_rate_g
        self._wasserstein_gp_dropout_decay_rate_d: float = dropout_decay_rate_d

        # Handle mutable default values safely
        self._wasserstein_gp_dense_layer_sizes_generator: list[int] = (
            dense_layer_sizes_generator
            if dense_layer_sizes_generator is not None
            else DEFAULT_WASSERSTEIN_GAN_GP_DENSE_LAYERS_SETTINGS_GENERATOR.copy()
        )
        self._wasserstein_gp_dense_layer_sizes_discriminator: list[int] = (
            dense_layer_sizes_discriminator
            if dense_layer_sizes_discriminator is not None
            else DEFAULT_WASSERSTEIN_GAN_GP_DENSE_LAYERS_SETTINGS_DISCRIMINATOR.copy()
        )

        self._wasserstein_gp_batch_size: int = batch_size
        self._wasserstein_gp_number_epochs: int = number_epochs
        self._wasserstein_gp_number_classes: int = number_classes
        self._wasserstein_gp_loss_function: str = loss_function
        self._wasserstein_gp_momentum: float = momentum
        self._wasserstein_gp_last_activation_layer: str = last_activation_layer
        self._wasserstein_gp_initializer_mean: float = initializer_mean
        self._wasserstein_gp_initializer_deviation: float = initializer_deviation
        self._wasserstein_gp_optimizer_generator_learning_rate: float = optimizer_generator_learning_rate
        self._wasserstein_gp_optimizer_discriminator_learning_rate: float = optimizer_discriminator_learning_rate
        self._wasserstein_gp_optimizer_generator_beta: float = optimizer_generator_beta
        self._wasserstein_gp_optimizer_discriminator_beta: float = optimizer_discriminator_beta
        self._wasserstein_gp_discriminator_steps: int = discriminator_steps
        self._wasserstein_gp_smoothing_rate: float = smoothing_rate
        self._wasserstein_gp_latent_mean_distribution: float = latent_mean_distribution
        self._wasserstein_gp_latent_standard_deviation: float = latent_standard_deviation
        self._wasserstein_gp_gradient_penalty: float = gradient_penalty
        self._wasserstein_gp_file_name_discriminator: str = file_name_discriminator
        self._wasserstein_gp_file_name_generator: str = file_name_generator
        self._wasserstein_gp_path_output_models: str = path_output_models

        # Flags to indicate if instances were provided
        self._has_external_algorithm: bool = algorithm is not None
        self._has_external_model: bool = model is not None

        # Storage for original input shape (for multi-dimensional data)
        self._original_input_shape = None
        # Flag to indicate if data was flattened during training
        self._data_was_flattened: bool = False

    @staticmethod
    def _calculate_samples_per_class(y_labels: np.ndarray) -> dict:
        """
        Calculate the distribution of samples per class from labels.

        Args:
            y_labels: Labels array

        Returns:
            dict: Dictionary with 'classes' and 'number_classes' keys
        """
        # Handle one-hot encoded labels
        if len(y_labels.shape) == 2 and y_labels.shape[1] > 1:
            y_labels = np.argmax(y_labels, axis=1)

        # Count samples per class
        unique, counts = np.unique(y_labels, return_counts=True)

        return {
            "classes": dict(zip(unique.tolist(), counts.tolist())),
            "number_classes": len(unique)
        }

    def _get_wasserstein_gp(self, input_shape: tuple[int, ...], number_samples_per_class: dict) -> None:
        """
        Initialize and configure the wasserstein_gp GAN model, including generator and critic components.

        This method sets up a wasserstein_gp Generative adversarial Network (WGAN) by configuring the generator and critic
        models using the `WassersteinGPModel` class and links them with the `WassersteinGPAlgorithm` class. The models
        are initialized with specified configurations such as latent dimension, activation functions, dropout rates,
        and layer sizes for both the generator and critic.

        If pre-initialized instances were provided in the constructor, they are used instead of creating new ones.

        Args:
            input_shape: The shape of the input data, which determines the output shape for the models.
            number_samples_per_class: Dictionary containing class distribution information

        Initializes:
            self._wasserstein_gp_model:
                An instance of the `WassersteinGPModel` class, including the generator and critic setup,
                with configurations for activation functions, layer sizes, dropout rates, and more.
            self._wasserstein_gp_algorithm:
                An instance of the `WassersteinGPAlgorithm` class, managing the WGAN-GP training process, including
                generator and critic loss functions, gradient penalty, and model parameters.
        """
        # Only create new model if none was provided
        if not self._has_external_model:
            # wasserstein_gp Model setup for the Generator and Discriminator
            # Pass number_samples_per_class during initialization
            self._wasserstein_gp_model = WassersteinGPModel(
                latent_dimension=self._wasserstein_gp_latent_dimension,
                output_shape=input_shape,
                activation_function=self._wasserstein_gp_activation_function,
                initializer_mean=self._wasserstein_gp_initializer_mean,
                initializer_deviation=self._wasserstein_gp_initializer_deviation,
                dropout_decay_rate_g=self._wasserstein_gp_dropout_decay_rate_g,
                dropout_decay_rate_d=self._wasserstein_gp_dropout_decay_rate_d,
                last_layer_activation=self._wasserstein_gp_last_activation_layer,
                dense_layer_sizes_g=self._wasserstein_gp_dense_layer_sizes_generator,
                dense_layer_sizes_d=self._wasserstein_gp_dense_layer_sizes_discriminator,
                dataset_type=np.float32,
                number_samples_per_class=number_samples_per_class
            )

        # Only create new algorithm if none was provided
        if not self._has_external_algorithm:
            # Ensure we have a model to get components from
            if self._wasserstein_gp_model is None:
                raise ValueError("WassersteinGPModel instance is required but was not provided.")

            # wasserstein_gp Algorithm setup for training and model operations
            self._wasserstein_gp_algorithm = WassersteinGPAlgorithm(
                generator_model=self._wasserstein_gp_model.get_generator(),
                discriminator_model=self._wasserstein_gp_model.get_discriminator(),
                latent_dimension=self._wasserstein_gp_latent_dimension,
                generator_loss_fn=self._wasserstein_gp_loss_function,
                discriminator_loss_fn=self._wasserstein_gp_loss_function,
                file_name_discriminator=self._wasserstein_gp_file_name_discriminator,
                file_name_generator=self._wasserstein_gp_file_name_generator,
                models_saved_path=self._wasserstein_gp_path_output_models,
                latent_mean_distribution=self._wasserstein_gp_latent_mean_distribution,
                latent_standard_deviation=self._wasserstein_gp_latent_standard_deviation,
                smoothing_rate=self._wasserstein_gp_smoothing_rate,
                gradient_penalty_weight=self._wasserstein_gp_gradient_penalty,
                discriminator_steps=self._wasserstein_gp_discriminator_steps
            )
        else:
            # If algorithm was provided externally, update its configuration if needed
            if hasattr(self._wasserstein_gp_algorithm, 'latent_dimension'):
                self._wasserstein_gp_algorithm.latent_dimension = self._wasserstein_gp_latent_dimension

            if hasattr(self._wasserstein_gp_algorithm, 'generator_loss_fn'):
                self._wasserstein_gp_algorithm.generator_loss_fn = self._wasserstein_gp_loss_function

            if hasattr(self._wasserstein_gp_algorithm, 'discriminator_loss_fn'):
                self._wasserstein_gp_algorithm.discriminator_loss_fn = self._wasserstein_gp_loss_function

            if hasattr(self._wasserstein_gp_algorithm, 'file_name_discriminator'):
                self._wasserstein_gp_algorithm.file_name_discriminator = self._wasserstein_gp_file_name_discriminator

            if hasattr(self._wasserstein_gp_algorithm, 'file_name_generator'):
                self._wasserstein_gp_algorithm.file_name_generator = self._wasserstein_gp_file_name_generator

            if hasattr(self._wasserstein_gp_algorithm, 'models_saved_path'):
                self._wasserstein_gp_algorithm.models_saved_path = self._wasserstein_gp_path_output_models

            if hasattr(self._wasserstein_gp_algorithm, 'latent_mean_distribution'):
                self._wasserstein_gp_algorithm.latent_mean_distribution = self._wasserstein_gp_latent_mean_distribution

            if hasattr(self._wasserstein_gp_algorithm, 'latent_standard_deviation'):
                self._wasserstein_gp_algorithm.latent_standard_deviation = self._wasserstein_gp_latent_standard_deviation

            if hasattr(self._wasserstein_gp_algorithm, 'smoothing_rate'):
                self._wasserstein_gp_algorithm.smoothing_rate = self._wasserstein_gp_smoothing_rate

            if hasattr(self._wasserstein_gp_algorithm, 'gradient_penalty_weight'):
                self._wasserstein_gp_algorithm.gradient_penalty_weight = self._wasserstein_gp_gradient_penalty

            if hasattr(self._wasserstein_gp_algorithm, 'discriminator_steps'):
                self._wasserstein_gp_algorithm.discriminator_steps = self._wasserstein_gp_discriminator_steps

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
        Executes the complete WGAN-GP training pipeline with optional data flattening.

        NOW SUPPORTS OPTIONAL FLATTENING:
        - flatten=True (default): Automatically flattens N-D input data (1D, 2D, 3D, etc.)
        - flatten=False: Uses data as-is, without flattening
        - Converts labels to categorical format
        - Stores original shape and flatten flag for reconstruction during generation

        Args:
            input_shape: Shape of input data samples (e.g., (784,) for 1D, (28, 28, 1) for 2D, (16, 16, 16) for 3D)
            x_real_samples: Training dataset samples (can be N-dimensional)
            y_real_samples: Corresponding sample labels (1D array of class indices)
            batch_size: Batch size for training. If None, uses the instance's default batch size.
            epochs: Number of training epochs. If None, uses the instance's default number of epochs.
            verbose: Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch)
            callbacks: List of callback instances to apply during training.
                       These will be merged with internally defined callbacks.
            flatten: If True, flattens multi-dimensional input data. If False, uses data as-is (default: True)

        Process:
            1. Stores original input shape for later reconstruction
            2. Optionally flattens multi-dimensional input data (if flatten=True)
            3. Calculates class distribution automatically from labels
            4. Initializes model architecture (or uses provided)
            5. Configures optimizers and loss functions
            6. Sets up training callbacks
            7. Alternates between critic and generator updates
            8. Applies gradient penalty during critic training
            9. Manages model saving and monitoring
        """
        # Use provided parameters or fall back to instance defaults
        effective_batch_size = batch_size if batch_size is not None else self._wasserstein_gp_batch_size
        effective_epochs = epochs if epochs is not None else self._wasserstein_gp_number_epochs

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

        # Calculate number_samples_per_class automatically from labels
        number_samples_per_class = self._calculate_samples_per_class(y_real_samples)

        # Initialize the wasserstein_gp model (or use provided) with appropriate dimension
        self._get_wasserstein_gp(model_input_dim, number_samples_per_class)

        # Ensure we have an algorithm
        if self._wasserstein_gp_algorithm is None:
            raise ValueError("WassersteinGPAlgorithm instance is required but was not provided or created.")

        # Define the custom loss functions for the discriminator and generator
        def discriminator_loss(real_img, fake_img):
            return tf.reduce_mean(fake_img) - tf.reduce_mean(real_img)

        def generator_loss(fake_img):
            return -tf.reduce_mean(fake_img)

        # Create optimizers
        generator_optimizer = Adam(
            learning_rate=self._wasserstein_gp_optimizer_generator_learning_rate,
            beta_1=self._wasserstein_gp_optimizer_generator_beta
        )
        discriminator_optimizer = Adam(
            learning_rate=self._wasserstein_gp_optimizer_discriminator_learning_rate,
            beta_1=self._wasserstein_gp_optimizer_discriminator_beta
        )

        # Compile the wasserstein_gp GAN algorithm
        self._wasserstein_gp_algorithm.compile(
            generator_optimizer,
            discriminator_optimizer,
            generator_loss,
            discriminator_loss
        )

        # Setup callbacks list
        callbacks_list = []

        # Add internally defined callbacks if they exist
        if hasattr(self, '_callback_model_monitor'):
            callbacks_list.append(self._callback_model_monitor)

        if hasattr(self, '_callback_early_stop'):
            callbacks_list.append(self._callback_early_stop)

        # Merge with user-provided callbacks
        if callbacks is not None:
            if isinstance(callbacks, list):
                callbacks_list.extend(callbacks)
            else:
                callbacks_list.append(callbacks)

        # Fit the wasserstein_gp GAN model with processed samples
        self._wasserstein_gp_algorithm.fit(
            x_real_samples_processed,
            to_categorical(y_real_samples, num_classes=number_samples_per_class["number_classes"]),
            epochs=effective_epochs,
            batch_size=effective_batch_size,
            verbose=verbose,
            callbacks=callbacks_list if callbacks_list else None
        )

    def get_samples(self, number_samples_per_class):
        """
        Generate synthetic samples and optionally reshape them to original input shape.

        NOW SUPPORTS OPTIONAL RESHAPING:
        - If data was flattened during training: Automatically reshapes to original dimensions
        - If data was not flattened: Returns data as-is from algorithm
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
        generated_data = self._wasserstein_gp_algorithm.get_samples(number_samples_per_class)

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
    def wasserstein_gp_algorithm(self) -> WassersteinGPAlgorithm | None:
        """Get the wasserstein_gp algorithm instance."""
        return self._wasserstein_gp_algorithm

    @property
    def wasserstein_gp_model(self) -> WassersteinGPModel | None:
        """Get the wasserstein_gp model instance."""
        return self._wasserstein_gp_model

    @property
    def data_was_flattened(self) -> bool:
        """Get flag indicating if data was flattened during training."""
        return self._data_was_flattened

    @property
    def original_input_shape(self) -> tuple:
        """Get the original input shape (before flattening)."""
        return self._original_input_shape

    # Property getters and setters
    @property
    def wasserstein_gp_latent_dimension(self) -> int:
        """Get the latent dimension."""
        return self._wasserstein_gp_latent_dimension

    @wasserstein_gp_latent_dimension.setter
    def wasserstein_gp_latent_dimension(self, value: int) -> None:
        """Set the latent dimension."""
        self._wasserstein_gp_latent_dimension = value

    @property
    def wasserstein_gp_training_algorithm(self) -> str:
        """Get the training algorithm."""
        return self._wasserstein_gp_training_algorithm

    @wasserstein_gp_training_algorithm.setter
    def wasserstein_gp_training_algorithm(self, value: str) -> None:
        """Set the training algorithm."""
        self._wasserstein_gp_training_algorithm = value

    @property
    def wasserstein_gp_activation_function(self) -> str:
        """Get the activation function."""
        return self._wasserstein_gp_activation_function

    @wasserstein_gp_activation_function.setter
    def wasserstein_gp_activation_function(self, value: str) -> None:
        """Set the activation function."""
        self._wasserstein_gp_activation_function = value

    @property
    def wasserstein_gp_dropout_decay_rate_g(self) -> float:
        """Get the generator dropout decay rate."""
        return self._wasserstein_gp_dropout_decay_rate_g

    @wasserstein_gp_dropout_decay_rate_g.setter
    def wasserstein_gp_dropout_decay_rate_g(self, value: float) -> None:
        """Set the generator dropout decay rate."""
        self._wasserstein_gp_dropout_decay_rate_g = value

    @property
    def wasserstein_gp_dropout_decay_rate_d(self) -> float:
        """Get the discriminator dropout decay rate."""
        return self._wasserstein_gp_dropout_decay_rate_d

    @wasserstein_gp_dropout_decay_rate_d.setter
    def wasserstein_gp_dropout_decay_rate_d(self, value: float) -> None:
        """Set the discriminator dropout decay rate."""
        self._wasserstein_gp_dropout_decay_rate_d = value

    @property
    def wasserstein_gp_dense_layer_sizes_generator(self) -> list[int]:
        """Get the generator dense layer sizes."""
        return self._wasserstein_gp_dense_layer_sizes_generator

    @wasserstein_gp_dense_layer_sizes_generator.setter
    def wasserstein_gp_dense_layer_sizes_generator(self, value: list[int]) -> None:
        """Set the generator dense layer sizes."""
        self._wasserstein_gp_dense_layer_sizes_generator = value

    @property
    def wasserstein_gp_dense_layer_sizes_discriminator(self) -> list[int]:
        """Get the discriminator dense layer sizes."""
        return self._wasserstein_gp_dense_layer_sizes_discriminator

    @wasserstein_gp_dense_layer_sizes_discriminator.setter
    def wasserstein_gp_dense_layer_sizes_discriminator(self, value: list[int]) -> None:
        """Set the discriminator dense layer sizes."""
        self._wasserstein_gp_dense_layer_sizes_discriminator = value

    @property
    def wasserstein_gp_batch_size(self) -> int:
        """Get the batch size."""
        return self._wasserstein_gp_batch_size

    @wasserstein_gp_batch_size.setter
    def wasserstein_gp_batch_size(self, value: int) -> None:
        """Set the batch size."""
        self._wasserstein_gp_batch_size = value

    @property
    def wasserstein_gp_number_epochs(self) -> int:
        """Get the number of epochs."""
        return self._wasserstein_gp_number_epochs

    @wasserstein_gp_number_epochs.setter
    def wasserstein_gp_number_epochs(self, value: int) -> None:
        """Set the number of epochs."""
        self._wasserstein_gp_number_epochs = value

    @property
    def wasserstein_gp_number_classes(self) -> int:
        """Get the number of classes."""
        return self._wasserstein_gp_number_classes

    @wasserstein_gp_number_classes.setter
    def wasserstein_gp_number_classes(self, value: int) -> None:
        """Set the number of classes."""
        self._wasserstein_gp_number_classes = value

    @property
    def wasserstein_gp_loss_function(self) -> str:
        """Get the loss function."""
        return self._wasserstein_gp_loss_function

    @wasserstein_gp_loss_function.setter
    def wasserstein_gp_loss_function(self, value: str) -> None:
        """Set the loss function."""
        self._wasserstein_gp_loss_function = value

    @property
    def wasserstein_gp_momentum(self) -> float:
        """Get the momentum."""
        return self._wasserstein_gp_momentum

    @wasserstein_gp_momentum.setter
    def wasserstein_gp_momentum(self, value: float) -> None:
        """Set the momentum."""
        self._wasserstein_gp_momentum = value

    @property
    def wasserstein_gp_last_activation_layer(self) -> str:
        """Get the last activation layer."""
        return self._wasserstein_gp_last_activation_layer

    @wasserstein_gp_last_activation_layer.setter
    def wasserstein_gp_last_activation_layer(self, value: str) -> None:
        """Set the last activation layer."""
        self._wasserstein_gp_last_activation_layer = value

    @property
    def wasserstein_gp_initializer_mean(self) -> float:
        """Get the initializer mean."""
        return self._wasserstein_gp_initializer_mean

    @wasserstein_gp_initializer_mean.setter
    def wasserstein_gp_initializer_mean(self, value: float) -> None:
        """Set the initializer mean."""
        self._wasserstein_gp_initializer_mean = value

    @property
    def wasserstein_gp_initializer_deviation(self) -> float:
        """Get the initializer deviation."""
        return self._wasserstein_gp_initializer_deviation

    @wasserstein_gp_initializer_deviation.setter
    def wasserstein_gp_initializer_deviation(self, value: float) -> None:
        """Set the initializer deviation."""
        self._wasserstein_gp_initializer_deviation = value

    @property
    def wasserstein_gp_optimizer_generator_learning_rate(self) -> float:
        """Get the generator optimizer learning rate."""
        return self._wasserstein_gp_optimizer_generator_learning_rate

    @wasserstein_gp_optimizer_generator_learning_rate.setter
    def wasserstein_gp_optimizer_generator_learning_rate(self, value: float) -> None:
        """Set the generator optimizer learning rate."""
        self._wasserstein_gp_optimizer_generator_learning_rate = value

    @property
    def wasserstein_gp_optimizer_discriminator_learning_rate(self) -> float:
        """Get the discriminator optimizer learning rate."""
        return self._wasserstein_gp_optimizer_discriminator_learning_rate

    @wasserstein_gp_optimizer_discriminator_learning_rate.setter
    def wasserstein_gp_optimizer_discriminator_learning_rate(self, value: float) -> None:
        """Set the discriminator optimizer learning rate."""
        self._wasserstein_gp_optimizer_discriminator_learning_rate = value

    @property
    def wasserstein_gp_optimizer_generator_beta(self) -> float:
        """Get the generator optimizer beta."""
        return self._wasserstein_gp_optimizer_generator_beta

    @wasserstein_gp_optimizer_generator_beta.setter
    def wasserstein_gp_optimizer_generator_beta(self, value: float) -> None:
        """Set the generator optimizer beta."""
        self._wasserstein_gp_optimizer_generator_beta = value

    @property
    def wasserstein_gp_optimizer_discriminator_beta(self) -> float:
        """Get the discriminator optimizer beta."""
        return self._wasserstein_gp_optimizer_discriminator_beta

    @wasserstein_gp_optimizer_discriminator_beta.setter
    def wasserstein_gp_optimizer_discriminator_beta(self, value: float) -> None:
        """Set the discriminator optimizer beta."""
        self._wasserstein_gp_optimizer_discriminator_beta = value

    @property
    def wasserstein_gp_discriminator_steps(self) -> int:
        """Get the discriminator steps."""
        return self._wasserstein_gp_discriminator_steps

    @wasserstein_gp_discriminator_steps.setter
    def wasserstein_gp_discriminator_steps(self, value: int) -> None:
        """Set the discriminator steps."""
        self._wasserstein_gp_discriminator_steps = value

    @property
    def wasserstein_gp_smoothing_rate(self) -> float:
        """Get the smoothing rate."""
        return self._wasserstein_gp_smoothing_rate

    @wasserstein_gp_smoothing_rate.setter
    def wasserstein_gp_smoothing_rate(self, value: float) -> None:
        """Set the smoothing rate."""
        self._wasserstein_gp_smoothing_rate = value

    @property
    def wasserstein_gp_latent_mean_distribution(self) -> float:
        """Get the latent mean distribution."""
        return self._wasserstein_gp_latent_mean_distribution

    @wasserstein_gp_latent_mean_distribution.setter
    def wasserstein_gp_latent_mean_distribution(self, value: float) -> None:
        """Set the latent mean distribution."""
        self._wasserstein_gp_latent_mean_distribution = value

    @property
    def wasserstein_gp_latent_standard_deviation(self) -> float:
        """Get the latent stander deviation."""
        return self._wasserstein_gp_latent_standard_deviation

    @wasserstein_gp_latent_standard_deviation.setter
    def wasserstein_gp_latent_standard_deviation(self, value: float) -> None:
        """Set the latent stander deviation."""
        self._wasserstein_gp_latent_standard_deviation = value

    @property
    def wasserstein_gp_gradient_penalty(self) -> float:
        """Get the gradient penalty."""
        return self._wasserstein_gp_gradient_penalty

    @wasserstein_gp_gradient_penalty.setter
    def wasserstein_gp_gradient_penalty(self, value: float) -> None:
        """Set the gradient penalty."""
        self._wasserstein_gp_gradient_penalty = value

    @property
    def wasserstein_gp_file_name_discriminator(self) -> str:
        """Get the discriminator file name."""
        return self._wasserstein_gp_file_name_discriminator

    @wasserstein_gp_file_name_discriminator.setter
    def wasserstein_gp_file_name_discriminator(self, value: str) -> None:
        """Set the discriminator file name."""
        self._wasserstein_gp_file_name_discriminator = value

    @property
    def wasserstein_gp_file_name_generator(self) -> str:
        """Get the generator file name."""
        return self._wasserstein_gp_file_name_generator

    @wasserstein_gp_file_name_generator.setter
    def wasserstein_gp_file_name_generator(self, value: str) -> None:
        """Set the generator file name."""
        self._wasserstein_gp_file_name_generator = value

    @property
    def wasserstein_gp_path_output_models(self) -> str:
        """Get the output models path."""
        return self._wasserstein_gp_path_output_models

    @wasserstein_gp_path_output_models.setter
    def wasserstein_gp_path_output_models(self, value: str) -> None:
        """Set the output models path."""
        self._wasserstein_gp_path_output_models = value