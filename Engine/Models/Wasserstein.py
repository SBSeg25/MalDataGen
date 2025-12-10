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
    import torch
    import torch.optim as optim
    import numpy as np
    import logging
    from Engine.Algorithms.Wasserstein.WassersteinAlgorithm import WassersteinAlgorithm
    from Engine.Architectures.Wasserstein.WassersteinModel import WassersteinModel
except ImportError as error:
    logging.error(error)
    sys.exit(-1)

# Default values from your constants file (note: need WGAN specific defaults)
DEFAULT_WASSERSTEIN_LATENT_DIMENSION = 64
DEFAULT_WASSERSTEIN_TRAINING_ALGORITHM = "Adam"
DEFAULT_WASSERSTEIN_ACTIVATION_FUNCTION = "swish"
DEFAULT_WASSERSTEIN_DROPOUT_DECAY_RATE_G = 0.25
DEFAULT_WASSERSTEIN_DROPOUT_DECAY_RATE_D = 0.25
DEFAULT_WASSERSTEIN_BATCH_SIZE = 64
DEFAULT_WASSERSTEIN_NUMBER_EPOCHS = 50
DEFAULT_WASSERSTEIN_NUMBER_CLASSES = 2
DEFAULT_WASSERSTEIN_DENSE_LAYERS_SETTINGS_GENERATOR = [256, 128, 64]
DEFAULT_WASSERSTEIN_DENSE_LAYERS_SETTINGS_DISCRIMINATOR = [64, 128, 256]
DEFAULT_WASSERSTEIN_LOSS_FUNCTION = "wasserstein"
DEFAULT_WASSERSTEIN_MOMENTUM = 0.5
DEFAULT_WASSERSTEIN_LAST_ACTIVATION_LAYER = "sigmoid"
DEFAULT_WASSERSTEIN_INITIALIZER_MEAN = 0.0
DEFAULT_WASSERSTEIN_INITIALIZER_DEVIATION = 0.02
DEFAULT_WASSERSTEIN_OPTIMIZER_GENERATOR_LEARNING_RATE = 0.0002
DEFAULT_WASSERSTEIN_OPTIMIZER_DISCRIMINATOR_LEARNING_RATE = 0.0002
DEFAULT_WASSERSTEIN_OPTIMIZER_GENERATOR_BETA = 0.5
DEFAULT_WASSERSTEIN_OPTIMIZER_DISCRIMINATOR_BETA = 0.5
DEFAULT_WASSERSTEIN_DISCRIMINATOR_STEPS = 5
DEFAULT_WASSERSTEIN_SMOOTHING_RATE = 0.0
DEFAULT_WASSERSTEIN_LATENT_MEAN_DISTRIBUTION = 0.0
DEFAULT_WASSERSTEIN_LATENT_STANDER_DEVIATION = 1.0
DEFAULT_WASSERSTEIN_FILE_NAME_DISCRIMINATOR = "discriminator_model"
DEFAULT_WASSERSTEIN_FILE_NAME_GENERATOR = "generator_model"
DEFAULT_WASSERSTEIN_PATH_OUTPUT_MODELS = "models_saved/"


class Wasserstein:
    """
    A class that implements a Wasserstein Generative Adversarial Network (WGAN).
    This implementation follows the Wasserstein GAN framework with improved training stability.

    Key Components:
    - Generator model for synthetic sample generation
    - Critic/Discriminator model (with Wasserstein loss)
    - Custom training loop with critic pre-training steps
    - Flexible architecture configuration via arguments

    Attributes:
        _wasserstein_algorithm: Orchestrates the WGAN-GP training process
        _wasserstein_model: Stores the generator and critic/discriminator models

        # WGAN Architecture Parameters
        _wasserstein_latent_dimension: Dimensionality of the latent space
        _wasserstein_training_algorithm: Type of training algorithm used
        _wasserstein_activation_function: Activation function for hidden layers
        _wasserstein_dropout_decay_rate_g: Dropout rate decay for generator
        _wasserstein_dropout_decay_rate_d: Dropout rate decay for discriminator
        _wasserstein_dense_layer_sizes_generator: Layer sizes for generator
        _wasserstein_dense_layer_sizes_discriminator: Layer sizes for discriminator
        _wasserstein_batch_size: Batch size for training
        _wasserstein_number_epochs: Number of training epochs
        _wasserstein_number_classes: Number of output classes
        _wasserstein_loss_function: Loss function used for optimization
        _wasserstein_momentum: Momentum parameter for optimizers
        _wasserstein_last_activation_layer: Activation for final layer
        _wasserstein_initializer_mean: Mean for weight initialization
        _wasserstein_initializer_deviation: Std dev for weight initialization

        # Optimization Parameters
        _wasserstein_optimizer_generator_learning_rate: Generator learning rate
        _wasserstein_optimizer_discriminator_learning_rate: Discriminator learning rate
        _wasserstein_optimizer_generator_beta: Beta parameter for generator optimizer
        _wasserstein_optimizer_discriminator_beta: Beta parameter for discriminator optimizer
        _wasserstein_discriminator_steps: Number of critic steps per generator step

        _wasserstein_smoothing_rate: Label smoothing rate
        _wasserstein_latent_mean_distribution: Distribution type for latent space
        _wasserstein_latent_stander_deviation: Std dev for latent distribution
        _wasserstein_file_name_discriminator: Filename for saving critic
        _wasserstein_file_name_generator: Filename for saving generator
        _wasserstein_path_output_models: Path for saving models
    """

    def __init__(
            self,
            latent_dimension: int = DEFAULT_WASSERSTEIN_LATENT_DIMENSION,
            training_algorithm: str = DEFAULT_WASSERSTEIN_TRAINING_ALGORITHM,
            activation_function: str = DEFAULT_WASSERSTEIN_ACTIVATION_FUNCTION,
            dropout_decay_rate_g: float = DEFAULT_WASSERSTEIN_DROPOUT_DECAY_RATE_G,
            dropout_decay_rate_d: float = DEFAULT_WASSERSTEIN_DROPOUT_DECAY_RATE_D,
            dense_layer_sizes_generator: list[int] = None,
            dense_layer_sizes_discriminator: list[int] = None,
            batch_size: int = DEFAULT_WASSERSTEIN_BATCH_SIZE,
            number_epochs: int = DEFAULT_WASSERSTEIN_NUMBER_EPOCHS,
            number_classes: int = DEFAULT_WASSERSTEIN_NUMBER_CLASSES,
            loss_function: str = DEFAULT_WASSERSTEIN_LOSS_FUNCTION,
            momentum: float = DEFAULT_WASSERSTEIN_MOMENTUM,
            last_activation_layer: str = DEFAULT_WASSERSTEIN_LAST_ACTIVATION_LAYER,
            initializer_mean: float = DEFAULT_WASSERSTEIN_INITIALIZER_MEAN,
            initializer_deviation: float = DEFAULT_WASSERSTEIN_INITIALIZER_DEVIATION,
            optimizer_generator_learning_rate: float = DEFAULT_WASSERSTEIN_OPTIMIZER_GENERATOR_LEARNING_RATE,
            optimizer_discriminator_learning_rate: float = DEFAULT_WASSERSTEIN_OPTIMIZER_DISCRIMINATOR_LEARNING_RATE,
            optimizer_generator_beta: float = DEFAULT_WASSERSTEIN_OPTIMIZER_GENERATOR_BETA,
            optimizer_discriminator_beta: float = DEFAULT_WASSERSTEIN_OPTIMIZER_DISCRIMINATOR_BETA,
            discriminator_steps: int = DEFAULT_WASSERSTEIN_DISCRIMINATOR_STEPS,
            smoothing_rate: float = DEFAULT_WASSERSTEIN_SMOOTHING_RATE,
            latent_mean_distribution: float = DEFAULT_WASSERSTEIN_LATENT_MEAN_DISTRIBUTION,
            latent_stander_deviation: float = DEFAULT_WASSERSTEIN_LATENT_STANDER_DEVIATION,
            file_name_discriminator: str = DEFAULT_WASSERSTEIN_FILE_NAME_DISCRIMINATOR,
            file_name_generator: str = DEFAULT_WASSERSTEIN_FILE_NAME_GENERATOR,
            path_output_models: str = DEFAULT_WASSERSTEIN_PATH_OUTPUT_MODELS,
            algorithm: WassersteinAlgorithm | None = None,
            model: WassersteinModel | None = None
    ) -> None:
        """
        Initializes the Wasserstein GAN instance with configuration parameters.

        Args:
            latent_dimension: Dimensionality of latent space (default: 64)
            training_algorithm: Training algorithm (default: "Adam")
            activation_function: Activation function (default: "swish")
            dropout_decay_rate_g: Generator dropout rate (default: 0.25)
            dropout_decay_rate_d: Discriminator dropout rate (default: 0.25)
            dense_layer_sizes_generator: Generator layer sizes (default: [256, 128, 64])
            dense_layer_sizes_discriminator: Discriminator layer sizes (default: [64, 128, 256])
            batch_size: Batch size (default: 64)
            number_epochs: Number of epochs (default: 50)
            number_classes: Number of classes (default: 2)
            loss_function: Loss function (default: "wasserstein")
            momentum: Momentum parameter (default: 0.5)
            last_activation_layer: Last layer activation (default: "sigmoid")
            initializer_mean: Weight init mean (default: 0.0)
            initializer_deviation: Weight init std dev (default: 0.02)
            optimizer_generator_learning_rate: Generator LR (default: 0.0002)
            optimizer_discriminator_learning_rate: Discriminator LR (default: 0.0002)
            optimizer_generator_beta: Generator beta (default: 0.5)
            optimizer_discriminator_beta: Discriminator beta (default: 0.5)
            discriminator_steps: Critic steps per generator (default: 5)
            smoothing_rate: Label smoothing rate (default: 0.0)
            latent_mean_distribution: Latent mean (default: 0.0)
            latent_stander_deviation: Latent std dev (default: 1.0)
            file_name_discriminator: Critic filename (default: "discriminator_model")
            file_name_generator: Generator filename (default: "generator_model")
            path_output_models: Model save path (default: "models_saved/")
            algorithm: Optional pre-initialized WassersteinAlgorithm (default: None)
            model: Optional pre-initialized WassersteinModelTorch (default: None)
        """
        # Store pre-initialized instances if provided
        self._wasserstein_algorithm: WassersteinAlgorithm | None = algorithm
        self._wasserstein_model: WassersteinModel | None = model

        # ** Wasserstein GAN Configuration Parameters **
        self._wasserstein_latent_dimension: int = latent_dimension
        self._wasserstein_training_algorithm: str = training_algorithm
        self._wasserstein_activation_function: str = activation_function
        self._wasserstein_dropout_decay_rate_g: float = dropout_decay_rate_g
        self._wasserstein_dropout_decay_rate_d: float = dropout_decay_rate_d

        # Handle mutable default values safely
        self._wasserstein_dense_layer_sizes_generator: list[int] = (
            dense_layer_sizes_generator
            if dense_layer_sizes_generator is not None
            else DEFAULT_WASSERSTEIN_DENSE_LAYERS_SETTINGS_GENERATOR.copy()
        )
        self._wasserstein_dense_layer_sizes_discriminator: list[int] = (
            dense_layer_sizes_discriminator
            if dense_layer_sizes_discriminator is not None
            else DEFAULT_WASSERSTEIN_DENSE_LAYERS_SETTINGS_DISCRIMINATOR.copy()
        )

        self._wasserstein_batch_size: int = batch_size
        self._wasserstein_number_epochs: int = number_epochs
        self._wasserstein_number_classes: int = number_classes
        self._wasserstein_loss_function: str = loss_function
        self._wasserstein_momentum: float = momentum
        self._wasserstein_last_activation_layer: str = last_activation_layer
        self._wasserstein_initializer_mean: float = initializer_mean
        self._wasserstein_initializer_deviation: float = initializer_deviation
        self._wasserstein_optimizer_generator_learning_rate: float = optimizer_generator_learning_rate
        self._wasserstein_optimizer_discriminator_learning_rate: float = optimizer_discriminator_learning_rate
        self._wasserstein_optimizer_generator_beta: float = optimizer_generator_beta
        self._wasserstein_optimizer_discriminator_beta: float = optimizer_discriminator_beta
        self._wasserstein_discriminator_steps: int = discriminator_steps
        self._wasserstein_smoothing_rate: float = smoothing_rate
        self._wasserstein_latent_mean_distribution: float = latent_mean_distribution
        self._wasserstein_latent_stander_deviation: float = latent_stander_deviation
        self._wasserstein_file_name_discriminator: str = file_name_discriminator
        self._wasserstein_file_name_generator: str = file_name_generator
        self._wasserstein_path_output_models: str = path_output_models

        # Initialize number_samples_per_class
        self._number_samples_per_class = {"number_classes": number_classes}

        # Flags to indicate if instances were provided
        self._has_external_algorithm: bool = algorithm is not None
        self._has_external_model: bool = model is not None

    def _get_wasserstein(self, input_shape: tuple[int, ...]) -> None:
        """
        Initializes and sets up a Wasserstein GAN model.

        This method sets up a Wasserstein Generative Adversarial Network (WGAN) by configuring
        the generator and discriminator models using custom WassersteinModelTorch class.
        The generator and discriminator are created and configured with their respective parameters.

        If pre-initialized instances were provided in the constructor, they are used instead of creating new ones.

        Args:
            input_shape: The shape of the input data, which determines the output shape for the models.

        Initializes:
            self._wasserstein_model: An instance of WassersteinModelTorch with generator and discriminator.
            self._wasserstein_algorithm: An instance of AlgorithmWassersteinModelTorch that manages training.
        """
        # Only create new model if none was provided
        if not self._has_external_model:
            # Wasserstein Model setup for the Generator and Discriminator
            self._wasserstein_model = WassersteinModel(
                latent_dimension=self._wasserstein_latent_dimension,
                output_shape=input_shape,
                activation_function=self._wasserstein_activation_function,
                initializer_mean=self._wasserstein_initializer_mean,
                initializer_deviation=self._wasserstein_initializer_deviation,
                dropout_decay_rate_g=self._wasserstein_dropout_decay_rate_g,
                dropout_decay_rate_d=self._wasserstein_dropout_decay_rate_d,
                last_layer_activation=self._wasserstein_last_activation_layer,
                dense_layer_sizes_g=self._wasserstein_dense_layer_sizes_generator,
                dense_layer_sizes_d=self._wasserstein_dense_layer_sizes_discriminator,
                dataset_type=torch.float32,
                number_samples_per_class=self._number_samples_per_class
            )

        # Only create new algorithm if none was provided
        if not self._has_external_algorithm:
            # Ensure we have a model to get generator and discriminator from
            if self._wasserstein_model is None:
                raise ValueError("WassersteinModelTorch instance is required but was not provided.")
            # Build the generator and discriminator models
            generator_model = self._wasserstein_model.get_generator()
            discriminator_model = self._wasserstein_model.get_discriminator()

            # Wasserstein Algorithm setup for training and model operations
            self._wasserstein_algorithm = WassersteinAlgorithm(
                generator_model=generator_model,
                discriminator_model=discriminator_model,
                latent_dimension=self._wasserstein_latent_dimension,
                generator_loss_fn=None,  # Will use default Wasserstein loss
                discriminator_loss_fn=None,  # Will use default Wasserstein loss
                file_name_discriminator=self._wasserstein_file_name_discriminator,
                file_name_generator=self._wasserstein_file_name_generator,
                models_saved_path=self._wasserstein_path_output_models,
                latent_mean_distribution=self._wasserstein_latent_mean_distribution,
                latent_standard_deviation=self._wasserstein_latent_stander_deviation,
                smoothing_rate=self._wasserstein_smoothing_rate,
                discriminator_steps=self._wasserstein_discriminator_steps,
                clip_value=0.01
            )
        else:
            # If algorithm was provided externally, update its configuration if needed
            if hasattr(self._wasserstein_algorithm, 'latent_dimension'):
                self._wasserstein_algorithm.latent_dimension = self._wasserstein_latent_dimension
            if hasattr(self._wasserstein_algorithm, 'file_name_discriminator'):
                self._wasserstein_algorithm.file_name_discriminator = self._wasserstein_file_name_discriminator
            if hasattr(self._wasserstein_algorithm, 'file_name_generator'):
                self._wasserstein_algorithm.file_name_generator = self._wasserstein_file_name_generator
            if hasattr(self._wasserstein_algorithm, 'models_saved_path'):
                self._wasserstein_algorithm.models_saved_path = self._wasserstein_path_output_models
            if hasattr(self._wasserstein_algorithm, 'latent_mean_distribution'):
                self._wasserstein_algorithm.latent_mean_distribution = self._wasserstein_latent_mean_distribution
            if hasattr(self._wasserstein_algorithm, 'latent_standard_deviation'):
                self._wasserstein_algorithm.latent_standard_deviation = self._wasserstein_latent_stander_deviation
            if hasattr(self._wasserstein_algorithm, 'smoothing_rate'):
                self._wasserstein_algorithm.smoothing_rate = self._wasserstein_smoothing_rate
            if hasattr(self._wasserstein_algorithm, 'discriminator_steps'):
                self._wasserstein_algorithm.discriminator_steps = self._wasserstein_discriminator_steps

    def _training_wasserstein_model(
            self,
            input_shape: tuple[int, ...],
            arguments: 'argparse.Namespace',
            x_real_samples: np.ndarray,
            y_real_samples: np.ndarray
    ) -> None:
        """
        Executes the complete training pipeline for Wasserstein GAN with Gradient Penalty.

        Process:
        1. Initializes generator and critic models (or uses provided)
        2. Configures optimizers with specified parameters
        3. Trains using alternating critic/generator updates
        4. Manages callbacks and monitoring

        Args:
            input_shape: Input data shape
            arguments: Training configuration
            x_real_samples: Training samples
            y_real_samples: Corresponding labels
        """
        # Initialize the WassersteinGP model (or use provided)
        self._get_wasserstein(input_shape)

        # Print the model summaries for the generator and discriminator if available
        if self._wasserstein_model is not None:
            print("\n" + "=" * 60)
            print("GENERATOR ARCHITECTURE")
            print("=" * 60)
            print(self._wasserstein_model.get_generator())
            print("\n" + "=" * 60)
            print("DISCRIMINATOR/CRITIC ARCHITECTURE")
            print("=" * 60)
            print(self._wasserstein_model.get_discriminator())
            print("=" * 60 + "\n")

        # Ensure we have an algorithm
        if self._wasserstein_algorithm is None:
            raise ValueError("WassersteinAlgorithm instance is required but was not provided or created.")

        # Setup optimizers - FIXED: Changed from .generator_model to .generator
        generator_optimizer = optim.Adam(
            self._wasserstein_algorithm.generator.parameters(),
            lr=self._wasserstein_optimizer_generator_learning_rate,
            betas=(self._wasserstein_optimizer_generator_beta, 0.9)
        )

        # FIXED: Changed from .discriminator_model to .discriminator
        discriminator_optimizer = optim.Adam(
            self._wasserstein_algorithm.discriminator.parameters(),
            lr=self._wasserstein_optimizer_discriminator_learning_rate,
            betas=(self._wasserstein_optimizer_discriminator_beta, 0.9)
        )

        # Compile the Wasserstein GAN algorithm
        self._wasserstein_algorithm.compile(
            generator_optimizer,
            discriminator_optimizer,
            generator_loss_fn=None,  # Use default Wasserstein loss
            discriminator_loss_fn=None  # Use default Wasserstein loss
        )

        # Prepare callbacks list with safety wrapper
        callbacks_list = []

        # Add callbacks if they exist
        if hasattr(self, '_callback_model_monitor') and self._callback_model_monitor:
            callbacks_list.append(self._callback_model_monitor)

        if hasattr(arguments, 'use_early_stop') and arguments.use_early_stop:
            if hasattr(self, '_callback_early_stop') and self._callback_early_stop:
                callbacks_list.append(self._callback_early_stop)

        # Ensure callbacks have proper initialization
        import time
        for callback in callbacks_list:
            if hasattr(callback, 'data'):
                if callback.data is None:
                    callback.data = {}
                callback.data['start_time'] = time.time()

        # Convert labels to one-hot if needed
        num_classes = self._number_samples_per_class["number_classes"]

        # Check if labels need conversion
        if len(y_real_samples.shape) == 1 or y_real_samples.shape[1] == 1:
            # Labels are in integer format or single column - convert to one-hot
            if len(y_real_samples.shape) == 2 and y_real_samples.shape[1] == 1:
                # Flatten from (N, 1) to (N,)
                y_real_samples = y_real_samples.flatten()

            # Convert to one-hot encoding
            y_one_hot = np.zeros((y_real_samples.shape[0], num_classes))
            y_one_hot[np.arange(y_real_samples.shape[0]), y_real_samples.astype(int)] = 1
            y_real_samples = y_one_hot
            print(f"Converted labels to one-hot encoding: shape {y_real_samples.shape}")
        elif y_real_samples.shape[1] != num_classes:
            raise ValueError(
                f"Label shape mismatch: got {y_real_samples.shape} but expected "
                f"either (N,) or (N, {num_classes}) for {num_classes} classes"
            )

        # Fit the WassersteinGP GAN model
        print(f"\nStarting training for {self._wasserstein_number_epochs} epochs...")
        print(f"Batch size: {self._wasserstein_batch_size}")
        print(f"Critic steps per generator step: {self._wasserstein_discriminator_steps}")
        print(f"Training samples: {len(x_real_samples)}\n")

        self._wasserstein_algorithm.fit(
            x_real_samples,
            y_real_samples,
            epochs=self._wasserstein_number_epochs,
            batch_size=self._wasserstein_batch_size,
            lambda_gp=10.0,
            callbacks=callbacks_list if callbacks_list else None,
            verbose=1
        )

        print("\nTraining completed successfully!")
    # Additional getters for the algorithm and model
    @property
    def wasserstein_algorithm(self) -> WassersteinAlgorithm | None:
        """Get the Wasserstein algorithm instance."""
        return self._wasserstein_algorithm

    @property
    def wasserstein_model(self) -> WassersteinModel | None:
        """Get the Wasserstein model instance."""
        return self._wasserstein_model

    # Getter and setter for wasserstein_latent_dimension
    @property
    def wasserstein_latent_dimension(self) -> int:
        """Get the latent dimension."""
        return self._wasserstein_latent_dimension

    @wasserstein_latent_dimension.setter
    def wasserstein_latent_dimension(self, value: int) -> None:
        """Set the latent dimension."""
        self._wasserstein_latent_dimension = value

    # Getter and setter for wasserstein_training_algorithm
    @property
    def wasserstein_training_algorithm(self) -> str:
        """Get the training algorithm."""
        return self._wasserstein_training_algorithm

    @wasserstein_training_algorithm.setter
    def wasserstein_training_algorithm(self, value: str) -> None:
        """Set the training algorithm."""
        self._wasserstein_training_algorithm = value

    # Getter and setter for wasserstein_activation_function
    @property
    def wasserstein_activation_function(self) -> str:
        """Get the activation function."""
        return self._wasserstein_activation_function

    @wasserstein_activation_function.setter
    def wasserstein_activation_function(self, value: str) -> None:
        """Set the activation function."""
        self._wasserstein_activation_function = value

    # Getter and setter for wasserstein_dropout_decay_rate_g
    @property
    def wasserstein_dropout_decay_rate_g(self) -> float:
        """Get the generator dropout decay rate."""
        return self._wasserstein_dropout_decay_rate_g

    @wasserstein_dropout_decay_rate_g.setter
    def wasserstein_dropout_decay_rate_g(self, value: float) -> None:
        """Set the generator dropout decay rate."""
        self._wasserstein_dropout_decay_rate_g = value

    # Getter and setter for wasserstein_dropout_decay_rate_d
    @property
    def wasserstein_dropout_decay_rate_d(self) -> float:
        """Get the discriminator dropout decay rate."""
        return self._wasserstein_dropout_decay_rate_d

    @wasserstein_dropout_decay_rate_d.setter
    def wasserstein_dropout_decay_rate_d(self, value: float) -> None:
        """Set the discriminator dropout decay rate."""
        self._wasserstein_dropout_decay_rate_d = value

    # Getter and setter for wasserstein_dense_layer_sizes_generator
    @property
    def wasserstein_dense_layer_sizes_generator(self) -> list[int]:
        """Get the generator layer sizes."""
        return self._wasserstein_dense_layer_sizes_generator

    @wasserstein_dense_layer_sizes_generator.setter
    def wasserstein_dense_layer_sizes_generator(self, value: list[int]) -> None:
        """Set the generator layer sizes."""
        self._wasserstein_dense_layer_sizes_generator = value

    # Getter and setter for wasserstein_dense_layer_sizes_discriminator
    @property
    def wasserstein_dense_layer_sizes_discriminator(self) -> list[int]:
        """Get the discriminator layer sizes."""
        return self._wasserstein_dense_layer_sizes_discriminator

    @wasserstein_dense_layer_sizes_discriminator.setter
    def wasserstein_dense_layer_sizes_discriminator(self, value: list[int]) -> None:
        """Set the discriminator layer sizes."""
        self._wasserstein_dense_layer_sizes_discriminator = value

    # Getter and setter for wasserstein_batch_size
    @property
    def wasserstein_batch_size(self) -> int:
        """Get the batch size."""
        return self._wasserstein_batch_size

    @wasserstein_batch_size.setter
    def wasserstein_batch_size(self, value: int) -> None:
        """Set the batch size."""
        self._wasserstein_batch_size = value

    # Getter and setter for wasserstein_number_classes
    @property
    def wasserstein_number_classes(self) -> int:
        """Get the number of classes."""
        return self._wasserstein_number_classes

    @wasserstein_number_classes.setter
    def wasserstein_number_classes(self, value: int) -> None:
        """Set the number of classes."""
        self._wasserstein_number_classes = value

    # Getter and setter for wasserstein_loss_function
    @property
    def wasserstein_loss_function(self) -> str:
        """Get the loss function."""
        return self._wasserstein_loss_function

    @wasserstein_loss_function.setter
    def wasserstein_loss_function(self, value: str) -> None:
        """Set the loss function."""
        self._wasserstein_loss_function = value

    # Getter and setter for wasserstein_momentum
    @property
    def wasserstein_momentum(self) -> float:
        """Get the momentum parameter."""
        return self._wasserstein_momentum

    @wasserstein_momentum.setter
    def wasserstein_momentum(self, value: float) -> None:
        """Set the momentum parameter."""
        self._wasserstein_momentum = value

    # Getter and setter for wasserstein_last_activation_layer
    @property
    def wasserstein_last_activation_layer(self) -> str:
        """Get the last layer activation."""
        return self._wasserstein_last_activation_layer

    @wasserstein_last_activation_layer.setter
    def wasserstein_last_activation_layer(self, value: str) -> None:
        """Set the last layer activation."""
        self._wasserstein_last_activation_layer = value

    # Getter and setter for wasserstein_initializer_mean
    @property
    def wasserstein_initializer_mean(self) -> float:
        """Get the initializer mean."""
        return self._wasserstein_initializer_mean

    @wasserstein_initializer_mean.setter
    def wasserstein_initializer_mean(self, value: float) -> None:
        """Set the initializer mean."""
        self._wasserstein_initializer_mean = value

    # Getter and setter for wasserstein_initializer_deviation
    @property
    def wasserstein_initializer_deviation(self) -> float:
        """Get the initializer deviation."""
        return self._wasserstein_initializer_deviation

    @wasserstein_initializer_deviation.setter
    def wasserstein_initializer_deviation(self, value: float) -> None:
        """Set the initializer deviation."""
        self._wasserstein_initializer_deviation = value

    # Getter and setter for wasserstein_optimizer_generator_learning_rate
    @property
    def wasserstein_optimizer_generator_learning_rate(self) -> float:
        """Get the generator learning rate."""
        return self._wasserstein_optimizer_generator_learning_rate

    @wasserstein_optimizer_generator_learning_rate.setter
    def wasserstein_optimizer_generator_learning_rate(self, value: float) -> None:
        """Set the generator learning rate."""
        self._wasserstein_optimizer_generator_learning_rate = value

    # Getter and setter for wasserstein_optimizer_discriminator_learning_rate
    @property
    def wasserstein_optimizer_discriminator_learning_rate(self) -> float:
        """Get the discriminator learning rate."""
        return self._wasserstein_optimizer_discriminator_learning_rate

    @wasserstein_optimizer_discriminator_learning_rate.setter
    def wasserstein_optimizer_discriminator_learning_rate(self, value: float) -> None:
        """Set the discriminator learning rate."""
        self._wasserstein_optimizer_discriminator_learning_rate = value

    # Getter and setter for wasserstein_optimizer_generator_beta
    @property
    def wasserstein_optimizer_generator_beta(self) -> float:
        """Get the generator beta parameter."""
        return self._wasserstein_optimizer_generator_beta

    @wasserstein_optimizer_generator_beta.setter
    def wasserstein_optimizer_generator_beta(self, value: float) -> None:
        """Set the generator beta parameter."""
        self._wasserstein_optimizer_generator_beta = value

    # Getter and setter for wasserstein_optimizer_discriminator_beta
    @property
    def wasserstein_optimizer_discriminator_beta(self) -> float:
        """Get the discriminator beta parameter."""
        return self._wasserstein_optimizer_discriminator_beta

    @wasserstein_optimizer_discriminator_beta.setter
    def wasserstein_optimizer_discriminator_beta(self, value: float) -> None:
        """Set the discriminator beta parameter."""
        self._wasserstein_optimizer_discriminator_beta = value

    # Getter and setter for wasserstein_discriminator_steps
    @property
    def wasserstein_discriminator_steps(self) -> int:
        """Get the discriminator steps."""
        return self._wasserstein_discriminator_steps

    @wasserstein_discriminator_steps.setter
    def wasserstein_discriminator_steps(self, value: int) -> None:
        """Set the discriminator steps."""
        self._wasserstein_discriminator_steps = value

    # Getter and setter for wasserstein_smoothing_rate
    @property
    def wasserstein_smoothing_rate(self) -> float:
        """Get the smoothing rate."""
        return self._wasserstein_smoothing_rate

    @wasserstein_smoothing_rate.setter
    def wasserstein_smoothing_rate(self, value: float) -> None:
        """Set the smoothing rate."""
        self._wasserstein_smoothing_rate = value

    # Getter and setter for wasserstein_latent_mean_distribution
    @property
    def wasserstein_latent_mean_distribution(self) -> float:
        """Get the latent mean distribution."""
        return self._wasserstein_latent_mean_distribution

    @wasserstein_latent_mean_distribution.setter
    def wasserstein_latent_mean_distribution(self, value: float) -> None:
        """Set the latent mean distribution."""
        self._wasserstein_latent_mean_distribution = value

    # Getter and setter for wasserstein_latent_stander_deviation
    @property
    def wasserstein_latent_stander_deviation(self) -> float:
        """Get the latent standard deviation."""
        return self._wasserstein_latent_stander_deviation

    @wasserstein_latent_stander_deviation.setter
    def wasserstein_latent_stander_deviation(self, value: float) -> None:
        """Set the latent standard deviation."""
        self._wasserstein_latent_stander_deviation = value

    # Getter and setter for wasserstein_file_name_discriminator
    @property
    def wasserstein_file_name_discriminator(self) -> str:
        """Get the discriminator filename."""
        return self._wasserstein_file_name_discriminator

    @wasserstein_file_name_discriminator.setter
    def wasserstein_file_name_discriminator(self, value: str) -> None:
        """Set the discriminator filename."""
        self._wasserstein_file_name_discriminator = value

    # Getter and setter for wasserstein_file_name_generator
    @property
    def wasserstein_file_name_generator(self) -> str:
        """Get the generator filename."""
        return self._wasserstein_file_name_generator

    @wasserstein_file_name_generator.setter
    def wasserstein_file_name_generator(self, value: str) -> None:
        """Set the generator filename."""
        self._wasserstein_file_name_generator = value

    # Getter and setter for wasserstein_path_output_models
    @property
    def wasserstein_path_output_models(self) -> str:
        """Get the output models path."""
        return self._wasserstein_path_output_models

    @wasserstein_path_output_models.setter
    def wasserstein_path_output_models(self, value: str) -> None:
        """Set the output models path."""
        self._wasserstein_path_output_models = value