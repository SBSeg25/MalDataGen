#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{2}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/06'
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
    import os
    import sys
    import numpy

    # Detecta o framework a partir da variável de ambiente
    ML_FRAMEWORK = os.getenv('ML_FRAMEWORK', 'tensorflow').lower()

    # Importa condicionalmente os frameworks
    if ML_FRAMEWORK == 'tensorflow':
        try:
            import tensorflow as tf
            from tensorflow.keras.utils import to_categorical
        except ImportError:
            raise ImportError("TensorFlow not found. Please install: pip install tensorflow")

    elif ML_FRAMEWORK == 'pytorch':
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("PyTorch not found. Please install: pip install torch")

    else:
        raise ValueError(f"Unsupported ML_FRAMEWORK: {ML_FRAMEWORK}. Use 'tensorflow' or 'pytorch'")

    from Engine.Algorithms.Autoencoder.AutoencoderAlgorithm import AutoencoderAlgorithm
    from Engine.Models.Autoencoder.ModelAutoencoder import AutoencoderModel


except ImportError as error:
    print(error)
    sys.exit(-1)


class AutoencoderInstance:
    """
    A class that instantiates and manages an Autoencoder model in a framework-agnostic way.
    This implementation provides complete configuration, training, and management capabilities
    for autoencoder-based learning tasks within the Synthetic Ocean ecosystem.

    Supports both TensorFlow/Keras and PyTorch frameworks through the ML_FRAMEWORK
    environment variable.

    Environment Variable:
        ML_FRAMEWORK: Define o framework a ser usado ('tensorflow' ou 'pytorch').
                     Padrão: 'tensorflow'

    Attributes:
        _autoencoder_model (AutoencoderModel): Contains encoder and decoder components
        _autoencoder_algorithm (AutoencoderAlgorithm): Manages the autoencoder training process
        _framework (str): The current framework being used

    Configuration Parameters (with getters/setters):
        _autoencoder_latent_dimension (int): Size of the latent space
        _autoencoder_training_algorithm (str): Training algorithm specification
        _autoencoder_activation_function (str): Activation function for hidden layers
        _autoencoder_dropout_decay_rate_encoder (float): Encoder dropout rate
        _autoencoder_dropout_decay_rate_decoder (float): Decoder dropout rate
        _autoencoder_dense_layer_sizes_encoder (list): Encoder layer sizes
        _autoencoder_dense_layer_sizes_decoder (list): Decoder layer sizes
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

    Example:
        >>> import os
        >>> os.environ['ML_FRAMEWORK'] = 'pytorch'  # or 'tensorflow'
        >>> autoencoder_instance = AutoencoderInstance(arguments)
        >>> autoencoder_instance._training_autoencoder_model(
        ...     input_shape, arguments, x_train, y_train
        ... )
    """

    def __init__(self, arguments):
        """
        Initializes the autoencoder instance with configuration parameters.

        Args:
            arguments (Namespace): Configuration object containing all required parameters:
                - autoencoder_latent_dimension: Latent space size
                - autoencoder_training_algorithm: Training algorithm
                - autoencoder_activation_function: Activation function
                - autoencoder_dropout_decay_rate_encoder: Encoder dropout rate
                - autoencoder_dropout_decay_rate_decoder: Decoder dropout rate
                - autoencoder_dense_layer_sizes_encoder: Encoder architecture
                - autoencoder_dense_layer_sizes_decoder: Decoder architecture
                - autoencoder_batch_size: Batch size for training
                - autoencoder_number_epochs: Number of epochs
                - autoencoder_number_classes: Number of classes
                - autoencoder_loss_function: Loss function name
                - autoencoder_momentum: Optimization momentum
                - autoencoder_last_activation_layer: Output activation
                - autoencoder_initializer_mean: Weight init mean
                - autoencoder_initializer_deviation: Weight init std
                - autoencoder_latent_mean_distribution: Latent mean
                - autoencoder_latent_stander_deviation: Latent std
                - autoencoder_file_name_encoder: Encoder save path
                - autoencoder_file_name_decoder: Decoder save path
                - autoencoder_path_output_models: Output directory
        """
        self._autoencoder_algorithm = None
        self._autoencoder_model = None
        self._framework = ML_FRAMEWORK

        # ** Autoencoder Model Configuration Parameters **
        self._autoencoder_latent_dimension = arguments.autoencoder_latent_dimension
        self._autoencoder_training_algorithm = arguments.autoencoder_training_algorithm
        self._autoencoder_activation_function = arguments.autoencoder_activation_function
        self._autoencoder_dropout_decay_rate_encoder = arguments.autoencoder_dropout_decay_rate_encoder
        self._autoencoder_dropout_decay_rate_decoder = arguments.autoencoder_dropout_decay_rate_decoder
        self._autoencoder_dense_layer_sizes_encoder = arguments.autoencoder_dense_layer_sizes_encoder
        self._autoencoder_dense_layer_sizes_decoder = arguments.autoencoder_dense_layer_sizes_decoder
        self._autoencoder_batch_size = arguments.autoencoder_batch_size
        self._autoencoder_number_epochs = arguments.autoencoder_number_epochs
        self._autoencoder_number_classes = arguments.autoencoder_number_classes
        self._autoencoder_loss_function = arguments.autoencoder_loss_function
        self._autoencoder_momentum = arguments.autoencoder_momentum
        self._autoencoder_last_activation_layer = arguments.autoencoder_last_activation_layer
        self._autoencoder_initializer_mean = arguments.autoencoder_initializer_mean
        self._autoencoder_initializer_deviation = arguments.autoencoder_initializer_deviation
        self._autoencoder_latent_mean_distribution = arguments.autoencoder_latent_mean_distribution
        self._autoencoder_latent_stander_deviation = arguments.autoencoder_latent_stander_deviation
        self._autoencoder_file_name_encoder = arguments.autoencoder_file_name_encoder
        self._autoencoder_file_name_decoder = arguments.autoencoder_file_name_decoder
        self._autoencoder_path_output_models = arguments.autoencoder_path_output_models

        # Atributo necessário para armazenar informações de classes
        # Assume que será definido externamente ou através de arguments
        if hasattr(arguments, 'number_samples_per_class'):
            self._number_samples_per_class = arguments.number_samples_per_class
        else:
            self._number_samples_per_class = {"number_classes": self._autoencoder_number_classes}

    def _get_autoencoder(self, input_shape):
        """
        Initialize and configure the Autoencoder model, including encoder and decoder components.

        This method sets up an Autoencoder model by configuring both the encoder and decoder using the
        `AutoencoderModel` class and links them with the `AutoencoderAlgorithm` class. The model is
        initialized with specified configurations such as latent dimension, activation functions,
        dropout rates, and layer sizes for both the encoder and decoder.

        Args:
            input_shape (tuple): The shape of the input data, which determines the output shape for the models.

        Initializes:
            self._autoencoder_model:
                An instance of the `AutoencoderModel` class, including the encoder and decoder setup,
                with configurations for activation functions, layer sizes, dropout rates, and more.

            self._autoencoder_algorithm:
                An instance of the `AutoencoderAlgorithm` class, managing the autoencoder training process,
                including the encoder and decoder models, loss function, latent distributions, and model file paths.

        Note:
            This method is framework-agnostic and will work with both TensorFlow and PyTorch
            based on the ML_FRAMEWORK environment variable.
        """
        # Autoencoder Model setup for Encoder and Decoder
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
            dataset_type=numpy.float32,
            number_samples_per_class=self._number_samples_per_class
        )

        # Autoencoder Algorithm setup for training and model operations
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

    def _convert_labels_to_categorical(self, y_labels, num_classes):
        """
        Converts labels to categorical format based on the current framework.

        Args:
            y_labels: Input labels (numpy array or tensor)
            num_classes (int): Number of classes

        Returns:
            Categorical labels in the appropriate format for the framework
        """
        if self._framework == 'tensorflow':
            return to_categorical(y_labels, num_classes=num_classes)
        elif self._framework == 'pytorch':
            # PyTorch: converte para one-hot encoding
            if isinstance(y_labels, numpy.ndarray):
                y_labels = torch.from_numpy(y_labels).long()
            return F.one_hot(y_labels, num_classes=num_classes).float()
        else:
            raise ValueError(f"Unsupported framework: {self._framework}")

    def _print_model_summary(self, input_shape):
        """
        Prints model summaries in a framework-appropriate way.

        Args:
            input_shape: Shape of the input data
        """
        encoder = self._autoencoder_model.get_encoder(input_shape)
        decoder = self._autoencoder_model.get_decoder(input_shape)

        if self._framework == 'tensorflow':
            print("\n=== Encoder Summary ===")
            encoder.summary()
            print("\n=== Decoder Summary ===")
            decoder.summary()
        elif self._framework == 'pytorch':
            print("\n=== Encoder Summary ===")
            print(encoder)
            print(f"\nEncoder parameters: {sum(p.numel() for p in encoder.parameters())}")

            print("\n=== Decoder Summary ===")
            print(decoder)
            print(f"\nDecoder parameters: {sum(p.numel() for p in decoder.parameters())}")

    def _training_autoencoder_model(self, input_shape, arguments, x_real_samples, y_real_samples):
        """
        Executes the complete autoencoder training process in a framework-agnostic way.

        This method handles the entire training pipeline including model initialization,
        compilation, callback setup, and training execution. It automatically adapts
        to the framework specified by ML_FRAMEWORK.

        Args:
            input_shape (tuple): Shape of input data samples
            arguments (Namespace): Configuration parameters including:
                - autoencoder_loss_function: Loss function to use
                - use_early_stop: Whether to use early stopping
            x_real_samples (ndarray or Tensor): Training data samples
            y_real_samples (ndarray or Tensor): Corresponding class labels

        Process:
            1. Initializes model architecture (encoder + decoder)
            2. Prints model summaries
            3. Configures loss function
            4. Sets up training callbacks (model monitor, early stopping)
            5. Converts labels to categorical format
            6. Executes autoencoder training
            7. Manages model saving and monitoring

        Note:
            - For TensorFlow: Uses keras callbacks and fit method
            - For PyTorch: Uses custom training loop with PyTorch callbacks
            - Callbacks are expected to be set up elsewhere (self._callback_model_monitor, etc.)
        """
        # Initialize the autoencoder model
        self._get_autoencoder(input_shape)

        # Print the model summaries for the encoder and decoder
        self._print_model_summary(input_shape)

        # Compile the autoencoder algorithm with the specified loss function
        self._autoencoder_algorithm.compile(loss=arguments.autoencoder_loss_function)

        # Setup callbacks list
        # callbacks_list = [self._callback_resources_monitor, self._callback_model_monitor]
        callbacks_list = [self._callback_model_monitor]

        if hasattr(arguments, 'use_early_stop') and arguments.use_early_stop:
            callbacks_list.append(self._callback_early_stop)

        # Convert labels to categorical format based on framework
        y_categorical = self._convert_labels_to_categorical(
            y_real_samples,
            num_classes=self._number_samples_per_class["number_classes"]
        )

        # Fit the autoencoder model
        # The AutoencoderAlgorithm.fit() method should handle framework differences internally
        self._autoencoder_algorithm.fit(
            (x_real_samples, y_categorical),
            x_real_samples,
            epochs=self._autoencoder_number_epochs,
            batch_size=self._autoencoder_batch_size,
            callbacks=callbacks_list
        )

    @property
    def framework(self) -> str:
        """
        Returns the framework being used.

        Returns:
            str: The name of the framework ('tensorflow' or 'pytorch')
        """
        return self._framework

    # Getter and setter for autoencoder_latent_dimension
    @property
    def autoencoder_latent_dimension(self):
        """Gets the latent dimension size."""
        return self._autoencoder_latent_dimension

    @autoencoder_latent_dimension.setter
    def autoencoder_latent_dimension(self, value):
        """Sets the latent dimension size."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("autoencoder_latent_dimension must be a positive integer.")
        self._autoencoder_latent_dimension = value

    # Getter and setter for autoencoder_training_algorithm
    @property
    def autoencoder_training_algorithm(self):
        """Gets the training algorithm."""
        return self._autoencoder_training_algorithm

    @autoencoder_training_algorithm.setter
    def autoencoder_training_algorithm(self, value):
        """Sets the training algorithm."""
        if not isinstance(value, str):
            raise ValueError("autoencoder_training_algorithm must be a string.")
        self._autoencoder_training_algorithm = value

    # Getter and setter for autoencoder_activation_function
    @property
    def autoencoder_activation_function(self):
        """Gets the activation function."""
        return self._autoencoder_activation_function

    @autoencoder_activation_function.setter
    def autoencoder_activation_function(self, value):
        """Sets the activation function."""
        if not isinstance(value, str):
            raise ValueError("autoencoder_activation_function must be a string.")
        self._autoencoder_activation_function = value

    # Getter and setter for autoencoder_dropout_decay_rate_encoder
    @property
    def autoencoder_dropout_decay_rate_encoder(self):
        """Gets the encoder dropout rate."""
        return self._autoencoder_dropout_decay_rate_encoder

    @autoencoder_dropout_decay_rate_encoder.setter
    def autoencoder_dropout_decay_rate_encoder(self, value):
        """Sets the encoder dropout rate."""
        if not isinstance(value, (int, float)) or not (0 <= value <= 1):
            raise ValueError("autoencoder_dropout_decay_rate_encoder must be between 0 and 1.")
        self._autoencoder_dropout_decay_rate_encoder = value

    # Getter and setter for autoencoder_dropout_decay_rate_decoder
    @property
    def autoencoder_dropout_decay_rate_decoder(self):
        """Gets the decoder dropout rate."""
        return self._autoencoder_dropout_decay_rate_decoder

    @autoencoder_dropout_decay_rate_decoder.setter
    def autoencoder_dropout_decay_rate_decoder(self, value):
        """Sets the decoder dropout rate."""
        if not isinstance(value, (int, float)) or not (0 <= value <= 1):
            raise ValueError("autoencoder_dropout_decay_rate_decoder must be between 0 and 1.")
        self._autoencoder_dropout_decay_rate_decoder = value

    # Getter and setter for autoencoder_dense_layer_sizes_encoder
    @property
    def autoencoder_dense_layer_sizes_encoder(self):
        """Gets the encoder layer sizes."""
        return self._autoencoder_dense_layer_sizes_encoder

    @autoencoder_dense_layer_sizes_encoder.setter
    def autoencoder_dense_layer_sizes_encoder(self, value):
        """Sets the encoder layer sizes."""
        if not isinstance(value, list) or not all(isinstance(x, int) and x > 0 for x in value):
            raise ValueError("autoencoder_dense_layer_sizes_encoder must be a list of positive integers.")
        self._autoencoder_dense_layer_sizes_encoder = value

    # Getter and setter for autoencoder_dense_layer_sizes_decoder
    @property
    def autoencoder_dense_layer_sizes_decoder(self):
        """Gets the decoder layer sizes."""
        return self._autoencoder_dense_layer_sizes_decoder

    @autoencoder_dense_layer_sizes_decoder.setter
    def autoencoder_dense_layer_sizes_decoder(self, value):
        """Sets the decoder layer sizes."""
        if not isinstance(value, list) or not all(isinstance(x, int) and x > 0 for x in value):
            raise ValueError("autoencoder_dense_layer_sizes_decoder must be a list of positive integers.")
        self._autoencoder_dense_layer_sizes_decoder = value

    # Getter and setter for autoencoder_batch_size
    @property
    def autoencoder_batch_size(self):
        """Gets the batch size."""
        return self._autoencoder_batch_size

    @autoencoder_batch_size.setter
    def autoencoder_batch_size(self, value):
        """Sets the batch size."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("autoencoder_batch_size must be a positive integer.")
        self._autoencoder_batch_size = value

    # Getter and setter for autoencoder_number_classes
    @property
    def autoencoder_number_classes(self):
        """Gets the number of classes."""
        return self._autoencoder_number_classes

    @autoencoder_number_classes.setter
    def autoencoder_number_classes(self, value):
        """Sets the number of classes."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("autoencoder_number_classes must be a positive integer.")
        self._autoencoder_number_classes = value

    # Getter and setter for autoencoder_loss_function
    @property
    def autoencoder_loss_function(self):
        """Gets the loss function."""
        return self._autoencoder_loss_function

    @autoencoder_loss_function.setter
    def autoencoder_loss_function(self, value):
        """Sets the loss function."""
        if not isinstance(value, str):
            raise ValueError("autoencoder_loss_function must be a string.")
        self._autoencoder_loss_function = value

    # Getter and setter for autoencoder_momentum
    @property
    def autoencoder_momentum(self):
        """Gets the momentum parameter."""
        return self._autoencoder_momentum

    @autoencoder_momentum.setter
    def autoencoder_momentum(self, value):
        """Sets the momentum parameter."""
        if not isinstance(value, (int, float)) or not (0 <= value <= 1):
            raise ValueError("autoencoder_momentum must be between 0 and 1.")
        self._autoencoder_momentum = value

    # Getter and setter for autoencoder_last_activation_layer
    @property
    def autoencoder_last_activation_layer(self):
        """Gets the last activation layer."""
        return self._autoencoder_last_activation_layer

    @autoencoder_last_activation_layer.setter
    def autoencoder_last_activation_layer(self, value):
        """Sets the last activation layer."""
        if not isinstance(value, str):
            raise ValueError("autoencoder_last_activation_layer must be a string.")
        self._autoencoder_last_activation_layer = value

    # Getter and setter for autoencoder_initializer_mean
    @property
    def autoencoder_initializer_mean(self):
        """Gets the weight initializer mean."""
        return self._autoencoder_initializer_mean

    @autoencoder_initializer_mean.setter
    def autoencoder_initializer_mean(self, value):
        """Sets the weight initializer mean."""
        if not isinstance(value, (int, float)):
            raise ValueError("autoencoder_initializer_mean must be a number.")
        self._autoencoder_initializer_mean = value

    # Getter and setter for autoencoder_initializer_deviation
    @property
    def autoencoder_initializer_deviation(self):
        """Gets the weight initializer standard deviation."""
        return self._autoencoder_initializer_deviation

    @autoencoder_initializer_deviation.setter
    def autoencoder_initializer_deviation(self, value):
        """Sets the weight initializer standard deviation."""
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError("autoencoder_initializer_deviation must be a non-negative number.")
        self._autoencoder_initializer_deviation = value

    # Getter and setter for autoencoder_latent_mean_distribution
    @property
    def autoencoder_latent_mean_distribution(self):
        """Gets the latent space mean distribution."""
        return self._autoencoder_latent_mean_distribution

    @autoencoder_latent_mean_distribution.setter
    def autoencoder_latent_mean_distribution(self, value):
        """Sets the latent space mean distribution."""
        if not isinstance(value, (int, float)):
            raise ValueError("autoencoder_latent_mean_distribution must be a number.")
        self._autoencoder_latent_mean_distribution = value

    # Getter and setter for autoencoder_latent_stander_deviation
    @property
    def autoencoder_latent_stander_deviation(self):
        """Gets the latent space standard deviation."""
        return self._autoencoder_latent_stander_deviation

    @autoencoder_latent_stander_deviation.setter
    def autoencoder_latent_stander_deviation(self, value):
        """Sets the latent space standard deviation."""
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError("autoencoder_latent_stander_deviation must be a non-negative number.")
        self._autoencoder_latent_stander_deviation = value

    # Getter and setter for autoencoder_file_name_encoder
    @property
    def autoencoder_file_name_encoder(self):
        """Gets the encoder filename."""
        return self._autoencoder_file_name_encoder

    @autoencoder_file_name_encoder.setter
    def autoencoder_file_name_encoder(self, value):
        """Sets the encoder filename."""
        if not isinstance(value, str):
            raise ValueError("autoencoder_file_name_encoder must be a string.")
        self._autoencoder_file_name_encoder = value

    # Getter and setter for autoencoder_file_name_decoder
    @property
    def autoencoder_file_name_decoder(self):
        """Gets the decoder filename."""
        return self._autoencoder_file_name_decoder

    @autoencoder_file_name_decoder.setter
    def autoencoder_file_name_decoder(self, value):
        """Sets the decoder filename."""
        if not isinstance(value, str):
            raise ValueError("autoencoder_file_name_decoder must be a string.")
        self._autoencoder_file_name_decoder = value

    # Getter and setter for autoencoder_path_output_models
    @property
    def autoencoder_path_output_models(self):
        """Gets the output models path."""
        return self._autoencoder_path_output_models

    @autoencoder_path_output_models.setter
    def autoencoder_path_output_models(self, value):
        """Sets the output models path."""
        if not isinstance(value, str):
            raise ValueError("autoencoder_path_output_models must be a string.")
        self._autoencoder_path_output_models = value