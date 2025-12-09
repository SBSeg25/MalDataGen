#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Kayuã Oleques']

from Engine.Algorithms.VariationalAutoencoder.VariationalAutoencoderAlgorithm import VariationalAutoencoderAlgorithm
from Engine.Architectures.VariationalAutoencoder.VariationalAutoencoderModel import VariationalAutoencoderModel

try:
    import sys
    import numpy
    import logging


except ImportError as error:
    logging.error(error)
    sys.exit(-1)


class VariationalAutoencoder:
    """
    A class that instantiates and manages a Variational Autoencoder (VAE) model.
    This implementation provides complete configuration, training, and management capabilities
    for variational autoencoder-based learning tasks.

    Attributes:
        _variational_model (VariationalAutoencoderModelTorch): Contains encoder and decoder components
        _variational_algorithm (VariationalAlgorithmTorch): Manages the VAE training process

    Configuration Parameters (with getters/setters):
        _variational_latent_dimension (int): Size of the latent space
        _variational_training_algorithm (str): Training algorithm specification
        _variational_activation_function (str): Activation function for hidden layers
        _variational_dropout_decay_rate_encoder (float): Encoder dropout rate
        _variational_dropout_decay_rate_decoder (float): Decoder dropout rate
        _variational_dense_layer_sizes_encoder (list): Encoder layer sizes
        _variational_dense_layer_sizes_decoder (list): Decoder layer sizes
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
    """

    def __init__(self, arguments):
        """
        Initializes the variational autoencoder instance with configuration parameters.

        Args:
            arguments (Namespace): Configuration object containing all required parameters.
        """
        self._variational_algorithm = None
        self._variational_model = None

        # ** Variational Autoencoder Model Configuration Parameters **
        self._variational_latent_dimension = arguments.variational_autoencoder_latent_dimension
        self._variational_training_algorithm = arguments.variational_autoencoder_training_algorithm
        self._variational_activation_function = arguments.variational_autoencoder_activation_function
        self._variational_dropout_decay_rate_encoder = arguments.variational_autoencoder_dropout_decay_rate_encoder
        self._variational_dropout_decay_rate_decoder = arguments.variational_autoencoder_dropout_decay_rate_decoder
        self._variational_dense_layer_sizes_encoder = arguments.variational_autoencoder_dense_layer_sizes_encoder
        self._variational_dense_layer_sizes_decoder = arguments.variational_autoencoder_dense_layer_sizes_decoder
        self._variational_batch_size = arguments.variational_autoencoder_batch_size
        self._variational_number_epochs = arguments.variational_autoencoder_number_epochs
        self._variational_number_classes = arguments.variational_autoencoder_number_classes
        self._variational_loss_function = arguments.variational_autoencoder_loss_function
        self._variational_momentum = arguments.variational_autoencoder_momentum
        self._variational_last_activation_layer = arguments.variational_autoencoder_last_activation_layer
        self._variational_initializer_mean = arguments.variational_autoencoder_initializer_mean
        self._variational_initializer_deviation = arguments.variational_autoencoder_initializer_deviation
        self._variational_latent_mean_distribution = arguments.variational_autoencoder_mean_distribution
        self._variational_latent_stander_deviation = arguments.variational_autoencoder_stander_deviation
        self._variational_file_name_encoder = arguments.variational_autoencoder_file_name_encoder
        self._variational_file_name_decoder = arguments.variational_autoencoder_file_name_decoder
        self._variational_path_output_models = arguments.variational_autoencoder_path_output_models

        # Get number_samples_per_class from arguments if available
        self._number_samples_per_class = getattr(arguments, 'number_samples_per_class', None)

    def _get_variational(self, input_shape):
        """
        Initialize and configure the Variational Autoencoder model.

        Args:
            input_shape (tuple):
                The shape of the input data, which determines the output shape for the models.
        """
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
            dataset_type=numpy.float32,
            number_samples_per_class=self._number_samples_per_class
        )

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

    def _training_variational_autoencoder_model(self, input_shape, arguments, x_real_samples, y_real_samples):
        """
        Executes the complete variational autoencoder training process.
        """
        # Initialize the variational autoencoder model
        self._get_variational(input_shape)

        # Configure optimizer
        self._variational_algorithm.configure_optimizer(
            learning_rate=arguments.variational_learning_rate if hasattr(arguments,
                                                                         'variational_learning_rate') else 0.001,
            beta_1=arguments.variational_beta_1 if hasattr(arguments, 'variational_beta_1') else 0.9,
            beta_2=arguments.variational_beta_2 if hasattr(arguments, 'variational_beta_2') else 0.999
        )

        # Prepare callbacks list
        callbacks_list = []

        # DEBUG: Verificar shapes
        print(f"DEBUG Training: x_real_samples shape = {x_real_samples.shape}")
        print(f"DEBUG Training: y_real_samples shape = {y_real_samples.shape}")

        # One-hot encode dos labels
        y_labels_one_hot = self._one_hot_encode(y_real_samples, self._number_samples_per_class["number_classes"])
        print(f"DEBUG Training: y_labels_one_hot shape = {y_labels_one_hot.shape}")

        # O target (y_data) deve ser x_real_samples para reconstrução
        y_data = x_real_samples
        print(f"DEBUG Training: y_data (target) shape = {y_data.shape}")

        # Fit the variational autoencoder model
        self._variational_algorithm.fit(
            (x_real_samples, y_labels_one_hot),  # x e labels
            y_data,  # y (target para reconstrução)
            epochs=self._variational_number_epochs,
            batch_size=self._variational_batch_size,
            callbacks=callbacks_list
        )

    def _one_hot_encode(self, labels, num_classes):
        """
        One-hot encode labels for PyTorch compatibility.

        Args:
            labels (ndarray): Integer labels with shape [batch_size] or [batch_size, 1]
            num_classes (int): Number of classes

        Returns:
            torch.Tensor: One-hot encoded labels with shape [batch_size, num_classes]
        """
        import torch

        print(f"DEBUG _one_hot_encode: Input labels shape = {labels.shape}")
        print(f"DEBUG _one_hot_encode: num_classes = {num_classes}")

        if isinstance(labels, numpy.ndarray):
            # Remover dimensão extra se for (n, 1) em vez de (n,)
            if len(labels.shape) == 2 and labels.shape[1] == 1:
                print(f"Flattening labels from shape {labels.shape}...")
                labels = labels.flatten()  # Converter de (n, 1) para (n,)
                print(f"Flattened labels shape = {labels.shape}")

            # Verificar valores únicos
            unique_vals = numpy.unique(labels)
            print(f"Unique label values: {unique_vals}")

            # Agora labels deve ser 1D
            if len(labels.shape) == 1:
                # Converter para tensor e fazer one-hot
                labels_tensor = torch.from_numpy(labels).long()
                one_hot = torch.nn.functional.one_hot(labels_tensor, num_classes).float()
                print(f"One-hot output shape = {one_hot.shape}")
                print(f"One-hot first 5 rows:\n{one_hot[:5]}")
                return one_hot
            else:
                raise ValueError(f"Labels should be 1D after flattening, but got shape {labels.shape}")

        elif isinstance(labels, torch.Tensor):
            # Mesma lógica para tensores
            if len(labels.shape) == 2 and labels.shape[1] == 1:
                labels = labels.squeeze(1)

            if len(labels.shape) == 1:
                return torch.nn.functional.one_hot(labels.long(), num_classes).float()
            else:
                raise ValueError(f"Tensor labels should be 1D, but got shape {labels.shape}")

        else:
            raise ValueError(f"Unsupported labels type: {type(labels)}")

    # Property getters and setters

    @property
    def variational_latent_dimension(self):
        return self._variational_latent_dimension

    @variational_latent_dimension.setter
    def variational_latent_dimension(self, value):
        self._variational_latent_dimension = value

    @property
    def variational_training_algorithm(self):
        return self._variational_training_algorithm

    @variational_training_algorithm.setter
    def variational_training_algorithm(self, value):
        self._variational_training_algorithm = value

    @property
    def variational_activation_function(self):
        return self._variational_activation_function

    @variational_activation_function.setter
    def variational_activation_function(self, value):
        self._variational_activation_function = value

    @property
    def variational_dropout_decay_rate_encoder(self):
        return self._variational_dropout_decay_rate_encoder

    @variational_dropout_decay_rate_encoder.setter
    def variational_dropout_decay_rate_encoder(self, value):
        self._variational_dropout_decay_rate_encoder = value

    @property
    def variational_dropout_decay_rate_decoder(self):
        return self._variational_dropout_decay_rate_decoder

    @variational_dropout_decay_rate_decoder.setter
    def variational_dropout_decay_rate_decoder(self, value):
        self._variational_dropout_decay_rate_decoder = value

    @property
    def variational_dense_layer_sizes_encoder(self):
        return self._variational_dense_layer_sizes_encoder

    @variational_dense_layer_sizes_encoder.setter
    def variational_dense_layer_sizes_encoder(self, value):
        self._variational_dense_layer_sizes_encoder = value

    @property
    def variational_dense_layer_sizes_decoder(self):
        return self._variational_dense_layer_sizes_decoder

    @variational_dense_layer_sizes_decoder.setter
    def variational_dense_layer_sizes_decoder(self, value):
        self._variational_dense_layer_sizes_decoder = value

    @property
    def variational_batch_size(self):
        return self._variational_batch_size

    @variational_batch_size.setter
    def variational_batch_size(self, value):
        self._variational_batch_size = value

    @property
    def variational_number_epochs(self):
        return self._variational_number_epochs

    @variational_number_epochs.setter
    def variational_number_epochs(self, value):
        self._variational_number_epochs = value

    @property
    def variational_number_classes(self):
        return self._variational_number_classes

    @variational_number_classes.setter
    def variational_number_classes(self, value):
        self._variational_number_classes = value

    @property
    def variational_loss_function(self):
        return self._variational_loss_function

    @variational_loss_function.setter
    def variational_loss_function(self, value):
        self._variational_loss_function = value

    @property
    def variational_momentum(self):
        return self._variational_momentum

    @variational_momentum.setter
    def variational_momentum(self, value):
        self._variational_momentum = value

    @property
    def variational_last_activation_layer(self):
        return self._variational_last_activation_layer

    @variational_last_activation_layer.setter
    def variational_last_activation_layer(self, value):
        self._variational_last_activation_layer = value

    @property
    def variational_initializer_mean(self):
        return self._variational_initializer_mean

    @variational_initializer_mean.setter
    def variational_initializer_mean(self, value):
        self._variational_initializer_mean = value

    @property
    def variational_initializer_deviation(self):
        return self._variational_initializer_deviation

    @variational_initializer_deviation.setter
    def variational_initializer_deviation(self, value):
        self._variational_initializer_deviation = value

    @property
    def variational_latent_mean_distribution(self):
        return self._variational_latent_mean_distribution

    @variational_latent_mean_distribution.setter
    def variational_latent_mean_distribution(self, value):
        self._variational_latent_mean_distribution = value

    @property
    def variational_latent_stander_deviation(self):
        return self._variational_latent_stander_deviation

    @variational_latent_stander_deviation.setter
    def variational_latent_stander_deviation(self, value):
        self._variational_latent_stander_deviation = value

    @property
    def variational_file_name_encoder(self):
        return self._variational_file_name_encoder

    @variational_file_name_encoder.setter
    def variational_file_name_encoder(self, value):
        self._variational_file_name_encoder = value

    @property
    def variational_file_name_decoder(self):
        return self._variational_file_name_decoder

    @variational_file_name_decoder.setter
    def variational_file_name_decoder(self, value):
        self._variational_file_name_decoder = value

    @property
    def variational_path_output_models(self):
        return self._variational_path_output_models

    @variational_path_output_models.setter
    def variational_path_output_models(self, value):
        self._variational_path_output_models = value

    @property
    def variational_algorithm(self):
        return self._variational_algorithm

    @property
    def variational_model(self):
        return self._variational_model