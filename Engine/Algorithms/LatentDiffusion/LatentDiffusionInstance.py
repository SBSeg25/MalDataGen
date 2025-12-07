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

    import keras
    import numpy

    import logging
    import tensorflow

    from tensorflow.keras.optimizers import Adam

    from tensorflow.keras.utils import to_categorical

    from tensorflow.python.keras.losses import MeanSquaredError

    from tensorflow.python.keras.losses import BinaryCrossentropy

    from Engine.Algorithms.LatentDiffusion.AlgorithmVAELatentDiffusion import VAELatentDiffusionAlgorithm
    from Engine.Algorithms.LatentDiffusion.GaussianLatentDiffusion import GaussianDiffusion
    from Engine.Models.LatentDiffusion.DiffusionModelUnet import UNetModel
    from Engine.Models.LatentDiffusion.VariationalAutoencoderModel import VariationalModelDiffusion
    from Engine.Models.VariationalAutoencoder.VariationalAutoencoderModel import VariationalModel

except ImportError as error:
    logging.error(error)
    sys.exit(-1)




class LatentDiffusionInstance:
    """
    A class that implements a Latent Denoising Probabilistic Diffusion (LDPD) model for generative tasks.
    This implementation combines variational autoencoders with diffusion models in latent space for
    high-quality sample generation.

    Key Components:
    - Two UNet models for the diffusion process
    - Variational Autoencoder for latent space representation
    - Gaussian diffusion utilities for noise scheduling
    - Complete training pipeline for both VAE and diffusion components
    - Highly configurable architecture via arguments for research experimentation

    Attributes:
        _latent_variational_algorithm_diffusion: Orchestrates training of the VAE within the diffusion context
        _latent_variation_model_diffusion: Stores the encoder and decoder of the VAE
        _latent_autoencoder_diffusion: Core autoencoder used for latent embedding and reconstruction
        _latent_gaussian_diffusion_util: Utility object for beta schedules and diffusion parameters
        _latent_second_unet_model: The second-stage UNet used in the denoising chain
        _latent_first_unet_model: The initial UNet used in early-stage denoising

        # Latent Diffusion - UNet Parameters
        _latent_diffusion_unet_last_layer_activation: Activation function used in UNet's final layer
        _latent_diffusion_latent_dimension: Dimensionality of latent space
        _latent_diffusion_unet_num_embedding_channels: Number of channels for positional/time embeddings
        _latent_diffusion_unet_channels_per_level: Channel config per U-Net level
        _latent_diffusion_unet_batch_size: Batch size used during UNet training
        _latent_diffusion_unet_attention_mode: Attention mechanism used in UNet (e.g., multi-head, cross-attn)
        _latent_diffusion_unet_num_residual_blocks: Number of residual blocks per level
        _latent_diffusion_unet_group_normalization: Whether to apply group norm in UNet layers
        _latent_diffusion_unet_intermediary_activation: Activation function for intermediate layers
        _latent_diffusion_unet_intermediary_activation_alpha: Alpha value (if using LeakyReLU, etc.)
        _latent_diffusion_unet_epochs: Number of epochs for UNet training

        # Latent Diffusion - VAE Parameters
        _latent_diffusion_VAE_mean_distribution: Type of distribution for latent mean (e.g., normal)
        _latent_diffusion_VAE_stander_deviation: Std deviation for latent distribution
        _latent_diffusion_VAE_file_name_encoder: File path to save/load the encoder
        _latent_diffusion_VAE_file_name_decoder: File path to save/load the decoder
        _latent_diffusion_VAE_path_output_models: Directory to store trained autoencoder components
        _latent_diffusion_VAE_loss_function: Loss used to optimize VAE (e.g., MSE + KL)
        _latent_diffusion_VAE_encoder_filters: Conv filter settings for encoder
        _latent_diffusion_VAE_decoder_filters: Conv filter settings for decoder
        _latent_diffusion_VAE_last_layer_activation: Output activation of decoder
        _latent_diffusion_VAE_latent_dimension: Size of compressed latent vector
        _latent_diffusion_VAE_batch_size_create_embedding: Batch size used for embedding generation
        _latent_diffusion_VAE_batch_size_training: Batch size during VAE training
        _latent_diffusion_VAE_epochs: Training epochs for the VAE
        _latent_diffusion_VAE_intermediary_activation_function: Activation in intermediate layers
        _latent_diffusion_VAE_intermediary_activation_alpha: Alpha parameter for the activation
        _latent_diffusion_VAE_activation_output_encoder: Activation at output of encoder

        # Latent Diffusion - Noise and Training Parameters
        _latent_diffusion_margin: Margin used in contrastive or reconstruction objectives
        _latent_diffusion_ema: Use of Exponential Moving Average in parameter updates
        _latent_diffusion_time_steps: Number of time steps for forward/reverse diffusion

        # Gaussian Diffusion - Scheduling and Initializer
        _latent_diffusion_gaussian_beta_start: Initial β value for the schedule
        _latent_diffusion_gaussian_beta_end: Final β value for the schedule
        _latent_diffusion_gaussian_time_steps: Number of diffusion steps
        _latent_diffusion_gaussian_clip_min: Minimum value for scheduled noise
        _latent_diffusion_gaussian_clip_max: Maximum value for scheduled noise
        _latent_diffusion_VAE_initializer_mean: Initial mean for model weight initialization
        _latent_diffusion_VAE_initializer_deviation: Initial std deviation for model weights
        _latent_diffusion_VAE_dropout_decay_rate_encoder: Dropout decay schedule for encoder
        _latent_diffusion_VAE_dropout_decay_rate_decoder: Dropout decay schedule for decoder

    """

    def __init__(self, arguments):
        """
        Initializes the latent diffusion instance with configuration parameters.

        Args:
            arguments (Namespace): Configuration object containing:
                - UNet architecture parameters
                - VAE configuration
                - Gaussian diffusion settings
                - Training hyperparameters
                - Model saving paths
        """

        self._latent_variational_algorithm_diffusion = None
        self._latent_variation_model_diffusion = None

        self._latent_autoencoder_diffusion = None
        self._latent_gaussian_diffusion_util = None

        self._latent_second_unet_model = None
        self._latent_first_unet_model = None

        # ** Latent Denoising Probabilistic LatentDiffusion (LDPD) Configuration Parameters **
        self._latent_diffusion_unet_last_layer_activation = arguments.latent_diffusion_unet_last_layer_activation
        self._latent_diffusion_latent_dimension = arguments.latent_diffusion_latent_dimension
        self._latent_diffusion_unet_num_embedding_channels = arguments.latent_diffusion_unet_num_embedding_channels
        self._latent_diffusion_unet_channels_per_level = arguments.latent_diffusion_unet_channels_per_level
        self._latent_diffusion_unet_batch_size = arguments.latent_diffusion_unet_batch_size
        self._latent_diffusion_unet_attention_mode = arguments.latent_diffusion_unet_attention_mode
        self._latent_diffusion_unet_num_residual_blocks = arguments.latent_diffusion_unet_num_residual_blocks
        self._latent_diffusion_unet_group_normalization = arguments.latent_diffusion_unet_group_normalization
        self._latent_diffusion_unet_intermediary_activation = arguments.latent_diffusion_unet_intermediary_activation
        self._latent_diffusion_unet_intermediary_activation_alpha = arguments.latent_diffusion_unet_intermediary_activation_alpha
        self._latent_diffusion_unet_epochs = arguments.latent_diffusion_unet_epochs

        self._latent_diffusion_VAE_mean_distribution = arguments.latent_diffusion_autoencoder_mean_distribution
        self._latent_diffusion_VAE_stander_deviation = arguments.latent_diffusion_autoencoder_stander_deviation
        self._latent_diffusion_VAE_file_name_encoder = arguments.latent_diffusion_autoencoder_file_name_encoder
        self._latent_diffusion_VAE_file_name_decoder = arguments.latent_diffusion_autoencoder_file_name_decoder
        self._latent_diffusion_VAE_path_output_models = arguments.latent_diffusion_autoencoder_path_output_models

        # Gaussian Diffusion - Scheduling and Initializer
        self._latent_diffusion_gaussian_beta_start = arguments.latent_diffusion_gaussian_beta_start
        self._latent_diffusion_gaussian_beta_end = arguments.latent_diffusion_gaussian_beta_end
        self._latent_diffusion_gaussian_time_steps = arguments.latent_diffusion_gaussian_time_steps
        self._latent_diffusion_gaussian_clip_min = arguments.latent_diffusion_gaussian_clip_min
        self._latent_diffusion_gaussian_clip_max = arguments.latent_diffusion_gaussian_clip_max


        self._latent_diffusion_VAE_loss_function = arguments.latent_diffusion_autoencoder_loss
        self._latent_diffusion_VAE_encoder_filters = arguments.latent_diffusion_autoencoder_encoder_filters
        self._latent_diffusion_VAE_decoder_filters = arguments.latent_diffusion_autoencoder_decoder_filters
        self._latent_diffusion_VAE_last_layer_activation = arguments.latent_diffusion_autoencoder_last_layer_activation
        self._latent_diffusion_VAE_latent_dimension = arguments.latent_diffusion_autoencoder_latent_dimension
        self._latent_diffusion_VAE_batch_size_create_embedding = arguments.latent_diffusion_autoencoder_batch_size_create_embedding
        self._latent_diffusion_VAE_batch_size_training = arguments.latent_diffusion_autoencoder_batch_size_training
        self._latent_diffusion_VAE_epochs = arguments.latent_diffusion_autoencoder_epochs
        self._latent_diffusion_VAE_intermediary_activation_function = arguments.latent_diffusion_autoencoder_intermediary_activation_function
        self._latent_diffusion_VAE_intermediary_activation_alpha = arguments.latent_diffusion_autoencoder_intermediary_activation_alpha
        self._latent_diffusion_VAE_activation_output_encoder = arguments.latent_diffusion_autoencoder_activation_output_encoder
        self._latent_diffusion_margin = arguments.latent_diffusion_margin
        self._latent_diffusion_ema = arguments.latent_diffusion_ema
        self._latent_diffusion_time_steps = arguments.latent_diffusion_time_steps

        # ** Gaussian LatentDiffusion Configuration Parameters **
        self._latent_diffusion_VAE_initializer_mean = arguments.latent_diffusion_autoencoder_initializer_mean
        self._latent_diffusion_VAE_initializer_deviation = arguments.latent_diffusion_autoencoder_initializer_deviation
        self._latent_diffusion_VAE_dropout_decay_rate_encoder = arguments.latent_diffusion_autoencoder_dropout_decay_rate_encoder
        self._latent_diffusion_VAE_dropout_decay_rate_decoder = arguments.latent_diffusion_autoencoder_dropout_decay_rate_decoder


    def _get_latent_diffusion(self, input_shape):
        """
         Initializes and configures the LatentDiffusion model using UNet architecture for image generation.

         This method initializes multiple components required for the diffusion process, including
         two UNet instances, a DiffusionAutoencoderModel, and a GaussianDiffusion utility. The UNet
         instances are configured with the specified hyperparameters for building the model. The
         weights of the second UNet model are synchronized with the first one. Additionally, the method
         sets up the variational model diffusion and the associated variational algorithm diffusion
         for image generation and embedding reconstruction.

         Args:
             input_shape (tuple):
              The shape of the input data, typically the dimensions of the images (height, width, channels).

         Initializes:
             self._first_instance_unet (UNetModel):
                The first instance of the UNet model used for the diffusion process.
             self._second_instance_unet (UNetModel):
                The second instance of the UNet model, which is a copy of the first one.
             self._first_unet_model (Model):
                The compiled UNet model for the first instance.
             self._second_unet_model (Model):
                The compiled UNet model for the second instance, with synchronized weights from the first model.
             self._gaussian_diffusion_util (GaussianDiffusion):
                Utility for managing the diffusion process with Gaussian noise.
             self._variation_model_diffusion (VariationalModelDiffusion):
                The diffusion model with variational autoencoder for latent representation learning.
             self._variational_algorithm_diffusion (VariationalAlgorithmDiffusion):
                The algorithm for variational inference during the diffusion process.
         """

        # Initialize the first instance of UNet for the diffusion model
        self._latent_first_instance_unet = UNetModel(embedding_dimension=self._latent_diffusion_latent_dimension,
                                                     embedding_channels= self._latent_diffusion_unet_num_embedding_channels,
                                                     list_neurons_per_level=self._latent_diffusion_unet_channels_per_level,
                                                     list_attentions=self._latent_diffusion_unet_attention_mode,
                                                     number_residual_blocks=self._latent_diffusion_unet_num_residual_blocks,
                                                     normalization_groups=self._latent_diffusion_unet_group_normalization,
                                                     intermediary_activation_function=self._latent_diffusion_unet_intermediary_activation,
                                                     intermediary_activation_alpha= self._latent_diffusion_unet_intermediary_activation_alpha,
                                                     last_layer_activation=self._latent_diffusion_unet_last_layer_activation,
                                                     number_samples_per_class=self._number_samples_per_class)

        # Initialize the second instance of UNet with the same configuration
        self._latent_second_instance_unet = UNetModel(embedding_dimension=self._latent_diffusion_latent_dimension,
                                                      embedding_channels= self._latent_diffusion_unet_num_embedding_channels,
                                                      list_neurons_per_level=self._latent_diffusion_unet_channels_per_level,
                                                      list_attentions=self._latent_diffusion_unet_attention_mode,
                                                      number_residual_blocks=self._latent_diffusion_unet_num_residual_blocks,
                                                      normalization_groups=self._latent_diffusion_unet_group_normalization,
                                                      intermediary_activation_function=self._latent_diffusion_unet_intermediary_activation,
                                                      intermediary_activation_alpha= self._latent_diffusion_unet_intermediary_activation_alpha,
                                                      last_layer_activation=self._latent_diffusion_unet_last_layer_activation,
                                                      number_samples_per_class=self._number_samples_per_class)

        # Build the models for both UNet instances
        self._latent_first_unet_model = self._latent_first_instance_unet.build_model()
        self._latent_second_unet_model = self._latent_second_instance_unet.build_model()

        # Synchronize the weights of the second UNet model with the first one
        self._latent_second_unet_model.set_weights(self._latent_first_unet_model.get_weights())

        # Initialize the GaussianDiffusion utility for the diffusion process
        self._latent_gaussian_diffusion_util = GaussianDiffusion(beta_start=self._latent_diffusion_gaussian_beta_start,
                                                                 beta_end=self._latent_diffusion_gaussian_beta_end,
                                                                 time_steps=self._latent_diffusion_gaussian_time_steps,
                                                                 clip_min=self._latent_diffusion_gaussian_clip_min,
                                                                 clip_max=self._latent_diffusion_gaussian_clip_max)

        # Initialize the VariationalModelDiffusion for embedding learning and reconstructor
        self._latent_variation_model_diffusion = VariationalModelDiffusion(latent_dimension=self._latent_diffusion_latent_dimension, output_shape=input_shape,
                                                                           activation_function=self._latent_diffusion_VAE_intermediary_activation_function,
                                                                           initializer_mean=self._latent_diffusion_VAE_initializer_mean,
                                                                           initializer_deviation=self._latent_diffusion_VAE_initializer_deviation,
                                                                           dropout_decay_encoder=self._latent_diffusion_VAE_dropout_decay_rate_encoder,
                                                                           dropout_decay_decoder=self._latent_diffusion_VAE_dropout_decay_rate_decoder,
                                                                           last_layer_activation=self._latent_diffusion_VAE_activation_output_encoder,
                                                                           number_neurons_encoder=self._latent_diffusion_VAE_encoder_filters,
                                                                           number_neurons_decoder=self._latent_diffusion_VAE_decoder_filters,
                                                                           dataset_type=numpy.float32,
                                                                           number_samples_per_class = self._number_samples_per_class)

        # Initialize the VariationalAlgorithmDiffusion for the training and diffusion process
        self._latent_variational_algorithm_diffusion = VAELatentDiffusionAlgorithm(encoder_model=self._latent_variation_model_diffusion.get_encoder(),
                                                                                   decoder_model=self._latent_variation_model_diffusion.get_decoder(),
                                                                                   loss_function=self._latent_diffusion_VAE_loss_function,
                                                                                   latent_dimension=self._latent_diffusion_latent_dimension,
                                                                                   decoder_latent_dimension = self._latent_diffusion_latent_dimension,
                                                                                   latent_mean_distribution=self._latent_diffusion_VAE_mean_distribution,
                                                                                   latent_stander_deviation=self._latent_diffusion_VAE_stander_deviation,
                                                                                   file_name_encoder=self._latent_diffusion_VAE_file_name_encoder,
                                                                                   file_name_decoder=self._latent_diffusion_VAE_file_name_decoder,
                                                                                   models_saved_path=self._latent_diffusion_VAE_path_output_models)

    def _get_variational_autoencoder(self, input_shape):
        """
        Initializes and sets up a Variational Autoencoder (VAE) model.

        This method creates an instance of a Variational Autoencoder (VAE) by configuring its encoder and decoder
        components. It uses a custom `VariationalModel` class to define and manage these components, and a `VariationalAlgorithm`
        to handle the training and operations of the VAE model. The VAE is designed for probabilistic inference and data generation.

        Args:
            input_shape (tuple):
                The shape of the input data, which is used to define the output shape of the model.

        Initializes:
            self._variation_model:
                An instance of the `VariationalModel` class that includes the encoder and decoder setup with
                configurations like latent dimension, activation functions, dropout rates, and neural network sizes.
            self._variational_algorithm:
                An instance of the `VariationalAlgorithm` class that handles the VAE's training process, loss function,
                and model parameters, including latent mean and standard deviation distributions.

        """

        # Variational Model setup for the VAE's encoder and decoder
        self._variation_model = VariationalModel(latent_dimension=self._latent_diffusion_VAE_latent_dimension,
                                                 output_shape=input_shape,
                                                 activation_function=self._latent_diffusion_VAE_intermediary_activation_function,
                                                 initializer_mean=self._latent_diffusion_VAE_initializer_mean,
                                                 initializer_deviation=self._latent_diffusion_VAE_initializer_deviation,
                                                 dropout_decay_encoder=self._latent_diffusion_VAE_dropout_decay_rate_encoder,
                                                 dropout_decay_decoder=self._latent_diffusion_VAE_dropout_decay_rate_decoder,
                                                 last_layer_activation=self._latent_diffusion_VAE_last_layer_activation,
                                                 number_neurons_encoder=self._latent_diffusion_VAE_encoder_filters,
                                                 number_neurons_decoder=self._latent_diffusion_VAE_decoder_filters,
                                                 dataset_type=numpy.float32,
                                                 number_samples_per_class = self._number_samples_per_class)

        # Variational Algorithm setup for training and model operations
        self._variational_algorithm = VariationalAlgorithm(encoder_model=self._variation_model.get_encoder(),
                                                           decoder_model=self._variation_model.get_decoder(),
                                                           loss_function=self._latent_diffusion_VAE_loss_function,
                                                           latent_dimension=self._latent_diffusion_VAE_latent_dimension,
                                                           decoder_latent_dimension = self._latent_diffusion_VAE_latent_dimension,
                                                           latent_mean_distribution=self._latent_diffusion_VAE_mean_distribution,
                                                           latent_stander_deviation=self._latent_diffusion_VAE_stander_deviation,
                                                           file_name_encoder=self._latent_diffusion_VAE_file_name_encoder,
                                                           file_name_decoder=self._latent_diffusion_VAE_file_name_decoder,
                                                           models_saved_path=self._latent_diffusion_VAE_path_output_models)



    def _training_latent_diffusion_model(self, input_shape, arguments, x_real_samples, y_real_samples):
        """
        Executes the complete training pipeline for latent diffusion.

        Process:
        1. Initializes diffusion models
        2. Trains variational autoencoder
        3. Creates latent embeddings
        4. Trains diffusion models on latent space
        5. Manages callbacks and monitoring

        Args:
            input_shape (tuple): Input data shape
            arguments (Namespace): Training configuration
            x_real_samples (ndarray): Training samples
            y_real_samples (ndarray): Corresponding labels
        """
        # Initialize the diffusion model
        self._get_latent_diffusion(input_shape)

        # Print the model summaries for the U-Net models
        self._latent_first_unet_model.summary()
        self._latent_second_unet_model.summary()

        # Initialize the variational autoencoder model for diffusion
        self._get_variational_autoencoder(input_shape)

        self._latent_variation_model_diffusion.get_encoder().summary()
        self._latent_variation_model_diffusion.get_decoder().summary()

        # Compile the variational algorithm for diffusion
        self._latent_variational_algorithm_diffusion.compile(loss=self._latent_diffusion_VAE_loss_function)

        # callbacks_list = [self._callback_resources_monitor, self._callback_model_monitor]
        callbacks_list = [self._callback_model_monitor]

        if arguments.use_early_stop:
            callbacks_list.append(self._callback_early_stop)

        # Fit the diffusion model with the training data
        self._latent_variational_algorithm_diffusion.fit((
            x_real_samples,
            to_categorical(y_real_samples, num_classes=self._number_samples_per_class["number_classes"])),
            x_real_samples, epochs=self._latent_diffusion_VAE_epochs,
            batch_size=self._latent_diffusion_VAE_batch_size_training,
            callbacks=callbacks_list)

        # Retrieve the trained encoder and decoder from the variational algorithm
        self._encoder_latent_diffusion = self._latent_variational_algorithm_diffusion.get_encoder_trained()
        self._decoder_latent_diffusion = self._latent_variational_algorithm_diffusion.get_decoder_trained()

        # Print summaries of the trained encoder and decoder
        self._encoder_latent_diffusion.summary()
        self._decoder_latent_diffusion.summary()

        # Initialize the final diffusion algorithm
        self._latent_diffusion_algorithm = LatentDiffusionAlgorithm(first_unet_model=self._latent_first_unet_model,
                                                                    second_unet_model=self._latent_second_unet_model,
                                                                    encoder_model_image=self._encoder_latent_diffusion,
                                                                    decoder_model_image=self._decoder_latent_diffusion,
                                                                    gdf_util=self._latent_gaussian_diffusion_util,
                                                                    optimizer_autoencoder=Adam(learning_rate=0.0001),
                                                                    optimizer_diffusion=Adam(learning_rate=0.0001),
                                                                    time_steps=self._latent_diffusion_gaussian_time_steps,
                                                                    ema=self._latent_diffusion_ema,
                                                                    margin=self._latent_diffusion_margin,
                                                                    embedding_dimension=self._latent_diffusion_latent_dimension)

        # Compile the diffusion model
        self._latent_diffusion_algorithm.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001))

        # Prepare the data embedding and train the diffusion model
        data_embedding = self._latent_variational_algorithm_diffusion.create_embedding([
            x_real_samples,
            to_categorical(y_real_samples, num_classes=self._number_samples_per_class["number_classes"])])

        data_embedding = numpy.array(data_embedding)
        data_embedding = tensorflow.expand_dims(data_embedding, axis=-1)

        # callbacks_list = [self._callback_resources_monitor, self._callback_model_monitor]
        callbacks_list = [self._callback_model_monitor]

        if arguments.use_early_stop:
            callbacks_list.append(self._callback_early_stop)

        self._latent_diffusion_algorithm.fit(
            data_embedding,
            to_categorical(y_real_samples, num_classes=self._number_samples_per_class["number_classes"]),
            epochs=self._latent_diffusion_unet_epochs, batch_size=self._latent_diffusion_unet_batch_size,
            callbacks=callbacks_list, verbose=2)

    # ** Latent Denoising Probabilistic LatentDiffusion (LDPD) Configuration Parameters **
    @property
    def latent_diffusion_unet_last_layer_activation(self):
        """Getter for UNET last layer activation function"""
        return self._latent_diffusion_unet_last_layer_activation

    @latent_diffusion_unet_last_layer_activation.setter
    def latent_diffusion_unet_last_layer_activation(self, value):
        """Setter for UNET last layer activation function"""
        self._latent_diffusion_unet_last_layer_activation = value

    @property
    def latent_diffusion_latent_dimension(self):
        """Getter for latent dimension size"""
        return self._latent_diffusion_latent_dimension

    @latent_diffusion_latent_dimension.setter
    def latent_diffusion_latent_dimension(self, value):
        """Setter for latent dimension size"""
        self._latent_diffusion_latent_dimension = value

    @property
    def latent_diffusion_unet_num_embedding_channels(self):
        """Getter for number of embedding channels in UNET"""
        return self._latent_diffusion_unet_num_embedding_channels

    @latent_diffusion_unet_num_embedding_channels.setter
    def latent_diffusion_unet_num_embedding_channels(self, value):
        """Setter for number of embedding channels in UNET"""
        self._latent_diffusion_unet_num_embedding_channels = value

    @property
    def latent_diffusion_unet_channels_per_level(self):
        """Getter for channels per level in UNET"""
        return self._latent_diffusion_unet_channels_per_level

    @latent_diffusion_unet_channels_per_level.setter
    def latent_diffusion_unet_channels_per_level(self, value):
        """Setter for channels per level in UNET"""
        self._latent_diffusion_unet_channels_per_level = value

    @property
    def latent_diffusion_unet_batch_size(self):
        """Getter for UNET batch size"""
        return self._latent_diffusion_unet_batch_size

    @latent_diffusion_unet_batch_size.setter
    def latent_diffusion_unet_batch_size(self, value):
        """Setter for UNET batch size"""
        self._latent_diffusion_unet_batch_size = value

    @property
    def latent_diffusion_unet_attention_mode(self):
        """Getter for UNET attention mode"""
        return self._latent_diffusion_unet_attention_mode

    @latent_diffusion_unet_attention_mode.setter
    def latent_diffusion_unet_attention_mode(self, value):
        """Setter for UNET attention mode"""
        self._latent_diffusion_unet_attention_mode = value

    @property
    def latent_diffusion_unet_num_residual_blocks(self):
        """Getter for number of residual blocks in UNET"""
        return self._latent_diffusion_unet_num_residual_blocks

    @latent_diffusion_unet_num_residual_blocks.setter
    def latent_diffusion_unet_num_residual_blocks(self, value):
        """Setter for number of residual blocks in UNET"""
        self._latent_diffusion_unet_num_residual_blocks = value

    @property
    def latent_diffusion_unet_group_normalization(self):
        """Getter for UNET group normalization setting"""
        return self._latent_diffusion_unet_group_normalization

    @latent_diffusion_unet_group_normalization.setter
    def latent_diffusion_unet_group_normalization(self, value):
        """Setter for UNET group normalization setting"""
        self._latent_diffusion_unet_group_normalization = value

    @property
    def latent_diffusion_unet_intermediary_activation(self):
        """Getter for UNET intermediary activation function"""
        return self._latent_diffusion_unet_intermediary_activation

    @latent_diffusion_unet_intermediary_activation.setter
    def latent_diffusion_unet_intermediary_activation(self, value):
        """Setter for UNET intermediary activation function"""
        self._latent_diffusion_unet_intermediary_activation = value

    @property
    def latent_diffusion_unet_intermediary_activation_alpha(self):
        """Getter for alpha parameter of UNET intermediary activation"""
        return self._latent_diffusion_unet_intermediary_activation_alpha

    @latent_diffusion_unet_intermediary_activation_alpha.setter
    def latent_diffusion_unet_intermediary_activation_alpha(self, value):
        """Setter for alpha parameter of UNET intermediary activation"""
        self._latent_diffusion_unet_intermediary_activation_alpha = value

    @property
    def latent_diffusion_unet_epochs(self):
        """Getter for number of training epochs for UNET"""
        return self._latent_diffusion_unet_epochs

    @latent_diffusion_unet_epochs.setter
    def latent_diffusion_unet_epochs(self, value):
        """Setter for number of training epochs for UNET"""
        self._latent_diffusion_unet_epochs = value

    # VAE-related properties
    @property
    def latent_diffusion_VAE_mean_distribution(self):
        """Getter for VAE mean distribution"""
        return self._latent_diffusion_VAE_mean_distribution

    @latent_diffusion_VAE_mean_distribution.setter
    def latent_diffusion_VAE_mean_distribution(self, value):
        """Setter for VAE mean distribution"""
        self._latent_diffusion_VAE_mean_distribution = value

    @property
    def latent_diffusion_VAE_stander_deviation(self):
        """Getter for VAE standard deviation"""
        return self._latent_diffusion_VAE_stander_deviation

    @latent_diffusion_VAE_stander_deviation.setter
    def latent_diffusion_VAE_stander_deviation(self, value):
        """Setter for VAE standard deviation"""
        self._latent_diffusion_VAE_stander_deviation = value

    @property
    def latent_diffusion_VAE_file_name_encoder(self):
        """Getter for VAE encoder filename"""
        return self._latent_diffusion_VAE_file_name_encoder

    @latent_diffusion_VAE_file_name_encoder.setter
    def latent_diffusion_VAE_file_name_encoder(self, value):
        """Setter for VAE encoder filename"""
        self._latent_diffusion_VAE_file_name_encoder = value

    @property
    def latent_diffusion_VAE_file_name_decoder(self):
        """Getter for VAE decoder filename"""
        return self._latent_diffusion_VAE_file_name_decoder

    @latent_diffusion_VAE_file_name_decoder.setter
    def latent_diffusion_VAE_file_name_decoder(self, value):
        """Setter for VAE decoder filename"""
        self._latent_diffusion_VAE_file_name_decoder = value

    @property
    def latent_diffusion_VAE_path_output_models(self):
        """Getter for VAE output models path"""
        return self._latent_diffusion_VAE_path_output_models

    @latent_diffusion_VAE_path_output_models.setter
    def latent_diffusion_VAE_path_output_models(self, value):
        """Setter for VAE output models path"""
        self._latent_diffusion_VAE_path_output_models = value

    # Gaussian diffusion properties
    @property
    def latent_diffusion_gaussian_beta_start(self):
        """Getter for Gaussian diffusion beta start value"""
        return self._latent_diffusion_gaussian_beta_start

    @latent_diffusion_gaussian_beta_start.setter
    def latent_diffusion_gaussian_beta_start(self, value):
        """Setter for Gaussian diffusion beta start value"""
        self._latent_diffusion_gaussian_beta_start = value

    @property
    def latent_diffusion_gaussian_beta_end(self):
        """Getter for Gaussian diffusion beta end value"""
        return self._latent_diffusion_gaussian_beta_end

    @latent_diffusion_gaussian_beta_end.setter
    def latent_diffusion_gaussian_beta_end(self, value):
        """Setter for Gaussian diffusion beta end value"""
        self._latent_diffusion_gaussian_beta_end = value

    @property
    def latent_diffusion_gaussian_time_steps(self):
        """Getter for number of Gaussian diffusion time steps"""
        return self._latent_diffusion_gaussian_time_steps

    @latent_diffusion_gaussian_time_steps.setter
    def latent_diffusion_gaussian_time_steps(self, value):
        """Setter for number of Gaussian diffusion time steps"""
        self._latent_diffusion_gaussian_time_steps = value

    @property
    def latent_diffusion_gaussian_clip_min(self):
        """Getter for Gaussian diffusion minimum clip value"""
        return self._latent_diffusion_gaussian_clip_min

    @latent_diffusion_gaussian_clip_min.setter
    def latent_diffusion_gaussian_clip_min(self, value):
        """Setter for Gaussian diffusion minimum clip value"""
        self._latent_diffusion_gaussian_clip_min = value

    @property
    def latent_diffusion_gaussian_clip_max(self):
        """Getter for Gaussian diffusion maximum clip value"""
        return self._latent_diffusion_gaussian_clip_max

    @latent_diffusion_gaussian_clip_max.setter
    def latent_diffusion_gaussian_clip_max(self, value):
        """Setter for Gaussian diffusion maximum clip value"""
        self._latent_diffusion_gaussian_clip_max = value

    # More VAE properties
    @property
    def latent_diffusion_VAE_loss_function(self):
        """Getter for VAE loss function"""
        return self._latent_diffusion_VAE_loss_function

    @latent_diffusion_VAE_loss_function.setter
    def latent_diffusion_VAE_loss_function(self, value):
        """Setter for VAE loss function"""
        self._latent_diffusion_VAE_loss_function = value

    @property
    def latent_diffusion_VAE_encoder_filters(self):
        """Getter for VAE encoder filters"""
        return self._latent_diffusion_VAE_encoder_filters

    @latent_diffusion_VAE_encoder_filters.setter
    def latent_diffusion_VAE_encoder_filters(self, value):
        """Setter for VAE encoder filters"""
        self._latent_diffusion_VAE_encoder_filters = value

    @property
    def latent_diffusion_VAE_decoder_filters(self):
        """Getter for VAE decoder filters"""
        return self._latent_diffusion_VAE_decoder_filters

    @latent_diffusion_VAE_decoder_filters.setter
    def latent_diffusion_VAE_decoder_filters(self, value):
        """Setter for VAE decoder filters"""
        self._latent_diffusion_VAE_decoder_filters = value

    @property
    def latent_diffusion_VAE_last_layer_activation(self):
        """Getter for VAE last layer activation"""
        return self._latent_diffusion_VAE_last_layer_activation

    @latent_diffusion_VAE_last_layer_activation.setter
    def latent_diffusion_VAE_last_layer_activation(self, value):
        """Setter for VAE last layer activation"""
        self._latent_diffusion_VAE_last_layer_activation = value

    @property
    def latent_diffusion_VAE_latent_dimension(self):
        """Getter for VAE latent dimension"""
        return self._latent_diffusion_VAE_latent_dimension

    @latent_diffusion_VAE_latent_dimension.setter
    def latent_diffusion_VAE_latent_dimension(self, value):
        """Setter for VAE latent dimension"""
        self._latent_diffusion_VAE_latent_dimension = value

    @property
    def latent_diffusion_VAE_batch_size_create_embedding(self):
        """Getter for VAE embedding creation batch size"""
        return self._latent_diffusion_VAE_batch_size_create_embedding

    @latent_diffusion_VAE_batch_size_create_embedding.setter
    def latent_diffusion_VAE_batch_size_create_embedding(self, value):
        """Setter for VAE embedding creation batch size"""
        self._latent_diffusion_VAE_batch_size_create_embedding = value

    @property
    def latent_diffusion_VAE_batch_size_training(self):
        """Getter for VAE training batch size"""
        return self._latent_diffusion_VAE_batch_size_training

    @latent_diffusion_VAE_batch_size_training.setter
    def latent_diffusion_VAE_batch_size_training(self, value):
        """Setter for VAE training batch size"""
        self._latent_diffusion_VAE_batch_size_training = value

    @property
    def latent_diffusion_VAE_epochs(self):
        """Getter for VAE training epochs"""
        return self._latent_diffusion_VAE_epochs

    @latent_diffusion_VAE_epochs.setter
    def latent_diffusion_VAE_epochs(self, value):
        """Setter for VAE training epochs"""
        self._latent_diffusion_VAE_epochs = value

    @property
    def latent_diffusion_VAE_intermediary_activation_function(self):
        """Getter for VAE intermediary activation function"""
        return self._latent_diffusion_VAE_intermediary_activation_function

    @latent_diffusion_VAE_intermediary_activation_function.setter
    def latent_diffusion_VAE_intermediary_activation_function(self, value):
        """Setter for VAE intermediary activation function"""
        self._latent_diffusion_VAE_intermediary_activation_function = value

    @property
    def latent_diffusion_VAE_intermediary_activation_alpha(self):
        """Getter for VAE intermediary activation alpha parameter"""
        return self._latent_diffusion_VAE_intermediary_activation_alpha

    @latent_diffusion_VAE_intermediary_activation_alpha.setter
    def latent_diffusion_VAE_intermediary_activation_alpha(self, value):
        """Setter for VAE intermediary activation alpha parameter"""
        self._latent_diffusion_VAE_intermediary_activation_alpha = value

    @property
    def latent_diffusion_VAE_activation_output_encoder(self):
        """Getter for VAE encoder output activation"""
        return self._latent_diffusion_VAE_activation_output_encoder

    @latent_diffusion_VAE_activation_output_encoder.setter
    def latent_diffusion_VAE_activation_output_encoder(self, value):
        """Setter for VAE encoder output activation"""
        self._latent_diffusion_VAE_activation_output_encoder = value

    @property
    def latent_diffusion_margin(self):
        """Getter for latent diffusion margin parameter"""
        return self._latent_diffusion_margin

    @latent_diffusion_margin.setter
    def latent_diffusion_margin(self, value):
        """Setter for latent diffusion margin parameter"""
        self._latent_diffusion_margin = value

    @property
    def latent_diffusion_ema(self):
        """Getter for EMA (Exponential Moving Average) setting"""
        return self._latent_diffusion_ema

    @latent_diffusion_ema.setter
    def latent_diffusion_ema(self, value):
        """Setter for EMA (Exponential Moving Average) setting"""
        self._latent_diffusion_ema = value

    @property
    def latent_diffusion_time_steps(self):
        """Getter for number of diffusion time steps"""
        return self._latent_diffusion_time_steps

    @latent_diffusion_time_steps.setter
    def latent_diffusion_time_steps(self, value):
        """Setter for number of diffusion time steps"""
        self._latent_diffusion_time_steps = value

    # ** Gaussian LatentDiffusion Configuration Parameters **
    @property
    def latent_diffusion_VAE_initializer_mean(self):
        """Getter for VAE initializer mean"""
        return self._latent_diffusion_VAE_initializer_mean

    @latent_diffusion_VAE_initializer_mean.setter
    def latent_diffusion_VAE_initializer_mean(self, value):
        """Setter for VAE initializer mean"""
        self._latent_diffusion_VAE_initializer_mean = value

    @property
    def latent_diffusion_VAE_initializer_deviation(self):
        """Getter for VAE initializer standard deviation"""
        return self._latent_diffusion_VAE_initializer_deviation

    @latent_diffusion_VAE_initializer_deviation.setter
    def latent_diffusion_VAE_initializer_deviation(self, value):
        """Setter for VAE initializer standard deviation"""
        self._latent_diffusion_VAE_initializer_deviation = value

    @property
    def latent_diffusion_VAE_dropout_decay_rate_encoder(self):
        """Getter for VAE encoder dropout decay rate"""
        return self._latent_diffusion_VAE_dropout_decay_rate_encoder

    @latent_diffusion_VAE_dropout_decay_rate_encoder.setter
    def latent_diffusion_VAE_dropout_decay_rate_encoder(self, value):
        """Setter for VAE encoder dropout decay rate"""
        self._latent_diffusion_VAE_dropout_decay_rate_encoder = value

    @property
    def latent_diffusion_VAE_dropout_decay_rate_decoder(self):
        """Getter for VAE decoder dropout decay rate"""
        return self._latent_diffusion_VAE_dropout_decay_rate_decoder

    @latent_diffusion_VAE_dropout_decay_rate_decoder.setter
    def latent_diffusion_VAE_dropout_decay_rate_decoder(self, value):
        """Setter for VAE decoder dropout decay rate"""
        self._latent_diffusion_VAE_dropout_decay_rate_decoder = value

