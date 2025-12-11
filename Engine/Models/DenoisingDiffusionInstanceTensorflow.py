#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

from Engine.Algorithms.DenoisingDiffusion.GaussianDenoisingDiffusion import GaussianDenoisingDiffusion

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

    from Engine.Algorithms.DenoisingDiffusion.Tensorflow.AlgorithmDenoisingDiffusionTensorflow import AlgorithmDenoisingDiffusionTensorflow
    from Engine.Algorithms.DenoisingDiffusion.Tensorflow.GaussianDenoisingDiffusionTensorflow import GaussianDiffusionTensorflow
    from Engine.Architectures.DenoisingDiffusion.Tensorflow.DenoisingDiffusionUNetModelTensorflow import DenoisingDiffusionUNetModelTensorflow

except ImportError as error:
    logging.error(error)
    sys.exit(-1)


class DenoisingDiffusionInstance:
    """
    A class that implements a Denoising Diffusion Probabilistic Model (DDPM) for image generation.
    This implementation uses a dual-UNet architecture with Gaussian diffusion to progressively
    denoise images through a Markov chain of diffusion steps.

    Key Components:
    - Two identical UNet models for the denoising process
    - Gaussian diffusion utilities for noise scheduling
    - Complete training pipeline for diffusion models
    - Exponential Moving Average (EMA) for model stability
    - Configurable architecture via hyperparameters

    Attributes:
        _denoising_gaussian_diffusion_util: Manages noise scheduling and diffusion process
        _denoising_diffusion_algorithm: Orchestrates the training and sampling process
        _denoising_second_unet_model: Second UNet in the denoising chain
        _denoising_first_unet_model: Primary UNet model for denoising

        # UNet Architecture Parameters
        _denoising_diffusion_unet_last_layer_activation: Final layer activation function
        _denoising_diffusion_latent_dimension: Dimensionality of latent space
        _denoising_diffusion_unet_num_embedding_channels: Channels for positional embeddings
        _denoising_diffusion_unet_channels_per_level: Channel configuration per UNet level
        _denoising_diffusion_unet_batch_size: Training batch size
        _denoising_diffusion_unet_attention_mode: Attention mechanism type
        _denoising_diffusion_unet_num_residual_blocks: Residual blocks per level
        _denoising_diffusion_unet_group_normalization: Whether to use group norm
        _denoising_diffusion_unet_intermediary_activation: Intermediate activation function
        _denoising_diffusion_unet_intermediary_activation_alpha: Alpha for activation (LeakyReLU etc.)
        _denoising_diffusion_unet_epochs: Number of training epochs

        # Diffusion Process Parameters
        _denoising_diffusion_gaussian_beta_start: Initial noise schedule value
        _denoising_diffusion_gaussian_beta_end: Final noise schedule value
        _denoising_diffusion_gaussian_time_steps: Number of diffusion steps
        _denoising_diffusion_gaussian_clip_min: Minimum noise value
        _denoising_diffusion_gaussian_clip_max: Maximum noise value
        _denoising_diffusion_margin: Margin for contrastive objectives
        _denoising_diffusion_ema: Whether to use EMA for model weights
        _denoising_diffusion_time_steps: Number of timesteps in diffusion process
    """

    def __init__(self, arguments):
        """
        Initializes the denoising diffusion instance with configuration parameters.

        Args:
            arguments (Namespace): Configuration object containing:
                - UNet architecture parameters
                - Diffusion process settings
                - Training hyperparameters
                - Optimization parameters
        """
        self._denoising_gaussian_diffusion_util = None
        self._denoising_diffusion_algorithm = None

        self._denoising_second_unet_model = None
        self._denoising_first_unet_model = None

        # ** Denoising Probabilistic LatentDiffusion (LDPD) Configuration Parameters **
        self._denoising_diffusion_unet_last_layer_activation = arguments.denoising_diffusion_unet_last_layer_activation
        self._denoising_diffusion_latent_dimension = arguments.denoising_diffusion_latent_dimension
        self._denoising_diffusion_unet_num_embedding_channels = arguments.denoising_diffusion_unet_num_embedding_channels
        self._denoising_diffusion_unet_channels_per_level = arguments.denoising_diffusion_unet_channels_per_level
        self._denoising_diffusion_unet_batch_size = arguments.denoising_diffusion_unet_batch_size
        self._denoising_diffusion_unet_attention_mode = arguments.denoising_diffusion_unet_attention_mode
        self._denoising_diffusion_unet_num_residual_blocks = arguments.denoising_diffusion_unet_num_residual_blocks
        self._denoising_diffusion_unet_group_normalization = arguments.denoising_diffusion_unet_group_normalization
        self._denoising_diffusion_unet_intermediary_activation = arguments.denoising_diffusion_unet_intermediary_activation
        self._denoising_diffusion_unet_intermediary_activation_alpha = arguments.denoising_diffusion_unet_intermediary_activation_alpha
        self._denoising_diffusion_unet_epochs = arguments.denoising_diffusion_unet_epochs

        # Diffusion Process Parameters
        self._denoising_diffusion_gaussian_beta_start = arguments.denoising_diffusion_gaussian_beta_start
        self._denoising_diffusion_gaussian_beta_end = arguments.denoising_diffusion_gaussian_beta_end
        self._denoising_diffusion_gaussian_time_steps = arguments.denoising_diffusion_gaussian_time_steps
        self._denoising_diffusion_gaussian_clip_min = arguments.denoising_diffusion_gaussian_clip_min
        self._denoising_diffusion_gaussian_clip_max = arguments.denoising_diffusion_gaussian_clip_max
        self._denoising_diffusion_margin = arguments.denoising_diffusion_margin
        self._denoising_diffusion_ema = arguments.denoising_diffusion_ema
        self._denoising_diffusion_time_steps = arguments.denoising_diffusion_time_steps

    def _get_denoising_diffusion(self, input_shape):
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
        self._denoising_first_instance_unet = DenoisingDiffusionUNetModelTensorflow(output_shape=input_shape,
                                                                                    embedding_channels= self._denoising_diffusion_unet_num_embedding_channels,
                                                                                    list_neurons_per_level=self._denoising_diffusion_unet_channels_per_level,
                                                                                    list_attentions=self._denoising_diffusion_unet_attention_mode,
                                                                                    number_residual_blocks=self._denoising_diffusion_unet_num_residual_blocks,
                                                                                    normalization_groups=self._denoising_diffusion_unet_group_normalization,
                                                                                    intermediary_activation_function=self._denoising_diffusion_unet_intermediary_activation,
                                                                                    intermediary_activation_alpha= self._denoising_diffusion_unet_intermediary_activation_alpha,
                                                                                    last_layer_activation=self._denoising_diffusion_unet_last_layer_activation,
                                                                                    number_samples_per_class=self._number_samples_per_class)

        # Initialize the second instance of UNet with the same configuration
        self._denoising_second_instance_unet = DenoisingDiffusionUNetModelTensorflow(output_shape=input_shape,
                                                                                     embedding_channels= self._denoising_diffusion_unet_num_embedding_channels,
                                                                                     list_neurons_per_level=self._denoising_diffusion_unet_channels_per_level,
                                                                                     list_attentions=self._denoising_diffusion_unet_attention_mode,
                                                                                     number_residual_blocks=self._denoising_diffusion_unet_num_residual_blocks,
                                                                                     normalization_groups=self._denoising_diffusion_unet_group_normalization,
                                                                                     intermediary_activation_function=self._denoising_diffusion_unet_intermediary_activation,
                                                                                     intermediary_activation_alpha= self._denoising_diffusion_unet_intermediary_activation_alpha,
                                                                                     last_layer_activation=self._denoising_diffusion_unet_last_layer_activation,
                                                                                     number_samples_per_class=self._number_samples_per_class)

        # Build the models for both UNet instances
        self._denoising_first_unet_model = self._denoising_first_instance_unet.build_model()
        self._denoising_second_unet_model = self._denoising_second_instance_unet.build_model()

        # Synchronize the weights of the second UNet model with the first one
        self._denoising_second_unet_model.set_weights(self._denoising_first_unet_model.get_weights())

        # Initialize the GaussianDiffusion utility for the diffusion process
        self._denoising_gaussian_diffusion_util = GaussianDenoisingDiffusion(beta_start=self._denoising_diffusion_gaussian_beta_start,
                                                                              beta_end=self._denoising_diffusion_gaussian_beta_end,
                                                                              time_steps=self._denoising_diffusion_gaussian_time_steps,
                                                                              clip_min=self._denoising_diffusion_gaussian_clip_min,
                                                                              clip_max=self._denoising_diffusion_gaussian_clip_max)




    def _training_denoising_diffusion_model(self, input_shape, arguments, x_real_samples, y_real_samples):
        """
        Executes the complete denoising diffusion training pipeline.

        The training process:
        1. Initializes dual UNet models and diffusion utilities
        2. Configures the diffusion algorithm with optimizers
        3. Prepares input data with proper dimensionality
        4. Trains using mean squared error on denoising objective
        5. Manages callbacks for monitoring and early stopping

        Args:
            input_shape (tuple): Input data dimensions
            arguments (Namespace): Training configuration
            x_real_samples (ndarray): Training samples
            y_real_samples (ndarray): Corresponding labels
        """
        # Initialize the diffusion model
        self._get_denoising_diffusion(input_shape)

        # Print the model summaries for the U-Net models
        self._denoising_first_unet_model.summary()
        self._denoising_second_unet_model.summary()

        # callbacks_list = [self._callback_resources_monitor, self._callback_model_monitor]
        callbacks_list = [self._callback_model_monitor]

        if arguments.use_early_stop:
            callbacks_list.append(self._callback_early_stop)

        # Initialize the final diffusion algorithm
        self._denoising_diffusion_algorithm = AlgorithmDenoisingDiffusionTensorflow(output_shape=input_shape,
                                                                                    first_unet_model=self._denoising_first_unet_model,
                                                                                    second_unet_model=self._denoising_second_unet_model,
                                                                                    gdf_util=self._denoising_gaussian_diffusion_util,
                                                                                    optimizer_autoencoder=Adam(
                                                                              learning_rate=0.0001),
                                                                                    optimizer_diffusion=Adam(
                                                                              learning_rate=0.0001),
                                                                                    time_steps=self._denoising_diffusion_gaussian_time_steps,
                                                                                    ema=self._denoising_diffusion_ema,
                                                                                    margin=self._denoising_diffusion_margin)

        # Compile the diffusion model
        self._denoising_diffusion_algorithm.compile(loss=MeanSquaredError(),
                                                    optimizer=Adam(learning_rate=0.0001))

        # callbacks_list = [self._callback_resources_monitor, self._callback_model_monitor]
        callbacks_list = [self._callback_model_monitor]

        if arguments.use_early_stop:
            callbacks_list.append(self._callback_early_stop)

        x_real_samples = numpy.array(x_real_samples)
        x_real_samples = tensorflow.expand_dims(x_real_samples, axis=-1)

        self._denoising_diffusion_algorithm.fit(
            x_real_samples,
            to_categorical(y_real_samples, num_classes=self._number_samples_per_class["number_classes"]),
            epochs=self._denoising_diffusion_unet_epochs, batch_size=self._denoising_diffusion_unet_batch_size,
            callbacks=callbacks_list)

    # Getter and setter for diffusion_unet_last_layer_activation
    @property
    def denoising_diffusion_unet_last_layer_activation(self):
        return self.denoising_diffusion_unet_last_layer_activation

    @denoising_diffusion_unet_last_layer_activation.setter
    def denoising_diffusion_unet_last_layer_activation(self, value):
        self.denoising_diffusion_unet_last_layer_activation = value

    # Getter and setter for diffusion_latent_dimension
    @property
    def denoising_diffusion_latent_dimension(self):
        return self.denoising_diffusion_latent_dimension

    @denoising_diffusion_latent_dimension.setter
    def denoising_diffusion_latent_dimension(self, value):
        self.denoising_diffusion_latent_dimension = value

    # Getter and setter for diffusion_unet_num_embedding_channels
    @property
    def denoising_diffusion_unet_num_embedding_channels(self):
        return self.denoising_diffusion_unet_num_embedding_channels

    @denoising_diffusion_unet_num_embedding_channels.setter
    def denoising_diffusion_unet_num_embedding_channels(self, value):
        self.denoising_diffusion_unet_num_embedding_channels = value

    # Getter and setter for diffusion_unet_channels_per_level
    @property
    def denoising_diffusion_unet_channels_per_level(self):
        return self.denoising_diffusion_unet_channels_per_level

    @denoising_diffusion_unet_channels_per_level.setter
    def denoising_diffusion_unet_channels_per_level(self, value):
        self.denoising_diffusion_unet_channels_per_level = value

    # Getter and setter for diffusion_unet_batch_size
    @property
    def denoising_diffusion_unet_batch_size(self):
        return self.denoising_diffusion_unet_batch_size

    @denoising_diffusion_unet_batch_size.setter
    def denoising_diffusion_unet_batch_size(self, value):
        self.denoising_diffusion_unet_batch_size = value

    # Getter and setter for diffusion_unet_attention_mode
    @property
    def denoising_diffusion_unet_attention_mode(self):
        return self.denoising_diffusion_unet_attention_mode

    @denoising_diffusion_unet_attention_mode.setter
    def denoising_diffusion_unet_attention_mode(self, value):
        self.denoising_diffusion_unet_attention_mode = value

    # Getter and setter for diffusion_unet_num_residual_blocks
    @property
    def denoising_diffusion_unet_num_residual_blocks(self):
        return self.denoising_diffusion_unet_num_residual_blocks

    @denoising_diffusion_unet_num_residual_blocks.setter
    def denoising_diffusion_unet_num_residual_blocks(self, value):
        self.denoising_diffusion_unet_num_residual_blocks = value

    # Getter and setter for diffusion_unet_group_normalization
    @property
    def denoising_diffusion_unet_group_normalization(self):
        return self.denoising_diffusion_unet_group_normalization

    @denoising_diffusion_unet_group_normalization.setter
    def denoising_diffusion_unet_group_normalization(self, value):
        self.denoising_diffusion_unet_group_normalization = value

    # Getter and setter for diffusion_unet_intermediary_activation
    @property
    def denoising_diffusion_unet_intermediary_activation(self):
        return self.denoising_diffusion_unet_intermediary_activation

    @denoising_diffusion_unet_intermediary_activation.setter
    def denoising_diffusion_unet_intermediary_activation(self, value):
        self.denoising_diffusion_unet_intermediary_activation = value

    # Getter and setter for diffusion_unet_intermediary_activation_alpha
    @property
    def denoising_diffusion_unet_intermediary_activation_alpha(self):
        return self.denoising_diffusion_unet_intermediary_activation_alpha

    @denoising_diffusion_unet_intermediary_activation_alpha.setter
    def denoising_diffusion_unet_intermediary_activation_alpha(self, value):
        self.denoising_diffusion_unet_intermediary_activation_alpha = value

    # Getter and setter for diffusion_unet_epochs
    @property
    def denoising_diffusion_unet_epochs(self):
        return self.denoising_diffusion_unet_epochs

    @denoising_diffusion_unet_epochs.setter
    def denoising_diffusion_unet_epochs(self, value):
        self.denoising_diffusion_unet_epochs = value

    # Getter and setter for diffusion_gaussian_beta_start
    @property
    def denoising_diffusion_gaussian_beta_start(self):
        return self.denoising_diffusion_gaussian_beta_start

    @denoising_diffusion_gaussian_beta_start.setter
    def denoising_diffusion_gaussian_beta_start(self, value):
        self.denoising_diffusion_gaussian_beta_start = value

    # Getter and setter for diffusion_gaussian_beta_end
    @property
    def denoising_diffusion_gaussian_beta_end(self):
        return self.denoising_diffusion_gaussian_beta_end

    @denoising_diffusion_gaussian_beta_end.setter
    def denoising_diffusion_gaussian_beta_end(self, value):
        self.denoising_diffusion_gaussian_beta_end = value

    # Getter and setter for diffusion_gaussian_time_steps
    @property
    def denoising_diffusion_gaussian_time_steps(self):
        return self.denoising_diffusion_gaussian_time_steps

    @denoising_diffusion_gaussian_time_steps.setter
    def denoising_diffusion_gaussian_time_steps(self, value):
        self.denoising_diffusion_gaussian_time_steps = value

    # Getter and setter for diffusion_gaussian_clip_min
    @property
    def denoising_diffusion_gaussian_clip_min(self):
        return self.denoising_diffusion_gaussian_clip_min

    @denoising_diffusion_gaussian_clip_min.setter
    def denoising_diffusion_gaussian_clip_min(self, value):
        self.denoising_diffusion_gaussian_clip_min = value

    # Getter and setter for diffusion_gaussian_clip_max
    @property
    def denoising_diffusion_gaussian_clip_max(self):
        return self.denoising_diffusion_gaussian_clip_max

    @denoising_diffusion_gaussian_clip_max.setter
    def denoising_diffusion_gaussian_clip_max(self, value):
        self.denoising_diffusion_gaussian_clip_max = value

    # Getter and setter for diffusion_autoencoder_loss
    @property
    def denoising_diffusion_autoencoder_loss(self):
        return self.denoising_diffusion_autoencoder_loss

    @denoising_diffusion_autoencoder_loss.setter
    def denoising_diffusion_autoencoder_loss(self, value):
        self.denoising_diffusion_autoencoder_loss = value

    # Getter and setter for diffusion_autoencoder_encoder_filters
    @property
    def denoising_diffusion_autoencoder_encoder_filters(self):
        return self.denoising_diffusion_autoencoder_encoder_filters

    @denoising_diffusion_autoencoder_encoder_filters.setter
    def denoising_diffusion_autoencoder_encoder_filters(self, value):
        self.denoising_diffusion_autoencoder_encoder_filters = value

    # Getter and setter for diffusion_autoencoder_decoder_filters
    @property
    def denoising_diffusion_autoencoder_decoder_filters(self):
        return self.denoising_diffusion_autoencoder_decoder_filters

    @denoising_diffusion_autoencoder_decoder_filters.setter
    def denoising_diffusion_autoencoder_decoder_filters(self, value):
        self.denoising_diffusion_autoencoder_decoder_filters = value

    # Getter and setter for diffusion_autoencoder_last_layer_activation
    @property
    def denoising_diffusion_autoencoder_last_layer_activation(self):
        return self.denoising_diffusion_autoencoder_last_layer_activation

    @denoising_diffusion_autoencoder_last_layer_activation.setter
    def denoising_diffusion_autoencoder_last_layer_activation(self, value):
        self.denoising_diffusion_autoencoder_last_layer_activation = value

    # Getter and setter for diffusion_autoencoder_latent_dimension
    @property
    def denoising_diffusion_autoencoder_latent_dimension(self):
        return self.denoising_diffusion_autoencoder_latent_dimension

    @denoising_diffusion_autoencoder_latent_dimension.setter
    def denoising_diffusion_autoencoder_latent_dimension(self, value):
        self.denoising_diffusion_autoencoder_latent_dimension = value

    # Getter and setter for diffusion_autoencoder_batch_size_create_embedding
    @property
    def denoising_diffusion_autoencoder_batch_size_create_embedding(self):
        return self.denoising_diffusion_autoencoder_batch_size_create_embedding

    @denoising_diffusion_autoencoder_batch_size_create_embedding.setter
    def denoising_diffusion_autoencoder_batch_size_create_embedding(self, value):
        self.denoising_diffusion_autoencoder_batch_size_create_embedding = value

    # Getter and setter for diffusion_autoencoder_batch_size_training
    @property
    def denoising_diffusion_autoencoder_batch_size_training(self):
        return self.denoising_diffusion_autoencoder_batch_size_training

    @denoising_diffusion_autoencoder_batch_size_training.setter
    def denoising_diffusion_autoencoder_batch_size_training(self, value):
        self.denoising_diffusion_autoencoder_batch_size_training = value

    # Getter and setter for diffusion_autoencoder_epochs
    @property
    def denoising_diffusion_autoencoder_epochs(self):
        return self.denoising_diffusion_autoencoder_epochs

    @denoising_diffusion_autoencoder_epochs.setter
    def denoising_diffusion_autoencoder_epochs(self, value):
        self.denoising_diffusion_autoencoder_epochs = value

    # Getter and setter for diffusion_autoencoder_intermediary_activation_function
    @property
    def denoising_diffusion_autoencoder_intermediary_activation_function(self):
        return self.denoising_diffusion_autoencoder_intermediary_activation_function

    @denoising_diffusion_autoencoder_intermediary_activation_function.setter
    def denoising_diffusion_autoencoder_intermediary_activation_function(self, value):
        self.denoising_diffusion_autoencoder_intermediary_activation_function = value

    # Getter and setter for diffusion_autoencoder_intermediary_activation_alpha
    @property
    def denoising_diffusion_autoencoder_intermediary_activation_alpha(self):
        return self.denoising_diffusion_autoencoder_intermediary_activation_alpha

    @denoising_diffusion_autoencoder_intermediary_activation_alpha.setter
    def denoising_diffusion_autoencoder_intermediary_activation_alpha(self, value):
        self.denoising_diffusion_autoencoder_intermediary_activation_alpha = value

    # Getter and setter for diffusion_autoencoder_activation_output_encoder
    @property
    def denoising_diffusion_autoencoder_activation_output_encoder(self):
        return self.denoising_diffusion_autoencoder_activation_output_encoder

    @denoising_diffusion_autoencoder_activation_output_encoder.setter
    def denoising_diffusion_autoencoder_activation_output_encoder(self, value):
        self.denoising_diffusion_autoencoder_activation_output_encoder = value

    # Getter and setter for diffusion_margin
    @property
    def denoising_diffusion_margin(self):
        return self.denoising_diffusion_margin

    @denoising_diffusion_margin.setter
    def denoising_diffusion_margin(self, value):
        self.denoising_diffusion_margin = value

    # Getter and setter for diffusion_ema
    @property
    def denoising_diffusion_ema(self):
        return self.denoising_diffusion_ema

    @denoising_diffusion_ema.setter
    def denoising_diffusion_ema(self, value):
        self.denoising_diffusion_ema = value

    # Getter and setter for diffusion_time_steps
    @property
    def denoising_diffusion_time_steps(self):
        return self.denoising_diffusion_time_steps

    @denoising_diffusion_time_steps.setter
    def denoising_diffusion_time_steps(self, value):
        self.denoising_diffusion_time_steps = value

