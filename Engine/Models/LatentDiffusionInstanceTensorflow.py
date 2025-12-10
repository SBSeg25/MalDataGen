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

    from Engine.Algorithms.LatentDiffusion.Tensorflow.AlgorithmVAELatentDiffusionTensorflow import VAELatentDiffusionAlgorithmTensorflow
    from Engine.Algorithms.LatentDiffusion.Tensorflow.GaussianLatentDiffusionTensorflow import GaussianDiffusionTensorflow
    from Engine.Architectures.LatentDiffusion.Tensorflow.DiffusionModelUnetTensorflow import UNetModel
    from Engine.Algorithms.LatentDiffusion.Tensorflow.AlgorithmLatentDiffusionTensorflow import \
        LatentDiffusionAlgorithmTensorflow

except ImportError as error:
    logging.error(error)
    sys.exit(-1)

# !/usr/bin/env python3
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
    import numpy as np
    import logging
    import tensorflow as tf

    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    from tensorflow.python.keras.losses import MeanSquaredError, BinaryCrossentropy

    from Engine.Algorithms.LatentDiffusion.Tensorflow.AlgorithmVAELatentDiffusionTensorflow import \
        VAELatentDiffusionAlgorithmTensorflow
    from Engine.Algorithms.LatentDiffusion.Tensorflow.GaussianLatentDiffusionTensorflow import \
        GaussianDiffusionTensorflow
    from Engine.Architectures.LatentDiffusion.Tensorflow.DiffusionModelUnetTensorflow import UNetModel
    from Engine.Architectures.LatentDiffusion.Tensorflow.VariationalAutoencoderModelTensorflow import VariationalModelDiffusionTensorflow
    from Engine.Algorithms.LatentDiffusion.Tensorflow.AlgorithmLatentDiffusionTensorflow import \
        LatentDiffusionAlgorithmTensorflow

except ImportError as error:
    logging.error(error)
    sys.exit(-1)

# Default values from your constants file
DEFAULT_LATENT_DIFFUSION_UNET_LAST_LAYER_ACTIVATION = 'linear'
DEFAULT_LATENT_DIFFUSION_LATENT_DIMENSION = 64
DEFAULT_LATENT_DIFFUSION_UNET_NUMBER_EMBEDDING_CHANNELS = 1
DEFAULT_LATENT_DIFFUSION_UNET_CHANNELS_PER_LEVEL = [1, 2, 4]
DEFAULT_LATENT_DIFFUSION_UNET_BATCH_SIZE = 128
DEFAULT_LATENT_DIFFUSION_UNET_ATTENTION_MODE = [False, True, True]
DEFAULT_LATENT_DIFFUSION_UNET_NUMBER_RESIDUAL_BLOCKS = 2
DEFAULT_LATENT_DIFFUSION_UNET_GROUP_NORMALIZATION = 1
DEFAULT_LATENT_DIFFUSION_UNET_INTERMEDIARY_ACTIVATION = 'swish'
DEFAULT_LATENT_DIFFUSION_UNET_INTERMEDIARY_ACTIVATION_ALPHA = 0.05
DEFAULT_LATENT_DIFFUSION_UNET_NUMBER_EPOCHS = 1000

DEFAULT_LATENT_DIFFUSION_GAUSSIAN_BETA_START = 1e-4
DEFAULT_LATENT_DIFFUSION_GAUSSIAN_BETA_END = 0.02
DEFAULT_LATENT_DIFFUSION_GAUSSIAN_TIME_STEPS = 1000
DEFAULT_LATENT_DIFFUSION_GAUSSIAN_CLIP_MIN = -1.0
DEFAULT_LATENT_DIFFUSION_GAUSSIAN_CLIP_MAX = 1.0

DEFAULT_LATENT_DIFFUSION_AUTOENCODER_LOSS = 'mse'
DEFAULT_LATENT_DIFFUSION_AUTOENCODER_ENCODER_FILTERS = [320, 160]
DEFAULT_LATENT_DIFFUSION_AUTOENCODER_DECODER_FILTERS = [160, 320]
DEFAULT_LATENT_DIFFUSION_AUTOENCODER_LAST_LAYER_ACTIVATION = 'sigmoid'
DEFAULT_LATENT_DIFFUSION_AUTOENCODER_LATENT_DIMENSION = 64
DEFAULT_LATENT_DIFFUSION_AUTOENCODER_BATCH_SIZE_CREATE_EMBEDDING = 128
DEFAULT_LATENT_DIFFUSION_AUTOENCODER_BATCH_SIZE_TRAINING = 64
DEFAULT_LATENT_DIFFUSION_AUTOENCODER_EPOCHS = 1000
DEFAULT_LATENT_DIFFUSION_AUTOENCODER_INTERMEDIARY_ACTIVATION = 'swish'
DEFAULT_LATENT_DIFFUSION_AUTOENCODER_INTERMEDIARY_ACTIVATION_ALPHA = 0.05
DEFAULT_LATENT_DIFFUSION_AUTOENCODER_ACTIVATION_OUTPUT_ENCODER = 'sigmoid'
DEFAULT_LATENT_DIFFUSION_AUTOENCODER_INITIALIZER_MEAN = 0.0
DEFAULT_LATENT_DIFFUSION_AUTOENCODER_INITIALIZER_DEVIATION = 0.125
DEFAULT_LATENT_DIFFUSION_AUTOENCODER_DROPOUT_DECAY_RATE_ENCODER = 0.2
DEFAULT_LATENT_DIFFUSION_AUTOENCODER_DROPOUT_DECAY_RATE_DECODER = 0.4
DEFAULT_LATENT_DIFFUSION_AUTOENCODER_FILE_NAME_ENCODER = "encoder_model"
DEFAULT_LATENT_DIFFUSION_AUTOENCODER_FILE_NAME_DECODER = "decoder_model"
DEFAULT_LATENT_DIFFUSION_AUTOENCODER_PATH_OUTPUT_MODELS = "models_saved/"
DEFAULT_LATENT_DIFFUSION_AUTOENCODER_MEAN_DISTRIBUTION = 0.5
DEFAULT_LATENT_DIFFUSION_AUTOENCODER_STANDER_DEVIATION = 0.125

DEFAULT_LATENT_DIFFUSION_MARGIN = 0.5
DEFAULT_LATENT_DIFFUSION_EMA = 0.999
DEFAULT_LATENT_DIFFUSION_TIME_STEPS = 1000


class LatentDiffusionTensorflow:
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
        _latent_variational_algorithm: Orchestrates training of the VAE within the diffusion context
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

    def __init__(self,
                 # UNet Parameters
                 unet_last_layer_activation: str = DEFAULT_LATENT_DIFFUSION_UNET_LAST_LAYER_ACTIVATION,
                 latent_dimension: int = DEFAULT_LATENT_DIFFUSION_LATENT_DIMENSION,
                 unet_num_embedding_channels: int = DEFAULT_LATENT_DIFFUSION_UNET_NUMBER_EMBEDDING_CHANNELS,
                 unet_channels_per_level: list[int] = None,
                 unet_batch_size: int = DEFAULT_LATENT_DIFFUSION_UNET_BATCH_SIZE,
                 unet_attention_mode: list[bool] = None,
                 unet_num_residual_blocks: int = DEFAULT_LATENT_DIFFUSION_UNET_NUMBER_RESIDUAL_BLOCKS,
                 unet_group_normalization: int = DEFAULT_LATENT_DIFFUSION_UNET_GROUP_NORMALIZATION,
                 unet_intermediary_activation: str = DEFAULT_LATENT_DIFFUSION_UNET_INTERMEDIARY_ACTIVATION,
                 unet_intermediary_activation_alpha: float = DEFAULT_LATENT_DIFFUSION_UNET_INTERMEDIARY_ACTIVATION_ALPHA,
                 unet_epochs: int = DEFAULT_LATENT_DIFFUSION_UNET_NUMBER_EPOCHS,

                 # Gaussian Diffusion Parameters
                 gaussian_beta_start: float = DEFAULT_LATENT_DIFFUSION_GAUSSIAN_BETA_START,
                 gaussian_beta_end: float = DEFAULT_LATENT_DIFFUSION_GAUSSIAN_BETA_END,
                 gaussian_time_steps: int = DEFAULT_LATENT_DIFFUSION_GAUSSIAN_TIME_STEPS,
                 gaussian_clip_min: float = DEFAULT_LATENT_DIFFUSION_GAUSSIAN_CLIP_MIN,
                 gaussian_clip_max: float = DEFAULT_LATENT_DIFFUSION_GAUSSIAN_CLIP_MAX,

                 # VAE Parameters
                 VAE_loss_function: str = DEFAULT_LATENT_DIFFUSION_AUTOENCODER_LOSS,
                 VAE_encoder_filters: list[int] = None,
                 VAE_decoder_filters: list[int] = None,
                 VAE_last_layer_activation: str = DEFAULT_LATENT_DIFFUSION_AUTOENCODER_LAST_LAYER_ACTIVATION,
                 VAE_latent_dimension: int = DEFAULT_LATENT_DIFFUSION_AUTOENCODER_LATENT_DIMENSION,
                 VAE_batch_size_create_embedding: int = DEFAULT_LATENT_DIFFUSION_AUTOENCODER_BATCH_SIZE_CREATE_EMBEDDING,
                 VAE_batch_size_training: int = DEFAULT_LATENT_DIFFUSION_AUTOENCODER_BATCH_SIZE_TRAINING,
                 VAE_epochs: int = DEFAULT_LATENT_DIFFUSION_AUTOENCODER_EPOCHS,
                 VAE_intermediary_activation_function: str = DEFAULT_LATENT_DIFFUSION_AUTOENCODER_INTERMEDIARY_ACTIVATION,
                 VAE_intermediary_activation_alpha: float = DEFAULT_LATENT_DIFFUSION_AUTOENCODER_INTERMEDIARY_ACTIVATION_ALPHA,
                 VAE_activation_output_encoder: str = DEFAULT_LATENT_DIFFUSION_AUTOENCODER_ACTIVATION_OUTPUT_ENCODER,

                 # Additional VAE Parameters
                 VAE_initializer_mean: float = DEFAULT_LATENT_DIFFUSION_AUTOENCODER_INITIALIZER_MEAN,
                 VAE_initializer_deviation: float = DEFAULT_LATENT_DIFFUSION_AUTOENCODER_INITIALIZER_DEVIATION,
                 VAE_dropout_decay_rate_encoder: float = DEFAULT_LATENT_DIFFUSION_AUTOENCODER_DROPOUT_DECAY_RATE_ENCODER,
                 VAE_dropout_decay_rate_decoder: float = DEFAULT_LATENT_DIFFUSION_AUTOENCODER_DROPOUT_DECAY_RATE_DECODER,
                 VAE_file_name_encoder: str = DEFAULT_LATENT_DIFFUSION_AUTOENCODER_FILE_NAME_ENCODER,
                 VAE_file_name_decoder: str = DEFAULT_LATENT_DIFFUSION_AUTOENCODER_FILE_NAME_DECODER,
                 VAE_path_output_models: str = DEFAULT_LATENT_DIFFUSION_AUTOENCODER_PATH_OUTPUT_MODELS,
                 VAE_mean_distribution: float = DEFAULT_LATENT_DIFFUSION_AUTOENCODER_MEAN_DISTRIBUTION,
                 VAE_stander_deviation: float = DEFAULT_LATENT_DIFFUSION_AUTOENCODER_STANDER_DEVIATION,

                 # Diffusion Training Parameters
                 margin: float = DEFAULT_LATENT_DIFFUSION_MARGIN,
                 ema: float = DEFAULT_LATENT_DIFFUSION_EMA,
                 time_steps: int = DEFAULT_LATENT_DIFFUSION_TIME_STEPS,

                 # Optional pre-initialized components
                 variational_algorithm: VAELatentDiffusionAlgorithmTensorflow | None = None,
                 variation_model: VariationalModelDiffusionTensorflow | None = None,
                 first_unet: UNetModel | None = None,
                 second_unet: UNetModel | None = None,
                 gaussian_diffusion_util: GaussianDiffusionTensorflow | None = None
                 ) -> None:
        """
        Initializes the latent diffusion instance with configuration parameters.

        Args:
            # UNet Parameters
            unet_last_layer_activation: Activation for the last layer of U-Net (default: 'linear')
            latent_dimension: Dimension of the latent space (default: 64)
            unet_num_embedding_channels: Number of embedding channels for U-Net (default: 1)
            unet_channels_per_level: List of channels per level in U-Net (default: [1, 2, 4])
            unet_batch_size: Batch size for U-Net training (default: 128)
            unet_attention_mode: Attention mode for U-Net (default: [False, True, True])
            unet_num_residual_blocks: Number of residual blocks in U-Net (default: 2)
            unet_group_normalization: Group normalization value for U-Net (default: 1)
            unet_intermediary_activation: Intermediary activation for U-Net (default: 'swish')
            unet_intermediary_activation_alpha: Alpha value for intermediary activation (default: 0.05)
            unet_epochs: Number of epochs for U-Net training (default: 1000)

            # Gaussian Diffusion Parameters
            gaussian_beta_start: Starting value of beta for Gaussian diffusion (default: 1e-4)
            gaussian_beta_end: Ending value of beta for Gaussian diffusion (default: 0.02)
            gaussian_time_steps: Number of time steps for Gaussian diffusion (default: 1000)
            gaussian_clip_min: Minimum clipping value for Gaussian noise (default: -1.0)
            gaussian_clip_max: Maximum clipping value for Gaussian noise (default: 1.0)

            # VAE Parameters
            VAE_loss_function: Loss function for Autoencoder (default: 'mse')
            VAE_encoder_filters: List of filters for Autoencoder encoder (default: [320, 160])
            VAE_decoder_filters: List of filters for Autoencoder decoder (default: [160, 320])
            VAE_last_layer_activation: Activation function for last layer of Autoencoder (default: 'sigmoid')
            VAE_latent_dimension: Dimension of the latent in Autoencoder (default: 64)
            VAE_batch_size_create_embedding: Batch size for creating embeddings in Autoencoder (default: 128)
            VAE_batch_size_training: Batch size for training Autoencoder (default: 64)
            VAE_epochs: Number of epochs for Autoencoder training (default: 1000)
            VAE_intermediary_activation_function: Intermediary activation function for Autoencoder (default: 'swish')
            VAE_intermediary_activation_alpha: Alpha value for intermediary activation (default: 0.05)
            VAE_activation_output_encoder: Activation function for output of encoder in Autoencoder (default: 'sigmoid')

            # Additional VAE Parameters
            VAE_initializer_mean: Initializer mean for VAE (default: 0.0)
            VAE_initializer_deviation: Initializer deviation for VAE (default: 0.125)
            VAE_dropout_decay_rate_encoder: Dropout decay rate for encoder (default: 0.2)
            VAE_dropout_decay_rate_decoder: Dropout decay rate for decoder (default: 0.4)
            VAE_file_name_encoder: File name for encoder model (default: 'encoder_model')
            VAE_file_name_decoder: File name for decoder model (default: 'decoder_model')
            VAE_path_output_models: Path for output models (default: 'models_saved/')
            VAE_mean_distribution: Mean distribution for VAE (default: 0.5)
            VAE_stander_deviation: Standard deviation for VAE (default: 0.125)

            # Diffusion Training Parameters
            margin: Margin for diffusion process (default: 0.5)
            ema: Exponential moving average for diffusion (default: 0.999)
            time_steps: Number of time steps for diffusion (default: 1000)

            # Optional pre-initialized components
            variational_algorithm: Optional pre-initialized VAELatentDiffusionAlgorithmTensorflow instance
            variation_model: Optional pre-initialized VariationalModelDiffusion instance
            first_unet: Optional pre-initialized UNetModel instance
            second_unet: Optional pre-initialized UNetModel instance
            gaussian_diffusion_util: Optional pre-initialized GaussianDiffusionTensorflow instance
        """
        # Store pre-initialized instances if provided
        self._latent_variational_algorithm: VAELatentDiffusionAlgorithmTensorflow | None = variational_algorithm
        self._latent_variation_model_diffusion: VariationalModelDiffusionTensorflow | None = variation_model
        self._latent_first_instance_unet: UNetModel | None = first_unet
        self._latent_second_instance_unet: UNetModel | None = second_unet
        self._latent_gaussian_diffusion_util: GaussianDiffusionTensorflow | None = gaussian_diffusion_util

        self._latent_first_unet_model = None
        self._latent_second_unet_model = None
        self._latent_autoencoder_diffusion = None

        # ** Latent Diffusion - UNet Parameters **
        self._latent_diffusion_unet_last_layer_activation: str = unet_last_layer_activation
        self._latent_diffusion_latent_dimension: int = latent_dimension
        self._latent_diffusion_unet_num_embedding_channels: int = unet_num_embedding_channels

        # Handle mutable default values safely
        self._latent_diffusion_unet_channels_per_level: list[int] = (
            unet_channels_per_level
            if unet_channels_per_level is not None
            else DEFAULT_LATENT_DIFFUSION_UNET_CHANNELS_PER_LEVEL.copy()
        )

        self._latent_diffusion_unet_batch_size: int = unet_batch_size

        self._latent_diffusion_unet_attention_mode: list[bool] = (
            unet_attention_mode
            if unet_attention_mode is not None
            else DEFAULT_LATENT_DIFFUSION_UNET_ATTENTION_MODE.copy()
        )

        self._latent_diffusion_unet_num_residual_blocks: int = unet_num_residual_blocks
        self._latent_diffusion_unet_group_normalization: int = unet_group_normalization
        self._latent_diffusion_unet_intermediary_activation: str = unet_intermediary_activation
        self._latent_diffusion_unet_intermediary_activation_alpha: float = unet_intermediary_activation_alpha
        self._latent_diffusion_unet_epochs: int = unet_epochs

        # ** Gaussian Diffusion Parameters **
        self._latent_diffusion_gaussian_beta_start: float = gaussian_beta_start
        self._latent_diffusion_gaussian_beta_end: float = gaussian_beta_end
        self._latent_diffusion_gaussian_time_steps: int = gaussian_time_steps
        self._latent_diffusion_gaussian_clip_min: float = gaussian_clip_min
        self._latent_diffusion_gaussian_clip_max: float = gaussian_clip_max

        # ** VAE Parameters **
        self._latent_diffusion_VAE_loss_function: str = VAE_loss_function

        self._latent_diffusion_VAE_encoder_filters: list[int] = (
            VAE_encoder_filters
            if VAE_encoder_filters is not None
            else DEFAULT_LATENT_DIFFUSION_AUTOENCODER_ENCODER_FILTERS.copy()
        )

        self._latent_diffusion_VAE_decoder_filters: list[int] = (
            VAE_decoder_filters
            if VAE_decoder_filters is not None
            else DEFAULT_LATENT_DIFFUSION_AUTOENCODER_DECODER_FILTERS.copy()
        )

        self._latent_diffusion_VAE_last_layer_activation: str = VAE_last_layer_activation
        self._latent_diffusion_VAE_latent_dimension: int = VAE_latent_dimension
        self._latent_diffusion_VAE_batch_size_create_embedding: int = VAE_batch_size_create_embedding
        self._latent_diffusion_VAE_batch_size_training: int = VAE_batch_size_training
        self._latent_diffusion_VAE_epochs: int = VAE_epochs
        self._latent_diffusion_VAE_intermediary_activation_function: str = VAE_intermediary_activation_function
        self._latent_diffusion_VAE_intermediary_activation_alpha: float = VAE_intermediary_activation_alpha
        self._latent_diffusion_VAE_activation_output_encoder: str = VAE_activation_output_encoder

        # ** Additional VAE Parameters **
        self._latent_diffusion_VAE_initializer_mean: float = VAE_initializer_mean
        self._latent_diffusion_VAE_initializer_deviation: float = VAE_initializer_deviation
        self._latent_diffusion_VAE_dropout_decay_rate_encoder: float = VAE_dropout_decay_rate_encoder
        self._latent_diffusion_VAE_dropout_decay_rate_decoder: float = VAE_dropout_decay_rate_decoder
        self._latent_diffusion_VAE_file_name_encoder: str = VAE_file_name_encoder
        self._latent_diffusion_VAE_file_name_decoder: str = VAE_file_name_decoder
        self._latent_diffusion_VAE_path_output_models: str = VAE_path_output_models
        self._latent_diffusion_VAE_mean_distribution: float = VAE_mean_distribution
        self._latent_diffusion_VAE_stander_deviation: float = VAE_stander_deviation

        # ** Diffusion Training Parameters **
        self._latent_diffusion_margin: float = margin
        self._latent_diffusion_ema: float = ema
        self._latent_diffusion_time_steps: int = time_steps

        # Flags to indicate if instances were provided externally
        self._has_external_variational_algorithm: bool = variational_algorithm is not None
        self._has_external_variation_model: bool = variation_model is not None
        self._has_external_first_unet: bool = first_unet is not None
        self._has_external_second_unet: bool = second_unet is not None
        self._has_external_gaussian_diffusion_util: bool = gaussian_diffusion_util is not None


    def _get_latent_diffusion(self, input_shape):
        """
        Initializes and configures the LatentDiffusion model using UNet architecture for image generation.

        Args:
            input_shape (tuple): The shape of the input data, typically the dimensions of the images (height, width, channels).
        """
        # Initialize UNet instances if not provided
        if not self._has_external_first_unet:
            self._latent_first_instance_unet = UNetModel(
                embedding_dimension=self._latent_diffusion_latent_dimension,
                embedding_channels=self._latent_diffusion_unet_num_embedding_channels,
                list_neurons_per_level=self._latent_diffusion_unet_channels_per_level,
                list_attentions=self._latent_diffusion_unet_attention_mode,
                number_residual_blocks=self._latent_diffusion_unet_num_residual_blocks,
                normalization_groups=self._latent_diffusion_unet_group_normalization,
                intermediary_activation_function=self._latent_diffusion_unet_intermediary_activation,
                intermediary_activation_alpha=self._latent_diffusion_unet_intermediary_activation_alpha,
                last_layer_activation=self._latent_diffusion_unet_last_layer_activation,
                number_samples_per_class=self._number_samples_per_class
            )
            self._latent_first_unet_model = self._latent_first_instance_unet.build_model()

        if not self._has_external_second_unet:
            self._latent_second_instance_unet = UNetModel(
                embedding_dimension=self._latent_diffusion_latent_dimension,
                embedding_channels=self._latent_diffusion_unet_num_embedding_channels,
                list_neurons_per_level=self._latent_diffusion_unet_channels_per_level,
                list_attentions=self._latent_diffusion_unet_attention_mode,
                number_residual_blocks=self._latent_diffusion_unet_num_residual_blocks,
                normalization_groups=self._latent_diffusion_unet_group_normalization,
                intermediary_activation_function=self._latent_diffusion_unet_intermediary_activation,
                intermediary_activation_alpha=self._latent_diffusion_unet_intermediary_activation_alpha,
                last_layer_activation=self._latent_diffusion_unet_last_layer_activation,
                number_samples_per_class=self._number_samples_per_class
            )
            self._latent_second_unet_model = self._latent_second_instance_unet.build_model()

        # Synchronize weights if both models are internal
        if not self._has_external_first_unet and not self._has_external_second_unet:
            self._latent_second_unet_model.set_weights(self._latent_first_unet_model.get_weights())

        # Initialize GaussianDiffusion utility if not provided
        if not self._has_external_gaussian_diffusion_util:
            self._latent_gaussian_diffusion_util = GaussianDiffusionTensorflow(
                beta_start=self._latent_diffusion_gaussian_beta_start,
                beta_end=self._latent_diffusion_gaussian_beta_end,
                time_steps=self._latent_diffusion_gaussian_time_steps,
                clip_min=self._latent_diffusion_gaussian_clip_min,
                clip_max=self._latent_diffusion_gaussian_clip_max
            )

        # Initialize VariationalModelDiffusion if not provided
        if not self._has_external_variation_model:
            self._latent_variation_model_diffusion = VariationalModelDiffusionTensorflow(
                latent_dimension=self._latent_diffusion_latent_dimension,
                output_shape=input_shape,
                activation_function=self._latent_diffusion_VAE_intermediary_activation_function,
                initializer_mean=self._latent_diffusion_VAE_initializer_mean,
                initializer_deviation=self._latent_diffusion_VAE_initializer_deviation,
                dropout_decay_encoder=self._latent_diffusion_VAE_dropout_decay_rate_encoder,
                dropout_decay_decoder=self._latent_diffusion_VAE_dropout_decay_rate_decoder,
                last_layer_activation=self._latent_diffusion_VAE_activation_output_encoder,
                number_neurons_encoder=self._latent_diffusion_VAE_encoder_filters,
                number_neurons_decoder=self._latent_diffusion_VAE_decoder_filters,
                dataset_type=np.float32,
                number_samples_per_class=self._number_samples_per_class
            )

        # Initialize VariationalAlgorithmDiffusion if not provided
        if not self._has_external_variational_algorithm:
            if self._latent_variation_model_diffusion is None:
                raise ValueError("VariationalModelDiffusion instance is required but was not provided.")

            self._latent_variational_algorithm = VAELatentDiffusionAlgorithmTensorflow(
                encoder_model=self._latent_variation_model_diffusion.get_encoder(),
                decoder_model=self._latent_variation_model_diffusion.get_decoder(),
                loss_function=self._latent_diffusion_VAE_loss_function,
                latent_dimension=self._latent_diffusion_latent_dimension,
                decoder_latent_dimension=self._latent_diffusion_latent_dimension,
                latent_mean_distribution=self._latent_diffusion_VAE_mean_distribution,
                latent_stander_deviation=self._latent_diffusion_VAE_stander_deviation,
                file_name_encoder=self._latent_diffusion_VAE_file_name_encoder,
                file_name_decoder=self._latent_diffusion_VAE_file_name_decoder,
                models_saved_path=self._latent_diffusion_VAE_path_output_models
            )

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

        # # Variational Model setup for the VAE's encoder and decoder
        self._variation_model = VariationalModelDiffusionTensorflow(latent_dimension=self._latent_diffusion_VAE_latent_dimension,
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
        self._variational_algorithm = VAELatentDiffusionAlgorithmTensorflow(encoder_model=self._variation_model.get_encoder(),
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
        print("---------------------------------------------------------")
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
        self._latent_variational_algorithm.compile(loss=self._latent_diffusion_VAE_loss_function)

        # callbacks_list = [self._callback_resources_monitor, self._callback_model_monitor]
        callbacks_list = [self._callback_model_monitor]

        if arguments.use_early_stop:
            callbacks_list.append(self._callback_early_stop)

        # Fit the diffusion model with the training data
        self._latent_variational_algorithm.fit((
            x_real_samples,
            to_categorical(y_real_samples, num_classes=self._number_samples_per_class["number_classes"])),
            x_real_samples, epochs=self._latent_diffusion_VAE_epochs,
            batch_size=self._latent_diffusion_VAE_batch_size_training,
            callbacks=callbacks_list)

        # Retrieve the trained encoder and decoder from the variational algorithm
        self._encoder_latent_diffusion = self._latent_variational_algorithm.get_encoder_trained()
        self._decoder_latent_diffusion = self._latent_variational_algorithm.get_decoder_trained()

        # Print summaries of the trained encoder and decoder
        self._encoder_latent_diffusion.summary()
        self._decoder_latent_diffusion.summary()

        # Initialize the final diffusion algorithm
        self._latent_diffusion_algorithm = LatentDiffusionAlgorithmTensorflow(first_unet_model=self._latent_first_unet_model,
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
        data_embedding = self._latent_variational_algorithm.create_embedding([
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
