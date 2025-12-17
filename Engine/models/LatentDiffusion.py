__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

from typing import Union, Optional, List

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
    import logging
    import numpy
    import numpy as np

    # Detect framework from environment
    ML_FRAMEWORK = os.environ.get('ML_FRAMEWORK', 'tensorflow').lower()

    from Engine.algorithms.latent_diffusion.AlgorithmLatentDiffusion import LatentDiffusionAlgorithm
    from Engine.algorithms.latent_diffusion.AlgorithmVAELatentDiffusion import AlgorithmVAELatentDiffusion
    from Engine.algorithms.latent_diffusion.GaussianLatentDiffusion import GaussianLatentDiffusion
    from Engine.architectures.latent_diffusion.DiffusionModelUNetModel import DiffusionModelUNetModel
    from Engine.architectures.latent_diffusion.VariationalModelDiffusion import VariationalModelDiffusion

    if ML_FRAMEWORK == 'tensorflow':
        import keras
        import tensorflow as tf
        import tensorflow
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.utils import to_categorical
        from tensorflow.python.keras.losses import MeanSquaredError, BinaryCrossentropy

    elif ML_FRAMEWORK == 'pytorch':
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.optim import Adam

    else:
        raise ValueError(f"Invalid ML_FRAMEWORK: '{ML_FRAMEWORK}'. Must be 'tensorflow' or 'pytorch'.")

except ImportError as error:
    logging.error(error)
    sys.exit(-1)

# Default values
DEFAULT_LATENT_DIFFUSION_UNET_LAST_LAYER_ACTIVATION = 'linear'
DEFAULT_LATENT_DIFFUSION_LATENT_DIMENSION = 64
DEFAULT_LATENT_DIFFUSION_UNET_NUMBER_EMBEDDING_CHANNELS = 1
DEFAULT_LATENT_DIFFUSION_UNET_CHANNELS_PER_LEVEL = [1, 2]
DEFAULT_LATENT_DIFFUSION_UNET_BATCH_SIZE = 256
DEFAULT_LATENT_DIFFUSION_UNET_ATTENTION_MODE = [False, True, True]
DEFAULT_LATENT_DIFFUSION_UNET_NUMBER_RESIDUAL_BLOCKS = 2
DEFAULT_LATENT_DIFFUSION_UNET_GROUP_NORMALIZATION = 1
DEFAULT_LATENT_DIFFUSION_UNET_INTERMEDIARY_ACTIVATION = 'swish'
DEFAULT_LATENT_DIFFUSION_UNET_INTERMEDIARY_ACTIVATION_ALPHA = 0.05
DEFAULT_LATENT_DIFFUSION_UNET_NUMBER_EPOCHS = 100

DEFAULT_LATENT_DIFFUSION_GAUSSIAN_BETA_START = 1e-4
DEFAULT_LATENT_DIFFUSION_GAUSSIAN_BETA_END = 0.02
DEFAULT_LATENT_DIFFUSION_GAUSSIAN_TIME_STEPS = 1000
DEFAULT_LATENT_DIFFUSION_GAUSSIAN_CLIP_MIN = -1.0
DEFAULT_LATENT_DIFFUSION_GAUSSIAN_CLIP_MAX = 1.0

DEFAULT_LATENT_DIFFUSION_AUTOENCODER_LOSS = 'mse'
DEFAULT_LATENT_DIFFUSION_AUTOENCODER_ENCODER_FILTERS = [512, 256]
DEFAULT_LATENT_DIFFUSION_AUTOENCODER_DECODER_FILTERS = [256, 512]
DEFAULT_LATENT_DIFFUSION_AUTOENCODER_LAST_LAYER_ACTIVATION = 'sigmoid'
DEFAULT_LATENT_DIFFUSION_AUTOENCODER_LATENT_DIMENSION = 64
DEFAULT_LATENT_DIFFUSION_AUTOENCODER_BATCH_SIZE_CREATE_EMBEDDING = 128
DEFAULT_LATENT_DIFFUSION_AUTOENCODER_BATCH_SIZE_TRAINING = 64
DEFAULT_LATENT_DIFFUSION_AUTOENCODER_EPOCHS = 100
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


class LatentDiffusion:
    """
    Framework-agnostic Latent Denoising Probabilistic Diffusion (LDPD) model.

    NOW SUPPORTS MULTI-DIMENSIONAL DATA: Automatically handles 1D, 2D, 3D, and N-D inputs
    by flattening them during training and reshaping them during generation.

    This unified class implements a latent diffusion model that works with both
    TensorFlow and PyTorch backends, automatically selecting the appropriate
    implementation based on the ML_FRAMEWORK environment variable.

    Key Components:
    - Two UNet models for the diffusion process
    - Variational autoencoder for latent space representation
    - Gaussian diffusion utilities for noise scheduling
    - Complete training pipeline for both VAE and diffusion components
    - Highly configurable architecture via arguments for research experimentation

    Environment Variables:
        ML_FRAMEWORK: Set to 'tensorflow' or 'pytorch' (default: 'tensorflow')

    Attributes:
        _framework: Current framework being used ('tensorflow' or 'pytorch')
        _device: Device for PyTorch ('cuda' or 'cpu')
        _original_input_shape: Stores the original shape of input data for reconstruction
        _latent_variational_algorithm: VAE training orchestrator
        _latent_variation_model_diffusion: VAE encoder/decoder models
        _latent_autoencoder_diffusion: Core autoencoder for latent space
        _latent_gaussian_diffusion_util: Gaussian diffusion utilities
        _latent_second_unet_model: Second-stage UNet
        _latent_first_unet_model: First-stage UNet

        [Additional attributes for all configuration parameters]
    """

    def __init__(self,
                 # UNet Parameters
                 unet_last_layer_activation: str = DEFAULT_LATENT_DIFFUSION_UNET_LAST_LAYER_ACTIVATION,
                 latent_dimension: int = DEFAULT_LATENT_DIFFUSION_LATENT_DIMENSION,
                 unet_num_embedding_channels: int = DEFAULT_LATENT_DIFFUSION_UNET_NUMBER_EMBEDDING_CHANNELS,
                 unet_channels_per_level: list = None,
                 unet_batch_size: int = DEFAULT_LATENT_DIFFUSION_UNET_BATCH_SIZE,
                 unet_attention_mode: list = None,
                 unet_num_residual_blocks: int = DEFAULT_LATENT_DIFFUSION_UNET_NUMBER_RESIDUAL_BLOCKS,
                 unet_group_normalization: int = DEFAULT_LATENT_DIFFUSION_UNET_GROUP_NORMALIZATION,
                 unet_intermediary_activation: str = DEFAULT_LATENT_DIFFUSION_UNET_INTERMEDIARY_ACTIVATION,
                 unet_intermediary_activation_alpha: float = DEFAULT_LATENT_DIFFUSION_UNET_INTERMEDIARY_ACTIVATION_ALPHA,
                 unet_epochs: int = DEFAULT_LATENT_DIFFUSION_UNET_NUMBER_EPOCHS,
                 number_classes: int = 3,
                 # Gaussian Diffusion Parameters
                 gaussian_beta_start: float = DEFAULT_LATENT_DIFFUSION_GAUSSIAN_BETA_START,
                 gaussian_beta_end: float = DEFAULT_LATENT_DIFFUSION_GAUSSIAN_BETA_END,
                 gaussian_time_steps: int = DEFAULT_LATENT_DIFFUSION_GAUSSIAN_TIME_STEPS,
                 gaussian_clip_min: float = DEFAULT_LATENT_DIFFUSION_GAUSSIAN_CLIP_MIN,
                 gaussian_clip_max: float = DEFAULT_LATENT_DIFFUSION_GAUSSIAN_CLIP_MAX,

                 # VAE Parameters
                 VAE_loss_function: str = DEFAULT_LATENT_DIFFUSION_AUTOENCODER_LOSS,
                 VAE_encoder_filters: list = None,
                 VAE_decoder_filters: list = None,
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

                 # Class Information
                 number_samples_per_class: dict = None,

                 # Optional pre-initialized components
                 variational_algorithm=None,
                 variation_model=None,
                 first_unet=None,
                 second_unet=None,
                 gaussian_diffusion_util=None,

                 # Framework-specific
                 device: str = None) -> None:
        """
        Initializes the latent diffusion model with automatic framework detection.

        Args:
            [All UNet, Gaussian, VAE, and training parameters as documented]
            number_samples_per_class: Dict with class information (e.g., {'number_classes': 10})
            device: Device for PyTorch ('cuda', 'cpu', or None for auto-detect)
        """

        # Store framework
        self._framework = ML_FRAMEWORK

        # Initialize device for PyTorch
        if self._framework == 'pytorch':
            if device is None:
                self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self._device = device

        # Store class information
        if number_samples_per_class is None:
            # Provide a default if not specified
            self._number_samples_per_class = {'number_classes': 2}

        else:
            self._number_samples_per_class = number_samples_per_class

        # Store pre-initialized instances if provided
        self._latent_variational_algorithm = variational_algorithm
        self._latent_variation_model_diffusion = variation_model
        self._latent_first_instance_unet = first_unet
        self._latent_second_instance_unet = second_unet
        self._latent_gaussian_diffusion_util = gaussian_diffusion_util

        self._latent_first_unet_model = None
        self._latent_second_unet_model = None
        self._latent_autoencoder_diffusion = None

        # Store all configuration parameters
        self._latent_diffusion_unet_last_layer_activation = unet_last_layer_activation
        self._latent_diffusion_latent_dimension = latent_dimension
        self._latent_diffusion_unet_num_embedding_channels = unet_num_embedding_channels

        self._latent_diffusion_unet_channels_per_level = (
            unet_channels_per_level if unet_channels_per_level is not None
            else DEFAULT_LATENT_DIFFUSION_UNET_CHANNELS_PER_LEVEL.copy()
        )

        self._latent_diffusion_unet_batch_size = unet_batch_size

        self._latent_diffusion_unet_attention_mode = (
            unet_attention_mode if unet_attention_mode is not None
            else DEFAULT_LATENT_DIFFUSION_UNET_ATTENTION_MODE.copy()
        )

        self._latent_diffusion_unet_num_residual_blocks = unet_num_residual_blocks
        self._latent_diffusion_unet_group_normalization = unet_group_normalization
        self._latent_diffusion_unet_intermediary_activation = unet_intermediary_activation
        self._latent_diffusion_unet_intermediary_activation_alpha = unet_intermediary_activation_alpha
        self._latent_diffusion_unet_epochs = unet_epochs

        # Gaussian Diffusion Parameters
        self._latent_diffusion_gaussian_beta_start = gaussian_beta_start
        self._latent_diffusion_gaussian_beta_end = gaussian_beta_end
        self._latent_diffusion_gaussian_time_steps = gaussian_time_steps
        self._latent_diffusion_gaussian_clip_min = gaussian_clip_min
        self._latent_diffusion_gaussian_clip_max = gaussian_clip_max

        # VAE Parameters
        self._latent_diffusion_VAE_loss_function = VAE_loss_function

        self._latent_diffusion_VAE_encoder_filters = (
            VAE_encoder_filters if VAE_encoder_filters is not None
            else DEFAULT_LATENT_DIFFUSION_AUTOENCODER_ENCODER_FILTERS.copy()
        )

        self._latent_diffusion_VAE_decoder_filters = (
            VAE_decoder_filters if VAE_decoder_filters is not None
            else DEFAULT_LATENT_DIFFUSION_AUTOENCODER_DECODER_FILTERS.copy()
        )

        self._latent_diffusion_VAE_last_layer_activation = VAE_last_layer_activation
        self._latent_diffusion_VAE_latent_dimension = VAE_latent_dimension
        self._latent_diffusion_VAE_batch_size_create_embedding = VAE_batch_size_create_embedding
        self._latent_diffusion_VAE_batch_size_training = VAE_batch_size_training
        self._latent_diffusion_VAE_epochs = VAE_epochs
        self._latent_diffusion_VAE_intermediary_activation_function = VAE_intermediary_activation_function
        self._latent_diffusion_VAE_intermediary_activation_alpha = VAE_intermediary_activation_alpha
        self._latent_diffusion_VAE_activation_output_encoder = VAE_activation_output_encoder

        # Additional VAE Parameters
        self._latent_diffusion_VAE_initializer_mean = VAE_initializer_mean
        self._latent_diffusion_VAE_initializer_deviation = VAE_initializer_deviation
        self._latent_diffusion_VAE_dropout_decay_rate_encoder = VAE_dropout_decay_rate_encoder
        self._latent_diffusion_VAE_dropout_decay_rate_decoder = VAE_dropout_decay_rate_decoder
        self._latent_diffusion_VAE_file_name_encoder = VAE_file_name_encoder
        self._latent_diffusion_VAE_file_name_decoder = VAE_file_name_decoder
        self._latent_diffusion_VAE_path_output_models = VAE_path_output_models
        self._latent_diffusion_VAE_mean_distribution = VAE_mean_distribution
        self._latent_diffusion_VAE_stander_deviation = VAE_stander_deviation

        # Diffusion Training Parameters
        self._latent_diffusion_margin = margin
        self._latent_diffusion_ema = ema
        self._latent_diffusion_time_steps = time_steps

        # Flags to indicate if instances were provided externally
        self._has_external_variational_algorithm = variational_algorithm is not None
        self._has_external_variation_model = variation_model is not None
        self._has_external_first_unet = first_unet is not None
        self._has_external_second_unet = second_unet is not None
        self._has_external_gaussian_diffusion_util = gaussian_diffusion_util is not None

        # Callback placeholders
        self._callback_model_monitor = None
        self._callback_early_stop = None
        self._callback_resources_monitor = None
        self._latent_diffusion_algorithm = None
        # Storage for original input shape (for multi-dimensional data)
        self._original_input_shape = None

    # ========================================================================
    # TENSORFLOW IMPLEMENTATION
    # ========================================================================

    def _get_latent_diffusion_tensorflow(self, input_shape):
        """TensorFlow-specific latent diffusion initialization."""
        # Initialize UNet instances if not provided
        if not self._has_external_first_unet:
            self._latent_first_instance_unet = DiffusionModelUNetModel(
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
            self._latent_second_instance_unet = DiffusionModelUNetModel(
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
            self._latent_gaussian_diffusion_util = GaussianLatentDiffusion(
                beta_start=self._latent_diffusion_gaussian_beta_start,
                beta_end=self._latent_diffusion_gaussian_beta_end,
                time_steps=self._latent_diffusion_gaussian_time_steps,
                clip_min=self._latent_diffusion_gaussian_clip_min,
                clip_max=self._latent_diffusion_gaussian_clip_max
            )

        # Initialize VariationalModelDiffusion if not provided
        if not self._has_external_variation_model:
            self._latent_variation_model_diffusion = VariationalModelDiffusion(
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

            self._latent_variational_algorithm = AlgorithmVAELatentDiffusion(
                encoder_model=self._latent_variation_model_diffusion.get_encoder(),
                decoder_model=self._latent_variation_model_diffusion.get_decoder(),
                loss_function=self._latent_diffusion_VAE_loss_function,
                latent_dimension=self._latent_diffusion_latent_dimension,
                decoder_latent_dimension=self._latent_diffusion_latent_dimension,
                latent_mean_distribution=self._latent_diffusion_VAE_mean_distribution,
                latent_standard_deviation=self._latent_diffusion_VAE_stander_deviation,
                file_name_encoder=self._latent_diffusion_VAE_file_name_encoder,
                file_name_decoder=self._latent_diffusion_VAE_file_name_decoder,
                models_saved_path=self._latent_diffusion_VAE_path_output_models
            )

    def _get_variational_autoencoder_tensorflow(self, input_shape):
        """TensorFlow-specific VAE initialization."""
        self._variation_model = VariationalModelDiffusion(
            latent_dimension=self._latent_diffusion_VAE_latent_dimension,
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
            number_samples_per_class=self._number_samples_per_class
        )

        self._variational_algorithm = AlgorithmVAELatentDiffusion(
            encoder_model=self._variation_model.get_encoder(),
            decoder_model=self._variation_model.get_decoder(),
            loss_function=self._latent_diffusion_VAE_loss_function,
            latent_dimension=self._latent_diffusion_VAE_latent_dimension,
            decoder_latent_dimension=self._latent_diffusion_VAE_latent_dimension,
            latent_mean_distribution=self._latent_diffusion_VAE_mean_distribution,
            latent_standard_deviation=self._latent_diffusion_VAE_stander_deviation,
            file_name_encoder=self._latent_diffusion_VAE_file_name_encoder,
            file_name_decoder=self._latent_diffusion_VAE_file_name_decoder,
            models_saved_path=self._latent_diffusion_VAE_path_output_models
        )

    def fit_model_tensorflow(
            self,
            input_shape: Union[int, tuple],
            x_real_samples: np.ndarray,
            y_real_samples: np.ndarray,
            batch_size: Optional[int] = None,
            epochs: Optional[int] = None,
            verbose: int = 1,
            callbacks: Optional[List] = None
    ) -> None:
        """
        TensorFlow-specific training pipeline.

        Args:
            input_shape (Union[int, tuple]): Input data shape
            x_real_samples (np.ndarray): Training samples
            y_real_samples (np.ndarray): Training labels
            batch_size (Optional[int]): Override default batch sizes if provided
            epochs (Optional[int]): Override default epoch counts if provided
            verbose (int): Verbosity mode
            callbacks (Optional[List]): List of callback objects for training monitoring
        """
        # Use provided parameters or fall back to defaults
        vae_batch_size = batch_size if batch_size is not None else self._latent_diffusion_VAE_batch_size_training
        vae_epochs = epochs if epochs is not None else self._latent_diffusion_VAE_epochs
        unet_batch_size = batch_size if batch_size is not None else self._latent_diffusion_unet_batch_size
        unet_epochs = epochs if epochs is not None else self._latent_diffusion_unet_epochs

        # Store callbacks
        if callbacks is not None:
            for callback in callbacks:
                callback_name = type(callback).__name__.lower()
                if 'early' in callback_name:
                    self._callback_early_stop = callback
                elif 'monitor' in callback_name and 'resource' not in callback_name:
                    self._callback_model_monitor = callback
                elif 'resource' in callback_name:
                    self._callback_resources_monitor = callback

        # Initialize latent diffusion components
        self._get_latent_diffusion_tensorflow(input_shape)

        # Prepare TensorFlow data
        y_real_samples_categorical = to_categorical(
            y_real_samples,
            num_classes=self._number_samples_per_class["number_classes"]
        )

        # Train VAE
        if verbose >= 1:
            print("Training VAE...")

        # Prepare callbacks for TensorFlow
        tf_callbacks = []
        if self._callback_early_stop is not None:
            tf_callbacks.append(self._callback_early_stop)
        if self._callback_model_monitor is not None:
            tf_callbacks.append(self._callback_model_monitor)
        if self._callback_resources_monitor is not None:
            tf_callbacks.append(self._callback_resources_monitor)

        self._latent_variational_algorithm.fit(
            x_real_samples,
            x_real_samples,
            y_real_samples_categorical,
            batch_size=vae_batch_size,
            epochs=vae_epochs,
            verbose=verbose,
            callbacks=tf_callbacks if tf_callbacks else None
        )

        # Get trained encoder and decoder
        self._encoder_latent_diffusion = self._latent_variational_algorithm.get_encoder_trained()
        self._decoder_latent_diffusion = self._latent_variational_algorithm.get_decoder_trained()

        # Create embeddings for diffusion training
        if verbose >= 1:
            print("Creating embeddings for diffusion training...")

        data_embedding = self._latent_variational_algorithm.create_embedding(
            x_real_samples,
            batch_size=self._latent_diffusion_VAE_batch_size_create_embedding
        )

        # Initialize and train diffusion algorithm
        self._latent_diffusion_algorithm = LatentDiffusionAlgorithm(
            first_unet_model=self._latent_first_unet_model,
            second_unet_model=self._latent_second_unet_model,
            encoder_model_image=self._encoder_latent_diffusion,
            decoder_model_image=self._decoder_latent_diffusion,
            gdf_util=self._latent_gaussian_diffusion_util,
            time_steps=self._latent_diffusion_gaussian_time_steps,
            ema=self._latent_diffusion_ema,
            margin=self._latent_diffusion_margin,
            embedding_dimension=self._latent_diffusion_latent_dimension
        )

        if verbose >= 1:
            print("Training diffusion model...")

        self._latent_diffusion_algorithm.fit(
            data_embedding,
            y_real_samples_categorical,
            epochs=unet_epochs,
            batch_size=unet_batch_size,
            verbose=verbose,
            callbacks=tf_callbacks if tf_callbacks else None
        )

    def fit_model_pytorch(
            self,
            input_shape: Union[int, tuple],
            x_real_samples: np.ndarray,
            y_real_samples: np.ndarray,
            batch_size: Optional[int] = None,
            epochs: Optional[int] = None,
            verbose: int = 1,
            callbacks: Optional[List] = None
    ) -> None:
        """
        PyTorch-specific training pipeline with proper model warm-up.

        Args:
            input_shape (Union[int, tuple]): Input data shape
            x_real_samples (np.ndarray): Training samples
            y_real_samples (np.ndarray): Training labels
            batch_size (Optional[int]): Override default batch sizes if provided
            epochs (Optional[int]): Override default epoch counts if provided
            verbose (int): Verbosity mode
            callbacks (Optional[List]): List of callback objects for training monitoring
        """
        # Use provided parameters or fall back to defaults
        vae_batch_size = batch_size if batch_size is not None else self._latent_diffusion_VAE_batch_size_training
        vae_epochs = epochs if epochs is not None else self._latent_diffusion_VAE_epochs
        unet_batch_size = batch_size if batch_size is not None else self._latent_diffusion_unet_batch_size
        unet_epochs = epochs if epochs is not None else self._latent_diffusion_unet_epochs

        # Store callbacks
        if callbacks is not None:
            for callback in callbacks:
                callback_name = type(callback).__name__.lower()
                if 'early' in callback_name:
                    self._callback_early_stop = callback
                elif 'monitor' in callback_name and 'resource' not in callback_name:
                    self._callback_model_monitor = callback
                elif 'resource' in callback_name:
                    self._callback_resources_monitor = callback

        # Initialize variation model if not provided
        if not self._has_external_variation_model:
            self._latent_variation_model_diffusion = VariationalModelDiffusion(
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
                dataset_type=numpy.float32,
                number_samples_per_class=self._number_samples_per_class
            ).to(self._device)

        # Initialize variational algorithm if not provided
        if not self._has_external_variational_algorithm:
            self._latent_variational_algorithm = AlgorithmVAELatentDiffusion(
                encoder_model=self._latent_variation_model_diffusion.get_encoder(),
                decoder_model=self._latent_variation_model_diffusion.get_decoder(),
                loss_function=self._latent_diffusion_VAE_loss_function,
                latent_dimension=self._latent_diffusion_latent_dimension,
                decoder_latent_dimension=self._latent_diffusion_latent_dimension,
                latent_mean_distribution=self._latent_diffusion_VAE_mean_distribution,
                latent_standard_deviation=self._latent_diffusion_VAE_stander_deviation,
                file_name_encoder=self._latent_diffusion_VAE_file_name_encoder,
                file_name_decoder=self._latent_diffusion_VAE_file_name_decoder,
                models_saved_path=self._latent_diffusion_VAE_path_output_models
            ).to(self._device)

        # Convert data to PyTorch tensors
        x_real_samples = torch.from_numpy(x_real_samples).float().to(self._device)
        y_real_samples = torch.from_numpy(y_real_samples).long().to(self._device)
        y_real_samples_onehot = F.one_hot(
            y_real_samples,
            num_classes=self._number_samples_per_class["number_classes"]
        ).float()

        # Create optimizer for VAE
        vae_optimizer = Adam(
            list(self._latent_variation_model_diffusion.get_encoder().parameters()) +
            list(self._latent_variation_model_diffusion.get_decoder().parameters()),
            lr=0.0001
        )

        num_batches = len(x_real_samples) // vae_batch_size

        # VAE Training Loop
        if verbose >= 1:
            print("Training VAE...")

        for epoch in range(vae_epochs):
            epoch_losses = []
            for batch_idx in range(num_batches):
                start_idx = batch_idx * vae_batch_size
                end_idx = start_idx + vae_batch_size
                batch_x = x_real_samples[start_idx:end_idx]
                batch_y = y_real_samples_onehot[start_idx:end_idx]

                loss_dict = self._latent_variational_algorithm.train_step(
                    (batch_x, batch_x), batch_y, vae_optimizer
                )
                epoch_losses.append(loss_dict['loss'])

            avg_loss = sum(epoch_losses) / len(epoch_losses)

            if verbose >= 1 and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"  VAE Epoch [{epoch + 1}/{vae_epochs}], Loss: {avg_loss:.4f}")

            # Call model monitor callback if available
            if self._callback_model_monitor is not None:
                self._callback_model_monitor.on_epoch_end(epoch, {'loss': avg_loss})

            if self._callback_early_stop and self._callback_early_stop.should_stop(avg_loss):
                if verbose >= 1:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

        # Get trained encoder and decoder
        self._encoder_latent_diffusion = self._latent_variational_algorithm.get_encoder_trained()
        self._decoder_latent_diffusion = self._latent_variational_algorithm.get_decoder_trained()

        # Initialize UNet models if not provided
        if not self._has_external_first_unet:
            self._latent_first_instance_unet = DiffusionModelUNetModel(
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
            ).to(self._device)

        if not self._has_external_second_unet:
            self._latent_second_instance_unet = DiffusionModelUNetModel(
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
            ).to(self._device)

            # CRITICAL FIX: Warm-up forward pass to create all dynamic layers
            if verbose >= 1:
                print("Initializing UNet architectures...")

            with torch.no_grad():
                # Create dummy inputs matching expected shapes
                dummy_batch_size = 2
                dummy_embedding = torch.randn(
                    dummy_batch_size,
                    self._latent_diffusion_latent_dimension,
                    self._latent_diffusion_unet_num_embedding_channels,
                    device=self._device
                )
                dummy_time = torch.randint(
                    0, self._latent_diffusion_gaussian_time_steps,
                    (dummy_batch_size,),
                    device=self._device
                )
                dummy_labels = torch.randn(
                    dummy_batch_size,
                    self._number_samples_per_class["number_classes"],
                    device=self._device
                )

                # Warm-up pass through first UNet to create all layers
                _ = self._latent_first_instance_unet(dummy_embedding, dummy_time, dummy_labels)

                # Warm-up pass through second UNet to create all layers
                _ = self._latent_second_instance_unet(dummy_embedding, dummy_time, dummy_labels)

            # NOW synchronize weights after both models have created all their layers
            self._latent_second_instance_unet.load_state_dict(
                self._latent_first_instance_unet.state_dict()
            )

            if verbose >= 1:
                print(
                    f"UNet models initialized with {len(list(self._latent_first_instance_unet.parameters()))} parameters each")

        # Initialize Gaussian diffusion utility if not provided
        if not self._has_external_gaussian_diffusion_util:
            self._latent_gaussian_diffusion_util = GaussianLatentDiffusion(
                beta_start=self._latent_diffusion_gaussian_beta_start,
                beta_end=self._latent_diffusion_gaussian_beta_end,
                time_steps=self._latent_diffusion_gaussian_time_steps,
                clip_min=self._latent_diffusion_gaussian_clip_min,
                clip_max=self._latent_diffusion_gaussian_clip_max
            ).to(self._device)

        # Create optimizers for the diffusion algorithm
        autoencoder_params = list(self._encoder_latent_diffusion.parameters()) + \
                             list(self._decoder_latent_diffusion.parameters())
        optimizer_autoencoder = Adam(autoencoder_params, lr=0.0001)
        optimizer_diffusion = Adam(self._latent_first_instance_unet.parameters(), lr=0.0001)

        # Initialize the final diffusion algorithm
        self._latent_diffusion_algorithm = LatentDiffusionAlgorithm(
            first_unet_model=self._latent_first_instance_unet,
            second_unet_model=self._latent_second_instance_unet,
            encoder_model_image=self._encoder_latent_diffusion,
            decoder_model_image=self._decoder_latent_diffusion,
            gdf_util=self._latent_gaussian_diffusion_util,
            optimizer_autoencoder=optimizer_autoencoder,
            optimizer_diffusion=optimizer_diffusion,
            time_steps=self._latent_diffusion_gaussian_time_steps,
            ema=self._latent_diffusion_ema,
            margin=self._latent_diffusion_margin,
            embedding_dimension=self._latent_diffusion_latent_dimension
        )

        # Create embeddings for diffusion training
        if verbose >= 1:
            print("Creating embeddings for diffusion training...")

        data_embedding = self._latent_variational_algorithm.create_embedding(
            x_real_samples.cpu().numpy(),
            batch_size=self._latent_diffusion_VAE_batch_size_create_embedding
        )

        data_embedding = torch.from_numpy(data_embedding).float().to(self._device)
        if len(data_embedding.shape) == 2:
            data_embedding = data_embedding.unsqueeze(-1)

        # Train diffusion model
        if verbose >= 1:
            print("Training diffusion model...")

        self._latent_diffusion_algorithm.fit(
            data_embedding,
            y_real_samples_onehot,
            epochs=unet_epochs,
            batch_size=unet_batch_size,
            verbose=verbose
        )

    def _detect_embedding_shape_pytorch(self, x_real_samples):
        """PyTorch-specific helper to detect embedding shape."""
        if len(x_real_samples) > 32:
            sample = x_real_samples[:32]
        else:
            sample = x_real_samples

        test_embedding = self._latent_variational_algorithm.create_embedding(
            sample,
            batch_size=min(16, len(sample))
        )

        test_embedding = torch.from_numpy(test_embedding).float()
        if len(test_embedding.shape) == 2:
            test_embedding = test_embedding.unsqueeze(-1)

        actual_shape = test_embedding.shape[1:]

        return actual_shape

    def fit_model(self, input_shape,
                  x_real_samples,
                  y_real_samples,
                  batch_size: Optional[int] = None,
                  epochs: Optional[int] = None,
                  verbose: int = 1,
                  callbacks: Optional[List] = None):
        """
        Executes the complete training pipeline for latent diffusion with automatic data flattening.

        NOW SUPPORTS MULTI-DIMENSIONAL DATA:
        - Automatically flattens N-D input data (1D, 2D, 3D, etc.)
        - Stores original shape for reconstruction during generation
        - Routes to framework-specific implementation

        Args:
            input_shape (tuple): Input data shape (e.g., (784,) for 1D, (28, 28, 1) for 2D, (16, 16, 16) for 3D)
            x_real_samples (ndarray): Training samples (can be N-dimensional)
            y_real_samples (ndarray): Corresponding labels (1D array of class indices)
            batch_size (Optional[int]): Override default batch sizes for training
            epochs (Optional[int]): Override default epoch counts for training
            verbose (int): Verbosity level (0=silent, 1=progress)
            callbacks (Optional[List]): List of callback objects for training monitoring
        """
        # Store original input shape for later reconstruction
        self._original_input_shape = input_shape

        # Calculate total flattened dimension
        flattened_dim = int(np.prod(input_shape))

        # CRITICAL: Set up number_samples_per_class from training labels
        # This must be done before building the models
        num_classes = int(y_real_samples.max()) + 1
        self._number_samples_per_class = {
            "number_classes": num_classes
        }
        # Flatten the input data if it has more than 2 dimensions
        # (batch_size, ...) -> (batch_size, flattened_features)
        if len(x_real_samples.shape) > 2:
            x_real_samples_flat = x_real_samples.reshape(x_real_samples.shape[0], -1)
        else:
            x_real_samples_flat = x_real_samples

        # Route to appropriate training method with flattened dimension
        if self._framework == 'tensorflow':
            self.fit_model_tensorflow(flattened_dim, x_real_samples_flat, y_real_samples,
                                      batch_size, epochs, verbose, callbacks)
        else:  # pytorch
            self.fit_model_pytorch(flattened_dim, x_real_samples_flat, y_real_samples,
                                   batch_size, epochs, verbose, callbacks)

    @property
    def framework(self):
        """Get the current framework being used."""
        return self._framework

    @property
    def device(self):
        """Get the device (PyTorch only)."""
        if self._framework == 'pytorch':
            return self._device
        return None

    @device.setter
    def device(self, value):
        """Set the device (PyTorch only)."""
        if self._framework == 'pytorch':
            self._device = value

    # UNet Properties
    @property
    def latent_diffusion_unet_last_layer_activation(self):
        return self._latent_diffusion_unet_last_layer_activation

    @latent_diffusion_unet_last_layer_activation.setter
    def latent_diffusion_unet_last_layer_activation(self, value):
        self._latent_diffusion_unet_last_layer_activation = value

    @property
    def latent_diffusion_latent_dimension(self):
        return self._latent_diffusion_latent_dimension

    @latent_diffusion_latent_dimension.setter
    def latent_diffusion_latent_dimension(self, value):
        self._latent_diffusion_latent_dimension = value

    @property
    def latent_diffusion_unet_num_embedding_channels(self):
        return self._latent_diffusion_unet_num_embedding_channels

    @latent_diffusion_unet_num_embedding_channels.setter
    def latent_diffusion_unet_num_embedding_channels(self, value):
        self._latent_diffusion_unet_num_embedding_channels = value

    @property
    def latent_diffusion_unet_channels_per_level(self):
        return self._latent_diffusion_unet_channels_per_level

    @latent_diffusion_unet_channels_per_level.setter
    def latent_diffusion_unet_channels_per_level(self, value):
        self._latent_diffusion_unet_channels_per_level = value

    @property
    def latent_diffusion_unet_batch_size(self):
        return self._latent_diffusion_unet_batch_size

    @latent_diffusion_unet_batch_size.setter
    def latent_diffusion_unet_batch_size(self, value):
        self._latent_diffusion_unet_batch_size = value

    @property
    def latent_diffusion_unet_attention_mode(self):
        return self._latent_diffusion_unet_attention_mode

    @latent_diffusion_unet_attention_mode.setter
    def latent_diffusion_unet_attention_mode(self, value):
        self._latent_diffusion_unet_attention_mode = value

    @property
    def latent_diffusion_unet_num_residual_blocks(self):
        return self._latent_diffusion_unet_num_residual_blocks

    @latent_diffusion_unet_num_residual_blocks.setter
    def latent_diffusion_unet_num_residual_blocks(self, value):
        self._latent_diffusion_unet_num_residual_blocks = value

    @property
    def latent_diffusion_unet_group_normalization(self):
        return self._latent_diffusion_unet_group_normalization

    @latent_diffusion_unet_group_normalization.setter
    def latent_diffusion_unet_group_normalization(self, value):
        self._latent_diffusion_unet_group_normalization = value

    @property
    def latent_diffusion_unet_intermediary_activation(self):
        return self._latent_diffusion_unet_intermediary_activation

    @latent_diffusion_unet_intermediary_activation.setter
    def latent_diffusion_unet_intermediary_activation(self, value):
        self._latent_diffusion_unet_intermediary_activation = value

    @property
    def latent_diffusion_unet_intermediary_activation_alpha(self):
        return self._latent_diffusion_unet_intermediary_activation_alpha

    @latent_diffusion_unet_intermediary_activation_alpha.setter
    def latent_diffusion_unet_intermediary_activation_alpha(self, value):
        self._latent_diffusion_unet_intermediary_activation_alpha = value

    @property
    def latent_diffusion_unet_epochs(self):
        return self._latent_diffusion_unet_epochs

    @latent_diffusion_unet_epochs.setter
    def latent_diffusion_unet_epochs(self, value):
        self._latent_diffusion_unet_epochs = value

    # VAE Properties
    @property
    def latent_diffusion_VAE_mean_distribution(self):
        return self._latent_diffusion_VAE_mean_distribution

    @latent_diffusion_VAE_mean_distribution.setter
    def latent_diffusion_VAE_mean_distribution(self, value):
        self._latent_diffusion_VAE_mean_distribution = value

    @property
    def latent_diffusion_VAE_stander_deviation(self):
        return self._latent_diffusion_VAE_stander_deviation

    @latent_diffusion_VAE_stander_deviation.setter
    def latent_diffusion_VAE_stander_deviation(self, value):
        self._latent_diffusion_VAE_stander_deviation = value

    @property
    def latent_diffusion_VAE_file_name_encoder(self):
        return self._latent_diffusion_VAE_file_name_encoder

    @latent_diffusion_VAE_file_name_encoder.setter
    def latent_diffusion_VAE_file_name_encoder(self, value):
        self._latent_diffusion_VAE_file_name_encoder = value

    @property
    def latent_diffusion_VAE_file_name_decoder(self):
        return self._latent_diffusion_VAE_file_name_decoder

    @latent_diffusion_VAE_file_name_decoder.setter
    def latent_diffusion_VAE_file_name_decoder(self, value):
        self._latent_diffusion_VAE_file_name_decoder = value

    @property
    def latent_diffusion_VAE_path_output_models(self):
        return self._latent_diffusion_VAE_path_output_models

    @latent_diffusion_VAE_path_output_models.setter
    def latent_diffusion_VAE_path_output_models(self, value):
        self._latent_diffusion_VAE_path_output_models = value

    # Gaussian Diffusion Properties
    @property
    def latent_diffusion_gaussian_beta_start(self):
        return self._latent_diffusion_gaussian_beta_start

    @latent_diffusion_gaussian_beta_start.setter
    def latent_diffusion_gaussian_beta_start(self, value):
        self._latent_diffusion_gaussian_beta_start = value

    @property
    def latent_diffusion_gaussian_beta_end(self):
        return self._latent_diffusion_gaussian_beta_end

    @latent_diffusion_gaussian_beta_end.setter
    def latent_diffusion_gaussian_beta_end(self, value):
        self._latent_diffusion_gaussian_beta_end = value

    @property
    def latent_diffusion_gaussian_time_steps(self):
        return self._latent_diffusion_gaussian_time_steps

    @latent_diffusion_gaussian_time_steps.setter
    def latent_diffusion_gaussian_time_steps(self, value):
        self._latent_diffusion_gaussian_time_steps = value

    @property
    def latent_diffusion_gaussian_clip_min(self):
        return self._latent_diffusion_gaussian_clip_min

    @latent_diffusion_gaussian_clip_min.setter
    def latent_diffusion_gaussian_clip_min(self, value):
        self._latent_diffusion_gaussian_clip_min = value

    @property
    def latent_diffusion_gaussian_clip_max(self):
        return self._latent_diffusion_gaussian_clip_max

    @latent_diffusion_gaussian_clip_max.setter
    def latent_diffusion_gaussian_clip_max(self, value):
        self._latent_diffusion_gaussian_clip_max = value

    # More VAE Properties
    @property
    def latent_diffusion_VAE_loss_function(self):
        return self._latent_diffusion_VAE_loss_function

    @latent_diffusion_VAE_loss_function.setter
    def latent_diffusion_VAE_loss_function(self, value):
        self._latent_diffusion_VAE_loss_function = value

    @property
    def latent_diffusion_VAE_encoder_filters(self):
        return self._latent_diffusion_VAE_encoder_filters

    @latent_diffusion_VAE_encoder_filters.setter
    def latent_diffusion_VAE_encoder_filters(self, value):
        self._latent_diffusion_VAE_encoder_filters = value

    @property
    def latent_diffusion_VAE_decoder_filters(self):
        return self._latent_diffusion_VAE_decoder_filters

    @latent_diffusion_VAE_decoder_filters.setter
    def latent_diffusion_VAE_decoder_filters(self, value):
        self._latent_diffusion_VAE_decoder_filters = value

    @property
    def latent_diffusion_VAE_last_layer_activation(self):
        return self._latent_diffusion_VAE_last_layer_activation

    @latent_diffusion_VAE_last_layer_activation.setter
    def latent_diffusion_VAE_last_layer_activation(self, value):
        self._latent_diffusion_VAE_last_layer_activation = value

    @property
    def latent_diffusion_VAE_latent_dimension(self):
        return self._latent_diffusion_VAE_latent_dimension

    @latent_diffusion_VAE_latent_dimension.setter
    def latent_diffusion_VAE_latent_dimension(self, value):
        self._latent_diffusion_VAE_latent_dimension = value

    @property
    def latent_diffusion_VAE_batch_size_create_embedding(self):
        return self._latent_diffusion_VAE_batch_size_create_embedding

    @latent_diffusion_VAE_batch_size_create_embedding.setter
    def latent_diffusion_VAE_batch_size_create_embedding(self, value):
        self._latent_diffusion_VAE_batch_size_create_embedding = value

    @property
    def latent_diffusion_VAE_batch_size_training(self):
        return self._latent_diffusion_VAE_batch_size_training

    @latent_diffusion_VAE_batch_size_training.setter
    def latent_diffusion_VAE_batch_size_training(self, value):
        self._latent_diffusion_VAE_batch_size_training = value

    @property
    def latent_diffusion_VAE_epochs(self):
        return self._latent_diffusion_VAE_epochs

    @latent_diffusion_VAE_epochs.setter
    def latent_diffusion_VAE_epochs(self, value):
        self._latent_diffusion_VAE_epochs = value

    @property
    def latent_diffusion_VAE_intermediary_activation_function(self):
        return self._latent_diffusion_VAE_intermediary_activation_function

    @latent_diffusion_VAE_intermediary_activation_function.setter
    def latent_diffusion_VAE_intermediary_activation_function(self, value):
        self._latent_diffusion_VAE_intermediary_activation_function = value

    @property
    def latent_diffusion_VAE_intermediary_activation_alpha(self):
        return self._latent_diffusion_VAE_intermediary_activation_alpha

    @latent_diffusion_VAE_intermediary_activation_alpha.setter
    def latent_diffusion_VAE_intermediary_activation_alpha(self, value):
        self._latent_diffusion_VAE_intermediary_activation_alpha = value

    @property
    def latent_diffusion_VAE_activation_output_encoder(self):
        return self._latent_diffusion_VAE_activation_output_encoder

    @latent_diffusion_VAE_activation_output_encoder.setter
    def latent_diffusion_VAE_activation_output_encoder(self, value):
        self._latent_diffusion_VAE_activation_output_encoder = value

    @property
    def latent_diffusion_margin(self):
        return self._latent_diffusion_margin

    @latent_diffusion_margin.setter
    def latent_diffusion_margin(self, value):
        self._latent_diffusion_margin = value

    @property
    def latent_diffusion_ema(self):
        return self._latent_diffusion_ema

    @latent_diffusion_ema.setter
    def latent_diffusion_ema(self, value):
        self._latent_diffusion_ema = value

    @property
    def latent_diffusion_time_steps(self):
        return self._latent_diffusion_time_steps

    @latent_diffusion_time_steps.setter
    def latent_diffusion_time_steps(self, value):
        self._latent_diffusion_time_steps = value

    @property
    def latent_diffusion_VAE_initializer_mean(self):
        return self._latent_diffusion_VAE_initializer_mean

    @latent_diffusion_VAE_initializer_mean.setter
    def latent_diffusion_VAE_initializer_mean(self, value):
        self._latent_diffusion_VAE_initializer_mean = value

    @property
    def latent_diffusion_VAE_initializer_deviation(self):
        return self._latent_diffusion_VAE_initializer_deviation

    @latent_diffusion_VAE_initializer_deviation.setter
    def latent_diffusion_VAE_initializer_deviation(self, value):
        self._latent_diffusion_VAE_initializer_deviation = value

    @property
    def latent_diffusion_VAE_dropout_decay_rate_encoder(self):
        return self._latent_diffusion_VAE_dropout_decay_rate_encoder

    @latent_diffusion_VAE_dropout_decay_rate_encoder.setter
    def latent_diffusion_VAE_dropout_decay_rate_encoder(self, value):
        self._latent_diffusion_VAE_dropout_decay_rate_encoder = value

    @property
    def latent_diffusion_VAE_dropout_decay_rate_decoder(self):
        return self._latent_diffusion_VAE_dropout_decay_rate_decoder

    @latent_diffusion_VAE_dropout_decay_rate_decoder.setter
    def latent_diffusion_VAE_dropout_decay_rate_decoder(self, value):
        self._latent_diffusion_VAE_dropout_decay_rate_decoder = value

    def __repr__(self):
        """String representation showing the active framework."""
        return f"LatentDiffusion(framework='{self._framework}')"

    def get_samples(self, number_samples_per_class):
        """
        Generate samples using the trained model and reshape them to original input shape.

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

        Raises:
            RuntimeError: If algorithm is not initialized
        """
        if self._latent_diffusion_algorithm is None:
            raise RuntimeError(
                "Cannot generate samples: algorithm is not initialized. "
                "Please train the model first using fit_model() or train()."
            )

        # Generate flattened samples from algorithm
        generated_data = self._latent_diffusion_algorithm.get_samples(number_samples_per_class)

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