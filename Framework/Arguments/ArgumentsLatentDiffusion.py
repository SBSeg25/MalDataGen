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


DEFAULT_LATENT_DIFFUSION_UNET_LAST_LAYER_ACTIVATION = 'linear'
DEFAULT_LATENT_DIFFUSION_LATENT_DIMENSION = 64
DEFAULT_LATENT_DIFFUSION_UNET_NUMBER_EMBEDDING_CHANNELS = 1
DEFAULT_LATENT_DIFFUSION_UNET_CHANNELS_PER_LEVEL = [1, 2, 4]
DEFAULT_LATENT_DIFFUSION_UNET_BATCH_SIZE = 128
DEFAULT_LATENT_DIFFUSION_UNET_ATTENTION_MODE = [False, True, True]
# DEFAULT_DIFFUSION_UNET_OPTIMIZER = tensorflow.keras.optimizers.Adam(learning_rate=0.0001)
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

# DEFAULT_DIFFUSION_AUTOENCODER_OPTIMIZER = tensorflow.keras.optimizers.Adam(learning_rate=0.0001)
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
 


def add_argument_latent_diffusion(parser):

    parser.add_argument('--latent_diffusion_unet_last_layer_activation', type=str,
                        default=DEFAULT_LATENT_DIFFUSION_UNET_LAST_LAYER_ACTIVATION,
                        help='Activation for the last layer of U-Net.')

    parser.add_argument('--latent_diffusion_latent_dimension', type=int,
                        default=DEFAULT_LATENT_DIFFUSION_LATENT_DIMENSION, help='Dimension of the latent space.')

    parser.add_argument('--latent_diffusion_unet_num_embedding_channels', type=int,
                        default=DEFAULT_LATENT_DIFFUSION_UNET_NUMBER_EMBEDDING_CHANNELS,
                        help='Number of embedding channels for U-Net.')

    parser.add_argument('--latent_diffusion_unet_channels_per_level', nargs='+', type=int,
                        default=DEFAULT_LATENT_DIFFUSION_UNET_CHANNELS_PER_LEVEL,
                        help='List of channels per level in U-Net.')

    parser.add_argument('--latent_diffusion_unet_batch_size', type=int,
                        default=DEFAULT_LATENT_DIFFUSION_UNET_BATCH_SIZE,
                        help='Batch size for U-Net training.')

    parser.add_argument('--latent_diffusion_unet_attention_mode', nargs='+', type=bool,
                        default=DEFAULT_LATENT_DIFFUSION_UNET_ATTENTION_MODE,
                        help='Attention mode for U-Net.')

    parser.add_argument('--latent_diffusion_unet_num_residual_blocks', type=int,
                        default=DEFAULT_LATENT_DIFFUSION_UNET_NUMBER_RESIDUAL_BLOCKS,
                        help='Number of residual blocks in U-Net.')

    parser.add_argument('--latent_diffusion_unet_group_normalization', type=int,
                        default=DEFAULT_LATENT_DIFFUSION_UNET_GROUP_NORMALIZATION,
                        help='Group normalization value for U-Net.')

    parser.add_argument('--latent_diffusion_unet_intermediary_activation', type=str,
                        default=DEFAULT_LATENT_DIFFUSION_UNET_INTERMEDIARY_ACTIVATION,
                        help='Intermediary activation for U-Net.')

    parser.add_argument('--latent_diffusion_unet_intermediary_activation_alpha', type=float,
                        default=DEFAULT_LATENT_DIFFUSION_UNET_INTERMEDIARY_ACTIVATION_ALPHA,
                        help='Alpha value for intermediary activation function in U-Net.')

    parser.add_argument('--latent_diffusion_unet_epochs', type=int,
                        default=DEFAULT_LATENT_DIFFUSION_UNET_NUMBER_EPOCHS,
                        help='Number of epochs for U-Net training.')

    parser.add_argument('--latent_diffusion_gaussian_beta_start', type=float,
                        default=DEFAULT_LATENT_DIFFUSION_GAUSSIAN_BETA_START,
                        help='Starting value of beta for Gaussian diffusion.')

    parser.add_argument('--latent_diffusion_gaussian_beta_end', type=float,
                        default=DEFAULT_LATENT_DIFFUSION_GAUSSIAN_BETA_END,
                        help='Ending value of beta for Gaussian diffusion.')

    parser.add_argument('--latent_diffusion_gaussian_time_steps', type=int,
                        default=DEFAULT_LATENT_DIFFUSION_GAUSSIAN_TIME_STEPS,
                        help='Number of time steps for Gaussian diffusion.')

    parser.add_argument('--latent_diffusion_gaussian_clip_min', type=float,
                        default=DEFAULT_LATENT_DIFFUSION_GAUSSIAN_CLIP_MIN,
                        help='Minimum clipping value for Gaussian noise.')

    parser.add_argument('--latent_diffusion_gaussian_clip_max', type=float,
                        default=DEFAULT_LATENT_DIFFUSION_GAUSSIAN_CLIP_MAX,
                        help='Maximum clipping value for Gaussian noise.')

    parser.add_argument('--latent_diffusion_autoencoder_loss', type=str,
                        default=DEFAULT_LATENT_DIFFUSION_AUTOENCODER_LOSS,
                        help='Loss function for Autoencoder.')

    parser.add_argument('--latent_diffusion_autoencoder_encoder_filters', nargs='+', type=int,
                        default=DEFAULT_LATENT_DIFFUSION_AUTOENCODER_ENCODER_FILTERS,
                        help='List of filters for Autoencoder encoder.')

    parser.add_argument('--latent_diffusion_autoencoder_decoder_filters', nargs='+', type=int,
                        default=DEFAULT_LATENT_DIFFUSION_AUTOENCODER_DECODER_FILTERS,
                        help='List of filters for Autoencoder decoder.')

    parser.add_argument('--latent_diffusion_autoencoder_last_layer_activation', type=str,
                        default=DEFAULT_LATENT_DIFFUSION_AUTOENCODER_LAST_LAYER_ACTIVATION,
                        help='Activation function for the last layer of Autoencoder.')

    parser.add_argument('--latent_diffusion_autoencoder_latent_dimension', type=int,
                        default=DEFAULT_LATENT_DIFFUSION_AUTOENCODER_LATENT_DIMENSION,
                        help='Dimension of the latent in Autoencoder.')

    parser.add_argument('--latent_diffusion_autoencoder_batch_size_create_embedding', type=int,
                        default=DEFAULT_LATENT_DIFFUSION_AUTOENCODER_BATCH_SIZE_CREATE_EMBEDDING,
                        help='Batch size for creating embeddings in Autoencoder.')

    parser.add_argument('--latent_diffusion_autoencoder_batch_size_training', type=int,
                        default=DEFAULT_LATENT_DIFFUSION_AUTOENCODER_BATCH_SIZE_TRAINING,
                        help='Batch size for training Autoencoder.')

    parser.add_argument('--latent_diffusion_autoencoder_epochs', type=int,
                        default=DEFAULT_LATENT_DIFFUSION_AUTOENCODER_EPOCHS,
                        help='Number of epochs for Autoencoder training.')

    parser.add_argument('--latent_diffusion_autoencoder_intermediary_activation_function', type=str,
                        default=DEFAULT_LATENT_DIFFUSION_AUTOENCODER_INTERMEDIARY_ACTIVATION,
                        help='Intermediary activation function for Autoencoder.')

    parser.add_argument('--latent_diffusion_autoencoder_intermediary_activation_alpha', type=float,
                        default=DEFAULT_LATENT_DIFFUSION_AUTOENCODER_INTERMEDIARY_ACTIVATION_ALPHA,
                        help='Alpha value for intermediary activation function in Autoencoder.')

    parser.add_argument('--latent_diffusion_autoencoder_activation_output_encoder', type=str,
                        default=DEFAULT_LATENT_DIFFUSION_AUTOENCODER_ACTIVATION_OUTPUT_ENCODER,
                        help='Activation function for the output of the encoder in Autoencoder.')

    parser.add_argument('--latent_diffusion_margin', type=float,
                        default=DEFAULT_LATENT_DIFFUSION_MARGIN, help='Margin for diffusion process.')

    parser.add_argument('--latent_diffusion_ema', type=float,
                        default=DEFAULT_LATENT_DIFFUSION_EMA, help='Exponential moving average for diffusion.')

    parser.add_argument('--latent_diffusion_time_steps', type=int,
                        default=DEFAULT_LATENT_DIFFUSION_TIME_STEPS, help='Number of time steps for diffusion.')

    parser.add_argument('--latent_diffusion_autoencoder_initializer_mean', type=float,
                        default=DEFAULT_LATENT_DIFFUSION_AUTOENCODER_INITIALIZER_MEAN)

    parser.add_argument('--latent_diffusion_autoencoder_initializer_deviation', type=float,
                        default=DEFAULT_LATENT_DIFFUSION_AUTOENCODER_INITIALIZER_DEVIATION)

    parser.add_argument('--latent_diffusion_autoencoder_dropout_decay_rate_encoder', type=float,
                        default=DEFAULT_LATENT_DIFFUSION_AUTOENCODER_DROPOUT_DECAY_RATE_ENCODER)

    parser.add_argument('--latent_diffusion_autoencoder_dropout_decay_rate_decoder', type=float,
                        default=DEFAULT_LATENT_DIFFUSION_AUTOENCODER_DROPOUT_DECAY_RATE_DECODER)

    parser.add_argument('--latent_diffusion_autoencoder_file_name_encoder', type=str,
                        default=DEFAULT_LATENT_DIFFUSION_AUTOENCODER_FILE_NAME_ENCODER)

    parser.add_argument('--latent_diffusion_autoencoder_file_name_decoder', type=str,
                        default=DEFAULT_LATENT_DIFFUSION_AUTOENCODER_FILE_NAME_DECODER)

    parser.add_argument('--latent_diffusion_autoencoder_path_output_models', type=str,
                        default=DEFAULT_LATENT_DIFFUSION_AUTOENCODER_PATH_OUTPUT_MODELS)

    parser.add_argument('--latent_diffusion_autoencoder_mean_distribution', type=float,
                        default=DEFAULT_LATENT_DIFFUSION_AUTOENCODER_MEAN_DISTRIBUTION)

    parser.add_argument('--latent_diffusion_autoencoder_stander_deviation', type=float,
                        default=DEFAULT_LATENT_DIFFUSION_AUTOENCODER_STANDER_DEVIATION)

    return parser
