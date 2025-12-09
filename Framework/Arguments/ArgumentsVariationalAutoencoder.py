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


DEFAULT_VARIATIONAL_AUTOENCODER_LATENT_DIMENSION = 32
DEFAULT_VARIATIONAL_AUTOENCODER_TRAINING_ALGORITHM = "Adam"
DEFAULT_VARIATIONAL_AUTOENCODER_ACTIVATION_INTERMEDIARY = "swish"
DEFAULT_VARIATIONAL_AUTOENCODER_DROPOUT_DECAY_RATE_ENCODER = 0.25
DEFAULT_VARIATIONAL_AUTOENCODER_DROPOUT_DECAY_RATE_DECODER = 0.25
DEFAULT_VARIATIONAL_AUTOENCODER_BATCH_SIZE = 64
DEFAULT_VARIATIONAL_AUTOENCODER_NUMBER_EPOCHS = 300
DEFAULT_VARIATIONAL_AUTOENCODER_NUMBER_CLASSES = 2
DEFAULT_VARIATIONAL_AUTOENCODER_DENSE_LAYERS_SETTINGS_ENCODER = [320, 160]
DEFAULT_VARIATIONAL_AUTOENCODER_DENSE_LAYERS_SETTINGS_DECODER = [160, 320]
DEFAULT_VARIATIONAL_AUTOENCODER_LOSS = "binary_crossentropy"
DEFAULT_VARIATIONAL_AUTOENCODER_MOMENTUM = 0.8
DEFAULT_VARIATIONAL_AUTOENCODER_LAST_ACTIVATION_LAYER = "sigmoid"
DEFAULT_VARIATIONAL_AUTOENCODER_INITIALIZER_MEAN = 0
DEFAULT_VARIATIONAL_AUTOENCODER_INITIALIZER_DEVIATION = 0.125
DEFAULT_VARIATIONAL_AUTOENCODER_LOSS_FUNCTION = 'binary_crossentropy'
DEFAULT_VARIATIONAL_AUTOENCODER_FILE_NAME_ENCODER = "encoder_model"
DEFAULT_VARIATIONAL_AUTOENCODER_FILE_NAME_DECODER = "decoder_model"
DEFAULT_VARIATIONAL_AUTOENCODER_PATH_OUTPUT_MODELS = "models_saved/"
DEFAULT_VARIATIONAL_AUTOENCODER_MEAN_DISTRIBUTION = 0.5
DEFAULT_VARIATIONAL_AUTOENCODER_STANDER_DEVIATION = 0.125


def add_argument_variation_autoencoder(parser):
    parser.add_argument("--variational_autoencoder_latent_dimension", type=int,
                        default=DEFAULT_VARIATIONAL_AUTOENCODER_LATENT_DIMENSION,
                        help="Latent space dimension for the Variational Autoencoder")

    parser.add_argument("--variational_autoencoder_training_algorithm", type=str,
                        default=DEFAULT_VARIATIONAL_AUTOENCODER_TRAINING_ALGORITHM,
                        help="Training algorithm for the Variational Autoencoder.")

    parser.add_argument("--variational_autoencoder_activation_function",
                        type=str, default=DEFAULT_VARIATIONAL_AUTOENCODER_ACTIVATION_INTERMEDIARY,
                        help="Intermediate activation function of the Variational Autoencoder.")

    parser.add_argument("--variational_autoencoder_dropout_decay_rate_encoder", type=float,
                        default=DEFAULT_VARIATIONAL_AUTOENCODER_DROPOUT_DECAY_RATE_ENCODER,
                        help="Dropout decay rate for the encoder of the Variational Autoencoder")

    parser.add_argument("--variational_autoencoder_dropout_decay_rate_decoder", type=float,
                        default=DEFAULT_VARIATIONAL_AUTOENCODER_DROPOUT_DECAY_RATE_DECODER,
                        help="Dropout decay rate for the discriminator of the Variational Autoencoder")

    parser.add_argument("--variational_autoencoder_dense_layer_sizes_encoder", type=int, nargs='+',
                        default=DEFAULT_VARIATIONAL_AUTOENCODER_DENSE_LAYERS_SETTINGS_ENCODER,
                        help="Sizes of dense layers in the encoder")

    parser.add_argument("--variational_autoencoder_dense_layer_sizes_decoder", type=int, nargs='+',
                        default=DEFAULT_VARIATIONAL_AUTOENCODER_DENSE_LAYERS_SETTINGS_DECODER,
                        help="Sizes of dense layers in the decoder")

    parser.add_argument('--variational_autoencoder_number_epochs', type=int,
                        default=DEFAULT_VARIATIONAL_AUTOENCODER_NUMBER_EPOCHS,
                        help='Number of classes for the Autoencoder.')


    parser.add_argument('--variational_autoencoder_batch_size', type=int,
                        default=DEFAULT_VARIATIONAL_AUTOENCODER_BATCH_SIZE,
                        help='Batch size for the Variational Autoencoder.')

    parser.add_argument('--variational_autoencoder_number_classes', type=int,
                        default=DEFAULT_VARIATIONAL_AUTOENCODER_NUMBER_CLASSES,
                        help='Number of classes for the Variational Autoencoder.')

    parser.add_argument("--variational_autoencoder_loss_function", type=str,
                        default=DEFAULT_VARIATIONAL_AUTOENCODER_LOSS,
                        help="Loss function for the Variational Autoencoder.")

    parser.add_argument("--variational_autoencoder_momentum", type=float,
                        default=DEFAULT_VARIATIONAL_AUTOENCODER_MOMENTUM,
                        help="Momentum for the training algorithm.")

    parser.add_argument("--variational_autoencoder_last_activation_layer", type=str,
                        default=DEFAULT_VARIATIONAL_AUTOENCODER_LAST_ACTIVATION_LAYER,
                        help="Activation function of the last layer.")

    parser.add_argument("--variational_autoencoder_initializer_mean", type=float,
                        default=DEFAULT_VARIATIONAL_AUTOENCODER_INITIALIZER_MEAN,
                        help='Mean value of the Gaussian initializer distribution.')

    parser.add_argument("--variational_autoencoder_initializer_deviation", type=float,
                        default=DEFAULT_VARIATIONAL_AUTOENCODER_INITIALIZER_DEVIATION,
                        help='Standard deviation of the Gaussian initializer distribution.')

    parser.add_argument("--variational_autoencoder_mean_distribution", type=float,
                        default=DEFAULT_VARIATIONAL_AUTOENCODER_MEAN_DISTRIBUTION,
                        help='Mean of the random noise distribution input')

    parser.add_argument("--variational_autoencoder_stander_deviation", type=float,
                        default=DEFAULT_VARIATIONAL_AUTOENCODER_STANDER_DEVIATION,
                        help='Standard deviation of the random noise input')

    parser.add_argument("--variational_autoencoder_file_name_encoder", type=str,
                        default=DEFAULT_VARIATIONAL_AUTOENCODER_FILE_NAME_ENCODER,
                        help='File name to save the encoder model.')

    parser.add_argument("--variational_autoencoder_file_name_decoder", type=str,
                        default=DEFAULT_VARIATIONAL_AUTOENCODER_FILE_NAME_DECODER,
                        help='File name to save the decoder model.')

    parser.add_argument("--variational_autoencoder_path_output_models", type=str,
                        default=DEFAULT_VARIATIONAL_AUTOENCODER_PATH_OUTPUT_MODELS,
                        help='Path to save the models.')

    return parser