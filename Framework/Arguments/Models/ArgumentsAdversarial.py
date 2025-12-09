#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Kayuã Oleques']

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


DEFAULT_ADVERSARIAL_NUMBER_EPOCHS = 20
DEFAULT_ADVERSARIAL_LATENT_DIMENSION = 128
DEFAULT_ADVERSARIAL_TRAINING_ALGORITHM = "Adam"
DEFAULT_ADVERSARIAL_INTERMEDIARY_ACTIVATION = "LeakyReLU"
DEFAULT_ADVERSARIAL_LAST_ACTIVATION_LAYER = "Sigmoid"
DEFAULT_ADVERSARIAL_DROPOUT_DECAY_RATE_G = 0.2
DEFAULT_ADVERSARIAL_DROPOUT_DECAY_RATE_D = 0.4
DEFAULT_ADVERSARIAL_INITIALIZER_MEAN = 0.0
DEFAULT_ADVERSARIAL_INITIALIZER_DEVIATION = 0.5
DEFAULT_ADVERSARIAL_BATCH_SIZE = 32
DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_G = [128]
DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_D = [128]
DEFAULT_ADVERSARIAL_RANDOM_LATENT_STANDER_DEVIATION = 1.0

DEFAULT_ADVERSARIAL_LOSS_GENERATOR = 'binary_crossentropy'
DEFAULT_ADVERSARIAL_LOSS_DISCRIMINATOR = 'binary_crossentropy'
DEFAULT_ADVERSARIAL_SMOOTHING_RATE = 0.15
DEFAULT_ADVERSARIAL_LATENT_MEAN_DISTRIBUTION = 0.0
DEFAULT_ADVERSARIAL_LATENT_STANDER_DEVIATION = 1.0
DEFAULT_ADVERSARIAL_FILE_NAME_DISCRIMINATOR = "discriminator_model"
DEFAULT_ADVERSARIAL_FILE_NAME_GENERATOR = "generator_model"
DEFAULT_ADVERSARIAL_PATH_OUTPUT_MODELS = "models_saved/"


def add_argument_adversarial(parser):

    parser.add_argument('--adversarial_number_epochs', type=int,
                        default=DEFAULT_ADVERSARIAL_NUMBER_EPOCHS,
                        help='Number of epochs (training iterations).')

    parser.add_argument('--adversarial_batch_size', type=int,
                        default=DEFAULT_ADVERSARIAL_BATCH_SIZE,
                        help='Number of epochs (training iterations).')

    parser.add_argument('--adversarial_initializer_mean', type=float,
                        default=DEFAULT_ADVERSARIAL_INITIALIZER_MEAN,
                        help='Mean value of the Gaussian initializer distribution.')

    parser.add_argument('--adversarial_initializer_deviation', type=float,
                        default=DEFAULT_ADVERSARIAL_INITIALIZER_DEVIATION,
                        help='Standard deviation of the Gaussian initializer distribution.')

    parser.add_argument("--adversarial_latent_dimension", type=int,
                        default=DEFAULT_ADVERSARIAL_LATENT_DIMENSION,
                        help="Latent space dimension for cGAN training.")

    parser.add_argument("--adversarial_training_algorithm", type=str,
                        default=DEFAULT_ADVERSARIAL_TRAINING_ALGORITHM,
                        help="Training algorithm for cGAN.")

    parser.add_argument("--adversarial_activation_function",
                        type=str, default=DEFAULT_ADVERSARIAL_INTERMEDIARY_ACTIVATION,
                        help="Activation function for the cGAN.")

    parser.add_argument("--adversarial_dropout_decay_rate_g",
                        type=float, default=DEFAULT_ADVERSARIAL_DROPOUT_DECAY_RATE_G,
                        help="Dropout decay rate for the cGAN generator.")

    parser.add_argument("--adversarial_dropout_decay_rate_d",
                        type=float, default=DEFAULT_ADVERSARIAL_DROPOUT_DECAY_RATE_D,
                        help="Dropout decay rate for the cGAN discriminator.")

    parser.add_argument("--adversarial_dense_layer_sizes_g", type=int, nargs='+',
                        default=DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_G,
                        help="Sizes of dense layers in the generator.")

    parser.add_argument("--adversarial_dense_layer_sizes_d", type=int, nargs='+',
                        default=DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_D,
                        help="Sizes of dense layers in the discriminator.")

    parser.add_argument("--adversarial_latent_mean_distribution", type=float,
                        help='Mean of the random noise input distribution.',
                        default=DEFAULT_ADVERSARIAL_LATENT_MEAN_DISTRIBUTION)
    
    parser.add_argument('--adversarial_latent_stander_deviation', type=float,
                        default=DEFAULT_ADVERSARIAL_LATENT_STANDER_DEVIATION,
                        help='Standard deviation of the latent space distribution.')

    parser.add_argument('--adversarial_loss_generator', type=str,
                        default=DEFAULT_ADVERSARIAL_LOSS_GENERATOR,
                        help='Loss function for the generator.')

    parser.add_argument('--adversarial_loss_discriminator', type=str,
                        default=DEFAULT_ADVERSARIAL_LOSS_DISCRIMINATOR,
                        help='Loss function for the discriminator.')

    parser.add_argument('--adversarial_smoothing_rate', type=float,
                        default=DEFAULT_ADVERSARIAL_SMOOTHING_RATE,
                        help='Label smoothing rate for the adversarial training.')

    parser.add_argument('--adversarial_file_name_discriminator', type=str,
                        default=DEFAULT_ADVERSARIAL_FILE_NAME_DISCRIMINATOR,
                        help='File name to save the trained discriminator model.')

    parser.add_argument('--adversarial_file_name_generator', type=str,
                        default=DEFAULT_ADVERSARIAL_FILE_NAME_GENERATOR,
                        help='File name to save the trained generator model.')

    parser.add_argument('--adversarial_path_output_models', type=str,
                        default=DEFAULT_ADVERSARIAL_PATH_OUTPUT_MODELS,
                        help='Path to save the trained models.')

    parser.add_argument('--adversarial_last_layer_activation', type=str,
                            default=DEFAULT_ADVERSARIAL_LAST_ACTIVATION_LAYER,
                            help='adversarial last layer activation.')



    return parser
