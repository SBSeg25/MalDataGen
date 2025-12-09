#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/04/25'
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


DEFAULT_WASSERSTEIN_GAN_GP_LATENT_DIMENSION = 128
DEFAULT_WASSERSTEIN_GAN_GP_TRAINING_ALGORITHM = "Adam"
DEFAULT_WASSERSTEIN_GAN_GP_ACTIVATION = "LeakyReLU"
DEFAULT_WASSERSTEIN_GAN_GP_DROPOUT_DECAY_RATE_G = 0.2
DEFAULT_WASSERSTEIN_GAN_GP_DROPOUT_DECAY_RATE_D = 0.4
DEFAULT_WASSERSTEIN_GAN_GP_BATCH_SIZE = 32
DEFAULT_WASSERSTEIN_GAN_GP_NUMBER_CLASSES = 2
DEFAULT_WASSERSTEIN_GAN_GP_NUMBER_EPOCHS = 20
DEFAULT_WASSERSTEIN_GAN_GP_DENSE_LAYERS_SETTINGS_GENERATOR = [128]
DEFAULT_WASSERSTEIN_GAN_GP_DENSE_LAYERS_SETTINGS_DISCRIMINATOR = [128]
DEFAULT_WASSERSTEIN_GAN_GP_LOSS = "binary_crossentropy"
DEFAULT_WASSERSTEIN_GAN_GP_MOMENTUM = 0.8
DEFAULT_WASSERSTEIN_GAN_GP_LAST_ACTIVATION_LAYER = "sigmoid"
DEFAULT_WASSERSTEIN_GAN_GP_INITIALIZER_MEAN = 0.0
DEFAULT_WASSERSTEIN_GAN_GP_INITIALIZER_DEVIATION = 0.125

DEFAULT_WASSERSTEIN_GAN_GP_OPTIMIZER_GENERATOR_LEARNING = 0.0001
DEFAULT_WASSERSTEIN_GAN_GP_OPTIMIZER_DISCRIMINATOR_LEARNING = 0.0001
DEFAULT_WASSERSTEIN_GAN_GP_OPTIMIZER_GENERATOR_BETA = 0.5
DEFAULT_WASSERSTEIN_GAN_GP_OPTIMIZER_DISCRIMINATOR_BETA = 0.5
DEFAULT_WASSERSTEIN_GAN_GP_ADAM_LEARNING_RATE = 0.0001
DEFAULT_WASSERSTEIN_GAN_GP_ADAM_BETA = 0.5
DEFAULT_WASSERSTEIN_GAN_GP_DISCRIMINATOR_STEPS = 3
DEFAULT_WASSERSTEIN_GAN_GP_SMOOTHING_RATE = 0.15
DEFAULT_WASSERSTEIN_GAN_GP_LATENT_MEAN_DISTRIBUTION = 0.0
DEFAULT_WASSERSTEIN_GAN_GP_LATENT_STANDER_DEVIATION = 0.125
DEFAULT_WASSERSTEIN_GAN_GP_GRADIENT_PENALTY = 10.0
DEFAULT_WASSERSTEIN_GAN_GP_FILE_NAME_DISCRIMINATOR = "discriminator_model"
DEFAULT_WASSERSTEIN_GAN_GP_FILE_NAME_GENERATOR = "generator_model"
DEFAULT_WASSERSTEIN_GAN_GP_PATH_OUTPUT_MODELS = "models_saved/"


#TODO
# DEFAULT_WASSERSTEIN_GAN_LOSS_GENERATOR = 'binary_crossentropy'
# DEFAULT_WASSERSTEIN_GAN_LOSS_DISCRIMINATOR = 'binary_crossentropy'
# DEFAULT_WASSERSTEIN_GAN_OPTIMIZER_GENERATOR = 'adam'
# DEFAULT_WASSERSTEIN_GAN_OPTIMIZER_DISCRIMINATOR = 'adam'


def add_argument_wasserstein_gan_gp(parser):

    parser.add_argument("--wasserstein_gp_latent_dimension", type=int,
                        default=DEFAULT_WASSERSTEIN_GAN_GP_LATENT_DIMENSION,
                        help="Latent space dimension for the WassersteinGP GAN.")

    parser.add_argument("--wasserstein_gp_training_algorithm", type=str,
                        default=DEFAULT_WASSERSTEIN_GAN_GP_TRAINING_ALGORITHM,
                        help="Training algorithm for the WassersteinGP GAN.")

    parser.add_argument("--wasserstein_gp_activation_function", type=str,
                        default=DEFAULT_WASSERSTEIN_GAN_GP_ACTIVATION,
                        help="Activation function for the WassersteinGP GAN.")

    parser.add_argument("--wasserstein_gp_dropout_decay_rate_g", type=float,
                        default=DEFAULT_WASSERSTEIN_GAN_GP_DROPOUT_DECAY_RATE_G,
                        help="Dropout decay rate for the generator.")

    parser.add_argument("--wasserstein_gp_dropout_decay_rate_d", type=float,
                        default=DEFAULT_WASSERSTEIN_GAN_GP_DROPOUT_DECAY_RATE_D,
                        help="Dropout decay rate for the discriminator.")

    parser.add_argument("--wasserstein_gp_dense_layer_sizes_generator", type=int, nargs='+',
                        default=DEFAULT_WASSERSTEIN_GAN_GP_DENSE_LAYERS_SETTINGS_GENERATOR,
                        help="Dense layer sizes for the generator.")

    parser.add_argument("--wasserstein_gp_dense_layer_sizes_discriminator", type=int, nargs='+',
                        default=DEFAULT_WASSERSTEIN_GAN_GP_DENSE_LAYERS_SETTINGS_DISCRIMINATOR,
                        help="Dense layer sizes for the discriminator.")

    parser.add_argument('--wasserstein_gp_batch_size', type=int,
                        default=DEFAULT_WASSERSTEIN_GAN_GP_BATCH_SIZE,
                        help='Batch size for the WassersteinGP GAN.')

    parser.add_argument('--wasserstein_gp_number_epochs', type=int,
                        default=DEFAULT_WASSERSTEIN_GAN_GP_NUMBER_EPOCHS,
                        help='Number epochs for the WassersteinGP GAN.')

    parser.add_argument('--wasserstein_gp_number_classes', type=int,
                        default=DEFAULT_WASSERSTEIN_GAN_GP_NUMBER_CLASSES,
                        help='Number of classes for the WassersteinGP GAN.')

    parser.add_argument("--wasserstein_gp_loss_function", type=str,
                        default=DEFAULT_WASSERSTEIN_GAN_GP_LOSS,
                        help="Loss function for the WassersteinGP GAN.",
                        choices=['binary_crossentropy', 'mean_squared_error'])

    parser.add_argument("--wasserstein_gp_momentum", type=float,
                        default=DEFAULT_WASSERSTEIN_GAN_GP_MOMENTUM,
                        help="Momentum for the training algorithm.")

    parser.add_argument("--wasserstein_gp_last_activation_layer", type=str,
                        default=DEFAULT_WASSERSTEIN_GAN_GP_LAST_ACTIVATION_LAYER,
                        help="Activation function for the last layer.")

    parser.add_argument("--wasserstein_gp_initializer_mean", type=float,
                        default=DEFAULT_WASSERSTEIN_GAN_GP_INITIALIZER_MEAN,
                        help="Mean value of the Gaussian initializer distribution.")

    parser.add_argument("--wasserstein_gp_initializer_deviation", type=float,
                        default=DEFAULT_WASSERSTEIN_GAN_GP_INITIALIZER_DEVIATION,
                        help="Standard deviation of the Gaussian initializer distribution.")

    parser.add_argument("--wasserstein_gp_optimizer_generator_learning_rate", type=float,
                        default=DEFAULT_WASSERSTEIN_GAN_GP_OPTIMIZER_GENERATOR_LEARNING,
                        help="Learning rate for the generator optimizer.")

    parser.add_argument("--wasserstein_gp_optimizer_discriminator_learning_rate", type=float,
                        default=DEFAULT_WASSERSTEIN_GAN_GP_OPTIMIZER_DISCRIMINATOR_LEARNING,
                        help="Learning rate for the discriminator optimizer.")

    parser.add_argument("--wasserstein_gp_optimizer_generator_beta", type=float,
                        default=DEFAULT_WASSERSTEIN_GAN_GP_OPTIMIZER_GENERATOR_BETA,
                        help="Beta value for the generator optimizer.")

    parser.add_argument("--wasserstein_gp_optimizer_discriminator_beta", type=float,
                        default=DEFAULT_WASSERSTEIN_GAN_GP_OPTIMIZER_DISCRIMINATOR_BETA,
                        help="Beta value for the discriminator optimizer.")

    parser.add_argument("--wasserstein_gp_discriminator_steps", type=int,
                        default=DEFAULT_WASSERSTEIN_GAN_GP_DISCRIMINATOR_STEPS,
                        help="Number of steps to update the discriminator per generator update.")

    parser.add_argument("--wasserstein_gp_smoothing_rate", type=float,
                        default=DEFAULT_WASSERSTEIN_GAN_GP_SMOOTHING_RATE,
                        help="Smoothing rate for the WassersteinGP GAN.")

    parser.add_argument("--wasserstein_gp_latent_mean_distribution", type=float,
                        default=DEFAULT_WASSERSTEIN_GAN_GP_LATENT_MEAN_DISTRIBUTION,
                        help="Mean of the random latent space distribution.")

    parser.add_argument("--wasserstein_gp_latent_stander_deviation", type=float,
                        default=DEFAULT_WASSERSTEIN_GAN_GP_LATENT_STANDER_DEVIATION,
                        help="Standard deviation of the random latent space distribution.")

    parser.add_argument("--wasserstein_gp_gradient_penalty", type=float,
                        default=DEFAULT_WASSERSTEIN_GAN_GP_GRADIENT_PENALTY,
                        help="Gradient penalty value for the WassersteinGP GAN.")

    parser.add_argument("--wasserstein_gp_file_name_discriminator", type=str,
                        default=DEFAULT_WASSERSTEIN_GAN_GP_FILE_NAME_DISCRIMINATOR,
                        help="File name to save the discriminator model.")

    parser.add_argument("--wasserstein_gp_file_name_generator", type=str,
                        default=DEFAULT_WASSERSTEIN_GAN_GP_FILE_NAME_GENERATOR,
                        help="File name to save the generator model.")

    parser.add_argument("--wasserstein_gp_path_output_models", type=str,
                        default=DEFAULT_WASSERSTEIN_GAN_GP_PATH_OUTPUT_MODELS,
                        help="Path to save the models.")

    return parser
