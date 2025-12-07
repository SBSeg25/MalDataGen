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
    import numpy

    from typing import List
    from typing import Tuple
    from typing import Union

    from typing import Optional
    from typing import Callable

    from Engine.Models.WassersteinGP.VanillaGenerator import VanillaGenerator
    from Engine.Models.WassersteinGP.VanillaDiscriminator import VanillaDiscriminator

except ImportError as error:
    print(error)
    print()
    sys.exit(-1)


DEFAULT_WASSERSTEIN_GAN_LATENT_DIMENSION = 128
DEFAULT_WASSERSTEIN_GAN_ACTIVATION = "LeakyReLU"
DEFAULT_WASSERSTEIN_GAN_DROPOUT_DECAY_RATE_G = 0.2
DEFAULT_WASSERSTEIN_GAN_DROPOUT_DECAY_RATE_D = 0.4
DEFAULT_WASSERSTEIN_GAN_DENSE_LAYERS_SETTINGS_GENERATOR = [128]
DEFAULT_WASSERSTEIN_GAN_DENSE_LAYERS_SETTINGS_DISCRIMINATOR = [128]
DEFAULT_WASSERSTEIN_GAN_LAST_ACTIVATION_LAYER = "sigmoid"
DEFAULT_WASSERSTEIN_GAN_INITIALIZER_MEAN = 0.0
DEFAULT_WASSERSTEIN_GAN_INITIALIZER_DEVIATION = 0.125


class WassersteinModel(VanillaDiscriminator, VanillaGenerator):
    """
    WassersteinGP Generative Adversarial Network (WGAN) with Gradient Penalty.

    This class implements a WassersteinGP GAN, a type of Generative Adversarial
    Network designed to improve training stability and provide a more meaningful
    loss metric by approximating the Earth Mover's Distance (WassersteinGP-1 Distance)
    between real and generated data distributions.

    The model integrates both the **generator** (which synthesizes new data samples)
    and the **critic** (which scores the realism of samples) into a single interface,
    ensuring consistency across architectural configuration and training routines.

    Unlike traditional GANs, the discriminator (referred to as "critic" in WGANs)
    does not classify inputs as "real" or "fake." Instead, it assigns a scalar score,
    which is optimized to approximate the WassersteinGP distance between the true data
    distribution and the distribution induced by the generator.

    To enforce the Lipschitz continuity condition required by the WGAN framework,
    this model supports **Gradient Penalty (GP)**, which penalizes deviations from
    unit gradient norms during training, following the approach introduced in
    Gulrajani et al., 2017.

    References:
        - Arjovsky, M., Chintala, S., & Bottou, L. (2017).
          WassersteinGP GAN. arXiv preprint arXiv:1701.07875.
          Available at: https://arxiv.org/abs/1701.07875

        - Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017).
          Improved Training of WassersteinGP GANs. arXiv preprint arXiv:1704.00028.
          Available at: https://arxiv.org/abs/1704.00028

    Attributes:
        @generator (VanillaGenerator): Instance of the generator network, responsible for mapping latent vectors to data samples.
        @critic (VanillaDiscriminator): Instance of the critic network, responsible for evaluating the realism of data samples.
        @latent_dimension (int): Dimensionality of the latent space, i.e., the length of the random input vector.
        @output_shape (Tuple[int, ...]): Shape of the generated samples, e.g., (28, 28, 1) for grayscale images.
        @activation_function (Union[str, Callable]): Activation function applied in hidden layers of both networks.
        @initializer_mean (float): Mean of the normal distribution used for weight initialization.
        @initializer_deviation (float): Standard deviation of the normal distribution used for weight initialization.
        @dropout_decay_rate_g (float): Dropout rate applied to the generator's dense layers.
        @dropout_decay_rate_d (float): Dropout rate applied to the critic's dense layers.
        @last_layer_activation (Union[str, Callable]): Activation function applied to the generator's output layer.
        @dense_layer_sizes_g (List[int]): List of integers specifying the number and size of dense layers in the generator.
        @dense_layer_sizes_d (List[int]): List of integers specifying the number and size of dense layers in the critic.
        @dataset_type (type): Data type used for the training data, e.g., numpy.float32.
        @number_samples_per_class (Optional[int]): Number of samples per class, if applicable for class-conditional generation.
    """

    def __init__(self, latent_dimension: int = DEFAULT_WASSERSTEIN_GAN_LATENT_DIMENSION,
                 output_shape: Tuple[int, ...] = (128, ),
                 activation_function: str = DEFAULT_WASSERSTEIN_GAN_ACTIVATION,
                 initializer_mean: float = DEFAULT_WASSERSTEIN_GAN_INITIALIZER_MEAN,
                 initializer_deviation: float = DEFAULT_WASSERSTEIN_GAN_INITIALIZER_DEVIATION,
                 dropout_decay_rate_g: float = DEFAULT_WASSERSTEIN_GAN_DROPOUT_DECAY_RATE_G,
                 dropout_decay_rate_d: float = DEFAULT_WASSERSTEIN_GAN_DROPOUT_DECAY_RATE_D,
                 last_layer_activation: str = DEFAULT_WASSERSTEIN_GAN_LAST_ACTIVATION_LAYER,
                 dense_layer_sizes_g=None,
                 dense_layer_sizes_d=None,
                 dataset_type: type = numpy.float32,
                 number_samples_per_class: Optional[int] = None):
        """
        Initializes a WassersteinModel, combining the generator and critic components.

        This constructor sets up both the generator and the critic networks, applying
        the provided architectural and training configurations. The generator maps
        random noise vectors into the data space, while the critic evaluates how
        realistic those samples are relative to real data.

        Args:
            @latent_dimension (int): Dimensionality of the latent space (random input vector).
            @output_shape (Tuple[int, ...]): Shape of the generated samples (including channels for images).
            @activation_function (Union[str, Callable]): Activation function for the hidden layers in both networks.
            @initializer_mean (float): Mean value for the normal distribution used in weight initialization.
            @initializer_deviation (float): Standard deviation for the normal distribution used in weight initialization.
            @dropout_decay_rate_g (float): Dropout rate for the generator's dense layers (to prevent overfitting).
            @dropout_decay_rate_d (float): Dropout rate for the critic's dense layers (to prevent overfitting).
            @last_layer_activation (Union[str, Callable]): Activation function for the generator's final layer (e.g., 'sigmoid').
            @dense_layer_sizes_g (List[int]): Sizes of the dense layers in the generator.
            @dense_layer_sizes_d (List[int]): Sizes of the dense layers in the critic.
            @dataset_type (type, optional): Data type used for the dataset (default: numpy.float32).
            @number_samples_per_class (Optional[int]): Number of samples per class (for class-conditional setups, if applicable).

        Raises:
            ValueError:
                Raised if any provided argument is invalid (e.g., negative dimensions,
                empty layer lists, etc.).
        """
        if dense_layer_sizes_d is None:
            dense_layer_sizes_d = DEFAULT_WASSERSTEIN_GAN_DENSE_LAYERS_SETTINGS_DISCRIMINATOR

        if dense_layer_sizes_g is None:
            dense_layer_sizes_g = DEFAULT_WASSERSTEIN_GAN_DENSE_LAYERS_SETTINGS_GENERATOR

        if latent_dimension <= 0:
            raise ValueError("Latent dimension must be a positive integer.")

        if not all(size > 0 for size in dense_layer_sizes_g):
            raise ValueError("All generator dense layer sizes must be positive integers.")

        if not all(size > 0 for size in dense_layer_sizes_d):
            raise ValueError("All discriminator dense layer sizes must be positive integers.")

        if dropout_decay_rate_g < 0 or dropout_decay_rate_g > 1:
            raise ValueError("Generator dropout decay rate must be between 0 and 1.")

        if dropout_decay_rate_d < 0 or dropout_decay_rate_d > 1:
            raise ValueError("Discriminator dropout decay rate must be between 0 and 1.")

        # Initialize the discriminator
        VanillaDiscriminator.__init__(self,
                                      latent_dimension,
                                      output_shape,
                                      activation_function,
                                      initializer_mean,
                                      initializer_deviation,
                                      dropout_decay_rate_d,
                                      last_layer_activation,
                                      dense_layer_sizes_d,
                                      dataset_type,
                                      number_samples_per_class)

        # Initialize the generator
        VanillaGenerator.__init__(self,
                                  latent_dimension,
                                  output_shape,
                                  activation_function,
                                  initializer_mean,
                                  initializer_deviation,
                                  dropout_decay_rate_g,
                                  last_layer_activation,
                                  dense_layer_sizes_g,
                                  dataset_type,
                                  number_samples_per_class)

    def latent_dimension(self, latent_dimension: int) -> None:
        """Sets the latent dimension for both the discriminator and generator."""

        if latent_dimension <= 0:
            raise ValueError("Latent dimension must be a positive integer.")

        self._discriminator_latent_dimension = latent_dimension
        self._generator_latent_dimension = latent_dimension

    def output_shape(self, output_shape: Tuple[int, ...]) -> None:
        """Sets the output shape for both the discriminator and generator."""

        if not all(dim > 0 for dim in output_shape):
            raise ValueError("Output shape dimensions must be positive integers.")

        self._discriminator_output_shape = output_shape
        self._generator_output_shape = output_shape

    def activation_function(self, activation_function: Union[str, Callable]) -> None:
        """Sets the activation function for both the discriminator and generator."""

        if not callable(activation_function) and not isinstance(activation_function, str):
            raise ValueError("Activation function must be a callable or a string.")

        self._discriminator_activation_function = activation_function
        self._generator_activation_function = activation_function

    def last_layer_activation(self, last_layer_activation: Union[str, Callable]) -> None:
        """Sets the last layer activation for both the discriminator and generator."""

        if not callable(last_layer_activation) and not isinstance(last_layer_activation, str):
            raise ValueError("Last layer activation must be a callable or a string.")

        self._discriminator_last_layer_activation = last_layer_activation
        self._generator_last_layer_activation = last_layer_activation

    def dataset_type(self, dataset_type: type) -> None:
        """Sets the data type for the input dataset for both the discriminator and generator."""

        if not isinstance(dataset_type, type):
            raise ValueError("Datasets type must be a valid type object.")

        self._discriminator_dataset_type = dataset_type
        self._generator_dataset_type = dataset_type

    def initializer_mean(self, initializer_mean: float) -> None:
        """Sets the mean value for the weights initializer for both the discriminator and generator."""

        if not isinstance(initializer_mean, (int, float)):
            raise ValueError("Initializer mean must be a numerical value.")

        self._discriminator_initializer_mean = initializer_mean
        self._generator_initializer_mean = initializer_mean
