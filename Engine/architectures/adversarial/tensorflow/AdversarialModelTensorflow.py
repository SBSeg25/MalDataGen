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

try:
    import sys
    import numpy

    from Engine.architectures.adversarial.tensorflow.VanillaGeneratorTensorflow import VanillaGenerator
    from Engine.architectures.adversarial.tensorflow.VanillaDiscriminatorTensorflow import VanillaDiscriminator

except ImportError as error:
    print(error)
    sys.exit(-1)

DEFAULT_ADVERSARIAL_LATENT_DIMENSION = 128
DEFAULT_ADVERSARIAL_INTERMEDIARY_ACTIVATION = "LeakyReLU"
DEFAULT_ADVERSARIAL_LAST_ACTIVATION_LAYER = "Sigmoid"
DEFAULT_ADVERSARIAL_DROPOUT_DECAY_RATE_G = 0.2
DEFAULT_ADVERSARIAL_DROPOUT_DECAY_RATE_D = 0.4
DEFAULT_ADVERSARIAL_INITIALIZER_MEAN = 0.0
DEFAULT_ADVERSARIAL_INITIALIZER_DEVIATION = 0.5
DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_G = [128]
DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_D = [128]


class AdversarialModelTensorflow(VanillaGenerator, VanillaDiscriminator):
    """
    AdversarialModel

    Combines a generator and a discriminator in a unified adversarial model.
    """

    def __init__(self,
                 latent_dimension: int = DEFAULT_ADVERSARIAL_LATENT_DIMENSION,
                 output_shape: int = 128,
                 activation_function: str = DEFAULT_ADVERSARIAL_INTERMEDIARY_ACTIVATION,
                 initializer_mean: float = DEFAULT_ADVERSARIAL_INITIALIZER_MEAN,
                 initializer_deviation: float = DEFAULT_ADVERSARIAL_INITIALIZER_DEVIATION,
                 dropout_decay_rate_g: float = DEFAULT_ADVERSARIAL_DROPOUT_DECAY_RATE_G,
                 dropout_decay_rate_d: float = DEFAULT_ADVERSARIAL_DROPOUT_DECAY_RATE_D,
                 last_layer_activation: str = DEFAULT_ADVERSARIAL_LAST_ACTIVATION_LAYER,
                 dense_layer_sizes_g=None,
                 dense_layer_sizes_d=None,
                 dataset_type: type = numpy.float32,
                 number_samples_per_class: dict | None = None,
                 optimizer="convolutional"):
        """
        Initializes the AdversarialModel.

        CORREÇÃO: Agora valida e armazena number_samples_per_class corretamente.
        """

        if dense_layer_sizes_d is None:
            dense_layer_sizes_d = DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_D

        if dense_layer_sizes_g is None:
            dense_layer_sizes_g = DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_G

        # CORREÇÃO: Armazenar number_samples_per_class antes de inicializar as classes pai
        self._stored_number_samples_per_class = number_samples_per_class

        # Initialize generator and discriminator using parent class constructors
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
                                  number_samples_per_class,
                                  optimizer=optimizer)

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
                                      number_samples_per_class,
                                      optimizer=optimizer)

    def set_model(self, model):
        """
        Set a custom discriminator model.

        CORREÇÃO: Mantém _generator_number_samples_per_class para geração posterior.
        """
        self._discriminator_model_dense = model
        # IMPORTANTE: Manter o _generator_number_samples_per_class configurado
        # para que get_generator() funcione corretamente

    def set_number_samples_per_class(self, number_samples_per_class: dict):
        """
        CORREÇÃO: Novo método para atualizar number_samples_per_class após criação.

        Args:
            number_samples_per_class (dict): Dictionary with class distribution info.
                Must include 'number_classes' key.
        """
        if not number_samples_per_class or "number_classes" not in number_samples_per_class:
            raise ValueError(
                "`number_samples_per_class` must include a 'number_classes' key."
            )

        self._generator_number_samples_per_class = number_samples_per_class
        self._discriminator_number_samples_per_class = number_samples_per_class
        self._stored_number_samples_per_class = number_samples_per_class

    def get_dense_generator_model(self):
        """Returns the generator's dense model."""
        return self._generator_model_dense

    def get_dense_discriminator_model(self):
        """Returns the discriminator's dense model."""
        return self._discriminator_model_dense

    def set_latent_dimension(self, latent_dimension):
        """Sets the latent dimension for both networks."""
        self._generator_latent_dimension = latent_dimension
        self._discriminator_latent_dimension = latent_dimension

    def set_output_shape(self, output_shape):
        """Sets the output shape for both networks."""
        self._generator_output_shape = output_shape
        self._discriminator_output_shape = output_shape

    def set_activation_function(self, activation_function):
        """Sets the activation function for both networks."""
        self._generator_activation_function = activation_function
        self._discriminator_activation_function = activation_function

    def set_last_layer_activation(self, last_layer_activation):
        """Sets the last layer activation function for both networks."""
        self._generator_last_layer_activation = last_layer_activation
        self._discriminator_last_layer_activation = last_layer_activation

    def set_dataset_type(self, dataset_type):
        """Sets the dataset type for both networks."""
        self._generator_dataset_type = dataset_type
        self._discriminator_dataset_type = dataset_type

    def set_initializer_mean(self, initializer_mean):
        """Sets the initializer mean for both networks."""
        self._generator_initializer_mean = initializer_mean
        self._discriminator_initializer_mean = initializer_mean

    def set_initializer_deviation(self, initializer_deviation):
        """Sets the initializer deviation for both networks."""
        self._generator_initializer_deviation = initializer_deviation
        self._discriminator_initializer_deviation = initializer_deviation