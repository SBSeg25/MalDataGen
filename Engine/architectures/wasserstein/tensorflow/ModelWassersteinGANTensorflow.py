#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/21'
__credits__ = ['Synthetic Ocean AI']

try:
    import sys
    import numpy

    from typing import List
    from typing import Tuple
    from typing import Union
    from typing import Optional
    from typing import Callable
    from Engine.architectures.wasserstein.tensorflow.VanillaDiscriminatorTensorflow import \
        VanillaDiscriminatorTensorflow
    from Engine.architectures.wasserstein.tensorflow.VanillaGeneratorTensorflow import VanillaGeneratorTensorflow
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


class WassersteinModelTensorflow(VanillaDiscriminatorTensorflow, VanillaGeneratorTensorflow):
    """
    Wasserstein Generative Adversarial Network (WGAN) with Gradient Penalty.

    CORREÇÃO: Ordem dos parâmetros agora corresponde aos construtores das classes base.
    """

    def __init__(self,
                 latent_dimension: int = DEFAULT_WASSERSTEIN_GAN_LATENT_DIMENSION,
                 output_shape: Tuple[int, ...] = (128,),
                 activation_function: str = DEFAULT_WASSERSTEIN_GAN_ACTIVATION,
                 initializer_mean: float = DEFAULT_WASSERSTEIN_GAN_INITIALIZER_MEAN,
                 initializer_deviation: float = DEFAULT_WASSERSTEIN_GAN_INITIALIZER_DEVIATION,
                 dropout_decay_rate_g: float = DEFAULT_WASSERSTEIN_GAN_DROPOUT_DECAY_RATE_G,
                 dropout_decay_rate_d: float = DEFAULT_WASSERSTEIN_GAN_DROPOUT_DECAY_RATE_D,
                 last_layer_activation: str = DEFAULT_WASSERSTEIN_GAN_LAST_ACTIVATION_LAYER,
                 dense_layer_sizes_g=None,
                 dense_layer_sizes_d=None,
                 dataset_type: type = numpy.float32,
                 number_samples_per_class: Optional[dict] = None):
        """
        Inicializa WassersteinModel com ordem correta de parâmetros.

        Args:
            latent_dimension: Dimensão do espaço latente
            output_shape: Forma da saída gerada
            activation_function: Função de ativação
            initializer_mean: Média do inicializador
            initializer_deviation: Desvio padrão do inicializador
            dropout_decay_rate_g: Taxa de dropout do gerador
            dropout_decay_rate_d: Taxa de dropout do discriminador
            last_layer_activation: Ativação da última camada
            dense_layer_sizes_g: Tamanhos das camadas do gerador
            dense_layer_sizes_d: Tamanhos das camadas do discriminador
            dataset_type: Tipo de dado do dataset
            number_samples_per_class: Dicionário com info de classes
        """
        if dense_layer_sizes_d is None:
            dense_layer_sizes_d = DEFAULT_WASSERSTEIN_GAN_DENSE_LAYERS_SETTINGS_DISCRIMINATOR

        if dense_layer_sizes_g is None:
            dense_layer_sizes_g = DEFAULT_WASSERSTEIN_GAN_DENSE_LAYERS_SETTINGS_GENERATOR

        # Validações
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

        # CORREÇÃO: Ordem correta dos parâmetros conforme o construtor esperado
        # VanillaDiscriminatorTensorflow espera:
        # (latent_dimension, output_shape, activation_function, initializer_mean,
        #  initializer_deviation, dropout_decay_rate_d, last_layer_activation,
        #  dense_layer_sizes_d, dataset_type, number_samples_per_class)

        VanillaDiscriminatorTensorflow.__init__(
            self,
            latent_dimension=latent_dimension,
            output_shape=output_shape,
            activation_function=activation_function,
            initializer_mean=initializer_mean,
            initializer_deviation=initializer_deviation,
            dropout_decay_rate_d=dropout_decay_rate_d,
            last_layer_activation=last_layer_activation,
            dense_layer_sizes_d=dense_layer_sizes_d,
            dataset_type=dataset_type,
            number_samples_per_class=number_samples_per_class
        )

        # VanillaGeneratorTensorflow espera:
        # (latent_dimension, output_shape, activation_function, initializer_mean,
        #  initializer_deviation, dropout_decay_rate_g, last_layer_activation,
        #  dataset_type, number_samples_per_class, ...)

        VanillaGeneratorTensorflow.__init__(
            self,
            latent_dimension=latent_dimension,
            output_shape=int(numpy.prod(output_shape)) if isinstance(output_shape, tuple) else output_shape,
            activation_function=activation_function,
            initializer_mean=initializer_mean,
            initializer_deviation=initializer_deviation,
            dropout_decay_rate_g=dropout_decay_rate_g,
            last_layer_activation=last_layer_activation,
            dataset_type=dataset_type,
            number_samples_per_class=number_samples_per_class
        )

    def latent_dimension(self, latent_dimension: int) -> None:
        """Define a dimensão latente para ambos discriminador e gerador."""
        if latent_dimension <= 0:
            raise ValueError("Latent dimension must be a positive integer.")

        self._discriminator_latent_dimension = latent_dimension
        self._generator_latent_dimension = latent_dimension

    def output_shape(self, output_shape: Tuple[int, ...]) -> None:
        """Define a forma de saída para ambos discriminador e gerador."""
        if not all(dim > 0 for dim in output_shape):
            raise ValueError("Output shape dimensions must be positive integers.")

        self._discriminator_output_shape = output_shape
        self._generator_output_shape = output_shape

    def activation_function(self, activation_function: Union[str, Callable]) -> None:
        """Define a função de ativação para ambos discriminador e gerador."""
        if not callable(activation_function) and not isinstance(activation_function, str):
            raise ValueError("Activation function must be a callable or a string.")

        self._discriminator_activation_function = activation_function
        self._generator_activation_function = activation_function

    def last_layer_activation(self, last_layer_activation: Union[str, Callable]) -> None:
        """Define a ativação da última camada para ambos discriminador e gerador."""
        if not callable(last_layer_activation) and not isinstance(last_layer_activation, str):
            raise ValueError("Last layer activation must be a callable or a string.")

        self._discriminator_last_layer_activation = last_layer_activation
        self._generator_last_layer_activation = last_layer_activation

    def dataset_type(self, dataset_type: type) -> None:
        """Define o tipo de dado para o dataset de entrada."""
        if not isinstance(dataset_type, type):
            raise ValueError("Dataset type must be a valid type object.")

        self._discriminator_dataset_type = dataset_type
        self._generator_dataset_type = dataset_type

    def initializer_mean(self, initializer_mean: float) -> None:
        """Define o valor médio para o inicializador de pesos."""
        if not isinstance(initializer_mean, (int, float)):
            raise ValueError("Initializer mean must be a numerical value.")

        self._discriminator_initializer_mean = initializer_mean
        self._generator_initializer_mean = initializer_mean