#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{2}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/06'
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
    import os
    import sys
    import numpy

    from typing import List
    from typing import Tuple
    from typing import Optional
    from typing import Any

    from Engine.Architectures.Autoencoder.Torch.VanillaDecoderTorch import VanillaDecoderTorch
    from Engine.Architectures.Autoencoder.Torch.VanillaEncoderTorch import VanillaEncoderTorch

    # Detecta o framework a partir da variável de ambiente
    ML_FRAMEWORK = os.getenv('ML_FRAMEWORK', 'pytorch').lower()

except ImportError as error:
    print(error)
    sys.exit(-1)

DEFAULT_AUTOENCODER_LATENT_DIMENSION = 64
DEFAULT_AUTOENCODER_ACTIVATION = "swish"
DEFAULT_AUTOENCODER_DROPOUT_DECAY_RATE_ENCODER = 0.25
DEFAULT_AUTOENCODER_DROPOUT_DECAY_RATE_DECODER = 0.25
DEFAULT_AUTOENCODER_DENSE_LAYERS_SETTINGS_ENCODER = [320, 160]
DEFAULT_AUTOENCODER_DENSE_LAYERS_SETTINGS_DECODER = [160, 320]
DEFAULT_AUTOENCODER_LAST_ACTIVATION_LAYER = "sigmoid"
DEFAULT_AUTOENCODER_INITIALIZER_MEAN = 0.0
DEFAULT_AUTOENCODER_INITIALIZER_DEVIATION = 0.125


class AutoencoderModelTorch(VanillaEncoderTorch, VanillaDecoderTorch):
    """
    AutoencoderModel

    Esta classe implementa um modelo Autoencoder herdando das classes VanillaEncoder e VanillaDecoder.
    Constrói uma arquitetura de autoencoder combinando encoder e decoder com hiperparâmetros customizáveis.
    Versão PyTorch do modelo.

    O autoencoder é tipicamente usado para tarefas como redução de dimensionalidade, aprendizado de features
    e denoising.

    Variável de Ambiente:
        ML_FRAMEWORK: Define o framework a ser usado ('tensorflow' ou 'pytorch').
                     Padrão: 'pytorch'

    Attributes:
        latent_dimension (int):
            A dimensionalidade do espaço latente (espaço de codificação).
        output_shape (tuple):
            A forma desejada da saída do decoder.
        activation_function (str):
            A função de ativação usada no encoder e decoder.
        initializer_mean (float):
            A média para inicialização dos pesos.
        initializer_deviation (float):
            O desvio padrão para inicialização dos pesos.
        dropout_decay_encoder (float):
            A taxa de dropout para o encoder.
        dropout_decay_decoder (float):
            A taxa de dropout para o decoder.
        last_layer_activation (str):
            A função de ativação para a última camada do encoder e decoder.
        number_neurons_encoder (List[int]):
            Uma lista especificando o número de neurônios para cada camada do encoder.
        number_neurons_decoder (List[int]):
            Uma lista especificando o número de neurônios para cada camada do decoder.
        dataset_type (numpy.dtype):
            O tipo de dados do dataset (padrão é numpy.float32).
        number_samples_per_class (Optional[int]):
            O número de amostras por classe, usado para gerenciamento do dataset (opcional).

    Raises:
        ValueError: Levantado quando argumentos inválidos são passados durante a inicialização.

    Example:
        >>> import os
        >>> os.environ['ML_FRAMEWORK'] = 'pytorch'
        >>> autoencoder_model = AutoencoderModel(
        ...     latent_dimension=64,
        ...     output_shape=(28, 28, 1),
        ...     activation_function="ReLU",
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.01,
        ...     dropout_decay_encoder=0.3,
        ...     dropout_decay_decoder=0.3,
        ...     last_layer_activation="sigmoid",
        ...     number_neurons_encoder=[128, 64, 32],
        ...     number_neurons_decoder=[32, 64, 128],
        ...     dataset_type=numpy.float32,
        ...     number_samples_per_class=1000
        ... )
        >>> encoder = autoencoder_model.get_encoder(input_shape=784)
        >>> decoder = autoencoder_model.get_decoder(output_shape=784)
    """

    def __init__(self,
                 latent_dimension: int = DEFAULT_AUTOENCODER_LATENT_DIMENSION,
                 output_shape: Tuple[int, ...] = (128,),
                 activation_function: str = DEFAULT_AUTOENCODER_ACTIVATION,
                 initializer_mean: float = DEFAULT_AUTOENCODER_INITIALIZER_MEAN,
                 initializer_deviation: float = DEFAULT_AUTOENCODER_INITIALIZER_DEVIATION,
                 dropout_decay_encoder: float = DEFAULT_AUTOENCODER_DROPOUT_DECAY_RATE_ENCODER,
                 dropout_decay_decoder: float = DEFAULT_AUTOENCODER_DROPOUT_DECAY_RATE_DECODER,
                 last_layer_activation: str = DEFAULT_AUTOENCODER_LAST_ACTIVATION_LAYER,
                 number_neurons_encoder=None,
                 number_neurons_decoder=None,
                 dataset_type: numpy.dtype = numpy.float32,
                 number_samples_per_class: Optional[int] = None):
        """
        Inicializa o AutoencoderModel configurando tanto o encoder quanto o decoder com os parâmetros fornecidos.

        Args:
            latent_dimension (int): A dimensionalidade do espaço latente.
            output_shape (tuple): A forma desejada da saída do decoder.
            activation_function (str): A função de ativação usada no encoder e decoder.
            initializer_mean (float): A média para inicialização dos pesos.
            initializer_deviation (float): O desvio padrão para inicialização dos pesos.
            dropout_decay_encoder (float): A taxa de dropout para o encoder.
            dropout_decay_decoder (float): A taxa de dropout para o decoder.
            last_layer_activation (str): A função de ativação para a última camada.
            number_neurons_encoder (list): Lista com o número de neurônios para cada camada do encoder.
            number_neurons_decoder (list): Lista com o número de neurônios para cada camada do decoder.
            dataset_type (numpy.dtype): O tipo de dados do dataset (padrão: numpy.float32).
            number_samples_per_class (int): O número de amostras por classe (opcional).

        Raises:
            ValueError: Se algum dos parâmetros fornecidos for inválido.
        """
        if number_neurons_decoder is None:
            number_neurons_decoder = DEFAULT_AUTOENCODER_DENSE_LAYERS_SETTINGS_DECODER

        if number_neurons_encoder is None:
            number_neurons_encoder = DEFAULT_AUTOENCODER_DENSE_LAYERS_SETTINGS_ENCODER

        # Validações
        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")

        if not isinstance(activation_function, str):
            raise ValueError("activation_function must be a string.")

        if not isinstance(initializer_mean, (float, int)):
            raise ValueError("initializer_mean must be a float or integer.")

        if not isinstance(initializer_deviation, (float, int)):
            raise ValueError("initializer_deviation must be a float or integer.")

        if not isinstance(dropout_decay_encoder, (float, int)) or not (0 <= dropout_decay_encoder <= 1):
            raise ValueError("dropout_decay_encoder must be a float or integer between 0 and 1.")

        if not isinstance(dropout_decay_decoder, (float, int)) or not (0 <= dropout_decay_decoder <= 1):
            raise ValueError("dropout_decay_decoder must be a float or integer between 0 and 1.")

        if not isinstance(last_layer_activation, str):
            raise ValueError("last_layer_activation must be a string.")

        if not isinstance(number_neurons_encoder, list) or not all(isinstance(x, int) for x in number_neurons_encoder):
            raise ValueError("number_neurons_encoder must be a list of integers.")

        if not isinstance(number_neurons_decoder, list) or not all(isinstance(x, int) for x in number_neurons_decoder):
            raise ValueError("number_neurons_decoder must be a list of integers.")

        # Inicializa o Decoder primeiro
        VanillaDecoderTorch.__init__(self,
                                latent_dimension,
                                output_shape,
                                activation_function,
                                initializer_mean,
                                initializer_deviation,
                                dropout_decay_decoder,
                                last_layer_activation,
                                number_neurons_decoder,
                                dataset_type,
                                number_samples_per_class)

        # Inicializa o Encoder depois
        VanillaEncoderTorch.__init__(self,
                                latent_dimension,
                                output_shape,
                                activation_function,
                                initializer_mean,
                                initializer_deviation,
                                dropout_decay_encoder,
                                last_layer_activation,
                                number_neurons_encoder,
                                dataset_type,
                                number_samples_per_class)

        # Armazena referências para os modelos (serão None até serem construídos)
        self._encoder_model = None
        self._decoder_model = None

        # Armazena o framework sendo usado
        self._framework = ML_FRAMEWORK

    @property
    def framework(self) -> str:
        """
        Retorna o framework sendo utilizado.

        Returns:
            str: O nome do framework ('tensorflow' ou 'pytorch').
        """
        return self._framework

    def get_encoder(self, input_shape: int) -> Any:
        """
        Constrói e retorna o modelo encoder.

        Args:
            input_shape (int): A forma dos dados de entrada.

        Returns:
            nn.Module: O modelo encoder construído.
        """
        self._encoder_model = VanillaEncoderTorch.get_encoder(self, input_shape)
        return self._encoder_model

    def get_decoder(self, output_shape: int) -> Any:
        """
        Constrói e retorna o modelo decoder.

        Args:
            output_shape (int): A dimensionalidade da saída do decoder.

        Returns:
            nn.Module: O modelo decoder construído.
        """
        self._decoder_model = VanillaDecoderTorch.get_decoder(self, output_shape)
        return self._decoder_model

    def get_dense_encoder_model(self) -> Any:
        """
        Retorna o modelo encoder previamente construído.

        Returns:
            nn.Module or None: O modelo encoder ou None se ainda não foi construído.
        """
        return self._encoder_model

    def get_dense_decoder_model(self) -> Any:
        """
        Retorna o modelo decoder previamente construído.

        Returns:
            nn.Module or None: O modelo decoder ou None se ainda não foi construído.
        """
        return self._decoder_model

    def set_latent_dimension(self, latent_dimension: int) -> None:
        """
        Define a dimensão latente para ambos encoder e decoder.

        Args:
            latent_dimension (int): A dimensionalidade do espaço latente.

        Raises:
            ValueError: Se latent_dimension não for um inteiro positivo.
        """
        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")

        self._encoder_latent_dimension = latent_dimension
        self._decoder_latent_dimension = latent_dimension

    def set_output_shape(self, output_shape: Tuple[int, ...]) -> None:
        """
        Define a forma de saída para ambos encoder e decoder.

        Args:
            output_shape (tuple): A forma da saída a ser gerada por ambos encoder e decoder.

        Raises:
            ValueError: Se output_shape não for uma tupla de inteiros.
        """
        if not isinstance(output_shape, tuple) or not all(isinstance(x, int) for x in output_shape):
            raise ValueError("output_shape must be a tuple of integers.")

        self._encoder_output_shape = output_shape
        self._decoder_output_shape = output_shape

    def set_activation_function(self, activation_function: str) -> None:
        """
        Define a função de ativação para ambos encoder e decoder.

        Args:
            activation_function (str): A função de ativação a ser aplicada no encoder e decoder.

        Raises:
            ValueError: Se activation_function não for uma string.
        """
        if not isinstance(activation_function, str):
            raise ValueError("activation_function must be a string.")

        self._encoder_activation_function = activation_function
        self._decoder_activation_function = activation_function

    def set_last_layer_activation(self, last_layer_activation: str) -> None:
        """
        Define a função de ativação para a última camada de ambos encoder e decoder.

        Args:
            last_layer_activation (str): A função de ativação para a última camada.

        Raises:
            ValueError: Se last_layer_activation não for uma string.
        """
        if not isinstance(last_layer_activation, str):
            raise ValueError("last_layer_activation must be a string.")

        self._encoder_last_layer_activation = last_layer_activation
        self._decoder_last_layer_activation = last_layer_activation

    # Mantém compatibilidade com código antigo que pode usar esses métodos sem 'set_'
    def latent_dimension(self, latent_dimension: int) -> None:
        """
        Método de compatibilidade. Use set_latent_dimension() em código novo.
        """
        self.set_latent_dimension(latent_dimension)

    def output_shape(self, output_shape: Tuple[int, ...]) -> None:
        """
        Método de compatibilidade. Use set_output_shape() em código novo.
        """
        self.set_output_shape(output_shape)

    def activation_function(self, activation_function: str) -> None:
        """
        Método de compatibilidade. Use set_activation_function() em código novo.
        """
        self.set_activation_function(activation_function)

    def last_layer_activation(self, last_layer_activation: str) -> None:
        """
        Método de compatibilidade. Use set_last_layer_activation() em código novo.
        """
        self.set_last_layer_activation(last_layer_activation)