#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{2}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/09'
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

    from typing import Any
    from typing import Dict
    from typing import Optional

    from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Concatenate
    from tensorflow.keras.models import Model
    from tensorflow.keras.initializers import RandomNormal
    from Engine.Activations.Activations import Activations

except ImportError as error:
    print(error)
    sys.exit(-1)


class VanillaEncoderTensorflow(Activations):
    """
    VanillaEncoder - Adaptativo Multi-Dimensional

    Um encoder condicional que se adapta automaticamente à dimensionalidade dos dados de entrada.
    Suporta dados 1D (vetor), 2D (imagem/matriz), 3D (volume) e N-D (tensor).
    Usa apenas camadas Dense com Flatten para processar dados de qualquer dimensionalidade.

    Attributes:
        @encoder_latent_dimension (int):
            Dimensionalidade do espaço latente que o modelo irá gerar.
        @encoder_output_shape (int or tuple):
            Forma de saída desejada do encoder (dimensionalidade do espaço latente).
        @encoder_activation_function (str):
            Função de ativação aplicada em cada camada (e.g., 'ReLU', 'LeakyReLU').
        @encoder_last_layer_activation (str):
            Função de ativação aplicada na camada de saída final.
        @encoder_dropout_decay_rate_encoder (float):
            Taxa de dropout aplicada durante codificação para melhorar generalização (entre 0 e 1).
        @encoder_number_neurons_encoder (list):
            Lista especificando o número de neurônios em cada camada da rede encoder.
        @encoder_dataset_type (dtype):
            Tipo de dados do dataset de entrada, padrão numpy.float32.
        @encoder_initializer_mean (float):
            Média para distribuição normal usada para inicializar os pesos.
        @encoder_initializer_deviation (float):
            Desvio padrão para distribuição normal usada para inicializar os pesos.
        @encoder_number_samples_per_class (Optional[dict]):
            Dicionário opcional contendo metadados sobre o número de amostras por classe.

    Raises:
        ValueError:
            Levantado quando argumentos inválidos são passados durante inicialização:
            - `latent_dimension` não é um inteiro positivo.
            - `output_shape` não é int positivo ou tuple de inteiros positivos.
            - `initializer_mean` ou `initializer_deviation` não são números.
            - `dropout_decay_encoder` está fora do intervalo válido [0, 1].
            - `number_neurons_encoder` não é uma lista não-vazia ou contém inteiros não-positivos.
            - `number_samples_per_class` é fornecido mas não é um dicionário.

    Example:
        >>> # Encoder 1D (vetor de entrada)
        >>> encoder_1d = VanillaEncoderTensorflow(
        ...     latent_dimension=128,
        ...     output_shape=128,
        ...     activation_function='ReLU',
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_encoder=0.3,
        ...     last_layer_activation='linear',
        ...     number_neurons_encoder=[256, 512],
        ...     number_samples_per_class={"number_classes": 10}
        ... )
        >>> 
        >>> # Encoder 2D (imagem de entrada)
        >>> encoder_2d = VanillaEncoderTensorflow(
        ...     latent_dimension=128,
        ...     output_shape=128,
        ...     activation_function='ReLU',
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_encoder=0.3,
        ...     last_layer_activation='linear',
        ...     number_neurons_encoder=[512, 256],
        ...     number_samples_per_class={"number_classes": 10}
        ... )
        >>>
        >>> # Encoder 3D (volume de entrada)
        >>> encoder_3d = VanillaEncoderTensorflow(
        ...     latent_dimension=256,
        ...     output_shape=256,
        ...     activation_function='ReLU',
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_encoder=0.4,
        ...     last_layer_activation='linear',
        ...     number_neurons_encoder=[1024, 512],
        ...     number_samples_per_class={"number_classes": 5}
        ... )
    """

    def __init__(self, latent_dimension: int, output_shape, activation_function: str, initializer_mean: float,
                 initializer_deviation: float, dropout_decay_encoder: float, last_layer_activation: str,
                 number_neurons_encoder: list, dataset_type: Any = numpy.float32,
                 number_samples_per_class: Optional[Dict[str, Any]] = None):
        """
        Inicializa o VanillaEncoder com os parâmetros fornecidos.

        Args:
            latent_dimension (int): Dimensão do espaço latente.
            output_shape (int or tuple): Forma de saída desejada do encoder.
            activation_function (str): Função de ativação a usar nas camadas.
            initializer_mean (float): Média para inicialização de pesos.
            initializer_deviation (float): Desvio padrão para inicialização de pesos.
            dropout_decay_encoder (float): Taxa de dropout aplicada durante codificação.
            last_layer_activation (str): Função de ativação para última camada.
            number_neurons_encoder (list): Lista especificando número de neurônios em cada camada.
            dataset_type (dtype, optional): Tipo de dados do dataset de entrada. Padrão numpy.float32.
            number_samples_per_class (dict, optional): Especifica número de amostras por classe.
        """

        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")

        # Validar output_shape (pode ser int ou tuple)
        if isinstance(output_shape, int):
            if output_shape <= 0:
                raise ValueError(f"Invalid value for output_shape: {output_shape}. It must be a positive integer.")
        elif isinstance(output_shape, tuple):
            if not all(isinstance(x, int) and x > 0 for x in output_shape):
                raise ValueError(f"Invalid value for output_shape: {output_shape}. All dimensions must be positive integers.")
        else:
            raise ValueError(f"Invalid value for output_shape: {output_shape}. It must be an int or tuple of ints.")

        if not isinstance(initializer_mean, (int, float)):
            raise ValueError("initializer_mean must be a number.")

        if not isinstance(initializer_deviation, (int, float)):
            raise ValueError("initializer_deviation must be a number.")

        if not isinstance(dropout_decay_encoder, (int, float)) or not (0 <= dropout_decay_encoder <= 1):
            raise ValueError("dropout_decay_encoder must be a float between 0 and 1.")

        if not isinstance(number_neurons_encoder, list) or len(number_neurons_encoder) == 0:
            raise ValueError("number_neurons_encoder must be a non-empty list.")

        for neurons in number_neurons_encoder:
            if not isinstance(neurons, int) or neurons <= 0:
                raise ValueError("Each element in number_neurons_encoder must be a positive integer.")

        if number_samples_per_class is not None:
            if not isinstance(number_samples_per_class, dict):
                raise ValueError("number_samples_per_class must be a dictionary.")

        self._encoder_latent_dimension = latent_dimension
        self._encoder_output_shape = output_shape
        self._encoder_activation_function = activation_function
        self._encoder_last_layer_activation = last_layer_activation
        self._encoder_dropout_decay_rate_encoder = dropout_decay_encoder
        self._encoder_dataset_type = dataset_type
        self._encoder_initializer_mean = initializer_mean
        self._encoder_initializer_deviation = initializer_deviation
        self._encoder_number_neurons_encoder = number_neurons_encoder
        self._encoder_number_samples_per_class = number_samples_per_class

    def _calculate_total_input_size(self, input_shape) -> int:
        """
        Calcula o tamanho total da entrada (número total de elementos).

        Args:
            input_shape (int or tuple): Forma da entrada.

        Returns:
            int: Número total de elementos na entrada.
        """
        if isinstance(input_shape, int):
            return input_shape
        else:
            return int(numpy.prod(input_shape))

    def _get_dimensionality(self, shape) -> int:
        """
        Determina a dimensionalidade dos dados baseado na forma.

        Args:
            shape (int or tuple): Forma dos dados.

        Returns:
            int: Número de dimensões (1 para 1D, 2 para 2D, 3 para 3D, etc.).
        """
        if isinstance(shape, int):
            return 1
        else:
            return len(shape)

    def get_encoder(self, input_shape) -> Model:
        """
        Cria e retorna o modelo encoder adaptado automaticamente à dimensionalidade.

        Este método constrói a rede neural empilhando camadas Dense com as configurações
        fornecidas (neurônios, dropout e ativação). Ele também concatena os dados de entrada
        e labels antes de passar pelas camadas. Se a entrada for N-D, usa Flatten automaticamente.

        Args:
            input_shape (int or tuple): Forma dos dados de entrada.
                - int: para dados 1D (ex: 784)
                - tuple: para dados N-D (ex: (28, 28, 1) para 2D, (16, 16, 16) para 3D)

        Returns:
            keras.Model: O modelo encoder que recebe dados de entrada e labels e gera
                         a representação latente codificada e labels.
        """

        # Validar input_shape
        if isinstance(input_shape, int):
            if input_shape <= 0:
                raise ValueError(f"Invalid value for input_shape: {input_shape}. It must be a positive integer.")
        elif isinstance(input_shape, tuple):
            if not all(isinstance(x, int) and x > 0 for x in input_shape):
                raise ValueError(f"Invalid value for input_shape: {input_shape}. All dimensions must be positive integers.")
        else:
            raise ValueError(f"Invalid value for input_shape: {input_shape}. It must be an int or tuple of ints.")

        # Determinar dimensionalidade
        dimensionality = self._get_dimensionality(input_shape)

        # Inicializar pesos usando distribuição normal com média e desvio especificados
        initialization = RandomNormal(mean=self._encoder_initializer_mean, stddev=self._encoder_initializer_deviation)

        # Definir camadas de entrada para dados e labels
        if dimensionality == 1:
            # Entrada 1D - usar shape diretamente
            neural_model_inputs = Input(shape=(input_shape,), dtype=self._encoder_dataset_type, name="first_input")
            flattened_input = neural_model_inputs
        else:
            # Entrada N-D - usar tuple e aplicar Flatten
            neural_model_inputs = Input(shape=input_shape, dtype=self._encoder_dataset_type, name="first_input")
            flattened_input = Flatten(name="Input_Flatten")(neural_model_inputs)

        label_input = Input(shape=(self._encoder_number_samples_per_class["number_classes"],),
                            dtype=self._encoder_dataset_type, name="second_input")

        # Concatenar dados e labels e aplicar primeira camada Dense com dropout e ativação
        concatenate_input = Concatenate()([flattened_input, label_input])
        conditional_encoder = Dense(self._encoder_number_neurons_encoder[0],
                                    kernel_initializer=initialization)(concatenate_input)
        conditional_encoder = Dropout(self._encoder_dropout_decay_rate_encoder)(conditional_encoder)
        conditional_encoder = self._add_activation_layer(conditional_encoder, self._encoder_activation_function)

        # Iterar sobre camadas Dense especificadas
        for number_neurons in self._encoder_number_neurons_encoder[1:]:
            conditional_encoder = Dense(number_neurons, kernel_initializer=initialization)(conditional_encoder)
            conditional_encoder = Dropout(self._encoder_dropout_decay_rate_encoder)(conditional_encoder)
            conditional_encoder = self._add_activation_layer(conditional_encoder, self._encoder_activation_function)

        # Mapear para o espaço latente
        conditional_encoder = Dense(self._encoder_latent_dimension, kernel_initializer=initialization,
                                    name="Latent_Space")(conditional_encoder)
        conditional_encoder = self._add_activation_layer(conditional_encoder, self._encoder_last_layer_activation)

        # Retornar o modelo encoder
        return Model([neural_model_inputs, label_input], [conditional_encoder, label_input],
                     name=f"Encoder_{dimensionality}D")

    @property
    def dropout_decay_rate_encoder(self) -> float:
        """
        Obtém a taxa de dropout para as camadas do encoder.

        Returns:
            float: Taxa de dropout aplicada às camadas do encoder.
        """
        return self._encoder_dropout_decay_rate_encoder

    @property
    def number_filters_encoder(self) -> list:
        """
        Obtém o número de neurônios para cada camada do encoder.

        Returns:
            list: Lista especificando o número de neurônios em cada camada do encoder.
        """
        return self._encoder_number_neurons_encoder

    @property
    def output_shape(self):
        """int or tuple: Obtém a forma de saída do encoder."""
        return self._encoder_output_shape

    @property
    def latent_dimension(self) -> int:
        """int: Obtém a dimensionalidade do espaço latente."""
        return self._encoder_latent_dimension

    @property
    def dimensionality(self) -> int:
        """int: Obtém a dimensionalidade dos dados configurados."""
        return self._get_dimensionality(self._encoder_output_shape)

    @dropout_decay_rate_encoder.setter
    def dropout_decay_rate_encoder(self, dropout_decay_rate_generator: float) -> None:
        """
        Define a taxa de dropout para as camadas do encoder.

        Args:
            dropout_decay_rate_generator (float): Nova taxa de dropout.

        Raises:
            ValueError: Se o valor não for um float entre 0 e 1.
        """
        if not (0 <= dropout_decay_rate_generator <= 1):
            raise ValueError("dropout_decay_rate_encoder must be a float between 0 and 1.")

        self._encoder_dropout_decay_rate_encoder = dropout_decay_rate_generator