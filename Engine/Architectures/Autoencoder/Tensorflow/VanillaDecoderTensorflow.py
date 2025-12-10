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

    from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Reshape
    from tensorflow.keras.layers import Concatenate
    from tensorflow.keras.models import Model
    from tensorflow.keras.initializers import RandomNormal
    from Engine.Activations.Activations import Activations

except ImportError as error:
    print(error)
    sys.exit(-1)


class VanillaDecoderTensorflow(Activations):
    """
    VanillaDecoder - Adaptativo Multi-Dimensional

    Um decoder condicional que se adapta automaticamente à dimensionalidade dos dados de entrada.
    Suporta dados 1D (vetor), 2D (imagem/matriz), 3D (volume) e N-D (tensor).
    Usa apenas camadas Dense com Reshape para adaptar a saída à forma desejada.

    Attributes:
        @decoder_latent_dimension (int):
            Dimensionalidade do espaço latente de entrada.
        @decoder_output_shape (int or tuple):
            Forma da saída do decoder.
            - int: para dados 1D (ex: 784)
            - tuple: para dados N-D (ex: (28, 28), (32, 32, 3), (16, 16, 16, 2))
        @decoder_activation_function (str):
            Função de ativação aplicada em cada camada (e.g., 'ReLU', 'LeakyReLU').
        @decoder_last_layer_activation (str):
            Função de ativação da camada de saída final.
        @decoder_dropout_decay_rate_decoder (float):
            Taxa de dropout aplicada durante decodificação (entre 0 e 1).
        @decoder_number_neurons_decoder (list):
            Lista especificando o número de neurônios em cada camada Dense.
        @decoder_dataset_type (dtype):
            Tipo de dados da entrada, padrão numpy.float32.
        @decoder_initializer_mean (float):
            Média para distribuição normal de inicialização dos pesos.
        @decoder_initializer_deviation (float):
            Desvio padrão para distribuição normal de inicialização.
        @decoder_number_samples_per_class (Optional[dict]):
            Dicionário contendo metadados sobre número de classes para entrada de labels.

    Raises:
        ValueError:
            Levantado quando argumentos inválidos são passados durante inicialização:
            - `latent_dimension` não é um inteiro positivo.
            - `output_shape` não é int positivo ou tuple de inteiros positivos.
            - `activation_function`, `last_layer_activation` não são strings.
            - `initializer_mean` ou `initializer_deviation` não são números.
            - `dropout_decay_decoder` está fora do intervalo válido [0, 1].
            - `number_neurons_decoder` não é uma lista de inteiros positivos.
            - `dataset_type` não é um tipo válido.
            - `number_samples_per_class` é fornecido mas não é um dict com 'number_classes'.

    Example:
        >>> # Decoder 1D (vetor)
        >>> decoder_1d = VanillaDecoderTensorflow(
        ...     latent_dimension=128,
        ...     output_shape=784,
        ...     activation_function='ReLU',
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_decoder=0.3,
        ...     last_layer_activation='sigmoid',
        ...     number_neurons_decoder=[512, 256],
        ...     number_samples_per_class={"number_classes": 10}
        ... )
        >>>
        >>> # Decoder 2D (imagem)
        >>> decoder_2d = VanillaDecoderTensorflow(
        ...     latent_dimension=128,
        ...     output_shape=(28, 28, 1),
        ...     activation_function='ReLU',
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_decoder=0.3,
        ...     last_layer_activation='sigmoid',
        ...     number_neurons_decoder=[256, 512],
        ...     number_samples_per_class={"number_classes": 10}
        ... )
        >>>
        >>> # Decoder 3D (volume)
        >>> decoder_3d = VanillaDecoderTensorflow(
        ...     latent_dimension=256,
        ...     output_shape=(16, 16, 16, 3),
        ...     activation_function='ReLU',
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_decoder=0.4,
        ...     last_layer_activation='tanh',
        ...     number_neurons_decoder=[512, 1024],
        ...     number_samples_per_class={"number_classes": 5}
        ... )
    """

    def __init__(self, latent_dimension: int, output_shape, activation_function: str, initializer_mean: float,
                 initializer_deviation: float, dropout_decay_decoder: float, last_layer_activation: str,
                 number_neurons_decoder: list[int], dataset_type: type = numpy.float32,
                 number_samples_per_class: dict = None):
        """
        Inicializa a classe VanillaDecoder com a configuração fornecida.

        Args:
            latent_dimension (int): Dimensionalidade do espaço latente.
            output_shape (int or tuple): Forma da saída (int para 1D, tuple para N-D).
            activation_function (str): Nome da função de ativação.
            initializer_mean (float): Média para o inicializador.
            initializer_deviation (float): Desvio padrão para o inicializador.
            dropout_decay_decoder (float): Taxa de dropout para camadas do decoder.
            last_layer_activation (str): Função de ativação para camada de saída.
            number_neurons_decoder (list[int]): Número de neurônios nas camadas do decoder.
            dataset_type (type): Tipo de dados para entradas/saídas (padrão numpy.float32).
            number_samples_per_class (dict, optional): Número de classes para entrada de label.

        Raises:
            ValueError: Se qualquer parâmetro fornecido for inválido.
        """

        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError(f"Invalid value for latent_dimension: {latent_dimension}. It must be a positive integer.")

        # Validar output_shape (pode ser int ou tuple)
        if isinstance(output_shape, int):
            if output_shape <= 0:
                raise ValueError(f"Invalid value for output_shape: {output_shape}. It must be a positive integer.")
        elif isinstance(output_shape, tuple):
            if not all(isinstance(x, int) and x > 0 for x in output_shape):
                raise ValueError(
                    f"Invalid value for output_shape: {output_shape}. All dimensions must be positive integers.")
        else:
            raise ValueError(f"Invalid value for output_shape: {output_shape}. It must be an int or tuple of ints.")

        if not isinstance(activation_function, str):
            raise ValueError(f"Invalid value for activation_function: {activation_function}. It must be a string.")

        if not isinstance(initializer_mean, (int, float)):
            raise ValueError(f"Invalid value for initializer_mean: {initializer_mean}. It must be a number.")

        if not isinstance(initializer_deviation, (int, float)):
            raise ValueError(f"Invalid value for initializer_deviation: {initializer_deviation}. It must be a number.")

        if not isinstance(dropout_decay_decoder, (int, float)) or not (0 <= dropout_decay_decoder <= 1):
            raise ValueError(
                f"Invalid value for dropout_decay_decoder: {dropout_decay_decoder}. It must be a number between 0 and 1.")

        if not isinstance(last_layer_activation, str):
            raise ValueError(f"Invalid value for last_layer_activation: {last_layer_activation}. It must be a string.")

        if not isinstance(number_neurons_decoder, list) or not all(
                isinstance(x, int) and x > 0 for x in number_neurons_decoder):
            raise ValueError(
                f"Invalid value for number_neurons_decoder: {number_neurons_decoder}. It must be a list of positive integers.")

        if not isinstance(dataset_type, type):
            raise ValueError(f"Invalid value for dataset_type: {dataset_type}. It must be a valid type.")

        if number_samples_per_class is not None and (
                not isinstance(number_samples_per_class, dict) or "number_classes" not in number_samples_per_class):
            raise ValueError(
                f"Invalid value for number_samples_per_class: {number_samples_per_class}. It must be a dictionary with 'number_classes'.")

        self._decoder_latent_dimension = latent_dimension
        self._decoder_output_shape = output_shape
        self._decoder_activation_function = activation_function
        self._decoder_last_layer_activation = last_layer_activation
        self._decoder_dropout_decay_rate_decoder = dropout_decay_decoder
        self._decoder_dataset_type = dataset_type
        self._decoder_initializer_mean = initializer_mean
        self._decoder_initializer_deviation = initializer_deviation
        self._decoder_number_neurons_decoder = number_neurons_decoder
        self._decoder_number_samples_per_class = number_samples_per_class

    def _calculate_total_output_size(self, output_shape) -> int:
        """
        Calcula o tamanho total da saída (número total de elementos).

        Args:
            output_shape (int or tuple): Forma da saída.

        Returns:
            int: Número total de elementos na saída.
        """
        if isinstance(output_shape, int):
            return output_shape
        else:
            return int(numpy.prod(output_shape))

    def _get_dimensionality(self, output_shape) -> int:
        """
        Determina a dimensionalidade dos dados baseado na forma da saída.

        Args:
            output_shape (int or tuple): Forma da saída.

        Returns:
            int: Número de dimensões (1 para 1D, 2 para 2D, 3 para 3D, etc.).
        """
        if isinstance(output_shape, int):
            return 1
        else:
            return len(output_shape)

    def get_decoder(self, output_shape):
        """
        Constrói e retorna o modelo decoder adaptado automaticamente à dimensionalidade.

        Args:
            output_shape (int or tuple): A forma de saída do decoder.

        Returns:
            keras.Model: O modelo decoder construído.

        Raises:
            ValueError: Se a forma de saída for inválida.
        """
        # Validar output_shape
        if isinstance(output_shape, int):
            if output_shape <= 0:
                raise ValueError(f"Invalid value for output_shape: {output_shape}. It must be a positive integer.")
        elif isinstance(output_shape, tuple):
            if not all(isinstance(x, int) and x > 0 for x in output_shape):
                raise ValueError(
                    f"Invalid value for output_shape: {output_shape}. All dimensions must be positive integers.")
        else:
            raise ValueError(f"Invalid value for output_shape: {output_shape}. It must be an int or tuple of ints.")

        # Determinar dimensionalidade e tamanho total
        dimensionality = self._get_dimensionality(output_shape)
        total_output_size = self._calculate_total_output_size(output_shape)

        # Inicializar pesos usando distribuição normal
        initialization = RandomNormal(mean=self._decoder_initializer_mean, stddev=self._decoder_initializer_deviation)

        # Definir camadas de entrada para espaço latente e labels
        neural_model_inputs = Input(shape=(self._decoder_latent_dimension,), dtype=self._decoder_dataset_type)
        label_input = Input(shape=(self._decoder_number_samples_per_class["number_classes"],),
                            dtype=self._decoder_dataset_type)

        # Concatenar espaço latente e labels
        concatenate_input = Concatenate()([neural_model_inputs, label_input])

        # Primeira camada Dense com dropout e ativação
        conditional_decoder = Dense(self._decoder_number_neurons_decoder[0], kernel_initializer=initialization)(
            concatenate_input)
        conditional_decoder = Dropout(self._decoder_dropout_decay_rate_decoder)(conditional_decoder)
        conditional_decoder = self._add_activation_layer(conditional_decoder, self._decoder_activation_function)

        # Iterar sobre as camadas Dense subsequentes
        for number_filters in self._decoder_number_neurons_decoder[1:]:
            conditional_decoder = Dense(number_filters, kernel_initializer=initialization)(conditional_decoder)
            conditional_decoder = Dropout(self._decoder_dropout_decay_rate_decoder)(conditional_decoder)
            conditional_decoder = self._add_activation_layer(conditional_decoder, self._decoder_activation_function)

        # Camada de saída Dense (produz vetor flat)
        conditional_decoder = Dense(total_output_size, kernel_initializer=initialization, name="Output_Dense")(
            conditional_decoder)
        conditional_decoder = self._add_activation_layer(conditional_decoder, self._decoder_last_layer_activation)

        # Reshape para a forma desejada (se não for 1D)
        if dimensionality > 1:
            conditional_decoder = Reshape(output_shape, name="Output_Reshape")(conditional_decoder)

        # Retornar o modelo construído
        return Model([neural_model_inputs, label_input], conditional_decoder, name=f"Decoder_{dimensionality}D")

    @property
    def dropout_decay_rate_decoder(self) -> float:
        """float: Obtém ou define a taxa de dropout para camadas do decoder."""
        return self._decoder_dropout_decay_rate_decoder

    @property
    def number_filters_decoder(self) -> list[int]:
        """list[int]: Obtém o número de neurônios nas camadas do decoder."""
        return self._decoder_number_neurons_decoder

    @property
    def output_shape(self):
        """int or tuple: Obtém a forma de saída do decoder."""
        return self._decoder_output_shape

    @property
    def dimensionality(self) -> int:
        """int: Obtém a dimensionalidade dos dados (1D, 2D, 3D, N-D)."""
        return self._get_dimensionality(self._decoder_output_shape)

    @dropout_decay_rate_decoder.setter
    def dropout_decay_rate_decoder(self, dropout_decay_rate_discriminator: float) -> None:
        """
        Define a taxa de dropout para as camadas do decoder.

        Args:
            dropout_decay_rate_discriminator (float): Taxa de dropout para camadas (entre 0 e 1).

        Raises:
            ValueError: Se a taxa de dropout não for um número válido entre 0 e 1.
        """
        if not isinstance(dropout_decay_rate_discriminator, (int, float)) or not (
                0 <= dropout_decay_rate_discriminator <= 1):
            raise ValueError(
                f"Invalid value for dropout_decay_rate_discriminator: {dropout_decay_rate_discriminator}. It must be a number between 0 and 1.")
        self._decoder_dropout_decay_rate_decoder = dropout_decay_rate_discriminator