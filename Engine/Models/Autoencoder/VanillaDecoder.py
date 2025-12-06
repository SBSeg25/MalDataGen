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
    from abc import ABC, abstractmethod
    from typing import Any, Dict, Optional, Union

    # Detecta o framework a partir da variável de ambiente
    ML_FRAMEWORK = os.getenv('ML_FRAMEWORK', 'tensorflow').lower()

    # Importa condicionalmente os frameworks
    tf = None
    torch = None
    nn = None
    F = None
    Dense = None
    Input = None
    Dropout = None
    Concatenate = None
    Model = None
    RandomNormal = None

    if ML_FRAMEWORK == 'tensorflow':
        try:
            from tensorflow.keras.layers import Dense, Input, Dropout, Concatenate
            from tensorflow.keras.models import Model
            from tensorflow.keras.initializers import RandomNormal
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow not found. Please install: pip install tensorflow")

    elif ML_FRAMEWORK == 'pytorch':
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("PyTorch not found. Please install: pip install torch")

    else:
        raise ValueError(f"Unsupported ML_FRAMEWORK: {ML_FRAMEWORK}. Use 'tensorflow' or 'pytorch'")

    from Engine.Activations.Activations import Activations

except ImportError as error:
    print(error)
    sys.exit(-1)


class BaseDecoderImplementation(ABC):
    """
    Classe abstrata base para implementações específicas de framework do decoder.
    """

    @abstractmethod
    def build_decoder_model(self, output_shape: int, config: dict) -> Any:
        """Constrói e retorna o modelo do decoder."""
        pass


if ML_FRAMEWORK == 'tensorflow':
    class TensorFlowDecoderImpl(BaseDecoderImplementation, Activations):
        """
        Implementação do Decoder usando TensorFlow/Keras.
        """

        def build_decoder_model(self, output_shape: int, config: dict) -> Any:
            """
            Constrói o modelo decoder usando TensorFlow/Keras.

            Args:
                output_shape (int): A dimensionalidade da saída do decoder.
                config (dict): Dicionário com configurações do modelo.

            Returns:
                keras.Model: O modelo decoder do TensorFlow.
            """
            initialization = RandomNormal(
                mean=config['initializer_mean'],
                stddev=config['initializer_deviation']
            )

            # Define camadas de entrada para espaço latente e labels
            neural_model_inputs = Input(
                shape=(config['latent_dimension'],),
                dtype=config['dataset_type']
            )
            label_input = Input(
                shape=(config['number_classes'],),
                dtype=config['dataset_type']
            )

            # Concatena espaço latente e labels, seguido por camada densa com dropout e ativação
            concatenate_input = Concatenate()([neural_model_inputs, label_input])
            x = Dense(
                config['number_neurons_decoder'][0],
                kernel_initializer=initialization
            )(concatenate_input)
            x = Dropout(config['dropout_decay_rate'])(x)
            x = self._add_activation_layer(x, config['activation_function'])

            # Itera sobre as camadas densas subsequentes
            for number_neurons in config['number_neurons_decoder'][1:]:
                x = Dense(number_neurons, kernel_initializer=initialization)(x)
                x = Dropout(config['dropout_decay_rate'])(x)
                x = self._add_activation_layer(x, config['activation_function'])

            # Adiciona a camada de saída
            x = Dense(output_shape, kernel_initializer=initialization, name="Output_1")(x)
            x = self._add_activation_layer(x, config['last_layer_activation'])

            return Model([neural_model_inputs, label_input], x, name="Decoder")

if ML_FRAMEWORK == 'pytorch':
    class PyTorchDecoderImpl(BaseDecoderImplementation, Activations):
        """
        Implementação do Decoder usando PyTorch.
        """

        class PyTorchDecoderModel(nn.Module):
            """
            Módulo PyTorch que implementa o Decoder.
            """

            def __init__(self, output_shape: int, config: dict, activation_handler):
                super().__init__()
                self.config = config
                self.activation_handler = activation_handler
                self.output_shape = output_shape

                # Calcula o tamanho da entrada concatenada (latente + labels)
                concat_size = config['latent_dimension'] + config['number_classes']

                # Constrói as camadas
                layers = []

                # Primeira camada
                layers.append(nn.Linear(concat_size, config['number_neurons_decoder'][0]))
                layers.append(nn.Dropout(config['dropout_decay_rate']))

                # Camadas intermediárias
                for i in range(len(config['number_neurons_decoder']) - 1):
                    layers.append(
                        nn.Linear(
                            config['number_neurons_decoder'][i],
                            config['number_neurons_decoder'][i + 1]
                        )
                    )
                    layers.append(nn.Dropout(config['dropout_decay_rate']))

                # Camada de saída
                layers.append(
                    nn.Linear(
                        config['number_neurons_decoder'][-1],
                        output_shape
                    )
                )

                self.layers = nn.ModuleList(layers)
                self._init_weights(config['initializer_mean'], config['initializer_deviation'])

            def _init_weights(self, mean: float, std: float):
                """Inicializa os pesos com distribuição normal."""
                for layer in self.layers:
                    if isinstance(layer, nn.Linear):
                        nn.init.normal_(layer.weight, mean=mean, std=std)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)

            def _get_activation(self, activation_name: str):
                """Retorna a função de ativação PyTorch apropriada."""
                activation_map = {
                    'relu': F.relu,
                    'leakyrelu': lambda x: F.leaky_relu(x, 0.2),
                    'sigmoid': torch.sigmoid,
                    'tanh': torch.tanh,
                    'softmax': lambda x: F.softmax(x, dim=-1),
                    'linear': lambda x: x,
                }
                return activation_map.get(activation_name.lower(), F.relu)

            def forward(self, latent_input, label_input):
                """
                Forward pass do decoder.

                Args:
                    latent_input: Tensor do espaço latente.
                    label_input: Tensor de labels.

                Returns:
                    Tensor: Saída reconstruída do decoder.
                """
                # Garante que os tensores tenham 2 dimensões
                # latent_input esperado: (batch_size, latent_dim) - 2D
                # label_input esperado: (batch_size, num_classes) - 2D

                # Se latent_input tem mais de 2 dimensões, achata para 2D
                if len(latent_input.shape) > 2:
                    latent_input = latent_input.view(latent_input.shape[0], -1)
                elif len(latent_input.shape) == 1:
                    latent_input = latent_input.unsqueeze(0)

                # Se label_input tem mais de 2 dimensões, achata para 2D
                if len(label_input.shape) > 2:
                    label_input = label_input.view(label_input.shape[0], -1)
                elif len(label_input.shape) == 1:
                    label_input = label_input.unsqueeze(0)

                # Verifica se batch sizes são compatíveis
                if latent_input.shape[0] != label_input.shape[0]:
                    raise ValueError(f"Batch size mismatch: latent_input has {latent_input.shape[0]} samples, "
                                     f"but label_input has {label_input.shape[0]} samples")

                # Concatena entradas
                x = torch.cat([latent_input, label_input], dim=-1)

                # Aplica primeira camada
                layer_idx = 0
                x = self.layers[layer_idx](x)
                layer_idx += 1
                x = self.layers[layer_idx](x)  # Dropout
                layer_idx += 1
                activation_fn = self._get_activation(self.config['activation_function'])
                x = activation_fn(x)

                # Aplica camadas intermediárias
                num_intermediate_layers = len(self.config['number_neurons_decoder']) - 1
                for _ in range(num_intermediate_layers):
                    x = self.layers[layer_idx](x)
                    layer_idx += 1
                    x = self.layers[layer_idx](x)  # Dropout
                    layer_idx += 1
                    x = activation_fn(x)

                # Camada de saída
                x = self.layers[layer_idx](x)
                final_activation = self._get_activation(self.config['last_layer_activation'])
                x = final_activation(x)

                return x

        def build_decoder_model(self, output_shape: int, config: dict) -> Any:
            """
            Constrói o modelo decoder usando PyTorch.

            Args:
                output_shape (int): A dimensionalidade da saída.
                config (dict): Dicionário com configurações do modelo.

            Returns:
                nn.Module: O modelo decoder do PyTorch.
            """
            return self.PyTorchDecoderModel(output_shape, config, self)


class VanillaDecoder:
    """
    VanillaDecoder

    Uma classe que representa um modelo decoder condicional com suporte para camadas densas
    customizadas, funções de ativação, dropout e entrada condicionada por labels.
    Suporta tanto TensorFlow quanto PyTorch através da variável de ambiente ML_FRAMEWORK.

    O decoder é projetado para processar uma representação latente e gerar a forma de saída
    desejada. Esta classe é tipicamente usada em tarefas como modelos generativos, autoencoders
    e modelos condicionais que geram dados a partir de um espaço latente.

    Variável de Ambiente:
        ML_FRAMEWORK: Define o framework a ser usado ('tensorflow' ou 'pytorch').
                     Padrão: 'tensorflow'

    Attributes:
        decoder_latent_dimension (int):
            A dimensionalidade da entrada do espaço latente.
        decoder_output_shape (int):
            A dimensionalidade da camada de saída.
        decoder_activation_function (str):
            A função de ativação aplicada a cada camada do decoder (ex: 'ReLU', 'LeakyReLU').
        decoder_last_layer_activation (str):
            A função de ativação aplicada à camada de saída final.
        decoder_dropout_decay_rate_decoder (float):
            A taxa de dropout aplicada durante a decodificação (deve estar entre 0 e 1).
        decoder_number_neurons_decoder (list):
            Uma lista especificando o número de neurônios em cada camada do decoder.
        decoder_dataset_type (type):
            O tipo de dados do dataset de entrada, padrão é numpy.float32.
        decoder_initializer_mean (float):
            A média para a distribuição normal usada para inicializar os pesos.
        decoder_initializer_deviation (float):
            O desvio padrão para a distribuição normal usada para inicializar os pesos.
        decoder_number_samples_per_class (Optional[dict]):
            Um dicionário opcional contendo metadados sobre o número de classes para entrada de labels.

    Raises:
        ValueError: Levantado quando argumentos inválidos são passados durante a inicialização.

    Example:
        >>> import os
        >>> os.environ['ML_FRAMEWORK'] = 'tensorflow'  # ou 'pytorch'
        >>> decoder = VanillaDecoder(
        ...     latent_dimension=128,
        ...     output_shape=64,
        ...     activation_function='ReLU',
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_decoder=0.5,
        ...     last_layer_activation='sigmoid',
        ...     number_neurons_decoder=[512, 256, 128],
        ...     dataset_type=numpy.float32,
        ...     number_samples_per_class={"number_classes": 10}
        ... )
        >>> model = decoder.get_decoder(output_shape=784)
    """

    def __init__(self, latent_dimension: int, output_shape: int, activation_function: str,
                 initializer_mean: float, initializer_deviation: float, dropout_decay_decoder: float,
                 last_layer_activation: str, number_neurons_decoder: list,
                 dataset_type: type = numpy.float32,
                 number_samples_per_class: dict = None):
        """
        Inicializa o VanillaDecoder com a configuração fornecida.

        Args:
            latent_dimension (int): Dimensionalidade da entrada do espaço latente.
            output_shape (int): Dimensionalidade da camada de saída.
            activation_function (str): Nome da função de ativação.
            initializer_mean (float): Média para o inicializador.
            initializer_deviation (float): Desvio padrão para o inicializador.
            dropout_decay_decoder (float): Taxa de dropout para camadas do decoder.
            last_layer_activation (str): Função de ativação para a camada de saída.
            number_neurons_decoder (list): Número de neurônios nas camadas do decoder.
            dataset_type (type): Tipo de dados para entradas/saídas (padrão: numpy.float32).
            number_samples_per_class (dict, optional): Número de classes para entrada de labels.

        Raises:
            ValueError: Se algum dos parâmetros fornecidos for inválido.
        """
        # Validações
        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError(f"Invalid value for latent_dimension: {latent_dimension}. It must be a positive integer.")

        if not isinstance(output_shape, int) or output_shape <= 0:
            raise ValueError(f"Invalid value for output_shape: {output_shape}. It must be a positive integer.")

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

        # Armazena parâmetros
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

        # Seleciona a implementação apropriada baseada no framework
        self._framework = ML_FRAMEWORK
        if self._framework == 'tensorflow':
            self._decoder_implementation = TensorFlowDecoderImpl()
        elif self._framework == 'pytorch':
            self._decoder_implementation = PyTorchDecoderImpl()
        else:
            raise ValueError(f"Unsupported framework: {self._framework}")

    def get_decoder(self, output_shape: int) -> Any:
        """
        Constrói e retorna o modelo decoder.

        Args:
            output_shape (int): A dimensionalidade da saída do decoder.

        Returns:
            Union[keras.Model, nn.Module]: O modelo decoder construído.

        Raises:
            ValueError: Se o output_shape for inválido.
        """
        if not isinstance(output_shape, int) or output_shape <= 0:
            raise ValueError(f"Invalid value for output_shape: {output_shape}. It must be a positive integer.")

        config = {
            'latent_dimension': self._decoder_latent_dimension,
            'activation_function': self._decoder_activation_function,
            'last_layer_activation': self._decoder_last_layer_activation,
            'dropout_decay_rate': self._decoder_dropout_decay_rate_decoder,
            'initializer_mean': self._decoder_initializer_mean,
            'initializer_deviation': self._decoder_initializer_deviation,
            'number_neurons_decoder': self._decoder_number_neurons_decoder,
            'dataset_type': self._decoder_dataset_type,
            'number_classes': self._decoder_number_samples_per_class["number_classes"]
        }

        return self._decoder_implementation.build_decoder_model(output_shape, config)

    @property
    def framework(self) -> str:
        """
        Retorna o framework sendo utilizado.

        Returns:
            str: O nome do framework ('tensorflow' ou 'pytorch').
        """
        return self._framework

    @property
    def dropout_decay_rate_decoder(self) -> float:
        """
        Obtém a taxa de dropout decay para as camadas do decoder.

        Returns:
            float: A taxa de dropout decay aplicada às camadas do decoder.
        """
        return self._decoder_dropout_decay_rate_decoder

    @property
    def number_filters_decoder(self) -> list:
        """
        Obtém o número de neurônios para cada camada do decoder.

        Returns:
            list: Uma lista especificando o número de neurônios em cada camada do decoder.
        """
        return self._decoder_number_neurons_decoder

    @dropout_decay_rate_decoder.setter
    def dropout_decay_rate_decoder(self, dropout_decay_rate_discriminator: float) -> None:
        """
        Define a taxa de dropout decay para as camadas do decoder.

        Args:
            dropout_decay_rate_discriminator (float): A nova taxa de dropout decay (entre 0 e 1).

        Raises:
            ValueError: Se o valor não for um número entre 0 e 1.
        """
        if not isinstance(dropout_decay_rate_discriminator, (int, float)) or not (
                0 <= dropout_decay_rate_discriminator <= 1):
            raise ValueError(
                f"Invalid value for dropout_decay_rate_discriminator: {dropout_decay_rate_discriminator}. It must be a number between 0 and 1.")
        self._decoder_dropout_decay_rate_decoder = dropout_decay_rate_discriminator
