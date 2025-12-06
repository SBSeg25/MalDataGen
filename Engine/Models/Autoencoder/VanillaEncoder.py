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


class BaseEncoderImplementation(ABC):
    """
    Classe abstrata base para implementações específicas de framework.
    """

    @abstractmethod
    def build_model(self, input_shape: tuple, config: dict) -> Any:
        """Constrói e retorna o modelo do encoder."""
        pass

    @abstractmethod
    def get_activation_layer(self, activation_function: str):
        """Retorna a camada de ativação apropriada para o framework."""
        pass


if ML_FRAMEWORK == 'tensorflow':
    class TensorFlowEncoderImpl(BaseEncoderImplementation, Activations):
        """
        Implementação do Encoder usando TensorFlow/Keras.
        """

        def build_model(self, input_shape: tuple, config: dict) -> Any:
            """
            Constrói o modelo encoder usando TensorFlow/Keras.

            Args:
                input_shape (tuple): A forma dos dados de entrada.
                config (dict): Dicionário com configurações do modelo.

            Returns:
                keras.Model: O modelo encoder do TensorFlow.
            """
            initialization = RandomNormal(
                mean=config['initializer_mean'],
                stddev=config['initializer_deviation']
            )

            # Define camadas de entrada
            neural_model_inputs = Input(
                shape=(input_shape,),
                dtype=config['dataset_type'],
                name="first_input"
            )
            label_input = Input(
                shape=(config['number_classes'],),
                dtype=config['dataset_type'],
                name="second_input"
            )

            # Concatena entradas e aplica primeira camada densa
            concatenate_input = Concatenate()([neural_model_inputs, label_input])
            x = Dense(
                config['number_neurons_encoder'][0],
                kernel_initializer=initialization
            )(concatenate_input)
            x = Dropout(config['dropout_decay_rate'])(x)
            x = self._add_activation_layer(x, config['activation_function'])

            # Itera sobre as camadas densas especificadas
            for number_neurons in config['number_neurons_encoder'][1:]:
                x = Dense(number_neurons, kernel_initializer=initialization)(x)
                x = Dropout(config['dropout_decay_rate'])(x)
                x = self._add_activation_layer(x, config['activation_function'])

            # Mapeia para o espaço latente
            x = Dense(config['latent_dimension'], kernel_initializer=initialization)(x)
            x = self._add_activation_layer(x, config['last_layer_activation'])

            return Model([neural_model_inputs, label_input], [x, label_input], name="Encoder")

        def get_activation_layer(self, activation_function: str):
            """Retorna a camada de ativação do TensorFlow."""
            return self._add_activation_layer

if ML_FRAMEWORK == 'pytorch':
    class PyTorchEncoderImpl(BaseEncoderImplementation, Activations):
        """
        Implementação do Encoder usando PyTorch.
        """

        class PyTorchEncoderModel(nn.Module):
            """
            Módulo PyTorch que implementa o Encoder.
            """

            def __init__(self, input_shape: int, config: dict, activation_handler):
                super().__init__()
                self.config = config
                self.activation_handler = activation_handler

                # Calcula o tamanho da entrada concatenada
                concat_size = input_shape + config['number_classes']

                # Constrói as camadas
                layers = []

                # Primeira camada
                layers.append(nn.Linear(concat_size, config['number_neurons_encoder'][0]))
                layers.append(nn.Dropout(config['dropout_decay_rate']))

                # Camadas intermediárias
                for i in range(len(config['number_neurons_encoder']) - 1):
                    layers.append(
                        nn.Linear(
                            config['number_neurons_encoder'][i],
                            config['number_neurons_encoder'][i + 1]
                        )
                    )
                    layers.append(nn.Dropout(config['dropout_decay_rate']))

                # Camada latente
                layers.append(
                    nn.Linear(
                        config['number_neurons_encoder'][-1],
                        config['latent_dimension']
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

            def forward(self, data_input, label_input):
                """
                Forward pass do encoder.

                Args:
                    data_input: Tensor de dados de entrada.
                    label_input: Tensor de labels.

                Returns:
                    tuple: (representação latente, labels)
                """
                # Garante que os tensores tenham 2 dimensões
                # data_input esperado: (batch_size, features) - 2D
                # label_input esperado: (batch_size, num_classes) - 2D

                # Se data_input tem mais de 2 dimensões, achata para 2D
                if len(data_input.shape) > 2:
                    data_input = data_input.view(data_input.shape[0], -1)
                elif len(data_input.shape) == 1:
                    data_input = data_input.unsqueeze(0)

                # Se label_input tem mais de 2 dimensões, achata para 2D
                if len(label_input.shape) > 2:
                    label_input = label_input.view(label_input.shape[0], -1)
                elif len(label_input.shape) == 1:
                    label_input = label_input.unsqueeze(0)

                # Verifica se batch sizes são compatíveis
                if data_input.shape[0] != label_input.shape[0]:
                    raise ValueError(f"Batch size mismatch: data_input has {data_input.shape[0]} samples, "
                                     f"but label_input has {label_input.shape[0]} samples")

                # Concatena entradas
                x = torch.cat([data_input, label_input], dim=-1)

                # Aplica primeira camada
                layer_idx = 0
                x = self.layers[layer_idx](x)
                layer_idx += 1
                x = self.layers[layer_idx](x)  # Dropout
                layer_idx += 1
                activation_fn = self._get_activation(self.config['activation_function'])
                x = activation_fn(x)

                # Aplica camadas intermediárias
                num_intermediate_layers = len(self.config['number_neurons_encoder']) - 1
                for _ in range(num_intermediate_layers):
                    x = self.layers[layer_idx](x)
                    layer_idx += 1
                    x = self.layers[layer_idx](x)  # Dropout
                    layer_idx += 1
                    x = activation_fn(x)

                # Camada latente
                x = self.layers[layer_idx](x)
                final_activation = self._get_activation(self.config['last_layer_activation'])
                x = final_activation(x)

                return x, label_input

        def build_model(self, input_shape: tuple, config: dict) -> Any:
            """
            Constrói o modelo encoder usando PyTorch.

            Args:
                input_shape (tuple): A forma dos dados de entrada.
                config (dict): Dicionário com configurações do modelo.

            Returns:
                nn.Module: O modelo encoder do PyTorch.
            """
            return self.PyTorchEncoderModel(input_shape, config, self)

        def get_activation_layer(self, activation_function: str):
            """Retorna a função de ativação do PyTorch."""
            return self._add_activation_layer


class VanillaEncoder:
    """
    VanillaEncoder

    Uma classe que representa um modelo Vanilla Encoder para aplicações de deep learning.
    Suporta tanto TensorFlow quanto PyTorch através da variável de ambiente ML_FRAMEWORK.

    O encoder é projetado para processar entradas e labels, aplicar uma série de camadas
    densas com ativações e dropout, e gerar uma representação latente dos dados de entrada.

    Variável de Ambiente:
        ML_FRAMEWORK: Define o framework a ser usado ('tensorflow' ou 'pytorch').
                     Padrão: 'tensorflow'

    Attributes:
        encoder_latent_dimension (int):
            A dimensionalidade do espaço latente que o modelo irá gerar.
        encoder_output_shape (tuple):
            A forma de saída desejada do encoder.
        encoder_activation_function (str):
            A função de ativação aplicada a cada camada do encoder (ex: 'ReLU', 'LeakyReLU').
        encoder_last_layer_activation (str):
            A função de ativação aplicada à camada de saída final.
        encoder_dropout_decay_rate_encoder (float):
            A taxa de dropout aplicada durante a codificação (deve estar entre 0 e 1).
        encoder_number_neurons_encoder (list):
            Uma lista especificando o número de neurônios em cada camada do encoder.
        encoder_dataset_type (dtype):
            O tipo de dados do dataset de entrada, padrão é numpy.float32.
        encoder_initializer_mean (float):
            A média para a distribuição normal usada para inicializar os pesos.
        encoder_initializer_deviation (float):
            O desvio padrão para a distribuição normal usada para inicializar os pesos.
        encoder_number_samples_per_class (Optional[dict]):
            Um dicionário opcional contendo metadados sobre o número de amostras por classe.

    Raises:
        ValueError: Levantado quando argumentos inválidos são passados durante a inicialização.

    Example:
        >>> import os
        >>> os.environ['ML_FRAMEWORK'] = 'tensorflow'  # ou 'pytorch'
        >>> encoder = VanillaEncoder(
        ...     latent_dimension=128,
        ...     output_shape=(64, 64, 1),
        ...     activation_function='ReLU',
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_encoder=0.5,
        ...     last_layer_activation='sigmoid',
        ...     number_neurons_encoder=[512, 256, 128],
        ...     dataset_type=numpy.float32,
        ...     number_samples_per_class={"number_classes": 10}
        ... )
        >>> model = encoder.get_encoder(input_shape=784)
    """

    def __init__(self, latent_dimension: int, output_shape: tuple, activation_function: str,
                 initializer_mean: float, initializer_deviation: float, dropout_decay_encoder: float,
                 last_layer_activation: str, number_neurons_encoder: list,
                 dataset_type: Any = numpy.float32,
                 number_samples_per_class: Optional[Dict[str, Any]] = None):
        """
        Inicializa o VanillaEncoder com os parâmetros fornecidos.

        Args:
            latent_dimension (int): A dimensão do espaço latente.
            output_shape (tuple): A forma de saída desejada do encoder.
            activation_function (str): A função de ativação para as camadas.
            initializer_mean (float): A média para inicialização dos pesos.
            initializer_deviation (float): O desvio padrão para inicialização dos pesos.
            dropout_decay_encoder (float): A taxa de dropout aplicada durante a codificação.
            last_layer_activation (str): A função de ativação para a última camada.
            number_neurons_encoder (list): Lista especificando o número de neurônios em cada camada.
            dataset_type (dtype, optional): O tipo de dados do dataset. Padrão: numpy.float32.
            number_samples_per_class (dict, optional): Especifica o número de amostras por classe.
        """
        # Validações
        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")

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

        # Armazena parâmetros
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

        # Seleciona a implementação apropriada baseada no framework
        self._framework = ML_FRAMEWORK
        if self._framework == 'tensorflow':
            self._implementation = TensorFlowEncoderImpl()
        elif self._framework == 'pytorch':
            self._implementation = PyTorchEncoderImpl()
        else:
            raise ValueError(f"Unsupported framework: {self._framework}")

    def get_encoder(self, input_shape: tuple) -> Any:
        """
        Cria e retorna o modelo encoder.

        Este método constrói a rede neural empilhando camadas densas com as configurações
        fornecidas (neurônios, dropout e ativação). Também concatena os dados de entrada
        e labels antes de passar pelas camadas.

        Args:
            input_shape (tuple): A forma dos dados de entrada.

        Returns:
            Union[keras.Model, nn.Module]: O modelo encoder que recebe dados de entrada
                                           e labels e retorna a representação latente
                                           codificada e os labels.
        """
        config = {
            'latent_dimension': self._encoder_latent_dimension,
            'activation_function': self._encoder_activation_function,
            'last_layer_activation': self._encoder_last_layer_activation,
            'dropout_decay_rate': self._encoder_dropout_decay_rate_encoder,
            'initializer_mean': self._encoder_initializer_mean,
            'initializer_deviation': self._encoder_initializer_deviation,
            'number_neurons_encoder': self._encoder_number_neurons_encoder,
            'dataset_type': self._encoder_dataset_type,
            'number_classes': self._encoder_number_samples_per_class["number_classes"]
        }

        return self._implementation.build_model(input_shape, config)

    @property
    def framework(self) -> str:
        """
        Retorna o framework sendo utilizado.

        Returns:
            str: O nome do framework ('tensorflow' ou 'pytorch').
        """
        return self._framework

    @property
    def dropout_decay_rate_encoder(self) -> float:
        """
        Obtém a taxa de dropout decay para as camadas do encoder.

        Returns:
            float: A taxa de dropout decay aplicada às camadas do encoder.
        """
        return self._encoder_dropout_decay_rate_encoder

    @property
    def number_filters_encoder(self) -> list:
        """
        Obtém o número de neurônios para cada camada do encoder.

        Returns:
            list: Uma lista especificando o número de neurônios em cada camada do encoder.
        """
        return self._encoder_number_neurons_encoder

    @dropout_decay_rate_encoder.setter
    def dropout_decay_rate_encoder(self, dropout_decay_rate_generator: float) -> None:
        """
        Define a taxa de dropout decay para as camadas do encoder.

        Args:
            dropout_decay_rate_generator (float): A nova taxa de dropout decay.

        Raises:
            ValueError: Se o valor não for um float entre 0 e 1.
        """
        if not (0 <= dropout_decay_rate_generator <= 1):
            raise ValueError("dropout_decay_rate_encoder must be a float between 0 and 1.")

        self._encoder_dropout_decay_rate_encoder = dropout_decay_rate_generator


#