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
    import tensorflow

    from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Reshape, Layer
    from tensorflow.keras.layers import Concatenate
    from tensorflow.keras.models import Model
    from tensorflow.keras.initializers import RandomNormal
    from Engine.Activations.Activations import Activations

except ImportError as error:
    print(error)
    sys.exit(-1)


class CrossAttentionLayer(Layer):
    """
    Cross-Attention layer for conditioning on label information.

    The latent vector acts as Query, while the label embedding provides Key and Value.
    """

    def __init__(self, embed_dim, num_heads=4, **kwargs):
        super(CrossAttentionLayer, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Query projection (from latent vector)
        self.query_dense = Dense(embed_dim, name='query')

        # Key and Value projections (from label embedding)
        self.key_dense = Dense(embed_dim, name='key')
        self.value_dense = Dense(embed_dim, name='value')

        # Output projection
        self.out_dense = Dense(embed_dim, name='output')

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, head_dim)"""
        x = tensorflow.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tensorflow.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query_input, key_value_input):
        batch_size = tensorflow.shape(query_input)[0]

        # Ensure inputs have sequence dimension
        if len(query_input.shape) == 2:
            query_input = tensorflow.expand_dims(query_input, axis=1)
        if len(key_value_input.shape) == 2:
            key_value_input = tensorflow.expand_dims(key_value_input, axis=1)

        # Linear projections
        Q = self.query_dense(query_input)
        K = self.key_dense(key_value_input)
        V = self.value_dense(key_value_input)

        # Split heads
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Scaled dot-product attention
        matmul_qk = tensorflow.matmul(Q, K, transpose_b=True)
        dk = tensorflow.cast(self.head_dim, tensorflow.float32)
        scaled_attention_logits = matmul_qk / tensorflow.math.sqrt(dk)

        attention_weights = tensorflow.nn.softmax(scaled_attention_logits, axis=-1)

        # Apply attention to values
        attention_output = tensorflow.matmul(attention_weights, V)

        # Concatenate heads
        attention_output = tensorflow.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tensorflow.reshape(attention_output, (batch_size, -1, self.embed_dim))

        # Final linear projection
        output = self.out_dense(concat_attention)

        # Remove sequence dimension if it was added
        output = tensorflow.squeeze(output, axis=1)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
        })
        return config


class VanillaDecoderTensorflow(Activations):
    """
    VanillaDecoder with Cross-Attention - Adaptativo Multi-Dimensional

    Um decoder condicional que se adapta automaticamente à dimensionalidade dos dados de entrada.
    Suporta dados 1D (vetor), 2D (imagem/matriz), 3D (volume) e N-D (tensor).
    Usa cross-attention para condicionamento sofisticado em vez de concatenação simples.
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
        @decoder_attention_embed_dim (int):
            Dimensão de embedding para mecanismo de cross-attention.
        @decoder_attention_num_heads (int):
            Número de cabeças de atenção na camada de cross-attention.

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
            - `attention_embed_dim` não é divisível por `attention_num_heads`.

    Example:
        >>> # Decoder 1D (vetor) com cross-attention
        >>> decoder_1d = VanillaDecoderTensorflow(
        ...     latent_dimension=128,
        ...     output_shape=784,
        ...     activation_function='ReLU',
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_decoder=0.3,
        ...     last_layer_activation='sigmoid',
        ...     number_neurons_decoder=[512, 256],
        ...     number_samples_per_class={"number_classes": 10},
        ...     attention_embed_dim=128,
        ...     attention_num_heads=4
        ... )
        >>>
        >>> # Decoder 2D (imagem) com cross-attention
        >>> decoder_2d = VanillaDecoderTensorflow(
        ...     latent_dimension=128,
        ...     output_shape=(28, 28, 1),
        ...     activation_function='ReLU',
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_decoder=0.3,
        ...     last_layer_activation='sigmoid',
        ...     number_neurons_decoder=[256, 512],
        ...     number_samples_per_class={"number_classes": 10},
        ...     attention_embed_dim=128,
        ...     attention_num_heads=4
        ... )
        >>>
        >>> # Decoder 3D (volume) com cross-attention
        >>> decoder_3d = VanillaDecoderTensorflow(
        ...     latent_dimension=256,
        ...     output_shape=(16, 16, 16, 3),
        ...     activation_function='ReLU',
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_decoder=0.4,
        ...     last_layer_activation='tanh',
        ...     number_neurons_decoder=[512, 1024],
        ...     number_samples_per_class={"number_classes": 5},
        ...     attention_embed_dim=256,
        ...     attention_num_heads=8
        ... )
    """

    def __init__(self, latent_dimension: int, output_shape, activation_function: str, initializer_mean: float,
                 initializer_deviation: float, dropout_decay_decoder: float, last_layer_activation: str,
                 number_neurons_decoder: list[int], dataset_type: type = numpy.float32,
                 number_samples_per_class: dict = None, attention_embed_dim: int = 128,
                 attention_num_heads: int = 4):
        """
        Inicializa a classe VanillaDecoder com cross-attention e a configuração fornecida.

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
            attention_embed_dim (int, optional): Dimensão de embedding para cross-attention (padrão: 128).
            attention_num_heads (int, optional): Número de cabeças de atenção (padrão: 4).

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

        if not isinstance(attention_embed_dim, int) or attention_embed_dim <= 0:
            raise ValueError(f"Invalid value for attention_embed_dim: {attention_embed_dim}. It must be a positive integer.")

        if not isinstance(attention_num_heads, int) or attention_num_heads <= 0:
            raise ValueError(f"Invalid value for attention_num_heads: {attention_num_heads}. It must be a positive integer.")

        if attention_embed_dim % attention_num_heads != 0:
            raise ValueError(f"attention_embed_dim ({attention_embed_dim}) must be divisible by attention_num_heads ({attention_num_heads}).")

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
        self._decoder_attention_embed_dim = attention_embed_dim
        self._decoder_attention_num_heads = attention_num_heads

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
        Constrói e retorna o modelo decoder com cross-attention adaptado automaticamente à dimensionalidade.

        O modelo usa cross-attention para condicionar o vetor latente na informação dos labels,
        onde o vetor latente atua como Query e o embedding dos labels fornece Key e Value.

        Args:
            output_shape (int or tuple): A forma de saída do decoder.

        Returns:
            keras.Model: O modelo decoder construído com cross-attention.

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
        neural_model_inputs = Input(shape=(self._decoder_latent_dimension,),
                                    dtype=self._decoder_dataset_type,
                                    name='latent_input')
        label_input = Input(shape=(self._decoder_number_samples_per_class["number_classes"],),
                           dtype=self._decoder_dataset_type,
                           name='label_input')

        # Label embedding
        label_input_embedding = Dense(self._decoder_attention_embed_dim,
                                      activation='relu',
                                      name='label_embedding')(label_input)

        # Projetar vetor latente para dimensão de embedding de atenção
        latent_projected = Dense(self._decoder_attention_embed_dim,
                                 kernel_initializer=initialization,
                                 name='latent_projection')(neural_model_inputs)

        # Cross-attention: latent consulta informação dos labels
        cross_attention = CrossAttentionLayer(
            embed_dim=self._decoder_attention_embed_dim,
            num_heads=self._decoder_attention_num_heads,
            name='cross_attention'
        )
        attended_features = cross_attention(latent_projected, label_input_embedding)

        # Primeira camada Dense após atenção
        conditional_decoder = Dense(self._decoder_number_neurons_decoder[0],
                                    kernel_initializer=initialization)(attended_features)
        conditional_decoder = self._add_activation_layer(conditional_decoder,
                                                         self._decoder_activation_function)
        conditional_decoder = Dropout(self._decoder_dropout_decay_rate_decoder)(conditional_decoder)

        # Camadas ocultas
        for number_filters in self._decoder_number_neurons_decoder[1:]:
            conditional_decoder = Dense(number_filters,
                                        kernel_initializer=initialization)(conditional_decoder)
            conditional_decoder = Dropout(self._decoder_dropout_decay_rate_decoder)(conditional_decoder)
            conditional_decoder = self._add_activation_layer(conditional_decoder,
                                                             self._decoder_activation_function)

        # Camada de saída Dense (produz vetor flat)
        conditional_decoder = Dense(total_output_size,
                                    kernel_initializer=initialization,
                                    name="Output_Dense")(conditional_decoder)
        conditional_decoder = self._add_activation_layer(conditional_decoder,
                                                         self._decoder_last_layer_activation)

        # Reshape para a forma desejada (se não for 1D)
        if dimensionality > 1:
            conditional_decoder = Reshape(output_shape, name="Output_Reshape")(conditional_decoder)

        # Retornar o modelo construído
        return Model([neural_model_inputs, label_input],
                    conditional_decoder,
                    name=f"Decoder_{dimensionality}D_CrossAttention")

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

    @property
    def attention_embed_dim(self) -> int:
        """int: Obtém a dimensão de embedding para cross-attention."""
        return self._decoder_attention_embed_dim

    @property
    def attention_num_heads(self) -> int:
        """int: Obtém o número de cabeças de atenção."""
        return self._decoder_attention_num_heads

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