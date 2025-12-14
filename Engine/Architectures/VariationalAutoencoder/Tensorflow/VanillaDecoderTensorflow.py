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

    import tensorflow
    from tensorflow import keras
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.models import Model

    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Concatenate

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
      VanillaDecoder with Cross-Attention

      This class implements a conditional fully connected decoder network with cross-attention
      mechanism for label conditioning. Instead of simple concatenation, the decoder uses
      cross-attention where the latent vector queries the label embedding information.

      The architecture consists of:
      1. Label embedding layer
      2. Cross-attention layer (latent queries label)
      3. Sequence of fully connected layers with activation and dropout
      4. Final output layer

      Attributes
      ----------
      @decoder_latent_dimension : int
          Dimensionality of the latent space (input to the decoder).
      @decoder_output_shape : int
          Dimensionality of the output data (reconstructed data).
      @decoder_intermediary_activation_function : Callable
          Activation function applied to intermediate layers.
      @decoder_last_layer_activation : Callable
          Activation function applied to the final layer.
      @decoder_dropout_decay_rate_decoder : float
          Dropout rate applied to the dense layers for regularization.
      @decoder_dataset_type : type
          Data type for inputs and outputs (default: numpy.float32).
      @decoder_initializer_mean : float
          Mean of the normal distribution used for weight initialization.
      @decoder_initializer_deviation : float
          Standard deviation of the normal distribution used for weight initialization.
      @decoder_number_neurons_decoder : List[int]
          List specifying the number of neurons in each dense layer.
      @decoder_number_samples_per_class : Optional[Dict[str, int]]
          Dictionary containing class metadata (e.g., number of samples per class).
      @decoder_attention_embed_dim : int
          Embedding dimension for cross-attention mechanism.
      @decoder_attention_num_heads : int
          Number of attention heads in cross-attention layer.
      @decoder_model : Optional[Model]
          Placeholder for the actual compiled Keras model (built later).

      Examples
      --------
      >>> decoder = VanillaDecoderTensorflow(
      ...     latent_dimension=128,
      ...     output_shape=784,
      ...     activation_function='relu',
      ...     initializer_mean=0.0,
      ...     initializer_deviation=0.02,
      ...     dropout_decay_decoder=0.3,
      ...     last_layer_activation='sigmoid',
      ...     number_neurons_decoder=[256, 128, 64],
      ...     dataset_type=numpy.float32,
      ...     number_samples_per_class={"number_classes": 10},
      ...     attention_embed_dim=128,
      ...     attention_num_heads=4
      ... )
      >>> decoder_model = decoder.get_decoder()
      """

    def __init__(self,
                 latent_dimension,
                 output_shape,
                 activation_function,
                 initializer_mean,
                 initializer_deviation,
                 dropout_decay_decoder,
                 last_layer_activation,
                 number_neurons_decoder,
                 dataset_type=numpy.float32,
                 number_samples_per_class=None,
                 attention_embed_dim=128,
                 attention_num_heads=4):
        """
        Initializes the VanillaDecoder with cross-attention.

        Parameters
        ----------
        @latent_dimension : int
            Dimensionality of the input latent space.
        @output_shape : int
            Dimensionality of the output.
        @activation_function : str or callable
            Activation function for intermediate layers.
        @initializer_mean : float
            Mean of the normal distribution for weight initialization.
        @initializer_deviation : float
            Standard deviation for weight initialization.
        @dropout_decay_decoder : float
            Dropout rate applied to intermediate layers.
        @last_layer_activation : str or callable
            Activation function for the final layer.
        @number_neurons_decoder : list of int
            Number of neurons in each fully connected layer.
        @dataset_type : type, optional
            Data type for inputs and outputs (default: numpy.float32).
        @number_samples_per_class : dict, optional
            Dictionary with class information (must contain 'number_classes' key).
        @attention_embed_dim : int, optional
            Embedding dimension for cross-attention (default: 128).
        @attention_num_heads : int, optional
            Number of attention heads (default: 4).
        """

        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")

        if not isinstance(output_shape, int) or output_shape <= 0:
            raise ValueError("output_shape must be a positive integer.")

        if not isinstance(activation_function, (str, callable)):
            raise ValueError("activation_function must be a string or a callable function.")

        if not isinstance(last_layer_activation, (str, callable)):
            raise ValueError("last_layer_activation must be a string or a callable function.")

        if not isinstance(initializer_mean, (int, float)):
            raise ValueError("initializer_mean must be a numeric value.")

        if not isinstance(initializer_deviation, (int, float)) or initializer_deviation <= 0:
            raise ValueError("initializer_deviation must be a positive numeric value.")

        if not isinstance(dropout_decay_decoder, (int, float)) or not (0.0 <= dropout_decay_decoder <= 1.0):
            raise ValueError("dropout_decay_decoder must be a float between 0 and 1.")

        if not isinstance(number_neurons_decoder, list) or not all(
                isinstance(n, int) and n > 0 for n in number_neurons_decoder):
            raise ValueError("number_neurons_decoder must be a list of positive integers.")

        if number_samples_per_class is not None:
            if not isinstance(number_samples_per_class, dict):
                raise ValueError("number_samples_per_class must be a dictionary or None.")

            if "number_classes" not in number_samples_per_class or not isinstance(
                    number_samples_per_class["number_classes"], int):
                raise ValueError("number_samples_per_class must contain a key 'number_classes' with an integer value.")

        if not isinstance(attention_embed_dim, int) or attention_embed_dim <= 0:
            raise ValueError("attention_embed_dim must be a positive integer.")

        if not isinstance(attention_num_heads, int) or attention_num_heads <= 0:
            raise ValueError("attention_num_heads must be a positive integer.")

        if attention_embed_dim % attention_num_heads != 0:
            raise ValueError("attention_embed_dim must be divisible by attention_num_heads.")

        self._decoder_latent_dimension = latent_dimension
        self._decoder_output_shape = output_shape
        self._decoder_intermediary_activation_function = activation_function
        self._decoder_last_layer_activation = last_layer_activation
        self._decoder_dropout_decay_rate_decoder = dropout_decay_decoder
        self._decoder_dataset_type = dataset_type
        self._decoder_initializer_mean = initializer_mean
        self._decoder_initializer_deviation = initializer_deviation
        self._decoder_number_neurons_decoder = number_neurons_decoder
        self._decoder_number_samples_per_class = number_samples_per_class
        self._decoder_attention_embed_dim = attention_embed_dim
        self._decoder_attention_num_heads = attention_num_heads
        self._decoder_model = None

    def get_decoder(self, input_shape=None):
        """
        Builds and returns the decoder model with cross-attention.

        The model uses cross-attention to condition the latent vector on label information,
        where the latent vector acts as Query and the label embedding provides Key and Value.

        Args:
            input_shape: Optional input shape (for API compatibility, not used)

        Returns:
            tensorflow.keras.Model: The decoder model with cross-attention conditioning.
        """
        initialization = RandomNormal(mean=self._decoder_initializer_mean,
                                      stddev=self._decoder_initializer_deviation)

        # Input layers
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

        # Project latent vector to attention embedding dimension
        latent_projected = Dense(self._decoder_attention_embed_dim,
                                 kernel_initializer=initialization,
                                 name='latent_projection')(neural_model_inputs)

        # Cross-attention: latent queries label information
        cross_attention = CrossAttentionLayer(
            embed_dim=self._decoder_attention_embed_dim,
            num_heads=self._decoder_attention_num_heads,
            name='cross_attention'
        )
        attended_features = cross_attention(latent_projected, label_input_embedding)

        # First dense layer after attention
        conditional_decoder = Dense(self._decoder_number_neurons_decoder[0],
                                    kernel_initializer=initialization)(attended_features)
        conditional_decoder = self._add_activation_layer(conditional_decoder,
                                                         self._decoder_intermediary_activation_function)
        conditional_decoder = Dropout(self._decoder_dropout_decay_rate_decoder)(conditional_decoder)

        # Hidden layers
        for number_filters in self._decoder_number_neurons_decoder[1:]:
            conditional_decoder = Dense(number_filters,
                                        kernel_initializer=initialization)(conditional_decoder)
            conditional_decoder = Dropout(self._decoder_dropout_decay_rate_decoder)(conditional_decoder)
            conditional_decoder = self._add_activation_layer(conditional_decoder,
                                                             self._decoder_intermediary_activation_function)

        # Output layer
        conditional_decoder = Dense(self._decoder_output_shape,
                                    kernel_initializer=initialization)(conditional_decoder)
        conditional_decoder = self._add_activation_layer(conditional_decoder,
                                                         self._decoder_last_layer_activation)

        self._decoder_model = Model([neural_model_inputs, label_input],
                                    conditional_decoder,
                                    name="Decoder_CrossAttention")

        return self._decoder_model

    @property
    def dropout_decay_rate_decoder(self):
        """float: Gets or sets the dropout decay rate for the decoder."""
        return self._decoder_dropout_decay_rate_decoder

    @property
    def number_filters_decoder(self):
        """list[int]: Gets the number of neurons for each layer in the decoder."""
        return self._decoder_number_neurons_decoder

    @property
    def attention_embed_dim(self):
        """int: Gets the embedding dimension for cross-attention."""
        return self._decoder_attention_embed_dim

    @property
    def attention_num_heads(self):
        """int: Gets the number of attention heads."""
        return self._decoder_attention_num_heads

    @dropout_decay_rate_decoder.setter
    def dropout_decay_rate_decoder(self, dropout_decay_rate_discriminator):
        """
        Sets the dropout decay rate for the decoder.

        Args:
            dropout_decay_rate_discriminator (float): New dropout decay rate.
        """
        self._decoder_dropout_decay_rate_decoder = dropout_decay_rate_discriminator