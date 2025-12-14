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
    from typing import Dict
    from typing import Union
    from typing import Optional

    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Concatenate
    from tensorflow.keras.initializers import RandomNormal

    import tensorflow as tf

    from Engine.Activations.Activations import Activations

except ImportError as error:
    print(error)
    sys.exit(-1)


class CrossAttentionLayer(Layer):
    """
    Cross-Attention layer for conditioning on label information (Keras implementation).

    The input features act as Query, while the label embedding provides Key and Value.
    """

    def __init__(self, embed_dim, num_heads=4, **kwargs):
        super(CrossAttentionLayer, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

    def build(self, input_shape):
        # Query projection (from input features)
        self.query_dense = Dense(self.embed_dim, name='query')

        # Key and Value projections (from label embedding)
        self.key_dense = Dense(self.embed_dim, name='key')
        self.value_dense = Dense(self.embed_dim, name='value')

        # Output projection
        self.out_dense = Dense(self.embed_dim, name='output')

        super(CrossAttentionLayer, self).build(input_shape)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, head_dim)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query_input, key_value_input = inputs
        batch_size = tf.shape(query_input)[0]

        # Ensure inputs have sequence dimension
        if len(query_input.shape) == 2:
            query_input = tf.expand_dims(query_input, 1)
        if len(key_value_input.shape) == 2:
            key_value_input = tf.expand_dims(key_value_input, 1)

        # Linear projections
        Q = self.query_dense(query_input)
        K = self.key_dense(key_value_input)
        V = self.value_dense(key_value_input)

        # Split heads
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Scaled dot-product attention
        matmul_qk = tf.matmul(Q, K, transpose_b=True)
        dk = tf.cast(self.head_dim, tf.float32)
        scaled_attention_logits = matmul_qk / tf.sqrt(dk)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        # Apply attention to values
        attention_output = tf.matmul(attention_weights, V)

        # Concatenate heads
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.embed_dim))

        # Final linear projection
        output = self.out_dense(concat_attention)

        # Remove sequence dimension if it was added
        output = tf.squeeze(output, axis=1)

        return output

    def get_config(self):
        config = super(CrossAttentionLayer, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads
        })
        return config


class VanillaDiscriminator(Activations):
    """
     VanillaDiscriminator with Cross-Attention

     Implements a fully-connected (dense) discriminator network with cross-attention mechanism
     for use in generative models, such as Generative Adversarial Networks (GANs). Instead of
     simple concatenation, the discriminator uses cross-attention where the input features query
     the label embedding information. This class provides flexibility in the design of the
     architecture, including customizable latent dimensions, output shapes, activation functions,
     dropout rates, and layer sizes.

     Attributes:
         @discriminator_latent_dimension (int):
             Dimensionality of the input latent space for the discriminator network.
         @discriminator_output_shape (int):
             The output shape of the network, typically used to define the shape of input data like images.
         @discriminator_activation_function (str):
             The activation function applied to all hidden layers (e.g., 'relu', 'leaky_relu').
         @discriminator_last_layer_activation (str):
             The activation function applied to the last layer (e.g., 'sigmoid').
         @discriminator_dropout_decay_rate_d (float):
             Dropout rate applied to layers in the network to help prevent overfitting.
         @discriminator_dense_layer_sizes_d (List[int]):
             List of integers defining the number of units in each dense layer.
         @discriminator_dataset_type (numpy.dtype):
             The data type of the dataset (default: numpy.float32).
         @discriminator_initializer_mean (float):
             Mean of the normal distribution used for weight initialization.
         @discriminator_initializer_deviation (float):
             Standard deviation of the normal distribution used for weight initialization.
         @discriminator_number_samples_per_class (Optional[Dict[str, int]]):
             Optional dictionary containing the number of samples per class.
         @discriminator_attention_embed_dim (int):
             Embedding dimension for cross-attention mechanism.
         @discriminator_attention_num_heads (int):
             Number of attention heads in cross-attention layer.
         @discriminator_model_dense (Optional[Model]):
             Placeholder for the compiled Keras model after building the network.

     Raises:
         ValueError:
             Raised if invalid arguments are passed during initialization, such as:
             - Non-positive `latent_dimension`
             - Dropout rate outside the range [0, 1]
             - Empty or invalid `dense_layer_sizes_d`
             - Missing required key "number_classes" in `number_samples_per_class`, if provided
             - `attention_embed_dim` not divisible by `attention_num_heads`

     Example:
         >>> discriminator = VanillaDiscriminator(
         ...     latent_dimension=100,
         ...     output_shape=784,
         ...     activation_function='leaky_relu',
         ...     initializer_mean=0.0,
         ...     initializer_deviation=0.02,
         ...     dropout_decay_rate_d=0.3,
         ...     last_layer_activation='sigmoid',
         ...     dense_layer_sizes_d=[512, 256, 128],
         ...     dataset_type=numpy.float32,
         ...     number_samples_per_class={"number_classes": 10},
         ...     attention_embed_dim=128,
         ...     attention_num_heads=4
         ... )
         >>> model = discriminator.get_discriminator()
     """

    def __init__(self,
                 latent_dimension: int,
                 output_shape: int,
                 activation_function: str,
                 initializer_mean: float,
                 initializer_deviation: float,
                 dropout_decay_rate_d: float,
                 last_layer_activation: str,
                 dense_layer_sizes_d: List[int],
                 dataset_type: numpy.dtype = numpy.float32,
                 number_samples_per_class: Optional[Dict[str, int]] = None,
                 attention_embed_dim: int = 128,
                 attention_num_heads: int = 4):
        """
        Initializes the VanillaDiscriminator with cross-attention and the provided parameters.

        Args:
            latent_dimension (int): The dimensionality of the input latent space.
            output_shape (int): The shape of the expected output data (e.g., flattened image size).
            activation_function (str): The activation function to apply to all hidden layers.
            initializer_mean (float): The mean for weight initialization.
            initializer_deviation (float): The standard deviation for weight initialization.
            dropout_decay_rate_d (float): Dropout rate for dense layers (should be between 0 and 1).
            last_layer_activation (str): The activation function for the last output layer.
            dense_layer_sizes_d (List[int]): A list of integers specifying the number of units in each dense layer.
            dataset_type (numpy.dtype, optional): The data type of the input data (default is numpy.float32).
            number_samples_per_class (Optional[Dict[str, int]], optional): A dictionary containing metadata
                about class distribution. It should include the key "number_classes" if provided.
            attention_embed_dim (int, optional): Embedding dimension for cross-attention (default: 128).
            attention_num_heads (int, optional): Number of attention heads (default: 4).

        Raises:
            ValueError: If any parameter validation fails.
        """

        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")

        if not isinstance(output_shape, int) or output_shape <= 0:
            raise ValueError("output_shape must be a positive integer.")

        if not isinstance(activation_function, str):
            raise ValueError("activation_function must be a string.")

        if not isinstance(initializer_mean, (float, int)):
            raise ValueError("initializer_mean must be a float or an integer.")

        if not isinstance(initializer_deviation, (float, int)) or initializer_deviation <= 0:
            raise ValueError("initializer_deviation must be a positive float or integer.")

        if not isinstance(dropout_decay_rate_d, (float, int)) or not (0 <= dropout_decay_rate_d <= 1):
            raise ValueError("dropout_decay_rate_d must be a float between 0 and 1.")

        if not isinstance(last_layer_activation, str):
            raise ValueError("last_layer_activation must be a string.")

        if not isinstance(dense_layer_sizes_d, list) or not all(
                isinstance(n, int) and n > 0 for n in dense_layer_sizes_d):
            raise ValueError("dense_layer_sizes_d must be a list of positive integers.")

        if number_samples_per_class is not None and not isinstance(number_samples_per_class, dict):
            raise ValueError("number_samples_per_class must be a dictionary if provided.")

        if not isinstance(attention_embed_dim, int) or attention_embed_dim <= 0:
            raise ValueError(
                f"Invalid value for attention_embed_dim: {attention_embed_dim}. It must be a positive integer.")

        if not isinstance(attention_num_heads, int) or attention_num_heads <= 0:
            raise ValueError(
                f"Invalid value for attention_num_heads: {attention_num_heads}. It must be a positive integer.")

        if attention_embed_dim % attention_num_heads != 0:
            raise ValueError(
                f"attention_embed_dim ({attention_embed_dim}) must be divisible by attention_num_heads ({attention_num_heads}).")

        self._discriminator_number_samples_per_class = number_samples_per_class
        self._discriminator_latent_dimension = latent_dimension
        self._discriminator_output_shape = output_shape
        self._discriminator_activation_function = activation_function
        self._discriminator_last_layer_activation = last_layer_activation
        self._discriminator_dropout_decay_rate_d = dropout_decay_rate_d
        self._discriminator_dense_layer_sizes_d = dense_layer_sizes_d
        self._discriminator_dataset_type = dataset_type
        self._discriminator_initializer_mean = initializer_mean
        self._discriminator_initializer_deviation = initializer_deviation
        self._discriminator_attention_embed_dim = attention_embed_dim
        self._discriminator_attention_num_heads = attention_num_heads
        self._discriminator_model_dense = None

    def get_discriminator(self) -> Model:
        """
        Build and return the complete discriminator model with cross-attention.

        This method constructs a neural network model using cross-attention to condition the
        input features on label information, where the input features act as Query and the
        label embedding provides Key and Value.

        Returns:
            Model: A Keras Model instance representing the discriminator with cross-attention.
        """
        # Weight initializer
        weight_initializer = RandomNormal(
            mean=self._discriminator_initializer_mean,
            stddev=self._discriminator_initializer_deviation
        )

        # Define the input layers
        discriminator_shape_input = Input(
            shape=(self._discriminator_output_shape,),
            dtype=self._discriminator_dataset_type,
            name='data_input'
        )
        label_input = Input(
            shape=(self._discriminator_number_samples_per_class["number_classes"],),
            dtype=self._discriminator_dataset_type,
            name='label_input'
        )

        # Label embedding layer
        label_embedded = Dense(
            self._discriminator_attention_embed_dim,
            activation='relu',
            kernel_initializer=weight_initializer,
            name='label_embedding'
        )(label_input)

        # Project input features to attention embedding dimension
        input_projected = Dense(
            self._discriminator_attention_embed_dim,
            kernel_initializer=weight_initializer,
            name='input_projection'
        )(discriminator_shape_input)

        # Cross-attention: input features query label information
        cross_attention = CrossAttentionLayer(
            embed_dim=self._discriminator_attention_embed_dim,
            num_heads=self._discriminator_attention_num_heads,
            name='cross_attention'
        )
        attended_features = cross_attention([input_projected, label_embedded])

        # Build the discriminator model with dense layers
        discriminator_model = Dense(
            self._discriminator_dense_layer_sizes_d[0],
            kernel_initializer=weight_initializer
        )(attended_features)
        discriminator_model = Dropout(self._discriminator_dropout_decay_rate_d)(discriminator_model)
        discriminator_model = self._add_activation_layer(
            discriminator_model,
            self._discriminator_activation_function
        )

        # Add additional dense layers with dropout and activations
        for layer_size in self._discriminator_dense_layer_sizes_d[1:]:
            discriminator_model = Dense(
                layer_size,
                kernel_initializer=weight_initializer
            )(discriminator_model)
            discriminator_model = Dropout(self._discriminator_dropout_decay_rate_d)(discriminator_model)
            discriminator_model = self._add_activation_layer(
                discriminator_model,
                self._discriminator_activation_function
            )

        # Final output layer with specified activation function
        discriminator_model = Dense(1, kernel_initializer=weight_initializer)(discriminator_model)
        validity = self._add_activation_layer(
            discriminator_model,
            self._discriminator_last_layer_activation
        )

        # Create the final model
        model = Model(
            inputs=[discriminator_shape_input, label_input],
            outputs=validity,
            name='Discriminator'
        )

        self._discriminator_model_dense = model

        return model

    def get_dense_discriminator_model(self) -> Optional[Model]:
        """
        Retrieve the dense discriminator model.

        Returns:
            Optional[Model]: The dense discriminator model, or None if not set.
        """
        return self._discriminator_model_dense

    @property
    def attention_embed_dim(self) -> int:
        """int: Gets the embedding dimension for cross-attention."""
        return self._discriminator_attention_embed_dim

    @property
    def attention_num_heads(self) -> int:
        """int: Gets the number of attention heads."""
        return self._discriminator_attention_num_heads

    def set_dropout_decay_rate_discriminator(self, dropout_decay_rate_discriminator: float):
        """
        Set the dropout decay rate for the discriminator network.

        Args:
            dropout_decay_rate_discriminator (float): The new dropout decay rate.

        Raises:
            ValueError: If the value is not between 0 and 1.
        """
        if not (0 <= dropout_decay_rate_discriminator <= 1):
            raise ValueError("dropout_decay_rate_discriminator must be a float between 0 and 1.")

        self._discriminator_dropout_decay_rate_d = dropout_decay_rate_discriminator

    def set_dense_layer_sizes_discriminator(self, dense_layer_sizes_discriminator: List[int]):
        """
        Set the sizes for the dense layers in the discriminator network.

        Args:
            dense_layer_sizes_discriminator (List[int]): A list of integers specifying the layer sizes.

        Raises:
            ValueError: If the list is empty or contains invalid values.
        """
        if not isinstance(dense_layer_sizes_discriminator, list) or not all(
                isinstance(n, int) and n > 0 for n in dense_layer_sizes_discriminator
        ):
            raise ValueError("dense_layer_sizes_discriminator must be a list of positive integers.")

        self._discriminator_dense_layer_sizes_d = dense_layer_sizes_discriminator