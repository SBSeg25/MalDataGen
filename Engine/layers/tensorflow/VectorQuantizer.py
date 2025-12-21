import math

from Engine.layers.tensorflow.RMSNorm import RMSNorm

try:
    import sys
    import numpy as np
    import tensorflow as tf
    from typing import Dict, List, Optional, Callable, Tuple
    from tensorflow.keras.layers import (
        Layer, Dense, Input, Dropout, Concatenate, Add,
        Lambda, LayerNormalization, Embedding, Reshape,
        Conv1D, DepthwiseConv1D, GlobalAveragePooling1D,
        Multiply, Activation, BatchNormalization
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.initializers import GlorotUniform, Orthogonal, HeNormal
    from tensorflow.keras import backend as K
    import tensorflow
    from Engine.activations.Activations import Activations

except ImportError as error:
    print(error)
    sys.exit(-1)

class VectorQuantizer(Layer):
    """VQ with reduced codebook for memory"""

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, **kwargs):
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

    def compute_output_shape(self, input_shape):
        batch, seq_len, dim = input_shape
        return [
            (batch, seq_len, dim),
            (batch * seq_len,)
        ]

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name='codebook',
            shape=(self.num_embeddings, self.embedding_dim),
            initializer='uniform',
            trainable=True
        )
        super().build(input_shape)

    def call(self, x, training=None):
        x_dtype = x.dtype
        flat_x = tf.reshape(x, [-1, self.embedding_dim])

        # Cast to float32 for distance computation (more stable)
        flat_x_f32 = tf.cast(flat_x, tf.float32)
        embeddings_f32 = tf.cast(self.embeddings, tf.float32)

        distances = (
                tf.reduce_sum(flat_x_f32 ** 2, axis=1, keepdims=True) +
                tf.reduce_sum(embeddings_f32 ** 2, axis=1) -
                2 * tf.matmul(flat_x_f32, embeddings_f32, transpose_b=True)
        )

        encoding_indices = tf.argmin(distances, axis=1)
        quantized_flat = tf.nn.embedding_lookup(self.embeddings, encoding_indices)

        # Cast back to original dtype
        quantized_flat = tf.cast(quantized_flat, x_dtype)
        quantized = tf.reshape(quantized_flat, tf.shape(x))
        quantized = x + tf.stop_gradient(quantized - x)

        if training:
            e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
            q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
            self.add_loss(q_latent_loss + self.commitment_cost * e_latent_loss)

        return quantized, encoding_indices
