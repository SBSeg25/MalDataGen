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

class FourierTimeEmbedding(Layer):
    """Fourier Time Embeddings"""

    def __init__(self, dim, max_period=10000, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.max_period = max_period

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dim)

    def build(self, input_shape):
        half_dim = self.dim // 2
        freqs = np.exp(
            -math.log(self.max_period) *
            np.arange(0, half_dim, dtype=np.float32) / half_dim
        )
        self.freqs = self.add_weight(
            name='fourier_freqs',
            shape=(half_dim,),
            initializer=tf.keras.initializers.Constant(freqs),
            trainable=False,
            dtype=tf.float32
        )
        self.proj = Dense(self.dim)
        super().build(input_shape)

    def call(self, timesteps):
        # Force float32 for time embeddings to avoid mixed precision issues
        timesteps = tf.cast(timesteps, tf.float32)
        timesteps = tf.expand_dims(timesteps, -1)
        freqs = tf.cast(self.freqs, tf.float32)
        args = timesteps * freqs[None, :]
        embedding = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
        return self.proj(embedding)
