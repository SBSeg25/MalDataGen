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

class RotaryPositionEmbedding(Layer):
    """Rotary Position Embeddings - Cached Version"""

    def __init__(self, dim, max_seq_len=1024, **kwargs):  # Reduced default seq_len
        super().__init__(**kwargs)
        self.dim = dim
        self.max_seq_len = max_seq_len

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        half_dim = self.dim // 2
        inv_freq = 1.0 / (10000 ** (np.arange(0, self.dim, 2, dtype=np.float32) / self.dim))
        self.inv_freq = self.add_weight(
            name='rope_inv_freq',
            shape=(half_dim,),
            initializer=tf.keras.initializers.Constant(inv_freq),
            trainable=False,
            dtype=tf.float32
        )
        super().build(input_shape)

    def call(self, x):
        if self.dim < 2:
            return x

        seq_len = tf.shape(x)[1]

        # Force float32 for positional embeddings
        x_dtype = x.dtype
        t = tf.cast(tf.range(seq_len), tf.float32)
        inv_freq = tf.cast(self.inv_freq, tf.float32)

        freqs = tf.einsum('i,j->ij', t, inv_freq)
        freqs = tf.reshape(freqs, [1, seq_len, 1, -1])

        cos_emb = tf.cos(freqs)
        sin_emb = tf.sin(freqs)

        # Cast back to input dtype for computation
        cos_emb = tf.cast(cos_emb, x_dtype)
        sin_emb = tf.cast(sin_emb, x_dtype)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_even_rot = x_even * cos_emb - x_odd * sin_emb
        x_odd_rot = x_odd * cos_emb + x_even * sin_emb

        x_rotated = tf.stack([x_even_rot, x_odd_rot], axis=-1)
        x_rotated = tf.reshape(x_rotated, tf.shape(x))

        return x_rotated
