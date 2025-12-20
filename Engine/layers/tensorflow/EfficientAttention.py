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

class EfficientAttention(Layer):
    """
    Efficient Multi-Head Attention - ULTRA STABLE
    Versão simplificada sem bugs de dimensão
    """

    def __init__(self, num_heads=4, head_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_model = num_heads * head_dim

        # Calculate scale in __init__ using Python math (not TensorFlow)
        import math
        self.scale_value = 1.0 / math.sqrt(float(head_dim))

    def build(self, input_shape):
        d_input = int(input_shape[-1])

        # Projeções QKV
        self.wq = Dense(self.d_model, use_bias=False)
        self.wk = Dense(self.d_model, use_bias=False)
        self.wv = Dense(self.d_model, use_bias=False)

        # Output projection
        self.wo = Dense(d_input, use_bias=False)

        super().build(input_shape)

    def call(self, x):
        batch = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # Project to Q, K, V
        q = self.wq(x)  # (batch, seq, d_model)
        k = self.wk(x)
        v = self.wv(x)

        # Reshape to multi-head: (batch, seq, heads, head_dim)
        q = tf.reshape(q, [batch, seq_len, self.num_heads, self.head_dim])
        k = tf.reshape(k, [batch, seq_len, self.num_heads, self.head_dim])
        v = tf.reshape(v, [batch, seq_len, self.num_heads, self.head_dim])

        # Transpose: (batch, heads, seq, head_dim)
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        # Attention: Q @ K^T / sqrt(d_k) - use pre-calculated scale
        scores = tf.matmul(q, k, transpose_b=True) * self.scale_value
        weights = tf.nn.softmax(scores, axis=-1)

        # Weighted sum: Attention @ V
        attended = tf.matmul(weights, v)

        # Reshape back: (batch, seq, d_model)
        attended = tf.transpose(attended, [0, 2, 1, 3])
        attended = tf.reshape(attended, [batch, seq_len, self.d_model])

        # Output projection
        output = self.wo(attended)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            **super().get_config(),
            'num_heads': self.num_heads,
            'head_dim': self.head_dim
        }
