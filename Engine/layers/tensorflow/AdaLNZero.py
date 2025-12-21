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

class AdaLNZero(Layer):
    """Adaptive Layer Norm with Zero Initialization"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        x_shape, cond_shape = input_shape
        batch, seq_len, dim = x_shape
        return [
            (batch, seq_len, dim),
            (batch, 1, dim),
            (batch, 1, dim),
            (batch, 1, dim),
            (batch, 1, dim)
        ]

    def build(self, input_shape):
        features_shape, condition_shape = input_shape
        dim = features_shape[-1]

        self.ada_proj = Dense(
            dim * 6,
            kernel_initializer='zeros',
            bias_initializer='zeros'
        )
        self.norm = LayerNormalization(epsilon=1e-6)
        super().build(input_shape)

    def call(self, inputs):
        x, cond = inputs

        # Ensure same dtype
        x_dtype = x.dtype
        cond = tf.cast(cond, x_dtype)

        x_norm = self.norm(x)

        ada_params = self.ada_proj(cond)
        ada_params = tf.expand_dims(ada_params, axis=1)

        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = tf.split(
            ada_params, 6, axis=-1
        )

        x_modulated = x_norm * (1 + scale_attn) + shift_attn

        return x_modulated, gate_attn, shift_ffn, scale_ffn, gate_ffn
