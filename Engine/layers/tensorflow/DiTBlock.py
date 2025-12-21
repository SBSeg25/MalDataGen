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

class DiTBlock(Layer):
    """Diffusion Transformer Block with gradient checkpointing"""

    def __init__(self, d_model, num_heads=4, num_kv_heads=1, mlp_ratio=2.5,
                 dropout=0.1, chunk_size=8, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout
        self.chunk_size = chunk_size

    def compute_output_shape(self, input_shape):
        x_shape, cond_shape = input_shape
        return x_shape

    def build(self, input_shape):
        self.adaln = AdaLNZero()
        self.attn = GroupedQueryAttention(
            self.d_model,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            dropout=self.dropout_rate,
            chunk_size=self.chunk_size
        )
        self.ffn = SwiGLU(
            self.d_model,
            hidden_dim=int(self.d_model * self.mlp_ratio)
        )
        self.norm_ffn = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, inputs, training=None):
        x, cond = inputs

        # Attention block (gradient checkpointing disabled for graph compatibility)
        x_mod, gate_attn, shift_ffn, scale_ffn, gate_ffn = self.adaln([x, cond])
        attn_out = self.attn(x_mod, training=training)
        attn_out = self.dropout(attn_out, training=training)
        x = x + gate_attn * attn_out

        # FFN block
        x_norm = self.norm_ffn(x)
        x_mod_ffn = x_norm * (1 + scale_ffn) + shift_ffn
        ffn_out = self.ffn(x_mod_ffn)
        ffn_out = self.dropout(ffn_out, training=training)
        x = x + gate_ffn * ffn_out

        return x
