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
    
class HybridConvTransformerBlock(Layer):
    """Hybrid Conv+Transformer with memory optimizations"""

    def __init__(self, dim, num_heads=4, num_kv_heads=1, dropout=0.1,
                 chunk_size=8, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dropout_rate = dropout
        self.chunk_size = chunk_size

    def compute_output_shape(self, input_shape):
        x_shape, cond_shape = input_shape
        return x_shape

    def build(self, input_shape):
        self.conv_block = ConvNextBlock(
            self.dim,
            mlp_ratio=2.5,
            dropout=self.dropout_rate
        )
        self.transformer_block = DiTBlock(
            self.dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            mlp_ratio=2.5,
            dropout=self.dropout_rate,
            chunk_size=self.chunk_size
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        x, cond = inputs
        x = self.conv_block(x, training=training)
        x = self.transformer_block([x, cond], training=training)
        return x