from Engine.architectures.adversarial.tensorflow.VanillaGeneratorTensorflow import DepthwiseSeparableConv, \
    SqueezeExcitation, EfficientAttention, GLU, RMSNorm

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

class ConvTransformerBlock(Layer):
    """Hybrid Conv-Transformer Block - ULTRA STABLE"""

    def __init__(self, filters, num_heads=4, head_dim=32, ff_ratio=2.0, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ff_ratio = ff_ratio
        self.dropout_rate = dropout

    def build(self, input_shape):
        # Conv path
        self.conv = DepthwiseSeparableConv(self.filters, kernel_size=3)
        self.se = SqueezeExcitation(ratio=4)

        # Attention path
        self.attn = EfficientAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim
        )

        # FFN with GLU
        ff_dim = int(self.filters * self.ff_ratio)
        self.ff1 = Dense(ff_dim * 2)
        self.glu = GLU()
        self.ff2 = Dense(self.filters)

        # Norms
        self.norm1 = RMSNorm()
        self.norm2 = RMSNorm()
        self.norm3 = RMSNorm()

        # Dropouts
        self.drop1 = Dropout(self.dropout_rate)
        self.drop2 = Dropout(self.dropout_rate)
        self.drop3 = Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, x, training=None):
        # 1. Conv path with residual
        conv_out = self.conv(x)
        conv_out = self.se(conv_out)
        conv_out = self.drop1(conv_out, training=training)
        x = self.norm1(x + conv_out)

        # 2. Attention path with residual
        attn_out = self.attn(x)
        attn_out = self.drop2(attn_out, training=training)
        x = self.norm2(x + attn_out)

        # 3. FFN with GLU and residual
        ff_out = self.ff1(x)
        ff_out = self.glu(ff_out)
        ff_out = self.ff2(ff_out)
        ff_out = self.drop3(ff_out, training=training)
        x = self.norm3(x + ff_out)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            **super().get_config(),
            'filters': self.filters,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'ff_ratio': self.ff_ratio,
            'dropout_rate': self.dropout_rate
        }
