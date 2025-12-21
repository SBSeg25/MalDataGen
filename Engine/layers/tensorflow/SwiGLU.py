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

class SwiGLU(Layer):
    """SwiGLU with reduced hidden dim for memory"""

    def __init__(self, dim, hidden_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        # Reduced multiplier: 2.5x instead of 8/3x (2.67x)
        self.hidden_dim = hidden_dim or int(dim * 2.5)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.dim)

    def build(self, input_shape):
        self.w1 = Dense(self.hidden_dim, use_bias=False, kernel_initializer=HeNormal())
        self.w2 = Dense(self.hidden_dim, use_bias=False, kernel_initializer=HeNormal())
        self.w3 = Dense(self.dim, use_bias=False, kernel_initializer=HeNormal())
        super().build(input_shape)

    def call(self, x):
        gate = tf.nn.swish(self.w1(x))
        value = self.w2(x)
        return self.w3(gate * value)