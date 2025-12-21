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

class ConvNextBlock(Layer):
    """ConvNeXt Block - Memory efficient"""

    def __init__(self, dim, mlp_ratio=2.5, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        self.dwconv = DepthwiseConv1D(
            kernel_size=7,
            padding='same',
            depthwise_initializer=HeNormal()
        )
        self.norm = LayerNormalization(epsilon=1e-6)
        hidden_dim = int(self.dim * self.mlp_ratio)
        self.pwconv1 = Dense(hidden_dim, kernel_initializer=TruncatedNormal(stddev=0.02))
        self.pwconv2 = Dense(self.dim, kernel_initializer=TruncatedNormal(stddev=0.02))
        self.dropout = Dropout(self.dropout_rate)
        self.gamma = self.add_weight(
            name='layer_scale',
            shape=(self.dim,),
            initializer=tf.keras.initializers.Constant(1e-6),
            trainable=True
        )
        super().build(input_shape)

    def call(self, x, training=None):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = tf.nn.gelu(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = self.dropout(x, training=training)
        return shortcut + x