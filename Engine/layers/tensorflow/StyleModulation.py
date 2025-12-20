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
    
class StyleModulation(Layer):
    """Adaptive Instance Normalization (AdaIN) for style injection"""

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        # Style affine transformation
        self.dense_scale = Dense(self.filters, name=f'{self.name}_scale')
        self.dense_bias = Dense(self.filters, name=f'{self.name}_bias')
        super().build(input_shape)

    def call(self, inputs):
        x, style = inputs  # x: features, style: w vector

        # Instance normalization
        mean = tf.reduce_mean(x, axis=[1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[1], keepdims=True)
        normalized = (x - mean) / tf.sqrt(variance + 1e-8)

        # Style modulation
        scale = self.dense_scale(style)
        bias = self.dense_bias(style)

        # Expand dimensions for broadcasting
        scale = tf.expand_dims(scale, axis=1)
        bias = tf.expand_dims(bias, axis=1)

        return normalized * (1.0 + scale) + bias