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

class SqueezeExcitation(Layer):
    """Squeeze-and-Excitation Block - STABLE"""

    def __init__(self, ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = int(input_shape[-1])
        reduced = max(channels // self.ratio, 8)

        self.pool = GlobalAveragePooling1D()
        self.fc1 = Dense(reduced, activation='relu')
        self.fc2 = Dense(channels, activation='sigmoid')
        super().build(input_shape)

    def call(self, x):
        scale = self.pool(x)
        scale = self.fc1(scale)
        scale = self.fc2(scale)
        scale = tf.expand_dims(scale, axis=1)
        return x * scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), 'ratio': self.ratio}
