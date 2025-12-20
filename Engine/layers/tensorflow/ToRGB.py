from Engine.layers.tensorflow.NoiseInjection import NoiseInjection
from Engine.layers.tensorflow.StyleModulation import StyleModulation

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

class ToRGB(Layer):
    """Converts features to RGB-like output at each resolution"""

    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.conv = Conv1D(self.output_dim, 1, padding='same', name=f'{self.name}_conv')
        super().build(input_shape)

    def call(self, inputs):
        x, style = inputs
        return self.conv(x)