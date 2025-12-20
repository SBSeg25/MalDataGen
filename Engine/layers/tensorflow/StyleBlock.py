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

class StyleBlock(Layer):
    """
    StyleGAN synthesis block:
    Conv → Noise → AdaIN (Style Modulation) → Activation
    """

    def __init__(self, filters, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = Conv1D(
            self.filters,
            self.kernel_size,
            padding='same',
            activation=None,
            name=f'{self.name}_conv'
        )
        self.noise = NoiseInjection(name=f'{self.name}_noise')
        self.style_mod = StyleModulation(self.filters, name=f'{self.name}_adain')
        self.activation = tf.keras.layers.LeakyReLU(0.2, name=f'{self.name}_act')
        super().build(input_shape)

    def call(self, inputs, training=None):
        x, style = inputs  # x: features, style: w vector

        x = self.conv(x)
        x = self.noise(x, training=training)
        x = self.style_mod([x, style])
        x = self.activation(x)

        return x