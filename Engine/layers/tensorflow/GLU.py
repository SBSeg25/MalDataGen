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

class GLU(Layer):
    """Gated Linear Unit - STABLE"""

    def call(self, x):
        half = tf.shape(x)[-1] // 2
        return x[..., :half] * tf.nn.sigmoid(x[..., half:])

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (input_shape[-1] // 2,)
