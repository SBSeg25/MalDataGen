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

class FiLM(Layer):
    """Feature-wise Linear Modulation - BATCH SAFE"""

    def build(self, input_shape):
        features_shape, condition_shape = input_shape
        channels = int(features_shape[-1])

        self.gamma_dense = Dense(channels, kernel_initializer='zeros')
        self.beta_dense = Dense(channels, kernel_initializer='zeros')
        super().build(input_shape)

    def call(self, inputs):
        features, condition = inputs

        # features: (Bf, T, C)
        # condition: (Bc, num_classes)

        bf = tf.shape(features)[0]
        bc = tf.shape(condition)[0]

        # Repetição segura do condition para alinhar batch
        condition = tf.cond(
            tf.not_equal(bf, bc),
            lambda: tf.repeat(condition, repeats=bf // bc, axis=0),
            lambda: condition
        )

        gamma = self.gamma_dense(condition)
        beta = self.beta_dense(condition)

        gamma = tf.expand_dims(gamma, axis=1)  # (Bf, 1, C)
        beta = tf.expand_dims(beta, axis=1)

        return features * (1.0 + gamma) + beta
