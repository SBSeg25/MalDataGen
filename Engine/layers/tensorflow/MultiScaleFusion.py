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


class MultiScaleFusion(Layer):
    """Multi-Scale Feature Fusion - BATCH SAFE"""

    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.num_scales = len(input_shape)

        self.projections = [
            Dense(self.output_dim, name=f'proj_{i}')
            for i in range(self.num_scales)
        ]

        self.scale_weights = self.add_weight(
            name='fusion_weights',
            shape=(self.num_scales,),
            initializer='ones',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        weights = tf.nn.softmax(self.scale_weights)

        # referência de batch: menor batch (batch real)
        base_batch = tf.reduce_min([tf.shape(x)[0] for x in inputs])

        fused = []

        for i, (x, proj) in enumerate(zip(inputs, self.projections)):
            # x: (Bi, Fi)

            bi = tf.shape(x)[0]

            # Agrupa de volta para base_batch
            x = tf.reshape(x, [base_batch, -1, x.shape[-1]])
            x = tf.reduce_mean(x, axis=1)  # pooling seguro

            x = proj(x)
            fused.append(x * weights[i])

        return tf.add_n(fused)
