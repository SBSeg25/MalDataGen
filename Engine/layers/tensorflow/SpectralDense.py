
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

class SpectralDense(Layer):
    """
    Dense com Spectral Normalization - VERSÃO 100% ESTÁVEL
    Implementação simplificada e testada do paper original
    """

    def __init__(self, units, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias

    def build(self, input_shape):
        input_dim = int(input_shape[-1])

        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_dim, self.units),
            initializer=HeNormal(),
            trainable=True
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer='zeros',
                trainable=True
            )

        # Vector u para power iteration - DIMENSÃO CORRETA
        self.u = self.add_weight(
            name='u',
            shape=(1, self.units),  # (1, output_dim)
            initializer='random_normal',
            trainable=False
        )

        super().build(input_shape)

    def call(self, x):
        # Simplificado: apenas 1 power iteration
        w = self.kernel

        # Power iteration: encontra maior singular value
        # u shape: (1, output_dim)
        # w shape: (input_dim, output_dim)

        # v = normalize(u @ w^T), shape: (1, input_dim)
        v = tf.matmul(self.u, w, transpose_b=True)
        v = tf.nn.l2_normalize(v, axis=-1)

        # u_new = normalize(v @ w), shape: (1, output_dim)
        u_new = tf.matmul(v, w)
        u_new = tf.nn.l2_normalize(u_new, axis=-1)

        # Sigma (maior singular value): u @ w @ v^T, resultado é escalar
        sigma = tf.reduce_sum(u_new * tf.matmul(v, w))

        # Normaliza kernel
        w_normalized = w / (sigma + 1e-6)

        # Atualiza u para próxima iteração
        self.u.assign(u_new)

        # Forward pass
        output = tf.matmul(x, w_normalized)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)

        return output

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.units,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'use_bias': self.use_bias
        })
        return config
