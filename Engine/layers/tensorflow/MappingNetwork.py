from Engine.architectures.adversarial.tensorflow.VanillaGeneratorTensorflow import RMSNorm

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

class MappingNetwork(Layer):
    """Maps latent z to intermediate latent w with better disentanglement"""

    def __init__(self, num_layers=8, hidden_dim=512, lr_multiplier=0.01, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lr_multiplier = lr_multiplier
        self.mapping_layers = []

    def build(self, input_shape):
        for i in range(self.num_layers):
            self.mapping_layers.append(
                Dense(
                    self.hidden_dim,
                    activation='linear',
                    kernel_initializer=tf.keras.initializers.RandomNormal(0, 1),
                    name=f'mapping_{i}'
                )
            )
        self.normalizer = RMSNorm(name='mapping_norm')
        super().build(input_shape)

    def call(self, inputs):
        # inputs: [z, y] (latent + labels)
        z, y = inputs
        x = Concatenate()([z, y])

        # Normalize input
        x = self.normalizer(x)

        # Pass through mapping layers
        for layer in self.mapping_layers:
            x = layer(x)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = x * self.lr_multiplier

        return x


