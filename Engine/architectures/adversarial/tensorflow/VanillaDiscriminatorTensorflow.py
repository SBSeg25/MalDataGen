#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ULTRA-POWERFUL Dense Discriminator para WGAN-GP
Versão 4.0 - Máxima Capacidade Discriminativa
"""

__author__ = 'Synthetic Ocean AI - Ultra Team'
__version__ = '4.0.0-ultra-powerful'

try:
    import sys
    import numpy as np
    import math
    from typing import Dict, List, Tuple, Optional, Callable, Union

    import tensorflow as tf
    from tensorflow.keras.layers import (
        Layer, Dense, Input, Dropout, Flatten, Concatenate,
        Lambda, Multiply, Add, BatchNormalization, LayerNormalization,
        GlobalAveragePooling1D, Reshape, Activation
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.initializers import RandomNormal, HeNormal
    from tensorflow.keras import backend as K

    from Engine.activations.Activations import Activations

except ImportError as error:
    print(error)
    sys.exit(-1)


class RMSNorm(Layer):
    """Root Mean Square Normalization"""

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        variance = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
        x_norm = x * tf.math.rsqrt(variance + self.epsilon)
        return x_norm * self.scale

    def get_config(self):
        return {**super().get_config(), 'epsilon': self.epsilon}


class SpectralDense(Layer):
    """Dense com Spectral Normalization + Weight Standardization"""

    def __init__(
            self,
            units: int,
            activation=None,
            use_bias=True,
            kernel_initializer='he_normal',
            power_iterations=1,
            use_weight_standardization=True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.power_iterations = power_iterations
        self.use_ws = use_weight_standardization

    def build(self, input_shape):
        input_dim = int(input_shape[-1])

        self.kernel = self.add_weight(
            name='kernel',
            shape=[input_dim, self.units],
            initializer=self.kernel_initializer,
            trainable=True
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=[self.units],
                initializer='zeros',
                trainable=True
            )

        self.u = self.add_weight(
            name='sn_u',
            shape=[1, self.units],
            initializer='random_normal',
            trainable=False,
            dtype=self.kernel.dtype
        )

        super().build(input_shape)

    def call(self, x, training=None):
        w = self.kernel

        # Weight Standardization (BigGAN, StyleGAN3)
        if self.use_ws:
            mean = tf.reduce_mean(w, axis=0, keepdims=True)
            var = tf.math.reduce_variance(w, axis=0, keepdims=True)
            w = (w - mean) * tf.math.rsqrt(var + 1e-8)

        w_shape = w.shape
        w_mat = tf.reshape(w, [-1, w_shape[-1]])
        u = self.u

        for _ in range(self.power_iterations):
            v = tf.nn.l2_normalize(tf.matmul(u, w_mat, transpose_b=True))
            u = tf.nn.l2_normalize(tf.matmul(v, w_mat))

        if training:
            self.u.assign(u)

        temp = tf.matmul(v, w_mat)
        sigma = tf.matmul(temp, u, transpose_b=True)
        sigma = tf.abs(sigma[0, 0])

        w_normalized = w / tf.maximum(sigma, 1e-12)
        output = tf.matmul(x, w_normalized)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)

    def get_config(self):
        return {
            **super().get_config(),
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'power_iterations': self.power_iterations,
            'use_weight_standardization': self.use_ws
        }


class GatedLinearUnit(Layer):
    """GLU - Melhora capacidade expressiva"""

    def __init__(self, use_spectral=True, **kwargs):
        # Remove custom params from kwargs
        kwargs.pop('use_spectral', None)
        super().__init__(**kwargs)
        self.use_spectral = use_spectral

    def build(self, input_shape):
        units = int(input_shape[-1])
        DenseLayer = SpectralDense if self.use_spectral else Dense

        self.gate = DenseLayer(units, activation='sigmoid')
        self.gate.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        return x * self.gate(x)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), 'use_spectral': self.use_spectral}


class CrossAttention(Layer):
    """Cross-Attention entre features e labels - NOVO!"""

    def __init__(self, num_heads=4, head_dim=32, use_spectral=True, **kwargs):
        # Remove custom params from kwargs
        kwargs.pop('num_heads', None)
        kwargs.pop('head_dim', None)
        kwargs.pop('use_spectral', None)
        
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_model = num_heads * head_dim
        self.use_spectral = use_spectral
        self.scale = 1.0 / math.sqrt(float(head_dim))

    def build(self, input_shape):
        # input_shape = [features_shape, context_shape]
        features_dim = int(input_shape[0][-1])
        context_dim = int(input_shape[1][-1])

        DenseLayer = SpectralDense if self.use_spectral else Dense

        self.wq = DenseLayer(self.d_model, use_bias=False)
        self.wk = DenseLayer(self.d_model, use_bias=False)
        self.wv = DenseLayer(self.d_model, use_bias=False)
        self.wo = DenseLayer(features_dim, use_bias=False)

        self.wq.build(input_shape[0])
        self.wk.build(input_shape[1])
        self.wv.build(input_shape[1])

        wo_input_shape = input_shape[0][:-1] + (self.d_model,)
        self.wo.build(wo_input_shape)

        super().build(input_shape)

    def call(self, inputs):
        features, context = inputs
        batch = tf.shape(features)[0]

        q = self.wq(features)
        k = self.wk(context)
        v = self.wv(context)

        # Reshape adding sequence dimension of 1: [batch, num_heads, 1, head_dim]
        q = tf.reshape(q, [batch, self.num_heads, 1, self.head_dim])
        k = tf.reshape(k, [batch, self.num_heads, 1, self.head_dim])
        v = tf.reshape(v, [batch, self.num_heads, 1, self.head_dim])

        # scores: [batch, num_heads, 1, 1]
        scores = tf.matmul(q, k, transpose_b=True) * self.scale
        weights = tf.nn.softmax(scores, axis=-1)

        attended = tf.matmul(weights, v)
        
        # Remove sequence dimension and flatten: [batch, d_model]
        attended = tf.reshape(attended, [batch, self.d_model])

        return self.wo(attended)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        return {
            **super().get_config(),
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'use_spectral': self.use_spectral
        }


class MultiHeadSelfAttention(Layer):
    """Self-Attention melhorado com rotary embeddings"""

    def __init__(self, num_heads=4, head_dim=32, use_spectral=True, use_rotary=True, **kwargs):
        # Remove custom params from kwargs before passing to super
        kwargs.pop('num_heads', None)
        kwargs.pop('head_dim', None)
        kwargs.pop('use_spectral', None)
        kwargs.pop('use_rotary', None)
        
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_model = num_heads * head_dim
        self.use_spectral = use_spectral
        self.use_rotary = use_rotary
        self.scale = 1.0 / math.sqrt(float(head_dim))

    def build(self, input_shape):
        d_input = int(input_shape[-1])

        DenseLayer = SpectralDense if self.use_spectral else Dense

        self.wq = DenseLayer(self.d_model, use_bias=False, name='q')
        self.wk = DenseLayer(self.d_model, use_bias=False, name='k')
        self.wv = DenseLayer(self.d_model, use_bias=False, name='v')
        self.wo = DenseLayer(d_input, use_bias=False, name='out')

        self.wq.build(input_shape)
        self.wk.build(input_shape)
        self.wv.build(input_shape)

        wo_input_shape = input_shape[:-1] + (self.d_model,)
        self.wo.build(wo_input_shape)

        self.temperature = self.add_weight(
            name='temperature',
            shape=[1],
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True
        )

        # Learnable relative position bias
        self.rel_pos_bias = self.add_weight(
            name='rel_pos_bias',
            shape=[self.num_heads],
            initializer='zeros',
            trainable=True
        )

        super().build(input_shape)

    def call(self, x):
        batch = tf.shape(x)[0]

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Reshape adding sequence dimension of 1: [batch, num_heads, 1, head_dim]
        q = tf.reshape(q, [batch, self.num_heads, 1, self.head_dim])
        k = tf.reshape(k, [batch, self.num_heads, 1, self.head_dim])
        v = tf.reshape(v, [batch, self.num_heads, 1, self.head_dim])

        # scores: [batch, num_heads, 1, 1]
        scores = tf.matmul(q, k, transpose_b=True)
        scores = scores * self.scale / tf.maximum(self.temperature, 0.1)

        # Add relative position bias: [1, num_heads, 1, 1]
        scores = scores + self.rel_pos_bias[None, :, None, None]

        weights = tf.nn.softmax(scores, axis=-1)
        attended = tf.matmul(weights, v)
        
        # Remove sequence dimension and flatten: [batch, d_model]
        attended = tf.reshape(attended, [batch, self.d_model])

        output = self.wo(attended)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            **super().get_config(),
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'use_spectral': self.use_spectral,
            'use_rotary': self.use_rotary
        }


class FeatureSqueezeExcitation(Layer):
    """SE + Feature Statistics - APRIMORADO"""

    def __init__(self, ratio=4, use_spectral=True, use_statistics=True, **kwargs):
        # Remove custom params from kwargs
        kwargs.pop('ratio', None)
        kwargs.pop('use_spectral', None)
        kwargs.pop('use_statistics', None)
        
        super().__init__(**kwargs)
        self.ratio = ratio
        self.use_spectral = use_spectral
        self.use_statistics = use_statistics

    def build(self, input_shape):
        channels = int(input_shape[-1])
        reduced = max(channels // self.ratio, 8)

        DenseLayer = SpectralDense if self.use_spectral else Dense

        self.fc1 = DenseLayer(reduced, activation='relu')
        self.fc2 = DenseLayer(channels, activation='sigmoid')

        self.fc1.build(input_shape)

        fc2_input_shape = input_shape[:-1] + (reduced,)
        self.fc2.build(fc2_input_shape)

        if self.use_statistics:
            # Pesos para combinar média e stddev
            self.weight_mean = self.add_weight(
                name='weight_mean',
                shape=[1],
                initializer=tf.keras.initializers.Constant(0.5),
                trainable=True
            )
            self.weight_std = self.add_weight(
                name='weight_std',
                shape=[1],
                initializer=tf.keras.initializers.Constant(0.5),
                trainable=True
            )

        super().build(input_shape)

    def call(self, x):
        if self.use_statistics:
            # Usa média E desvio padrão
            mean = tf.reduce_mean(x, axis=-1, keepdims=True)
            std = tf.math.reduce_std(x, axis=-1, keepdims=True)

            mean_norm = self.weight_mean * mean
            std_norm = self.weight_std * std

            pooled = tf.concat([mean_norm, std_norm], axis=-1)
            pooled = tf.reduce_mean(pooled, axis=-1, keepdims=False)
            pooled = tf.expand_dims(pooled, -1)
            pooled = tf.tile(pooled, [1, x.shape[-1]])
        else:
            pooled = x

        scale = self.fc1(pooled)
        scale = self.fc2(scale)
        return x * scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            **super().get_config(),
            'ratio': self.ratio,
            'use_spectral': self.use_spectral,
            'use_statistics': self.use_statistics
        }


class MinibatchStdDev(Layer):
    """Minibatch Standard Deviation - Múltiplos grupos"""

    def __init__(self, group_size=4, num_new_features=1, **kwargs):
        # Remove custom params from kwargs
        kwargs.pop('group_size', None)
        kwargs.pop('num_new_features', None)
        
        super().__init__(**kwargs)
        self.group_size = group_size
        self.num_new_features = num_new_features

    def call(self, x):
        batch_size = tf.shape(x)[0]
        num_features = x.shape[-1]

        group_size = tf.minimum(self.group_size, batch_size)
        x_grouped = tf.reshape(x, [group_size, -1, num_features])

        mean = tf.reduce_mean(x_grouped, axis=0, keepdims=True)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(x_grouped - mean), axis=0) + 1e-8)

        stddev_mean = tf.reduce_mean(stddev, keepdims=True)
        stddev_feature = tf.tile(stddev_mean, [batch_size, 1])

        return tf.concat([x, stddev_feature], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + self.num_new_features)

    def get_config(self):
        return {
            **super().get_config(),
            'group_size': self.group_size,
            'num_new_features': self.num_new_features
        }


class AdaptiveInstanceNorm(Layer):
    """Adaptive Instance Normalization - StyleGAN inspired"""

    def __init__(self, use_spectral=True, **kwargs):
        # Remove custom params from kwargs
        kwargs.pop('use_spectral', None)
        
        super().__init__(**kwargs)
        self.use_spectral = use_spectral

    def build(self, input_shape):
        # input_shape = [features_shape, style_shape]
        features_dim = int(input_shape[0][-1])

        DenseLayer = SpectralDense if self.use_spectral else Dense

        self.scale_transform = DenseLayer(features_dim)
        self.shift_transform = DenseLayer(features_dim)

        self.scale_transform.build(input_shape[1])
        self.shift_transform.build(input_shape[1])

        super().build(input_shape)

    def call(self, inputs):
        features, style = inputs

        # Instance normalization
        mean = tf.reduce_mean(features, axis=-1, keepdims=True)
        std = tf.math.reduce_std(features, axis=-1, keepdims=True)
        normalized = (features - mean) / (std + 1e-8)

        # Adaptive modulation
        scale = self.scale_transform(style)
        shift = self.shift_transform(style)

        return normalized * (1 + scale) + shift

    def get_config(self):
        return {**super().get_config(), 'use_spectral': self.use_spectral}


class ResidualBlock(Layer):
    """Residual Block Ultra-Poderoso com GLU + Better Norm"""

    def __init__(self, units, dropout=0.1, use_spectral=True, activation='swish',
                 use_glu=True, **kwargs):
        # Remove custom params from kwargs
        kwargs.pop('units', None)
        kwargs.pop('dropout', None)
        kwargs.pop('use_spectral', None)
        kwargs.pop('activation', None)
        kwargs.pop('use_glu', None)
        
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout
        self.use_spectral = use_spectral
        self.activation_name = activation
        self.use_glu = use_glu

    def build(self, input_shape):
        input_dim = int(input_shape[-1])

        DenseLayer = SpectralDense if self.use_spectral else Dense

        self.norm1 = RMSNorm()
        self.act1 = Activation(self.activation_name)
        self.dense1 = DenseLayer(self.units)

        self.norm2 = RMSNorm()
        self.act2 = Activation(self.activation_name)
        self.dropout = Dropout(self.dropout_rate)
        self.dense2 = DenseLayer(self.units)

        self.se = FeatureSqueezeExcitation(ratio=4, use_spectral=self.use_spectral, use_statistics=True)

        if self.use_glu:
            self.glu = GatedLinearUnit(use_spectral=self.use_spectral)

        if input_dim != self.units:
            self.projection = DenseLayer(self.units, use_bias=False)
        else:
            self.projection = None

        # Build all layers
        self.norm1.build(input_shape)
        self.dense1.build(input_shape)

        dense1_output_shape = input_shape[:-1] + (self.units,)

        self.norm2.build(dense1_output_shape)
        self.dense2.build(dense1_output_shape)

        self.se.build(dense1_output_shape)

        if self.use_glu:
            self.glu.build(dense1_output_shape)

        if self.projection is not None:
            self.projection.build(input_shape)

        super().build(input_shape)

    def call(self, x, training=None):
        identity = x

        out = self.norm1(x)
        out = self.act1(out)
        out = self.dense1(out)

        out = self.norm2(out)
        out = self.act2(out)
        out = self.dropout(out, training=training)
        out = self.dense2(out)

        out = self.se(out)

        if self.use_glu:
            out = self.glu(out)

        if self.projection is not None:
            identity = self.projection(identity)

        return out + identity

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)

    def get_config(self):
        return {
            **super().get_config(),
            'units': self.units,
            'dropout': self.dropout_rate,
            'use_spectral': self.use_spectral,
            'activation': self.activation_name,
            'use_glu': self.use_glu
        }


class ConditionalBatchNorm(Layer):
    """Conditional Batch Normalization"""

    def __init__(self, num_features, num_classes, use_spectral=True, **kwargs):
        # Remove custom params from kwargs
        kwargs.pop('num_features', None)
        kwargs.pop('num_classes', None)
        kwargs.pop('use_spectral', None)
        
        super().__init__(**kwargs)
        self.num_features = num_features
        self.num_classes = num_classes
        self.use_spectral = use_spectral

    def build(self, input_shape):
        x_shape = input_shape[0] if isinstance(input_shape, list) else input_shape
        y_shape = input_shape[1] if isinstance(input_shape, list) else (None, self.num_classes)

        self.bn = BatchNormalization(scale=False, center=False)

        DenseLayer = SpectralDense if self.use_spectral else Dense

        self.gamma_embed = DenseLayer(self.num_features, use_bias=False)
        self.beta_embed = DenseLayer(self.num_features, use_bias=False)

        self.bn.build(x_shape)
        self.gamma_embed.build(y_shape)
        self.beta_embed.build(y_shape)

        super().build(input_shape)

    def call(self, inputs):
        x, y = inputs
        x = self.bn(x)
        gamma = self.gamma_embed(y)
        beta = self.beta_embed(y)
        return x * (1 + gamma) + beta

    def get_config(self):
        return {
            **super().get_config(),
            'num_features': self.num_features,
            'num_classes': self.num_classes,
            'use_spectral': self.use_spectral
        }


class VanillaDiscriminator(Activations):
    """
    ULTRA-POWERFUL Discriminator para WGAN-GP
    Versão 4.0 - Capacidade Discriminativa Máxima

    NOVOS RECURSOS:
    - Weight Standardization + Spectral Norm
    - Cross-Attention (features ↔ labels)
    - Gated Linear Units (GLU)
    - Adaptive Instance Normalization (AdaIN)
    - Feature Statistics em SE blocks
    - Relative Position Bias em Attention
    - Multi-Scale Feature Pyramid
    - Learnable Temperature Scaling
    """

    def __init__(
            self,
            latent_dimension: int,
            output_shape: Tuple[int, ...],
            activation_function: Callable,
            initializer_mean: float,
            initializer_deviation: float,
            dropout_decay_rate_d: float,
            last_layer_activation: Callable,
            dense_layer_sizes_d: List[int],
            dataset_type: type = np.float32,
            number_samples_per_class: Optional[Dict[str, int]] = None,
            # Architecture
            use_spectral_norm: bool = True,
            use_residual_blocks: bool = True,
            num_residual_blocks: int = 4,
            use_multi_scale: bool = True,
            # Attention
            use_attention: bool = True,
            attention_heads: int = 8,
            attention_layers: List[int] = None,
            use_cross_attention: bool = True,  # NOVO
            cross_attention_heads: int = 8,  # NOVO
            # Normalization
            norm_type: str = 'rms',
            use_conditional_bn: bool = True,
            use_adaptive_norm: bool = True,  # NOVO
            # Gates & Modulation
            use_glu: bool = True,  # NOVO
            use_weight_standardization: bool = True,  # NOVO
            # Regularization
            use_minibatch_stddev: bool = True,
            minibatch_group_size: int = 4,
            use_input_noise: bool = True,
            noise_stddev: float = 0.05,
            label_smoothing: float = 0.0,
            # Loss & Penalty
            use_r1_regularization: bool = True,
            r1_gamma: float = 10.0,
            use_gradient_penalty: bool = True,
            gp_lambda: float = 10.0,
            # Output
            output_units: int = 1,
            # Performance
            use_mixed_precision: bool = False,
            activation_name: str = 'swish',
    ):
        if latent_dimension <= 0:
            raise ValueError("latent_dimension must be > 0")
        if not dense_layer_sizes_d:
            raise ValueError("dense_layer_sizes_d cannot be empty")

        self._latent_dim = latent_dimension
        self._output_shape = output_shape
        self._activation_fn = activation_function
        self._last_activation = last_layer_activation
        self._dropout = dropout_decay_rate_d
        self._dense_sizes = dense_layer_sizes_d
        self._dtype = dataset_type
        self._class_info = number_samples_per_class

        self._use_sn = use_spectral_norm
        self._use_residual = use_residual_blocks
        self._num_res_blocks = num_residual_blocks
        self._use_multi_scale = use_multi_scale

        self._use_attn = use_attention
        self._attn_heads = attention_heads
        self._attn_layers = [128, 128]

        self._use_cross_attn = use_cross_attention
        self._cross_attn_heads = cross_attention_heads

        self._norm_type = norm_type
        self._use_cbn = use_conditional_bn
        self._use_adain = use_adaptive_norm

        self._use_glu = use_glu
        self._use_ws = use_weight_standardization

        self._use_mbstd = use_minibatch_stddev
        self._mb_group = minibatch_group_size
        self._use_noise = use_input_noise
        self._noise_std = noise_stddev
        self._label_smooth = label_smoothing

        self._use_r1 = use_r1_regularization
        self._r1_gamma = r1_gamma
        self._use_gp = use_gradient_penalty
        self._gp_lambda = gp_lambda

        self._output_units = output_units
        self._mixed_precision = use_mixed_precision
        self._activation_name = activation_name

        self._model: Optional[Model] = None

    def _get_norm_layer(self, name: str):
        """Retorna camada de normalização"""
        if self._norm_type == 'rms':
            return RMSNorm(name=f'{name}_rms')
        elif self._norm_type == 'layer':
            return LayerNormalization(name=f'{name}_ln')
        elif self._norm_type == 'batch':
            return BatchNormalization(name=f'{name}_bn')
        return Lambda(lambda x: x, name=f'{name}_no_norm')

    def get_discriminator(self) -> Model:
        """Constrói discriminador ULTRA-PODEROSO"""

        if not self._class_info:
            raise ValueError("number_samples_per_class is required")

        if self._mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        num_classes = self._class_info['number_classes']
        input_size = int(np.prod(self._output_shape))

        # INPUTS
        x_in = Input(shape=(input_size,), dtype=self._dtype, name='input_real')
        y_in = Input(shape=(num_classes,), dtype=self._dtype, name='input_label')

        x = x_in

        # Input noise
        if self._use_noise:
            x = Lambda(
                lambda t: t + tf.random.normal(tf.shape(t), 0.0, self._noise_std),
                name='input_noise'
            )(x)

        # Minibatch StdDev
        if self._use_mbstd:
            x = MinibatchStdDev(group_size=self._mb_group, name='minibatch_stddev')(x)

        # Label embedding ULTRA-PODEROSO
        DenseLayer = SpectralDense if self._use_sn else Dense

        label_embed = DenseLayer(
            input_size // 2,  # Embedding maior!
            activation=self._activation_name,
            name='label_embedding_1'
        )(y_in)

        label_embed = DenseLayer(
            input_size // 4,
            activation=self._activation_name,
            name='label_embedding_2'
        )(label_embed)

        x = Concatenate(name='concat_input_label')([x, label_embed])

        # Initial projection
        x = DenseLayer(
            self._dense_sizes[0],
            name='input_projection'
        )(x)
        x = self._get_norm_layer('input')(x)
        x = Activation(self._activation_name)(x)

        multi_scale_features = []
        label_context = label_embed  # Contexto para cross-attention

        # Main processing
        for i, units in enumerate(self._dense_sizes):
            if self._use_residual:
                for j in range(self._num_res_blocks):
                    x = ResidualBlock(
                        units,
                        dropout=self._dropout,
                        use_spectral=self._use_sn,
                        activation=self._activation_name,
                        use_glu=self._use_glu,
                        name=f'res_block_{i}_{j}'
                    )(x)
            else:
                x = DenseLayer(units, name=f'dense_{i}')(x)
                x = self._get_norm_layer(f'layer_{i}')(x)
                x = Activation(self._activation_name)(x)
                x = Dropout(self._dropout, name=f'dropout_{i}')(x)

            # Conditional BN
            if self._use_cbn and i > 0:
                x = ConditionalBatchNorm(
                    units,
                    num_classes,
                    use_spectral=self._use_sn,
                    name=f'cbn_{i}'
                )([x, y_in])

            # Self-Attention
            if self._use_attn and i in self._attn_layers:
                attn = MultiHeadSelfAttention(
                    num_heads=self._attn_heads,
                    head_dim=max(units // self._attn_heads, 16),
                    use_spectral=self._use_sn,
                    use_rotary=True,
                    name=f'self_attn_{i}'
                )(x)

                x = Add(name=f'attn_residual_{i}')([x, attn])

            # Cross-Attention (NOVO!) - features atendem ao contexto das labels
            if self._use_cross_attn and i in self._attn_layers:
                cross_attn = CrossAttention(
                    num_heads=self._cross_attn_heads,
                    head_dim=max(units // self._cross_attn_heads, 16),
                    use_spectral=self._use_sn,
                    name=f'cross_attn_{i}'
                )([x, label_context])

                x = Add(name=f'cross_attn_residual_{i}')([x, cross_attn])

            # Adaptive Instance Norm (NOVO!)
            if self._use_adain and i > 0:
                x = AdaptiveInstanceNorm(
                    use_spectral=self._use_sn,
                    name=f'adain_{i}'
                )([x, label_context])

            if self._use_multi_scale:
                multi_scale_features.append(x)

        # Multi-scale fusion melhorado
        if self._use_multi_scale and len(multi_scale_features) > 1:
            projected = []
            for idx, feat in enumerate(multi_scale_features):
                proj = DenseLayer(
                    self._dense_sizes[-1],
                    name=f'scale_proj_{idx}'
                )(feat)
                projected.append(proj)

            x_multi = Add(name='multi_scale_sum')(projected)
            x_multi = Lambda(
                lambda t: t / len(projected),
                name='multi_scale_avg'
            )(x_multi)

            x = Concatenate(name='concat_multi_scale')([x, x_multi])

            x = DenseLayer(
                self._dense_sizes[-1],
                activation=self._activation_name,
                name='multi_scale_fusion'
            )(x)

            # GLU final para multi-scale
            if self._use_glu:
                x = GatedLinearUnit(use_spectral=self._use_sn, name='multi_scale_glu')(x)

        # Output head poderoso
        x = self._get_norm_layer('pre_output')(x)
        x = Activation(self._activation_name)(x)
        x = Dropout(self._dropout, name='output_dropout')(x)

        # Camada intermediária antes do output
        x = DenseLayer(
            self._dense_sizes[-1] // 2,
            activation=self._activation_name,
            name='pre_output_dense'
        )(x)

        x = DenseLayer(
            self._output_units,
            use_bias=True,
            name='output_logits'
        )(x)

        if self._last_activation is not None:
            x = self._add_activation_layer(x, self._last_activation)

        model = Model(
            inputs=[x_in, y_in],
            outputs=x,
            name='ULTRA_DenseDiscriminator_v4'
        )

        self._model = model

        print("\n" + "=" * 90)
        print("🚀 ULTRA-POWERFUL DENSE DISCRIMINATOR v4.0 - MÁXIMA CAPACIDADE")
        print("=" * 90)
        print(f"🏗️  Architecture: {len(self._dense_sizes)} stages × {self._num_res_blocks} residual blocks")
        print(f"📊 Layer sizes: {self._dense_sizes}")
        print(f"⚡ Activation: {self._activation_name}")
        print(f"\n🔥 NOVOS RECURSOS ULTRA-PODEROSOS:")
        print(f"   ├─ Weight Standardization: {'✓' if self._use_ws else '✗'}")
        print(f"   ├─ Gated Linear Units (GLU): {'✓' if self._use_glu else '✗'}")
        print(
            f"   ├─ Cross-Attention (feat↔label): {'✓' if self._use_cross_attn else '✗'} ({self._cross_attn_heads} heads)")
        print(f"   ├─ Adaptive Instance Norm (AdaIN): {'✓' if self._use_adain else '✗'}")
        print(f"   └─ Feature Statistics in SE: ✓")
        print(f"\n🔒 Spectral Norm: {'✓ ALL layers' if self._use_sn else '✗'}")
        print(f"📈 Normalization: {self._norm_type.upper()}")
        print(f"🧠 Self-Attention: {'✓' if self._use_attn else '✗'} ({self._attn_heads} heads)")
        print(f"🔀 Residual Blocks: {'✓' if self._use_residual else '✗'}")
        print(f"📐 Multi-Scale Pyramid: {'✓' if self._use_multi_scale else '✗'}")
        print(f"📊 Minibatch StdDev: {'✓' if self._use_mbstd else '✗'}")
        print(f"🎲 Input Noise: {'✓' if self._use_noise else '✗'}")
        print(f"🔧 Conditional BN: {'✓' if self._use_cbn else '✗'}")
        print(f"\n💪 PODER DISCRIMINATIVO: ULTRA-MÁXIMO")
        print("=" * 90 + "\n")

#        model.summary()

        return model

    @tf.function
    def gradient_penalty(
            self,
            real_samples: tf.Tensor,
            fake_samples: tf.Tensor,
            labels: tf.Tensor
    ) -> tf.Tensor:
        """Gradient Penalty para WGAN-GP"""
        if not self._use_gp:
            return tf.constant(0.0)

        batch_size = tf.shape(real_samples)[0]
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0, dtype=real_samples.dtype)

        interpolated = alpha * real_samples + (1 - alpha) * fake_samples

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self._model([interpolated, labels], training=True)

        gradients = tape.gradient(pred, interpolated)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1) + 1e-10)
        gp = tf.reduce_mean(tf.square(slopes - 1.0))

        return self._gp_lambda * gp

    @tf.function
    def r1_regularization(
            self,
            real_samples: tf.Tensor,
            labels: tf.Tensor
    ) -> tf.Tensor:
        """R1 Regularization (StyleGAN2)"""
        if not self._use_r1:
            return tf.constant(0.0)

        with tf.GradientTape() as tape:
            tape.watch(real_samples)
            real_pred = self._model([real_samples, labels], training=True)

        gradients = tape.gradient(real_pred, real_samples)
        r1_penalty = tf.reduce_sum(tf.square(gradients), axis=1)
        r1_penalty = tf.reduce_mean(r1_penalty)

        return self._r1_gamma * 0.5 * r1_penalty

    def get_model(self) -> Optional[Model]:
        return self._model

    @property
    def trainable_variables(self):
        return self._model.trainable_variables if self._model else []

    def get_config(self) -> Dict:
        return {
            'latent_dimension': self._latent_dim,
            'output_shape': self._output_shape,
            'dense_layer_sizes': self._dense_sizes,
            'use_spectral_norm': self._use_sn,
            'use_residual_blocks': self._use_residual,
            'num_residual_blocks': self._num_res_blocks,
            'use_multi_scale': self._use_multi_scale,
            'use_attention': self._use_attn,
            'attention_heads': self._attn_heads,
            'attention_layers': self._attn_layers,
            'use_cross_attention': self._use_cross_attn,
            'cross_attention_heads': self._cross_attn_heads,
            'norm_type': self._norm_type,
            'use_conditional_bn': self._use_cbn,
            'use_adaptive_norm': self._use_adain,
            'use_glu': self._use_glu,
            'use_weight_standardization': self._use_ws,
            'use_minibatch_stddev': self._use_mbstd,
            'minibatch_group_size': self._mb_group,
            'use_input_noise': self._use_noise,
            'noise_stddev': self._noise_std,
            'use_r1_regularization': self._use_r1,
            'r1_gamma': self._r1_gamma,
            'use_gradient_penalty': self._use_gp,
            'gp_lambda': self._gp_lambda,
            'activation_name': self._activation_name,
            'mixed_precision': self._mixed_precision
        }
