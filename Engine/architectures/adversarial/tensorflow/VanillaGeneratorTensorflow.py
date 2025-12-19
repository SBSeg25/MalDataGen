# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{2}.{2}.{0}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/19'
__credits__ = ['Synthetic Ocean AI']

# MIT License
# Copyright (c) 2025 Synthetic Ocean AI

try:
    import sys
    import numpy as np
    import math
    from typing import Dict, List, Optional, Callable, Tuple

    import tensorflow as tf
    from tensorflow.keras.layers import (
        Layer, Dense, Input, Dropout, Concatenate,
        Lambda, Multiply, Add, BatchNormalization, LayerNormalization,
        Reshape, Activation, MultiHeadAttention
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.initializers import RandomNormal, HeNormal, GlorotUniform

    from Engine.activations.Activations import Activations

except ImportError as error:
    print(error)
    sys.exit(-1)


# ==================================================================
# FOURIER FEATURES (NeRF, Implicit Neural Representations)
# ==================================================================

class FourierFeatures(Layer):
    """
    Fourier Features para melhor representação de frequências.
    Ref: "Fourier Features Let Networks Learn High Frequency Functions"

    Mapeia inputs para espaço de frequências usando funções seno/cosseno,
    permitindo melhor captura de detalhes de alta frequência.
    """

    def __init__(
            self,
            num_frequencies: int = 10,
            sigma: float = 1.0,
            learnable: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.num_frequencies = num_frequencies
        self.sigma = sigma
        self.learnable = learnable

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Matriz de frequências B ~ N(0, sigma^2)
        b_init = tf.random.normal(
            [input_dim, self.num_frequencies],
            stddev=self.sigma,
            dtype=tf.float32
        )

        self.B = self.add_weight(
            name='frequency_matrix',
            shape=[input_dim, self.num_frequencies],
            initializer=tf.keras.initializers.Constant(b_init),
            trainable=self.learnable,
            dtype=tf.float32
        )

        super().build(input_shape)

    def call(self, x):
        # x: [batch, input_dim]
        # Calcula 2π * B^T * x
        x_proj = 2.0 * np.pi * tf.matmul(x, self.B)

        # Concatena sin e cos
        return tf.concat([tf.sin(x_proj), tf.cos(x_proj)], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2 * self.num_frequencies)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_frequencies': self.num_frequencies,
            'sigma': self.sigma,
            'learnable': self.learnable
        })
        return config


# ==================================================================
# ADAPTIVE INSTANCE NORMALIZATION (AdaIN) IMPROVED
# ==================================================================

class AdaptiveInstanceNormalization(Layer):
    """
    AdaIN melhorado com opções de normalização e projeção adaptativa.
    Ref: StyleGAN2, "Arbitrary Style Transfer in Real-time"
    """

    def __init__(
            self,
            epsilon: float = 1e-8,
            use_learnable_epsilon: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.use_learnable_epsilon = use_learnable_epsilon

    def build(self, input_shape):
        # input_shape: [(batch, features), (batch, style_dim)]
        features_dim = input_shape[0][-1]

        if self.use_learnable_epsilon:
            self.eps = self.add_weight(
                name='epsilon',
                shape=[1],
                initializer=tf.keras.initializers.Constant(self.epsilon),
                trainable=True,
                dtype=tf.float32
            )

        super().build(input_shape)

    def call(self, inputs):
        x, style = inputs

        # Normaliza x
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.math.reduce_variance(x, axis=-1, keepdims=True)

        eps = self.eps if self.use_learnable_epsilon else self.epsilon
        x_norm = (x - mean) / tf.sqrt(variance + eps)

        # Aplica style (assumindo que style já foi projetado para [gamma, beta])
        style_dim = style.shape[-1]
        x_dim = x.shape[-1]

        # Se style_dim = 2 * x_dim, split em gamma e beta
        if style_dim == 2 * x_dim:
            gamma, beta = tf.split(style, 2, axis=-1)
        else:
            # Caso contrário, usa style como gamma e beta zero
            gamma = style
            beta = tf.zeros_like(x)

        return x_norm * (1 + gamma) + beta

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon,
            'use_learnable_epsilon': self.use_learnable_epsilon
        })
        return config


# ==================================================================
# CROSS-ATTENTION CONDITIONING
# ==================================================================

class CrossAttentionConditioning(Layer):
    """
    Cross-Attention para conditioning (melhor que concatenação simples).
    Features atendem ao style vector.
    Ref: Transformers, "Attention is All You Need"
    """

    def __init__(
            self,
            num_heads: int = 4,
            key_dim: int = 64,
            dropout: float = 0.0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout

    def build(self, input_shape):
        # input_shape: [(batch, features), (batch, style_dim)]
        features_dim = input_shape[0][-1]

        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout_rate,
            dtype=tf.float32
        )

        # Projection para garantir dimensões corretas
        self.query_proj = Dense(features_dim, dtype=tf.float32)
        self.key_value_proj = Dense(features_dim, dtype=tf.float32)
        self.output_proj = Dense(features_dim, dtype=tf.float32)

        super().build(input_shape)

    def call(self, inputs, training=None):
        features, style = inputs

        # Expande dimensões para attention
        # Query: features [batch, 1, features_dim]
        query = tf.expand_dims(features, axis=1)
        query = self.query_proj(query)

        # Key, Value: style [batch, 1, style_dim]
        kv = tf.expand_dims(style, axis=1)
        kv = self.key_value_proj(kv)

        # Cross-attention
        attended = self.attention(
            query=query,
            key=kv,
            value=kv,
            training=training
        )

        # Remove dimensão temporal
        attended = tf.squeeze(attended, axis=1)

        # Output projection
        output = self.output_proj(attended)

        # Residual connection
        return features + output

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'dropout': self.dropout_rate
        })
        return config


# ==================================================================
# MINIBATCH STANDARD DEVIATION (ProGAN, StyleGAN)
# ==================================================================

class MinibatchStdDev(Layer):
    """
    Minibatch Standard Deviation para discriminator/generator.
    Adiciona estatística de grupo como feature adicional.
    Ajuda a reduzir mode collapse.
    """

    def __init__(self, group_size: int = 4, epsilon: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.group_size = group_size
        self.epsilon = epsilon

    def call(self, x):
        batch_size = tf.shape(x)[0]
        features = x.shape[-1]

        # Divide em grupos
        group_size = tf.minimum(self.group_size, batch_size)

        # Reshape: [group_size, num_groups, features]
        num_groups = batch_size // group_size
        remainder = batch_size % group_size

        # Usa apenas parte divisível
        x_grouped = x[:num_groups * group_size]
        x_grouped = tf.reshape(x_grouped, [num_groups, group_size, features])

        # Calcula std dentro de cada grupo
        group_mean = tf.reduce_mean(x_grouped, axis=1, keepdims=True)
        group_variance = tf.reduce_mean(
            tf.square(x_grouped - group_mean),
            axis=1,
            keepdims=True
        )
        group_std = tf.sqrt(group_variance + self.epsilon)

        # Média sobre features e grupos
        minibatch_std = tf.reduce_mean(group_std, keepdims=True)

        # Broadcast para batch size original
        minibatch_std = tf.tile(minibatch_std, [batch_size, 1])

        # Concatena com features originais
        return tf.concat([x, minibatch_std], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'group_size': self.group_size,
            'epsilon': self.epsilon
        })
        return config


# ==================================================================
# LEARNABLE NOISE INJECTION
# ==================================================================

class LearnableNoiseInjection(Layer):
    """
    Noise Injection com parâmetros aprendíveis (StyleGAN2).
    Cada canal tem seu próprio scaling factor para o ruído.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        features = input_shape[-1]

        self.noise_weight = self.add_weight(
            name='noise_weight',
            shape=[1, features],
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32
        )

        super().build(input_shape)

    def call(self, x, training=None):
        if not training:
            return x

        # Gera ruído
        noise = tf.random.normal(
            tf.shape(x),
            mean=0.0,
            stddev=1.0,
            dtype=tf.float32
        )

        # Aplica scaling aprendível
        return x + noise * self.noise_weight

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super().get_config()


# ==================================================================
# MULTI-SCALE SKIP CONNECTIONS
# ==================================================================

class MultiScaleSkipConnection(Layer):
    """
    Skip connections multi-escala para melhor fluxo de gradiente.
    Conecta features de diferentes resoluções/profundidades.
    """

    def __init__(self, target_dim: int, num_scales: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.target_dim = target_dim
        self.num_scales = num_scales

    def build(self, input_shape):
        # Cria projeções para cada escala
        self.projections = []
        for i in range(self.num_scales):
            proj = Dense(
                self.target_dim,
                name=f'scale_{i}_projection',
                dtype=tf.float32
            )
            self.projections.append(proj)

        # Learnable weights para combinar escalas
        self.scale_weights = self.add_weight(
            name='scale_weights',
            shape=[self.num_scales],
            initializer=tf.keras.initializers.Constant(1.0 / self.num_scales),
            trainable=True,
            dtype=tf.float32
        )

        super().build(input_shape)

    def call(self, inputs):
        # inputs: lista de tensors de diferentes escalas
        if not isinstance(inputs, list):
            inputs = [inputs]

        # Limita ao número de escalas configurado
        inputs = inputs[:self.num_scales]

        # Projeta cada escala
        projected = []
        for i, (x, proj) in enumerate(zip(inputs, self.projections)):
            projected.append(proj(x))

        # Combina com pesos aprendíveis
        weights = tf.nn.softmax(self.scale_weights)
        combined = tf.zeros_like(projected[0])

        for i, proj_x in enumerate(projected):
            combined = combined + weights[i] * proj_x

        return combined

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0] if isinstance(input_shape, list) else input_shape[0],
                self.target_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            'target_dim': self.target_dim,
            'num_scales': self.num_scales
        })
        return config


# ==================================================================
# EXISTING LAYERS (mantidos do código original)
# ==================================================================

class KaiserWindow:
    """Kaiser window para filtros low-pass (StyleGAN3)."""

    @staticmethod
    def kaiser_attenuation(numtaps: int, width: float, ripple_db: float = 60.0) -> np.ndarray:
        if ripple_db > 50:
            beta = 0.1102 * (ripple_db - 8.7)
        elif ripple_db >= 21:
            beta = 0.5842 * (ripple_db - 21) ** 0.4 + 0.07886 * (ripple_db - 21)
        else:
            beta = 0.0

        n = np.arange(numtaps)
        alpha = (numtaps - 1) / 2.0
        kaiser = np.i0(beta * np.sqrt(1 - ((n - alpha) / alpha) ** 2)) / np.i0(beta)
        cutoff = 0.5 / width
        sinc = np.sinc(2 * cutoff * (n - alpha))
        window = kaiser * sinc
        window /= np.sum(window)

        return window.astype(np.float32)


class LowPassFilter(Layer):
    """Low-pass filter para operações alias-free (StyleGAN3)."""

    def __init__(self, channels: int, cutoff: float = 1.0, width: float = 1.0,
                 kernel_size: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.cutoff = cutoff
        self.width = width
        self.kernel_size = kernel_size

    def build(self, input_shape):
        filter_1d = KaiserWindow.kaiser_attenuation(self.kernel_size, self.width) * self.cutoff

        self.filter = self.add_weight(
            name='lowpass_filter',
            shape=(self.kernel_size,),
            initializer=tf.keras.initializers.Constant(filter_1d),
            trainable=False,
            dtype=tf.float32
        )
        super().build(input_shape)

    def call(self, x):
        original_shape = tf.shape(x)
        batch_size = original_shape[0]
        features = original_shape[1]

        padding = self.kernel_size // 2
        x_padded = tf.pad(x, [[0, 0], [padding, padding]], mode='REFLECT')
        x_reshaped = tf.expand_dims(x_padded, axis=-1)
        kernel = tf.reshape(self.filter, [self.kernel_size, 1, 1])

        filtered = tf.nn.conv1d(x_reshaped, kernel, stride=1, padding='VALID')
        filtered = tf.squeeze(filtered, axis=-1)
        filtered = filtered[:, :features]

        current_features = tf.shape(filtered)[1]
        if_needs_padding = tf.less(current_features, features)

        def pad_tensor():
            padding_needed = features - current_features
            return tf.pad(filtered, [[0, 0], [0, padding_needed]])

        filtered = tf.cond(if_needs_padding, pad_tensor, lambda: filtered)
        filtered.set_shape([None, x.shape[-1]])

        return filtered

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'cutoff': self.cutoff,
            'width': self.width,
            'kernel_size': self.kernel_size
        })
        return config


class ModulatedDense(Layer):
    """Dense layer com Weight Demodulation (StyleGAN2)."""

    def __init__(self, units: int, demodulate: bool = True, epsilon: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.demodulate = demodulate
        self.epsilon = epsilon
        self._built_in_features = None

    def build(self, input_shape):
        features_shape = input_shape[0]
        self._built_in_features = features_shape[-1]

        self.kernel = self.add_weight(
            name='kernel',
            shape=(self._built_in_features, self.units),
            initializer=GlorotUniform(),
            trainable=True,
            dtype=tf.float32
        )

        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            dtype=tf.float32
        )

        self.style_proj = Dense(
            self._built_in_features,
            kernel_initializer='ones',
            use_bias=True,
            name='style_projection',
            dtype=tf.float32
        )

        super().build(input_shape)

    def call(self, inputs):
        x, style = inputs

        s = self.style_proj(style)

        if s.shape[-1] != x.shape[-1]:
            style_features = s.shape[-1]
            x_features = x.shape[-1]

            if style_features < x_features:
                padding = x_features - style_features
                s = tf.pad(s, [[0, 0], [0, padding]], constant_values=0.0)
            else:
                s = s[:, :x_features]

        if self._built_in_features != x.shape[-1]:
            kernel_to_use = self.kernel
            if self._built_in_features < x.shape[-1]:
                x = x[:, :self._built_in_features]
            else:
                padding = self._built_in_features - x.shape[-1]
                x = tf.pad(x, [[0, 0], [0, padding]])

        s_reshaped = tf.expand_dims(s + 1.0, axis=-1)
        w = tf.expand_dims(self.kernel, axis=0)
        w_modulated = w * s_reshaped

        if self.demodulate:
            demod = tf.math.rsqrt(
                tf.reduce_sum(tf.square(w_modulated), axis=1, keepdims=True) + self.epsilon
            )
            w_final = w_modulated * demod
        else:
            w_final = w_modulated

        x_expanded = tf.expand_dims(x, axis=1)
        output = tf.matmul(x_expanded, w_final)
        output = tf.squeeze(output, axis=1)
        output = output + self.bias
        output = tf.ensure_shape(output, [None, self.units])

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.units)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'demodulate': self.demodulate,
            'epsilon': self.epsilon
        })
        return config


class SpectralNormalization(Layer):
    """Spectral Normalization para estabilização."""

    def __init__(self, layer: Layer, power_iterations: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.layer = layer
        self.power_iterations = power_iterations

    def build(self, input_shape):
        self.layer.build(input_shape)

        self.w = self.layer.kernel
        w_shape = self.w.shape.as_list()
        self.w_shape = w_shape
        self.w_reshaped_shape = (w_shape[-1], np.prod(w_shape[:-1]))

        self.u = self.add_weight(
            shape=(1, self.w_reshaped_shape[0]),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name='spectral_norm_u',
            dtype=tf.float32
        )

        super().build(input_shape)

    def call(self, inputs, training=None):
        w_normalized = self._normalize_weights(training)
        self.layer.kernel.assign(w_normalized)
        output = self.layer(inputs)
        return output

    def _normalize_weights(self, training):
        w_reshaped = tf.reshape(self.w, self.w_reshaped_shape)

        u_hat = self.u
        v_hat = None

        for _ in range(self.power_iterations):
            v_hat = tf.nn.l2_normalize(
                tf.matmul(u_hat, w_reshaped, transpose_b=True),
                axis=1
            )

            u_hat = tf.nn.l2_normalize(
                tf.matmul(v_hat, w_reshaped),
                axis=1
            )

        if training:
            self.u.assign(u_hat)

        sigma = tf.matmul(
            tf.matmul(u_hat, w_reshaped),
            v_hat,
            transpose_b=True
        )

        w_normalized = self.w / sigma

        return w_normalized

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'power_iterations': self.power_iterations,
            'layer': tf.keras.layers.serialize(self.layer)
        })
        return config


class PixelNormalization(Layer):
    """Pixel Normalization (ProGAN, StyleGAN)."""

    def __init__(self, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, x):
        return x / tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.epsilon)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'epsilon': self.epsilon})
        return config


class SelfAttentionGenerator(Layer):
    """Self-Attention otimizado para generator."""

    def __init__(self, reduction: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self._built_channels = None

    def build(self, input_shape):
        self._built_channels = input_shape[-1]
        self.hidden_dim = max(self._built_channels // self.reduction, 8)

        self.query = Dense(self.hidden_dim, use_bias=False, dtype=tf.float32)
        self.key = Dense(self.hidden_dim, use_bias=False, dtype=tf.float32)
        self.value = Dense(self.hidden_dim, use_bias=False, dtype=tf.float32)
        self.out_proj = Dense(self._built_channels, use_bias=False, dtype=tf.float32)

        self.gamma = self.add_weight(
            name='gamma',
            shape=[1],
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True,
            dtype=tf.float32
        )

        super().build(input_shape)

    def call(self, x):
        input_features = x.shape[-1]

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attention_weights = tf.nn.softmax(
            q * k / tf.sqrt(tf.cast(self.hidden_dim, tf.float32)),
            axis=-1
        )

        attended = attention_weights * v
        out = self.out_proj(attended)

        out_features = out.shape[-1]
        if out_features != input_features:
            if out_features < input_features:
                padding = input_features - out_features
                out = tf.pad(out, [[0, 0], [0, padding]])
            else:
                out = out[:, :input_features]

        return self.gamma * out + x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'reduction': self.reduction})
        return config


class ConditionalBatchNormGenerator(Layer):
    """Conditional Batch Normalization para generator."""

    def __init__(self, num_features: int, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.num_classes = num_classes

    def build(self, input_shape):
        self.bn = BatchNormalization(scale=False, center=False, dtype=tf.float32)

        self.gamma_embed = Dense(
            self.num_features,
            kernel_initializer='ones',
            use_bias=False,
            name='cbn_gamma',
            dtype=tf.float32
        )
        self.beta_embed = Dense(
            self.num_features,
            kernel_initializer='zeros',
            use_bias=False,
            name='cbn_beta',
            dtype=tf.float32
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        x, y = inputs
        x_norm = self.bn(x, training=training)
        gamma = self.gamma_embed(y)
        beta = self.beta_embed(y)
        return x_norm * (1 + gamma) + beta

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_features': self.num_features,
            'num_classes': self.num_classes
        })
        return config


# ==================================================================
# ADVANCED GENERATOR WITH SOTA IMPROVEMENTS
# ==================================================================

class VanillaGenerator(Activations):
    """
    Advanced Generator com técnicas SOTA:

    ORIGINAL FEATURES:
    - Weight Demodulation (StyleGAN2)
    - Alias-free operations (StyleGAN3)
    - Pixel Normalization (ProGAN, StyleGAN)
    - Spectral Normalization
    - Self-Attention otimizado
    - Residual connections com scaling
    - Noise injection (StyleGAN)
    - Mixed precision ready

    NEW SOTA IMPROVEMENTS:
    - Fourier Features (NeRF-style) para latent space
    - Cross-Attention Conditioning (melhor que concatenação)
    - Adaptive Instance Normalization melhorado
    - Learnable Noise Injection (StyleGAN2)
    - Minibatch Standard Deviation (mode collapse reduction)
    - Multi-Scale Skip Connections (melhor gradient flow)
    """

    def __init__(
            self,
            latent_dimension: int,
            output_shape: int,
            activation_function: Callable,
            initializer_mean: float,
            initializer_deviation: float,
            dropout_decay_rate_g: float,
            last_layer_activation: Callable,
            dense_layer_sizes_g: List[int],
            dataset_type: type = np.float32,
            number_samples_per_class: Optional[Dict[str, int]] = None,
            # ========== ORIGINAL HYPERPARAMETERS ==========
            use_pixel_norm: bool = True,
            use_spectral_norm: bool = False,
            spectral_norm_power_iterations: int = 1,
            norm_type: str = 'conditional_batch',
            use_weight_demodulation: bool = False,
            use_alias_free: bool = False,
            alias_free_cutoff: float = 1.0,
            conditioning_method: str = 'cross_attention',  # Mudado para cross_attention
            use_self_attention: bool = True,
            attention_layers: List[int] = None,
            use_residual_connections: bool = True,
            residual_scaling: float = 0.2,
            bottleneck_factor: float = 0.25,
            use_noise_injection: bool = True,
            noise_strength: float = 0.05,
            latent_noise_strength: float = 0.01,
            use_dropout: bool = True,
            latent_mapping_layers: int = 1,
            latent_mapping_units: int = None,
            use_tanh_output: bool = False,
            use_mixed_precision: bool = False,
            # ========== NEW SOTA HYPERPARAMETERS ==========
            use_fourier_features: bool = True,
            fourier_frequencies: int = 10,
            fourier_sigma: float = 0.5,
            fourier_learnable: bool = False,
            use_cross_attention: bool = True,
            cross_attention_heads: int = 4,
            cross_attention_key_dim: int = 64,
            use_improved_adain: bool = True,
            adain_learnable_epsilon: bool = True,
            use_learnable_noise: bool = True,
            use_minibatch_std: bool = False,  # Geralmente melhor para discriminator
            minibatch_std_group_size: int = 4,
            use_multi_scale_skip: bool = True,
            multi_scale_num_scales: int = 3,
    ) -> None:

        # Validações
        if latent_dimension <= 0:
            raise ValueError("latent_dimension must be > 0")
        if output_shape <= 0:
            raise ValueError("output_shape must be > 0")
        if not dense_layer_sizes_g:
            raise ValueError("dense_layer_sizes_g cannot be empty")
        if not 0.0 <= dropout_decay_rate_g <= 1.0:
            raise ValueError("dropout_decay_rate_g must be in [0, 1]")

        # Atributos básicos
        self._latent_dim = latent_dimension
        self._output_shape = output_shape
        self._activation_fn = activation_function
        self._last_activation = last_layer_activation
        self._dropout_rate = dropout_decay_rate_g
        self._dense_sizes = dense_layer_sizes_g
        self._dtype = dataset_type
        self._init_mean = initializer_mean
        self._init_std = initializer_deviation
        self._class_info = number_samples_per_class

        # Original Hyperparameters
        self._use_pixel_norm = use_pixel_norm
        self._use_spectral_norm = use_spectral_norm
        self._spectral_power_iters = spectral_norm_power_iterations
        self._norm_type = norm_type
        self._use_weight_demod = use_weight_demodulation
        self._use_alias_free = use_alias_free
        self._alias_cutoff = alias_free_cutoff
        self._cond_method = conditioning_method
        self._use_attn = use_self_attention
        self._attn_layers = attention_layers or [3]
        self._use_residual = use_residual_connections
        self._res_scale = residual_scaling
        self._bottleneck = bottleneck_factor
        self._use_noise_inj = use_noise_injection
        self._noise_str = noise_strength
        self._latent_noise_str = latent_noise_strength
        self._use_dropout = use_dropout
        self._latent_map_layers = latent_mapping_layers
        self._latent_map_units = latent_mapping_units or latent_dimension
        self._use_tanh = use_tanh_output
        self._mixed_precision = use_mixed_precision

        # New SOTA Hyperparameters
        self._use_fourier = use_fourier_features
        self._fourier_freq = fourier_frequencies
        self._fourier_sigma = fourier_sigma
        self._fourier_learn = fourier_learnable
        self._use_cross_attn = use_cross_attention
        self._cross_attn_heads = cross_attention_heads
        self._cross_attn_key_dim = cross_attention_key_dim
        self._use_improved_adain = use_improved_adain
        self._adain_learn_eps = adain_learnable_epsilon
        self._use_learnable_noise = use_learnable_noise
        self._use_minibatch_std = use_minibatch_std
        self._minibatch_std_group = minibatch_std_group_size
        self._use_multi_scale = use_multi_scale_skip
        self._multi_scale_num = multi_scale_num_scales

        self._model: Optional[Model] = None
        self._mapping_network: Optional[Model] = None
        self._multi_scale_features: List[Layer] = []

    # ==================================================================
    # UTILITY METHODS
    # ==================================================================

    def _get_dense_layer(self, units: int, name: str, use_bias: bool = True):
        """Retorna Dense layer com inicialização adequada e spectral norm opcional."""
        dense = Dense(
            units,
            kernel_initializer=GlorotUniform(),
            use_bias=use_bias,
            name=name,
            dtype=tf.float32
        )

        if self._use_spectral_norm:
            return SpectralNormalization(
                dense,
                power_iterations=self._spectral_power_iters,
                name=f'{name}_spectral_norm'
            )

        return dense

    def _add_noise_layer(self, x: Layer, name: str = 'noise') -> Layer:
        """Adiciona ruído ao layer com opção learnable."""
        if not self._use_noise_inj:
            return x

        if self._use_learnable_noise:
            return LearnableNoiseInjection(name=name)(x)
        else:
            @tf.function
            def inject_noise(features):
                noise = tf.random.normal(
                    tf.shape(features),
                    stddev=self._noise_str,
                    dtype=tf.float32
                )
                return features + noise

            return Lambda(inject_noise, name=name)(x)

    def _normalization_layer(self, x: Layer, name: str) -> Layer:
        """Adiciona normalização baseada em configuração."""
        if self._norm_type == 'pixel':
            return PixelNormalization(name=f'{name}_pn')(x)
        elif self._norm_type == 'batch':
            return BatchNormalization(name=f'{name}_bn', dtype=tf.float32)(x)
        elif self._norm_type == 'layer':
            return LayerNormalization(name=f'{name}_ln', dtype=tf.float32)(x)
        return x

    def _alias_free_activation(self, x: Layer, name: str) -> Layer:
        """Aplica ativação com filtro anti-aliasing (StyleGAN3)."""
        if not self._use_alias_free:
            return self._add_activation_layer(x, self._activation_fn)

        x = self._add_activation_layer(x, self._activation_fn)
        units = x.shape[-1]
        x = LowPassFilter(
            channels=units,
            cutoff=self._alias_cutoff,
            name=f'{name}_lowpass'
        )(x)

        return x

    # ==================================================================
    # BUILDING BLOCKS
    # ==================================================================

    def _dense_block(
            self,
            x: Layer,
            units: int,
            name: str,
            use_residual: bool = False,
            style: Optional[Layer] = None
    ) -> Layer:
        """Bloco denso avançado com múltiplas opções de conditioning."""
        identity = x

        # Salva feature para multi-scale skip
        if self._use_multi_scale:
            self._multi_scale_features.append(x)

        # Cross-Attention Conditioning (NOVO)
        if self._use_cross_attn and style is not None:
            x = CrossAttentionConditioning(
                num_heads=self._cross_attn_heads,
                key_dim=self._cross_attn_key_dim,
                name=f'{name}_cross_attn'
            )([x, style])

        # Weight Demodulation
        if self._use_weight_demod and style is not None:
            x = ModulatedDense(
                units=units,
                demodulate=True,
                name=f'{name}_modulated_dense'
            )([x, style])
        else:
            x = self._get_dense_layer(units, f'{name}_dense')(x)
            x = self._normalization_layer(x, name)

        # AdaIN melhorado (NOVO)
        if self._use_improved_adain and style is not None:
            # Projeta style para ter 2x as dimensões (gamma + beta)
            style_proj = Dense(
                2 * units,
                name=f'{name}_style_adain_proj',
                dtype=tf.float32
            )(style)

            x = AdaptiveInstanceNormalization(
                use_learnable_epsilon=self._adain_learn_eps,
                name=f'{name}_adain'
            )([x, style_proj])

        # Ativação com alias-free
        x = self._alias_free_activation(x, name)

        # Noise injection (learnable ou fixo)
        x = self._add_noise_layer(x, f'{name}_noise')

        # Dropout
        if self._use_dropout:
            x = Dropout(self._dropout_rate, name=f'{name}_dropout')(x)

        # Residual connection
        if use_residual and identity.shape[-1] == units:
            x = Lambda(
                lambda t: t[0] + self._res_scale * t[1],
                name=f'{name}_residual'
            )([identity, x])

        return x

    def _attention_block(self, x: Layer, layer_idx: int) -> Layer:
        """Adiciona self-attention se configurado."""
        if self._use_attn and layer_idx in self._attn_layers:
            x = SelfAttentionGenerator(
                reduction=8,
                name=f'gen_layer_{layer_idx}_attn'
            )(x)
        return x

    def _latent_mapping_network(
            self,
            z_in: Layer,
            num_classes: int
    ) -> Tuple[Layer, Model]:
        """Latent Mapping Network com Fourier Features (NOVO)."""
        x = z_in

        # Fourier Features para latent space (NOVO - NeRF-style)
        if self._use_fourier:
            x = FourierFeatures(
                num_frequencies=self._fourier_freq,
                sigma=self._fourier_sigma,
                learnable=self._fourier_learn,
                name='latent_fourier'
            )(x)

        # Adiciona ruído ao latent input
        if self._latent_noise_str > 0:
            @tf.function
            def add_latent_noise(z):
                noise = tf.random.normal(
                    tf.shape(z),
                    stddev=self._latent_noise_str,
                    dtype=tf.float32
                )
                return z + noise

            x = Lambda(add_latent_noise, name='latent_noise')(x)

        # Pixel normalization no input
        if self._use_pixel_norm:
            x = PixelNormalization(name='latent_pn_input')(x)

        # Mapping layers
        for i in range(self._latent_map_layers):
            x = self._get_dense_layer(
                self._latent_map_units,
                f'mapping_{i}'
            )(x)

            if self._use_pixel_norm:
                x = PixelNormalization(name=f'mapping_pn_{i}')(x)

            x = self._add_activation_layer(x, self._activation_fn)

        mapping_model = Model(z_in, x, name='LatentMapping')
        return x, mapping_model

    # ==================================================================
    # MODEL BUILDING
    # ==================================================================

    def get_generator(self) -> Model:
        """Constrói o generator avançado com SOTA improvements."""

        if not self._class_info:
            raise ValueError("number_samples_per_class is required")

        # Mixed precision
        if self._mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)

        num_classes = self._class_info['number_classes']

        # Inputs
        z_in = Input(shape=(self._latent_dim,), dtype=tf.float32, name='latent_input')
        y_in = Input(shape=(num_classes,), dtype=tf.float32, name='label_input')

        # Latent mapping network com Fourier Features
        w, mapping_model = self._latent_mapping_network(z_in, num_classes)
        self._mapping_network = mapping_model

        # Combina latent code com label para criar style
        style = Concatenate(name='style_concat')([w, y_in])

        # Style projection
        style = self._get_dense_layer(
            self._latent_map_units,
            'style_projection'
        )(style)

        if self._use_pixel_norm:
            style = PixelNormalization(name='style_pn')(style)

        # Initial dense layer
        x = self._get_dense_layer(128, 'initial_dense')(w)

        if self._use_pixel_norm:
            x = PixelNormalization(name='initial_pn')(x)

        # Reset multi-scale features
        self._multi_scale_features = [x]

        # Processing blocks
        for i, units in enumerate([32]):
            use_res = self._use_residual and i > 0
            x = self._dense_block(
                x=x,
                units=units,
                name=f'block_{i}',
                use_residual=use_res,
                style=style
            )

            # Attention
            x = self._attention_block(x, i)

            # Minibatch Standard Deviation (NOVO)
            if self._use_minibatch_std:
                x = MinibatchStdDev(
                    group_size=self._minibatch_std_group,
                    name=f'minibatch_std_{i}'
                )(x)

        # Multi-Scale Skip Connection (NOVO)
        if self._use_multi_scale and len(self._multi_scale_features) > 1:
            x = MultiScaleSkipConnection(
                target_dim=x.shape[-1],
                num_scales=min(self._multi_scale_num, len(self._multi_scale_features)),
                name='multi_scale_skip'
            )(self._multi_scale_features[-self._multi_scale_num:])

        # Output block
        if self._use_weight_demod and style is not None:
            x = ModulatedDense(
                units=self._output_shape,
                demodulate=False,
                name='output_modulated'
            )([x, style])
        else:
            x = self._get_dense_layer(
                self._output_shape,
                'output_dense'
            )(x)

        # Output activation
        if self._use_tanh:
            x = Activation('tanh', name='output_tanh', dtype=tf.float32)(x)
        elif self._last_activation is not None:
            x = self._add_activation_layer(x, self._last_activation)

        # Build model
        model = Model(
            inputs=[z_in, y_in],
            outputs=x,
            name='AdvancedGenerator'
        )

        self._model = model

        # Mostra informações
        print("\n" + "=" * 70)
        print("🚀 Advanced Generator with SOTA Improvements:")
        print("=" * 70)

        print("\n📦 ORIGINAL FEATURES:")
        if self._use_weight_demod:
            print("  ✓ Weight Demodulation (StyleGAN2)")
        if self._use_alias_free:
            print(f"  ✓ Alias-free operations (StyleGAN3, cutoff={self._alias_cutoff})")
        if self._use_spectral_norm:
            print(f"  ✓ Spectral Normalization ({self._spectral_power_iters} iterations)")
        if self._use_pixel_norm:
            print("  ✓ Pixel Normalization")
        if self._use_attn:
            print(f"  ✓ Self-Attention (layers {self._attn_layers})")
        if self._use_noise_inj:
            print(f"  ✓ Noise Injection (strength={self._noise_str})")

        print("\n✨ NEW SOTA IMPROVEMENTS:")
        if self._use_fourier:
            learnable_str = "learnable" if self._fourier_learn else "fixed"
            print(f"  ✓ Fourier Features ({self._fourier_freq} freq, {learnable_str})")
        if self._use_cross_attn:
            print(f"  ✓ Cross-Attention Conditioning ({self._cross_attn_heads} heads)")
        if self._use_improved_adain:
            eps_str = "learnable ε" if self._adain_learn_eps else "fixed ε"
            print(f"  ✓ Improved AdaIN ({eps_str})")
        if self._use_learnable_noise:
            print("  ✓ Learnable Noise Injection (per-channel scaling)")
        if self._use_minibatch_std:
            print(f"  ✓ Minibatch StdDev (group size={self._minibatch_std_group})")

        return model
    def create_advanced_generator_from_config(config: Dict):
        """
        Cria um generator avançado a partir de um dicionário de configuração.

        Args:
            config: Dicionário com configuração

        Returns:
            Instância do VanillaGenerator
        """
        required_keys = [
            'latent_dimension',
            'output_shape',
            'activation_function',
            'initializer_mean',
            'initializer_deviation',
            'dropout_decay_rate_g',
            'last_layer_activation',
            'dense_layer_sizes_g',
            'number_samples_per_class'
        ]

        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in config: {key}")

        return VanillaGenerator(**config)