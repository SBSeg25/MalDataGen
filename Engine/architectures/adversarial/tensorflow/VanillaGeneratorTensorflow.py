#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{2}.{1}.{0}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/18'
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
        Reshape, Activation
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.initializers import RandomNormal, HeNormal, GlorotUniform

    from Engine.activations.Activations import Activations

except ImportError as error:
    print(error)
    sys.exit(-1)


class KaiserWindow:
    """
    Kaiser window para filtros low-pass (StyleGAN3).
    Usado para operações alias-free.
    """

    @staticmethod
    def kaiser_attenuation(numtaps: int, width: float, ripple_db: float = 60.0) -> np.ndarray:
        """Calcula janela Kaiser para filtro low-pass."""
        # Kaiser beta parameter
        if ripple_db > 50:
            beta = 0.1102 * (ripple_db - 8.7)
        elif ripple_db >= 21:
            beta = 0.5842 * (ripple_db - 21) ** 0.4 + 0.07886 * (ripple_db - 21)
        else:
            beta = 0.0

        # Calcula janela
        n = np.arange(numtaps)
        alpha = (numtaps - 1) / 2.0

        # Kaiser window
        kaiser = np.i0(beta * np.sqrt(1 - ((n - alpha) / alpha) ** 2)) / np.i0(beta)

        # Sinc filter
        cutoff = 0.5 / width
        sinc = np.sinc(2 * cutoff * (n - alpha))

        # Combine
        window = kaiser * sinc
        window /= np.sum(window)

        return window.astype(np.float32)


class LowPassFilter(Layer):
    """
    Low-pass filter para operações alias-free (StyleGAN3).
    Previne aliasing durante transformações.
    """

    def __init__(
            self,
            channels: int,
            cutoff: float = 1.0,
            width: float = 1.0,
            kernel_size: int = 6,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.cutoff = cutoff
        self.width = width
        self.kernel_size = kernel_size

    def build(self, input_shape):
        # Cria filtro Kaiser
        filter_1d = KaiserWindow.kaiser_attenuation(
            self.kernel_size,
            self.width
        ) * self.cutoff

        # Armazena como peso não-treinável
        self.filter = self.add_weight(
            name='lowpass_filter',
            shape=(self.kernel_size,),
            initializer=tf.keras.initializers.Constant(filter_1d),
            trainable=False,
            dtype=tf.float32
        )

        super().build(input_shape)

    def call(self, x):
        # Para camadas Dense, aplica filtro como convolução 1D nos features
        # x shape: (batch, features)

        original_shape = tf.shape(x)
        batch_size = original_shape[0]
        features = original_shape[1]

        # Pad
        padding = self.kernel_size // 2
        x_padded = tf.pad(x, [[0, 0], [padding, padding]], mode='REFLECT')

        # Reshape para conv1d: (batch, features + 2*padding, 1)
        x_reshaped = tf.expand_dims(x_padded, axis=-1)

        # Cria kernel para conv1d: (kernel_size, 1, 1)
        kernel = tf.reshape(self.filter, [self.kernel_size, 1, 1])

        # Aplica convolução
        filtered = tf.nn.conv1d(
            x_reshaped,
            kernel,
            stride=1,
            padding='VALID'
        )

        # Remove dimensão extra: (batch, some_features)
        filtered = tf.squeeze(filtered, axis=-1)

        # CRÍTICO: Garante que o output tem EXATAMENTE o mesmo shape que o input
        # Se necessário, crop ou pad
        filtered_features = tf.shape(filtered)[1]

        # Crop se necessário
        filtered = filtered[:, :features]

        # Se ainda falta features (improvável), pad com zeros
        current_features = tf.shape(filtered)[1]
        if_needs_padding = tf.less(current_features, features)

        def pad_tensor():
            padding_needed = features - current_features
            return tf.pad(filtered, [[0, 0], [0, padding_needed]])

        def no_pad():
            return filtered

        filtered = tf.cond(if_needs_padding, pad_tensor, no_pad)

        # Garante shape estático
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
    """
    Dense layer com Weight Demodulation (StyleGAN2).

    Substitui AdaIN por modulação direta dos pesos:
    1. Modula pesos: W' = W * style
    2. Demodula: W'' = W' / ||W'||

    Vantagens:
    - Remove artefatos de normalização
    - Mais estável que AdaIN
    - Melhor controle de estilo
    """

    def __init__(
            self,
            units: int,
            demodulate: bool = True,
            epsilon: float = 1e-8,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.demodulate = demodulate
        self.epsilon = epsilon
        self._built_in_features = None

    def build(self, input_shape):
        # input_shape é lista: [features_shape, style_shape]
        features_shape = input_shape[0]
        self._built_in_features = features_shape[-1]

        # Weight matrix
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self._built_in_features, self.units),
            initializer=GlorotUniform(),
            trainable=True,
            dtype=tf.float32
        )

        # Bias
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            dtype=tf.float32
        )

        # Style projection (se necessário)
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

        # Detecta número de features do input REAL
        actual_in_features = tf.shape(x)[-1]

        # Projeta style para dimensão dos input features
        s = self.style_proj(style)  # [batch, in_features]

        # Se o número de features não bater, ajusta o style
        if s.shape[-1] != x.shape[-1]:
            # Crop ou pad o style para bater com x
            style_features = s.shape[-1]
            x_features = x.shape[-1]

            if style_features < x_features:
                # Pad: adiciona uns (neutro para multiplicação)
                padding = x_features - style_features
                s = tf.pad(s, [[0, 0], [0, padding]], constant_values=0.0)
            else:
                # Crop
                s = s[:, :x_features]

        # Se kernel também não bater, recria dinamicamente
        if self._built_in_features != x.shape[-1]:
            # Usa apenas subset do kernel ou faz padding
            kernel_to_use = self.kernel

            if self._built_in_features < x.shape[-1]:
                # Input tem mais features que esperado - usa apenas primeiras N
                x = x[:, :self._built_in_features]
            else:
                # Input tem menos features - pad com zeros
                padding = self._built_in_features - x.shape[-1]
                x = tf.pad(x, [[0, 0], [0, padding]])

        # Modula pesos: W' = W * (s + 1)
        s_reshaped = tf.expand_dims(s + 1.0, axis=-1)  # [batch, in_features, 1]
        w = tf.expand_dims(self.kernel, axis=0)  # [1, in_features, units]
        w_modulated = w * s_reshaped  # [batch, in_features, units]

        if self.demodulate:
            # Demodula: W'' = W' / ||W'||_2
            demod = tf.math.rsqrt(
                tf.reduce_sum(tf.square(w_modulated), axis=1, keepdims=True) + self.epsilon
            )
            w_final = w_modulated * demod
        else:
            w_final = w_modulated

        # Aplica transformação: y = x @ W_final + b
        x_expanded = tf.expand_dims(x, axis=1)  # [batch, 1, in_features]
        output = tf.matmul(x_expanded, w_final)  # [batch, 1, units]
        output = tf.squeeze(output, axis=1)  # [batch, units]
        output = output + self.bias

        # Garante que o output tem o shape correto
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
    """
    Spectral Normalization para estabilização de GANs.
    """

    def __init__(
            self,
            layer: Layer,
            power_iterations: int = 1,
            **kwargs
    ):
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
    """Self-Attention otimizado para generator com auto-detecção de dimensões."""

    def __init__(self, reduction: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self._built_channels = None

    def build(self, input_shape):
        # Detecta automaticamente o número de canais do input
        self._built_channels = input_shape[-1]
        self.hidden_dim = max(self._built_channels // self.reduction, 8)

        # Cria layers com dimensões corretas
        self.query = Dense(
            self.hidden_dim,
            use_bias=False,
            dtype=tf.float32
        )
        self.key = Dense(
            self.hidden_dim,
            use_bias=False,
            dtype=tf.float32
        )
        self.value = Dense(
            self.hidden_dim,
            use_bias=False,
            dtype=tf.float32
        )

        # CRÍTICO: Output projection DEVE ter exatamente self._built_channels
        self.out_proj = Dense(
            self._built_channels,
            use_bias=False,
            dtype=tf.float32
        )

        self.gamma = self.add_weight(
            name='gamma',
            shape=[1],
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True,
            dtype=tf.float32
        )

        super().build(input_shape)

    def call(self, x):
        # x shape: (batch, features)
        input_features = x.shape[-1]

        q = self.query(x)  # [batch, hidden_dim]
        k = self.key(x)  # [batch, hidden_dim]
        v = self.value(x)  # [batch, hidden_dim]

        attention_weights = tf.nn.softmax(
            q * k / tf.sqrt(tf.cast(self.hidden_dim, tf.float32)),
            axis=-1
        )  # [batch, hidden_dim]

        attended = attention_weights * v  # [batch, hidden_dim]
        out = self.out_proj(attended)  # Should be [batch, built_channels]

        # Se por algum motivo o shape não bater, faz padding ou crop
        out_features = out.shape[-1]
        if out_features != input_features:
            if out_features < input_features:
                # Padding: adiciona zeros
                padding = input_features - out_features
                out = tf.pad(out, [[0, 0], [0, padding]])
            else:
                # Crop: remove excesso
                out = out[:, :input_features]

        # Residual connection
        return self.gamma * out + x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'reduction': self.reduction
        })
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


class VanillaGenerator(Activations):
    """
    Advanced Generator com técnicas SOTA:

    - Weight Demodulation (StyleGAN2) - substitui AdaIN
    - Alias-free operations (StyleGAN3)
    - Pixel Normalization (ProGAN, StyleGAN)
    - Spectral Normalization (estabilização)
    - Self-Attention otimizado
    - Residual connections com scaling
    - Noise injection (StyleGAN)
    - Mixed precision ready
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
            # ========== HYPERPARAMETERS ==========
            # Normalization
            use_pixel_norm: bool = True,
            use_spectral_norm: bool = False,
            spectral_norm_power_iterations: int = 1,
            norm_type: str = 'conditional_batch',
            # StyleGAN2/3
            use_weight_demodulation: bool = False,  # Experimental - desabilitado por padrão
            # NOTA: Weight Demod requer arquitetura muito específica (todas as camadas devem ter shapes consistentes)
            # Ative apenas se você entender completamente o funcionamento interno do StyleGAN2
            use_alias_free: bool = False,  # Experimental - desabilitado por padrão
            # NOTA: Alias-free operations podem causar problemas de shape em arquiteturas Dense
            # Funciona melhor com Conv2D (imagens). Para Dense, mantenha desabilitado.
            alias_free_cutoff: float = 1.0,
            # Conditioning
            conditioning_method: str = 'weight_demod',  # 'film', 'cbn', 'weight_demod'
            # Architecture
            use_self_attention: bool = True,
            attention_layers: List[int] = None,  # Se None, usa [3] (última camada)
            use_residual_connections: bool = True,
            residual_scaling: float = 0.2,
            bottleneck_factor: float = 0.25,
            # Regularization
            use_noise_injection: bool = True,
            noise_strength: float = 0.05,
            latent_noise_strength: float = 0.1,
            use_dropout: bool = True,
            # Latent space
            latent_mapping_layers: int = 1,
            latent_mapping_units: int = None,
            # Output
            use_tanh_output: bool = False,
            # Performance
            use_mixed_precision: bool = False,
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

        # Hyperparameters
        self._use_pixel_norm = use_pixel_norm
        self._use_spectral_norm = use_spectral_norm
        self._spectral_power_iters = spectral_norm_power_iterations
        self._norm_type = norm_type
        self._use_weight_demod = use_weight_demodulation
        self._use_alias_free = use_alias_free
        self._alias_cutoff = alias_free_cutoff
        self._cond_method = conditioning_method
        self._use_attn = use_self_attention
        # Aplica attention apenas na última camada por padrão (mais estável)
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

        self._model: Optional[Model] = None
        self._mapping_network: Optional[Model] = None

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
        """Adiciona ruído ao layer (StyleGAN noise injection)."""
        if not self._use_noise_inj:
            return x

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

        # Ativação
        x = self._add_activation_layer(x, self._activation_fn)

        # Low-pass filter
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
        """
        Bloco denso com Weight Demodulation opcional.

        Args:
            x: Input tensor
            units: Number of units
            name: Layer name
            use_residual: Whether to use residual connections
            style: Optional style tensor for Weight Demodulation
        """
        identity = x

        if self._use_weight_demod and style is not None:
            # Weight Demodulation (StyleGAN2)
            x = ModulatedDense(
                units=units,
                demodulate=True,
                name=f'{name}_modulated_dense'
            )([x, style])
        else:
            # Dense normal
            x = self._get_dense_layer(units, f'{name}_dense')(x)
            x = self._normalization_layer(x, name)

        # Ativação com alias-free
        x = self._alias_free_activation(x, name)

        # Noise injection
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
            # Self-attention detecta dimensões automaticamente no build()
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
        """
        Latent Mapping Network (StyleGAN).
        Mapeia latent code Z para latent code W (mais disentangled).
        """
        x = z_in

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
        """Constrói o generator avançado."""

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

        # Latent mapping network (StyleGAN-style)
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
        x = self._get_dense_layer(
            128,
            'initial_dense'
        )(w)

        if self._use_pixel_norm:
            x = PixelNormalization(name='initial_pn')(x)

        # Processing blocks com Weight Demodulation
        for i, units in enumerate([32]):
            # Dense block com style modulation
            use_res = self._use_residual and i > 0
            x = self._dense_block(
                x=x,
                units=units,
                name=f'block_{i}',
                use_residual=use_res,
                style=style  # Passa style para Weight Demodulation
            )

            # Attention
            x = self._attention_block(x, i)

        # Output block
        if self._use_weight_demod and style is not None:
            x = ModulatedDense(
                units=self._output_shape,
                demodulate=False,  # No demodulation no output
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
        print("\n" + "=" * 60)
        print("🚀 Advanced Generator Configuration:")
        print("=" * 60)
        if self._use_weight_demod:
            print("✓ Weight Demodulation (StyleGAN2) enabled")
        else:
            print("✗ Weight Demodulation disabled (using standard Dense)")
        if self._use_alias_free:
            print(f"✓ Alias-free operations (StyleGAN3) enabled (cutoff={self._alias_cutoff})")
        else:
            print("✗ Alias-free operations disabled (standard activations)")
        if self._use_spectral_norm:
            print(f"✓ Spectral Normalization enabled ({self._spectral_power_iters} iterations)")
        if self._use_pixel_norm:
            print("✓ Pixel Normalization enabled")
        if self._use_attn:
            print(f"✓ Self-Attention enabled at layers {self._attn_layers}")
        if self._use_noise_inj:
            print(f"✓ Noise Injection enabled (strength={self._noise_str})")
        print("=" * 60 + "\n")

        model.summary()

        return model

    # ==================================================================
    # UTILITY METHODS
    # ==================================================================

    def sample_latent(self, batch_size: int, seed: Optional[int] = None) -> np.ndarray:
        """Gera amostras do espaço latente."""
        if seed is not None:
            np.random.seed(seed)
        return np.random.randn(batch_size, self._latent_dim).astype(np.float32)

    def truncation_trick(
            self,
            z: np.ndarray,
            truncation_psi: float = 0.7,
            truncation_cutoff: Optional[int] = None
    ) -> np.ndarray:
        """Truncation trick (BigGAN, StyleGAN) para melhor qualidade."""
        z_mean = np.mean(z, axis=0, keepdims=True)

        if truncation_cutoff is not None:
            z_truncated = z.copy()
            z_truncated[:, :truncation_cutoff] = (
                    z_mean[:, :truncation_cutoff] +
                    truncation_psi * (z[:, :truncation_cutoff] - z_mean[:, :truncation_cutoff])
            )
            return z_truncated
        else:
            return z_mean + truncation_psi * (z - z_mean)

    # ==================================================================
    # GETTERS
    # ==================================================================

    def get_model(self) -> Optional[Model]:
        """Retorna o modelo construído."""
        return self._model

    def get_mapping_network(self) -> Optional[Model]:
        """Retorna a rede de mapeamento latente."""
        return self._mapping_network

    @property
    def trainable_variables(self):
        """Retorna variáveis treináveis."""
        return self._model.trainable_variables if self._model else []

    def get_config(self) -> Dict:
        """Retorna configuração do generator."""
        return {
            'latent_dimension': self._latent_dim,
            'output_shape': self._output_shape,
            'dense_layer_sizes': self._dense_sizes,
            'use_pixel_norm': self._use_pixel_norm,
            'use_spectral_norm': self._use_spectral_norm,
            'spectral_norm_power_iterations': self._spectral_power_iters,
            'use_weight_demodulation': self._use_weight_demod,
            'use_alias_free': self._use_alias_free,
            'alias_free_cutoff': self._alias_cutoff,
            'norm_type': self._norm_type,
            'conditioning_method': self._cond_method,
            'use_self_attention': self._use_attn,
            'use_noise_injection': self._use_noise_inj,
            'use_residual_connections': self._use_residual,
            'latent_mapping_layers': self._latent_map_layers,
            'mixed_precision': self._mixed_precision
        }