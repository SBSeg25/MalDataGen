#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{2}.{0}.{0}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/18'
__credits__ = ['Synthetic Ocean AI']

# MIT License
# Copyright (c) 2025 Synthetic Ocean AI

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


class SpectralDense(Layer):
    """
    Dense layer com Spectral Normalization integrada.
    Mais eficiente e compatível com tf.function.
    """

    def __init__(
            self,
            units: int,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            use_spectral_norm=True,
            n_iterations=1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.use_spectral_norm = use_spectral_norm
        self.n_iterations = n_iterations

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=[input_shape[-1], self.units],
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

        if self.use_spectral_norm:
            self.u = self.add_weight(
                name='sn_u',
                shape=[1, self.units],
                initializer='random_normal',
                trainable=False
            )

        super().build(input_shape)

    def call(self, x, training=None):
        kernel = self.kernel

        # Apply spectral normalization
        if self.use_spectral_norm:
            w_shape = kernel.shape
            w_mat = tf.reshape(kernel, [-1, w_shape[-1]])

            u = self.u

            # Power iteration
            for _ in range(self.n_iterations):
                v = tf.nn.l2_normalize(tf.matmul(u, tf.transpose(w_mat)))
                u = tf.nn.l2_normalize(tf.matmul(v, w_mat))

            # Update u during training
            if training:
                self.u.assign(u)

            # Compute spectral norm
            sigma = tf.matmul(tf.matmul(v, w_mat), tf.transpose(u))
            kernel = kernel / (sigma + 1e-8)

        # Dense operation
        output = tf.matmul(x, kernel)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'use_spectral_norm': self.use_spectral_norm,
            'n_iterations': self.n_iterations
        })
        return config


class EfficientChannelAttention(Layer):
    """
    ECA adaptado para features densas (1D).
    Usa convolução 1D ao invés de 2D para vetores de features.
    """

    def __init__(self, gamma: int = 2, b: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.b = b

    def build(self, input_shape):
        channels = input_shape[-1]
        # Calcula kernel adaptativo
        t = int(abs(math.log2(channels) + self.b) / self.gamma)
        self.k_size = max(3, t if t % 2 else t + 1)

        # Convolução 1D para features densas
        self.conv = tf.keras.layers.Conv1D(
            1,
            self.k_size,
            padding='same',
            use_bias=False,
            name='eca_conv1d'
        )
        super().build(input_shape)

    def call(self, x):
        # x shape: (batch, features)
        # Expande para (batch, features, 1) para Conv1D
        x_expanded = tf.expand_dims(x, axis=-1)

        # Apply 1D convolution
        attention = self.conv(x_expanded)  # (batch, features, 1)
        attention = tf.nn.sigmoid(attention)
        attention = tf.squeeze(attention, axis=-1)  # (batch, features)

        # Apply attention
        return x * attention

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'b': self.b
        })
        return config


class SqueezeExcitation(Layer):
    """
    Squeeze-and-Excitation block adaptado para features densas.
    Alternativa mais simples e robusta ao ECA.
    """

    def __init__(self, ratio: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.fc1 = Dense(
            max(channels // self.ratio, 8),
            activation='relu',
            use_bias=False,
            name='se_fc1'
        )
        self.fc2 = Dense(
            channels,
            activation='sigmoid',
            use_bias=False,
            name='se_fc2'
        )
        super().build(input_shape)

    def call(self, x):
        # Global average pooling já está implícito (features densas)
        squeeze = self.fc1(x)
        excitation = self.fc2(squeeze)
        return x * excitation

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'ratio': self.ratio})
        return config


class ConditionalBatchNorm(Layer):
    """Conditional Batch Normalization - melhor que FiLM para GANs."""

    def __init__(self, num_features: int, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.num_classes = num_classes

    def build(self, input_shape):
        self.bn = BatchNormalization(scale=False, center=False)
        self.gamma_embed = Dense(self.num_features, use_bias=False)
        self.beta_embed = Dense(self.num_features, use_bias=False)
        super().build(input_shape)

    def call(self, inputs):
        x, y = inputs
        x = self.bn(x)
        gamma = self.gamma_embed(y)
        beta = self.beta_embed(y)
        return x * (1 + gamma) + beta

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_features': self.num_features,
            'num_classes': self.num_classes
        })
        return config


class SelfAttention(Layer):
    """
    Self-Attention para features densas (feature-wise attention).
    Calcula atenção entre diferentes dimensões do vetor de features.
    """

    def __init__(self, channels: int, reduction: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.hidden_dim = max(channels // reduction, 8)

    def build(self, input_shape):
        # Query, Key, Value projections
        self.query = Dense(self.hidden_dim, use_bias=False, name='attn_q')
        self.key = Dense(self.hidden_dim, use_bias=False, name='attn_k')
        self.value = Dense(self.hidden_dim, use_bias=False, name='attn_v')

        # Output projection
        self.out_proj = Dense(self.channels, use_bias=False, name='attn_out')

        # Learnable residual weight
        self.gamma = self.add_weight(
            name='attn_gamma',
            shape=[1],
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        # x shape: (batch, channels)
        batch_size = tf.shape(x)[0]

        # Project to query, key, value
        q = self.query(x)  # [B, H']
        k = self.key(x)  # [B, H']
        v = self.value(x)  # [B, H']

        # Compute attention: queremos atenção "global" sobre as features
        # Para batch de features, fazemos atenção entre samples
        # Expandir para fazer matmul: [B, 1, H'] @ [B, H', 1] = [B, 1, 1]

        # Alternativa: atenção elemento-wise (mais simples e eficiente)
        attention_weights = tf.nn.softmax(
            q * k / tf.sqrt(tf.cast(self.hidden_dim, tf.float32)),
            axis=-1
        )  # [B, H']

        # Apply attention
        attended = attention_weights * v  # [B, H']

        # Project back to original dimension
        out = self.out_proj(attended)  # [B, C]

        # Residual connection with learnable weight
        return self.gamma * out + x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'reduction': self.reduction
        })
        return config


class VanillaDiscriminator(Activations):
    """
    Advanced Discriminator com técnicas SOTA:

    - Spectral Normalization (SNGAN, 2018)
    - Conditional Batch Normalization (cGAN improvements)
    - Efficient Channel Attention ou Squeeze-Excitation
    - Self-Attention otimizado
    - Minibatch Discrimination otimizado
    - Progressive Growing ready
    - Mixed Precision ready
    - Gradient Penalty ready
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
            # ========== HYPERPARAMETERS ==========
            # Regularization
            use_spectral_norm: bool = False,
            noise_stddev: float = 0.0,
            use_input_noise: bool = False,
            label_smoothing: float = 0.0,
            # Normalization
            norm_type: str = 'layer',  # 'batch', 'layer', 'none'
            use_conditional_bn: bool = True,
            # Attention & Features
            use_self_attention: bool = True,
            attention_layers: List[int] = None,  # Indices de layers com attention
            use_channel_attention: bool = True,
            channel_attention_type: str = 'se',  # 'se' (Squeeze-Excitation) ou 'eca'
            # Minibatch Discrimination
            use_minibatch_stddev: bool = True,
            minibatch_stddev_group_size: int = 8,
            minibatch_stddev_averaging: str = 'all',  # 'all' ou 'spatial'
            # Architecture
            use_residual_connections: bool = True,
            residual_scaling: float = 0.1,
            bottleneck_factor: float = 0.25,
            # Output
            use_gradient_penalty: bool = False,
            output_units: int = 1,
            # Performance
            use_mixed_precision: bool = False,
    ) -> None:

        # Validações
        if latent_dimension <= 0:
            raise ValueError("latent_dimension must be > 0")
        if not 0.0 <= dropout_decay_rate_d <= 1.0:
            raise ValueError("dropout_decay_rate_d must be in [0, 1]")
        if not dense_layer_sizes_d:
            raise ValueError("dense_layer_sizes_d cannot be empty")

        # Atributos básicos
        self._latent_dim = latent_dimension
        self._output_shape = output_shape
        self._activation_fn = activation_function
        self._last_activation = last_layer_activation
        self._dropout = dropout_decay_rate_d
        self._dense_sizes = dense_layer_sizes_d
        self._dtype = dataset_type
        self._init_mean = initializer_mean
        self._init_std = initializer_deviation
        self._class_info = number_samples_per_class

        # Hyperparameters
        self._use_sn = use_spectral_norm
        self._noise_std = noise_stddev
        self._use_noise = use_input_noise
        self._label_smooth = label_smoothing
        self._norm_type = norm_type
        self._use_cbn = use_conditional_bn
        self._use_attn = use_self_attention
        self._attn_layers = attention_layers or [len(dense_layer_sizes_d) // 2]
        self._use_ch_attn = use_channel_attention
        self._ch_attn_type = channel_attention_type
        self._use_mbstd = use_minibatch_stddev
        self._mb_group = minibatch_stddev_group_size
        self._mb_avg = minibatch_stddev_averaging
        self._use_residual = use_residual_connections
        self._res_scale = residual_scaling
        self._bottleneck = bottleneck_factor
        self._use_gp = use_gradient_penalty
        self._output_units = output_units
        self._mixed_precision = use_mixed_precision

        self._model: Optional[Model] = None

    # ==================================================================
    # UTILITY LAYERS
    # ==================================================================

    def _get_dense_layer(self, units: int, use_bias: bool = True, name: str = None):
        """Retorna Dense layer com ou sem Spectral Normalization."""
        init = self._get_initializer()

        if self._use_sn:
            return SpectralDense(
                units,
                use_bias=use_bias,
                kernel_initializer=init,
                use_spectral_norm=True,
                name=name
            )
        else:
            return Dense(
                units,
                use_bias=use_bias,
                kernel_initializer=init,
                name=name
            )

    def _get_kernel_constraint(self):
        """Retorna constraint baseado em configurações."""
        # Não usado mais, mas mantido para compatibilidade
        return None

    def _get_initializer(self):
        """Retorna inicializador otimizado."""
        if self._use_sn:
            return HeNormal()  # Melhor com Spectral Norm
        return RandomNormal(self._init_mean, self._init_std)

    def _add_noise_layer(self, x: Layer) -> Layer:
        """Adiciona ruído gaussiano no input (regularização)."""
        if not self._use_noise:
            return x

        @tf.function
        def add_noise(t):
            noise = tf.random.normal(
                tf.shape(t),
                stddev=self._noise_std,
                dtype=t.dtype
            )
            return t + noise

        return Lambda(add_noise, name='input_noise')(x)

    def _minibatch_stddev_layer(self, x: Layer) -> Layer:
        """
        Minibatch Standard Deviation (ProGAN, StyleGAN)
        Otimizado para performance.
        """
        if not self._use_mbstd:
            return x

        @tf.function
        def compute_mbstd(features):
            group_size = tf.minimum(self._mb_group, tf.shape(features)[0])

            # Reshape para grupos
            s = tf.shape(features)
            y = tf.reshape(features, [group_size, -1, s[1]])
            y = tf.cast(y, tf.float32)

            # Compute stddev
            mean = tf.reduce_mean(y, axis=0, keepdims=True)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(y - mean), axis=0) + 1e-8)

            # Average over features e spatial
            if self._mb_avg == 'all':
                stddev = tf.reduce_mean(stddev)
            else:
                stddev = tf.reduce_mean(stddev, axis=1, keepdims=True)

            # Broadcast to batch
            stddev = tf.tile(
                tf.reshape(stddev, [1, -1]),
                [s[0], 1]
            )

            return tf.concat([features, stddev], axis=1)

        return Lambda(compute_mbstd, name='minibatch_stddev')(x)

    def _normalization_layer(self, x: Layer, name: str) -> Layer:
        """Adiciona normalização baseada em configuração."""
        if self._norm_type == 'batch':
            return BatchNormalization(name=f'{name}_bn')(x)
        elif self._norm_type == 'layer':
            return LayerNormalization(name=f'{name}_ln')(x)
        return x

    # ==================================================================
    # BUILDING BLOCKS
    # ==================================================================

    def _dense_block(
            self,
            x: Layer,
            units: int,
            name: str,
            use_residual: bool = False
    ) -> Layer:
        """
        Bloco denso otimizado com:
        - Spectral Normalization opcional
        - Bottleneck architecture para eficiência
        - Residual connections
        """
        identity = x

        # Bottleneck (reduz computação)
        if self._bottleneck < 1.0 and units > 64:
            bottleneck_units = max(int(units * self._bottleneck), 32)

            # Down-projection
            x = self._get_dense_layer(
                bottleneck_units,
                name=f'{name}_bottleneck_down'
            )(x)
            x = self._normalization_layer(x, f'{name}_bottleneck')
            x = self._add_activation_layer(x, self._activation_fn)

            # Up-projection
            x = self._get_dense_layer(
                units,
                name=f'{name}_bottleneck_up'
            )(x)
        else:
            x = self._get_dense_layer(
                units,
                name=f'{name}_dense'
            )(x)

        x = self._normalization_layer(x, name)
        x = self._add_activation_layer(x, self._activation_fn)
        x = Dropout(self._dropout, name=f'{name}_dropout')(x)

        # Residual connection
        if use_residual and identity.shape[-1] == units:
            x = Lambda(
                lambda t: t[0] + self._res_scale * t[1],
                name=f'{name}_residual'
            )([identity, x])

        return x

    def _attention_block(self, x: Layer, layer_idx: int) -> Layer:
        """Adiciona attention mechanisms."""
        name = f'layer_{layer_idx}'

        # Self-Attention
        if self._use_attn and layer_idx in self._attn_layers:
            units = x.shape[-1]
            x = SelfAttention(units, name=f'{name}_self_attn')(x)

        # Channel Attention (SE ou ECA)
        if self._use_ch_attn:
            if self._ch_attn_type == 'eca':
                x = EfficientChannelAttention(name=f'{name}_eca')(x)
            else:  # 'se' (default, mais robusto)
                x = SqueezeExcitation(ratio=8, name=f'{name}_se')(x)

        return x

    # ==================================================================
    # MODEL BUILDING
    # ==================================================================

    def get_discriminator(self) -> Model:
        """Constrói o discriminador avançado."""

        if not self._class_info:
            raise ValueError("number_samples_per_class is required")

        # Mixed precision
        if self._mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)

        num_classes = self._class_info['number_classes']
        input_size = int(np.prod(self._output_shape))

        # Inputs
        x_in = Input(shape=(input_size,), dtype=self._dtype, name='input_real')
        y_in = Input(shape=(num_classes,), dtype=self._dtype, name='input_label')

        # Adiciona ruído ao input
        x = self._add_noise_layer(x_in)

        # Minibatch stddev
        x = self._minibatch_stddev_layer(x)

        # Modulation inicial com label (conditional)
        if self._use_cbn:
            # Embedding da label
            label_embed = self._get_dense_layer(
                input_size,
                name='label_embedding'
            )(y_in)

            # Concatena com features
            x = Concatenate(name='concat_label')([x, label_embed])

            # Projection layer
            x = self._get_dense_layer(
                32,
                name='input_projection'
            )(x)

        # Processing blocks
        for i, units in enumerate([32]):
            # Dense block
            use_res = self._use_residual and i > 0
            x = self._dense_block(x, units, f'block_{i}', use_res)

            # Conditional BN (se aplicável)
            if self._use_cbn and i < len(self._dense_sizes) - 1:
                x = ConditionalBatchNorm(units, num_classes, name=f'cbn_{i}')(
                    [x, y_in]
                )

            # Attention mechanisms
            x = self._attention_block(x, i)

        # Output layer
        x = self._get_dense_layer(
            self._output_units,
            name='output_logits'
        )(x)

        if self._last_activation is not None:
            x = self._add_activation_layer(x, self._last_activation)

        # Build model
        model = Model(
            inputs=[x_in, y_in],
            outputs=x,
            name='AdvancedDiscriminator'
        )

        self._model = model
        model.summary()

        return model

    # ==================================================================
    # GRADIENT PENALTY (para WGAN-GP)
    # ==================================================================

    @tf.function
    def gradient_penalty(
            self,
            real_samples: tf.Tensor,
            fake_samples: tf.Tensor,
            labels: tf.Tensor,
            lambda_gp: float = 10.0
    ) -> tf.Tensor:
        """
        Computa gradient penalty para WGAN-GP.
        Uso: adicione ao loss do discriminador.
        """
        if not self._use_gp:
            return tf.constant(0.0)

        batch_size = tf.shape(real_samples)[0]
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)

        # Interpolação
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples

        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self._model([interpolated, labels], training=True)

        gradients = tape.gradient(pred, interpolated)

        # Compute gradient penalty
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1) + 1e-8)
        gp = tf.reduce_mean(tf.square(slopes - 1.0))

        return lambda_gp * gp

    # ==================================================================
    # GETTERS
    # ==================================================================

    def get_model(self) -> Optional[Model]:
        """Retorna o modelo construído."""
        return self._model

    @property
    def trainable_variables(self):
        """Retorna variáveis treináveis."""
        return self._model.trainable_variables if self._model else []

    def get_config(self) -> Dict:
        """Retorna configuração do discriminador."""
        return {
            'latent_dimension': self._latent_dim,
            'output_shape': self._output_shape,
            'dense_layer_sizes': self._dense_sizes,
            'use_spectral_norm': self._use_sn,
            'use_self_attention': self._use_attn,
            'use_channel_attention': self._use_ch_attn,
            'channel_attention_type': self._ch_attn_type,
            'use_minibatch_stddev': self._use_mbstd,
            'norm_type': self._norm_type,
            'use_conditional_bn': self._use_cbn,
            'use_residual_connections': self._use_residual,
            'mixed_precision': self._mixed_precision
        }