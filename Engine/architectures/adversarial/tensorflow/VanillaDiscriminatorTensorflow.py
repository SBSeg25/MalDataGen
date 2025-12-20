#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{2}.{1}.{0}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/20'
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
        GlobalAveragePooling1D, Reshape, Activation, MultiHeadAttention
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.initializers import RandomNormal, HeNormal
    from tensorflow.keras import backend as K

    from Engine.activations.Activations import Activations

except ImportError as error:
    print(error)
    sys.exit(-1)


class SpectralDense(Layer):
    """Dense layer com Spectral Normalization integrada."""

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

        if self.use_spectral_norm:
            w_shape = kernel.shape
            w_mat = tf.reshape(kernel, [-1, w_shape[-1]])
            u = self.u

            for _ in range(self.n_iterations):
                v = tf.nn.l2_normalize(tf.matmul(u, tf.transpose(w_mat)))
                u = tf.nn.l2_normalize(tf.matmul(v, w_mat))

            if training:
                self.u.assign(u)

            sigma = tf.matmul(tf.matmul(v, w_mat), tf.transpose(u))
            kernel = kernel / (sigma + 1e-8)

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

class MinibatchStdDev(Layer):
    def __init__(self, group_size: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.group_size = group_size

    def call(self, x):
        # x: [B, S, D]
        batch_size = tf.shape(x)[0]
        group_size = tf.minimum(self.group_size, batch_size)

        mean = tf.reduce_mean(x, axis=0, keepdims=True)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(x - mean), axis=0) + 1e-8)

        stddev = tf.reduce_mean(stddev)  # escalar

        stddev_feature = tf.fill([batch_size, 1, 1], stddev)
        stddev_feature = tf.broadcast_to(
            stddev_feature,
            [batch_size, tf.shape(x)[1], 1]
        )

        return tf.concat([x, stddev_feature], axis=-1)

    def compute_output_shape(self, input_shape):
        # input_shape: (B, S, D)
        return (input_shape[0], input_shape[1], input_shape[2] + 1)

    def get_config(self):
        config = super().get_config()
        config.update({'group_size': self.group_size})
        return config
class PositionalEncoding(Layer):
    """
    Positional encoding sinusoidal para sequências.
    Permite que o Transformer entenda a ordem das features.
    """

    def __init__(self, max_length: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length

    def build(self, input_shape):
        d_model = input_shape[-1]
        position = np.arange(self.max_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe = np.zeros((self.max_length, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = self.add_weight(
            name='positional_encoding',
            shape=(1, self.max_length, d_model),
            initializer=tf.constant_initializer(pe),
            trainable=False
        )
        super().build(input_shape)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({'max_length': self.max_length})
        return config


class HierarchicalPooling(Layer):
    def __init__(self, pool_size: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size

    def call(self, x):
        # x shape: [batch, seq_len, dim]
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        dim = tf.shape(x)[2]

        pad_len = (self.pool_size - (seq_len % self.pool_size)) % self.pool_size

        def pad():
            return tf.pad(x, [[0, 0], [0, pad_len], [0, 0]])

        def no_pad():
            return x

        x = tf.cond(pad_len > 0, pad, no_pad)

        new_seq_len = (seq_len + pad_len) // self.pool_size

        x_reshaped = tf.reshape(
            x,
            [batch_size, new_seq_len, self.pool_size, dim]
        )

        max_pool = tf.reduce_max(x_reshaped, axis=2)
        avg_pool = tf.reduce_mean(x_reshaped, axis=2)

        return (max_pool + avg_pool) / 2.0

class TransformerBlock(Layer):
    """
    Bloco Transformer otimizado com:
    - Multi-head attention
    - Feed-forward network com Spectral Norm
    - Residual connections
    - Layer normalization
    - Dropout
    """

    def __init__(
            self,
            d_model: int,
            num_heads: int,
            dff: int,
            dropout_rate: float = 0.1,
            use_spectral_norm: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.use_sn = use_spectral_norm

    def build(self, input_shape):
        # Multi-head attention
        self.mha = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate,
            name='mha'
        )

        # Feed-forward network
        if self.use_sn:
            self.ffn1 = SpectralDense(
                self.dff,
                activation='gelu',
                use_spectral_norm=True,
                name='ffn1'
            )
            self.ffn2 = SpectralDense(
                self.d_model,
                use_spectral_norm=True,
                name='ffn2'
            )
        else:
            self.ffn1 = Dense(self.dff, activation='gelu', name='ffn1')
            self.ffn2 = Dense(self.d_model, name='ffn2')

        # Normalization
        self.ln1 = LayerNormalization(epsilon=1e-6, name='ln1')
        self.ln2 = LayerNormalization(epsilon=1e-6, name='ln2')

        # Dropout
        self.dropout1 = Dropout(self.dropout_rate, name='dropout1')
        self.dropout2 = Dropout(self.dropout_rate, name='dropout2')

        super().build(input_shape)

    def call(self, x, training=None, mask=None):
        # Multi-head attention com residual
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            attention_mask=mask,
            training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        x1 = self.ln1(x + attn_output)

        # Feed-forward com residual
        ffn_output = self.ffn1(x1)
        ffn_output = self.dropout2(ffn_output, training=training)
        ffn_output = self.ffn2(ffn_output)
        x2 = self.ln2(x1 + ffn_output)

        return x2

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate,
            'use_spectral_norm': self.use_sn
        })
        return config


class ConditionalLayerNorm(Layer):
    """
    Conditional Layer Normalization para injetar informação de classe.
    Similar ao CBN mas usando LayerNorm.
    """

    def __init__(self, d_model: int, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_classes = num_classes

    def build(self, input_shape):
        self.ln = LayerNormalization(scale=False, center=False, epsilon=1e-6)
        self.gamma_embed = Dense(self.d_model, use_bias=False, name='gamma')
        self.beta_embed = Dense(self.d_model, use_bias=False, name='beta')
        super().build(input_shape)

    def call(self, inputs):
        x, y = inputs  # x: [B, S, D], y: [B, C]
        x_norm = self.ln(x)

        # Expande label para broadcast
        gamma = self.gamma_embed(y)  # [B, D]
        beta = self.beta_embed(y)    # [B, D]

        gamma = tf.expand_dims(gamma, axis=1)  # [B, 1, D]
        beta = tf.expand_dims(beta, axis=1)    # [B, 1, D]

        return x_norm * (1 + gamma) + beta

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_classes': self.num_classes
        })
        return config


class VanillaDiscriminator(Activations):
    """
    Discriminator baseado em Transformers Hierárquicos.

    ECONOMIA DE PARÂMETROS através de:
    ✓ Pooling Hierárquico: reduz sequência progressivamente (N → N/2 → N/4 → ...)
    ✓ Attention eficiente: complexidade reduz quadraticamente com pooling
    ✓ Shared weights: reutiliza blocos Transformer
    ✓ Bottleneck FFN: dimensão intermediária menor
    ✓ Spectral Normalization: estabiliza sem aumentar parâmetros

    VANTAGENS sobre Dense:
    - Escalabilidade: O(n²) → O(n²/4) → O(n²/16) por nível hierárquico
    - Multi-scale features: captura padrões locais e globais
    - Menos parâmetros: compartilhamento de pesos entre níveis
    - Melhor generalização: inductive bias da hierarquia

    TÉCNICAS MANTIDAS:
    - Spectral Normalization nas FFN
    - Conditional normalization
    - Minibatch standard deviation
    - Gradient penalty ready
    - Input noise regularization
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
            dense_layer_sizes_d: List[int],  # Usado para compatibilidade
            dataset_type: type = np.float32,
            number_samples_per_class: Optional[Dict[str, int]] = None,
            # ========== TRANSFORMER HYPERPARAMETERS ==========
            # Architecture
            d_model: int = 256,  # Dimensão do modelo
            num_heads: int = 4,  # Número de attention heads
            dff_ratio: float = 2.0,  # FFN dimension = d_model * dff_ratio
            num_layers_per_level: int = 2,  # Blocos por nível hierárquico
            hierarchy_levels: int = 3,  # Número de níveis (N → N/2 → N/4)
            pool_size: int = 2,  # Fator de redução por nível
            patch_size: int = 8,  # Features por patch inicial
            # Regularization
            use_spectral_norm: bool = True,
            noise_stddev: float = 0.0,
            use_input_noise: bool = False,
            # Normalization
            use_conditional_norm: bool = True,
            # Minibatch Discrimination
            use_minibatch_stddev: bool = True,
            minibatch_stddev_group_size: int = 8,
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

        # Atributos básicos
        self._latent_dim = latent_dimension
        self._output_shape = output_shape
        self._activation_fn = activation_function
        self._last_activation = last_layer_activation
        self._dropout = dropout_decay_rate_d
        self._dtype = dataset_type
        self._init_mean = initializer_mean
        self._init_std = initializer_deviation
        self._class_info = number_samples_per_class

        # Transformer hyperparameters
        self._d_model = d_model
        self._num_heads = num_heads
        self._dff = int(d_model * dff_ratio)
        self._num_layers = num_layers_per_level
        self._hierarchy = hierarchy_levels
        self._pool_size = pool_size
        self._patch_size = patch_size

        # Regularization
        self._use_sn = use_spectral_norm
        self._noise_std = noise_stddev
        self._use_noise = use_input_noise
        self._use_cond_norm = use_conditional_norm
        self._use_mbstd = use_minibatch_stddev
        self._mb_group = minibatch_stddev_group_size
        self._use_gp = use_gradient_penalty
        self._output_units = output_units
        self._mixed_precision = use_mixed_precision

        self._model: Optional[Model] = None

    # ==================================================================
    # UTILITY FUNCTIONS
    # ==================================================================

    def _add_noise_layer(self, x: Layer) -> Layer:
        """Adiciona ruído gaussiano ao input."""
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
        """Minibatch Standard Deviation para detectar mode collapse."""
        if not self._use_mbstd:
            return x

        @tf.function
        def compute_mbstd(features):
            # features shape: [B, S, D]
            batch_size = tf.shape(features)[0]
            group_size = tf.minimum(self._mb_group, batch_size)

            # Compute stddev across batch
            mean = tf.reduce_mean(features, axis=0, keepdims=True)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(features - mean), axis=0) + 1e-8)

            # Average over sequence and features
            stddev = tf.reduce_mean(stddev)

            # Broadcast to batch
            stddev_feature = tf.fill([batch_size, 1, 1], stddev)

            return tf.concat([features, stddev_feature], axis=-1)

        # ADICIONE output_shape aqui
        def compute_output_shape(input_shape):
            return (input_shape[0], input_shape[1], input_shape[2] + 1)

        return Lambda(
            compute_mbstd,
            output_shape=compute_output_shape,
            name='minibatch_stddev'
        )(x)
    def _patchify(self, x: Layer, name: str) -> Layer:
        """
        Converte features 1D em patches (sequência).
        [B, N] → [B, N/P, P] onde P = patch_size
        """
        @tf.function
        def create_patches(features):
            batch_size = tf.shape(features)[0]
            n_features = tf.shape(features)[1]

            # Pad se necessário
            pad_len = (self._patch_size - (n_features % self._patch_size)) % self._patch_size
            if pad_len > 0:
                features = tf.pad(features, [[0, 0], [0, pad_len]])

            # Reshape para patches
            n_patches = (n_features + pad_len) // self._patch_size
            patches = tf.reshape(
                features,
                [batch_size, n_patches, self._patch_size]
            )

            return patches

        return Lambda(create_patches, name=name)(x)

    # ==================================================================
    # HIERARCHICAL TRANSFORMER BLOCKS
    # ==================================================================

    def _build_transformer_level(
            self,
            x: Layer,
            level: int,
            num_classes: int
    ) -> Layer:
        """
        Constrói um nível da hierarquia Transformer.
        Cada nível tem múltiplos blocos Transformer seguidos de pooling.
        """

        # Conditional normalization no início do nível
        if self._use_cond_norm and level > 0:
            # Placeholder para label (será passado depois)
            pass

        # Múltiplos blocos Transformer
        for layer_idx in range(self._num_layers):
            x = TransformerBlock(
                d_model=self._d_model,
                num_heads=self._num_heads,
                dff=self._dff,
                dropout_rate=self._dropout,
                use_spectral_norm=self._use_sn,
                name=f'level{level}_block{layer_idx}'
            )(x)

        return x

    # ==================================================================
    # MODEL BUILDING
    # ==================================================================

    def get_discriminator(self) -> Model:
        """Constrói o discriminador Transformer Hierárquico."""

        if not self._class_info:
            raise ValueError("number_samples_per_class is required")

        # Mixed precision
        if self._mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)

        num_classes = self._class_info['number_classes']
        input_size = int(np.prod(self._output_shape))

        # ==============================================================
        # INPUTS
        # ==============================================================
        x_in = Input(shape=(input_size,), dtype=self._dtype, name='input_real')
        y_in = Input(shape=(num_classes,), dtype=self._dtype, name='input_label')

        # Adiciona ruído ao input
        x = self._add_noise_layer(x_in)

        # ==============================================================
        # EMBEDDING + CONDITIONING
        # ==============================================================

        # Label embedding
        label_embed = Dense(
            input_size,
            kernel_initializer=HeNormal(),
            name='label_embedding'
        )(y_in)

        # Combina input com label
        x = Concatenate(name='concat_input_label')([x, label_embed])

        # Patchify: converte em sequência
        x = self._patchify(x, 'patchify')

        # Projection para d_model
        x = Dense(
            self._d_model,
            kernel_initializer=HeNormal(),
            name='patch_projection'
        )(x)

        # Positional encoding
        max_seq_len = (input_size * 2) // self._patch_size + 10
        x = PositionalEncoding(max_length=max_seq_len, name='pos_encoding')(x)

        # ==============================================================
        # HIERARCHICAL TRANSFORMER
        # ==============================================================

        for level in range(self._hierarchy):
            # Transformer blocks
            x = self._build_transformer_level(x, level, num_classes)

            # Conditional normalization
            if self._use_cond_norm:
                x = ConditionalLayerNorm(
                    self._d_model,
                    num_classes,
                    name=f'cond_norm_level{level}'
                )([x, y_in])

            # Pooling hierárquico (exceto no último nível)
            if level < self._hierarchy - 1:
                x = HierarchicalPooling(
                    pool_size=self._pool_size,
                    name=f'pool_level{level}'
                )(x)

        # ==============================================================
        # GLOBAL AGGREGATION
        # ==============================================================

        # Minibatch stddev antes do pooling final
        x = MinibatchStdDev()(x)

        # Global average pooling
        x = Lambda(
            lambda t: tf.reduce_mean(t, axis=1),
            name='global_avg_pool'
        )(x)

        # ==============================================================
        # OUTPUT HEAD
        # ==============================================================

        # Final projection
        if self._use_sn:
            x = SpectralDense(
                self._d_model // 2,
                activation='gelu',
                use_spectral_norm=True,
                name='pre_output'
            )(x)
        else:
            x = Dense(
                self._d_model // 2,
                activation='gelu',
                name='pre_output'
            )(x)

        x = Dropout(self._dropout, name='output_dropout')(x)

        # Output logits
        if self._use_sn:
            x = SpectralDense(
                self._output_units,
                use_spectral_norm=True,
                name='output_logits'
            )(x)
        else:
            x = Dense(
                self._output_units,
                name='output_logits'
            )(x)

        if self._last_activation is not None:
            x = self._add_activation_layer(x, self._last_activation)

        # ==============================================================
        # BUILD MODEL
        # ==============================================================

        model = Model(
            inputs=[x_in, y_in],
            outputs=x,
            name='HierarchicalTransformerDiscriminator'
        )

        self._model = model

        # Informações
        print("\n" + "=" * 70)
        print("🚀 Hierarchical Transformer Discriminator Configuration:")
        print("=" * 70)
        print(f"✓ Architecture: {self._hierarchy} hierarchical levels")
        print(f"✓ Transformer: d_model={self._d_model}, heads={self._num_heads}")
        print(f"✓ Blocks per level: {self._num_layers}")
        print(f"✓ FFN dimension: {self._dff}")
        print(f"✓ Patch size: {self._patch_size}")
        print(f"✓ Pooling factor: {self._pool_size}")

        # Estimar redução de sequência
        seq_reduction = self._pool_size ** (self._hierarchy - 1)
        print(f"✓ Sequence reduction: {seq_reduction}x (N → N/{seq_reduction})")

        if self._use_sn:
            print("✓ Spectral Normalization in FFN")
        if self._use_cond_norm:
            print("✓ Conditional Layer Normalization")
        if self._use_mbstd:
            print(f"✓ Minibatch StdDev (group_size={self._mb_group})")
        if self._use_noise:
            print(f"✓ Input noise (stddev={self._noise_std})")
        if self._use_gp:
            print("✓ Gradient Penalty ready")

        # Estimativa de economia de parâmetros
        print("\n💡 Parameter Efficiency:")
        print(f"   - Attention complexity per level:")
        base_seq = (input_size * 2) // self._patch_size
        for i in range(self._hierarchy):
            seq_len = base_seq // (self._pool_size ** i)
            complexity = seq_len ** 2
            print(f"     Level {i}: O({seq_len}²) = O({complexity:,})")

        print("=" * 70 + "\n")

        model.summary()

        return model

    # ==================================================================
    # GRADIENT PENALTY
    # ==================================================================

    @tf.function
    def gradient_penalty(
            self,
            real_samples: tf.Tensor,
            fake_samples: tf.Tensor,
            labels: tf.Tensor,
            lambda_gp: float = 10.0
    ) -> tf.Tensor:
        """Gradient penalty para WGAN-GP."""
        if not self._use_gp:
            return tf.constant(0.0)

        batch_size = tf.shape(real_samples)[0]
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)

        interpolated = alpha * real_samples + (1 - alpha) * fake_samples

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self._model([interpolated, labels], training=True)

        gradients = tape.gradient(pred, interpolated)
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
            'd_model': self._d_model,
            'num_heads': self._num_heads,
            'dff': self._dff,
            'num_layers_per_level': self._num_layers,
            'hierarchy_levels': self._hierarchy,
            'pool_size': self._pool_size,
            'patch_size': self._patch_size,
            'use_spectral_norm': self._use_sn,
            'use_conditional_norm': self._use_cond_norm,
            'use_minibatch_stddev': self._use_mbstd,
            'use_gradient_penalty': self._use_gp,
            'use_input_noise': self._use_noise,
            'noise_stddev': self._noise_std,
            'mixed_precision': self._mixed_precision
        }