#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid Convolutional-Transformer Generator com Vector Quantization Hierárquica
Versão ESTÁVEL com SPECTRAL NORMALIZATION COMPLETA para WGAN-GP
"""

__author__ = 'Synthetic Ocean AI - Enhanced Team'
__version__ = '5.2.0-stable-hierarchical-vq-spectral'

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

    from Engine.activations.Activations import Activations

except ImportError as error:
    print(error)
    sys.exit(-1)


class SpectralNormalization(Layer):
    """
    Spectral Normalization Wrapper - UNIVERSAL
    Pode ser aplicado a qualquer camada com kernel
    """

    def __init__(self, layer, power_iterations=1, **kwargs):
        super().__init__(**kwargs)
        self.layer = layer
        self.power_iterations = power_iterations

    def build(self, input_shape):
        self.layer.build(input_shape)

        # Pega o kernel da camada wrapped
        self.kernel = self.layer.kernel
        kernel_shape = self.kernel.shape

        # Cria vetor u para power iteration
        self.u = self.add_weight(
            name='u',
            shape=(1, kernel_shape[-1]),
            initializer='random_normal',
            trainable=False,
            dtype=self.kernel.dtype
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        # Power iteration para encontrar singular value dominante
        w = self.kernel
        w_shape = w.shape
        w_reshaped = tf.reshape(w, [-1, w_shape[-1]])

        u = self.u
        for _ in range(self.power_iterations):
            v = tf.nn.l2_normalize(tf.matmul(u, w_reshaped, transpose_b=True))
            u = tf.nn.l2_normalize(tf.matmul(v, w_reshaped))

        # Calcula sigma (singular value)
        sigma = tf.reduce_sum(tf.matmul(u, w_reshaped) * v)

        # Normaliza o kernel
        w_normalized = w / (sigma + 1e-12)

        # Atualiza u
        self.u.assign(u)

        # Substitui kernel temporariamente
        self.layer.kernel = w_normalized
        output = self.layer(inputs)
        self.layer.kernel = w

        return output

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)


class SpectralDense(Layer):
    """Dense com Spectral Normalization - OTIMIZADA"""

    def __init__(self, units, use_bias=True, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)

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

        self.u = self.add_weight(
            name='u',
            shape=(1, self.units),
            initializer='random_normal',
            trainable=False
        )

        super().build(input_shape)

    def call(self, x, training=None):
        # Power iteration
        w = self.kernel
        v = tf.matmul(self.u, w, transpose_b=True)
        v = tf.nn.l2_normalize(v, axis=-1)

        u_new = tf.matmul(v, w)
        u_new = tf.nn.l2_normalize(u_new, axis=-1)

        sigma = tf.reduce_sum(u_new * tf.matmul(v, w))
        w_normalized = w / (sigma + 1e-12)

        self.u.assign(u_new)

        output = tf.matmul(x, w_normalized)
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.units,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'use_bias': self.use_bias,
            'activation': tf.keras.activations.serialize(self.activation)
        })
        return config


class SpectralConv1D(Layer):
    """Conv1D com Spectral Normalization - CRÍTICO PARA GANs"""

    def __init__(self, filters, kernel_size, strides=1, padding='same',
                 activation=None, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])

        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.kernel_size, input_dim, self.filters),
            initializer=HeNormal(),
            trainable=True
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer='zeros',
                trainable=True
            )

        # u para spectral norm
        self.u = self.add_weight(
            name='u',
            shape=(1, self.filters),
            initializer='random_normal',
            trainable=False
        )

        super().build(input_shape)

    def call(self, x, training=None):
        # Reshape kernel para matrix
        w = self.kernel
        w_reshaped = tf.reshape(w, [-1, self.filters])

        # Power iteration
        v = tf.matmul(self.u, w_reshaped, transpose_b=True)
        v = tf.nn.l2_normalize(v, axis=-1)

        u_new = tf.matmul(v, w_reshaped)
        u_new = tf.nn.l2_normalize(u_new, axis=-1)

        sigma = tf.reduce_sum(u_new * tf.matmul(v, w_reshaped))
        w_normalized = w / (sigma + 1e-12)

        self.u.assign(u_new)

        # Convolução com kernel normalizado
        output = tf.nn.conv1d(
            x,
            w_normalized,
            stride=self.strides,
            padding=self.padding.upper()
        )

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        if self.padding == 'same':
            length = input_shape[1] // self.strides
        else:
            length = (input_shape[1] - self.kernel_size) // self.strides + 1
        return (input_shape[0], length, self.filters)

    def get_config(self):
        return {
            **super().get_config(),
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'use_bias': self.use_bias,
            'activation': tf.keras.activations.serialize(self.activation)
        }


class VectorQuantization(Layer):
    """Vector Quantization Layer - STABLE"""

    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25, **kwargs):
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name='codebook',
            shape=(self.num_embeddings, self.embedding_dim),
            initializer='uniform',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]

        distances = (
                tf.reduce_sum(inputs ** 2, axis=1, keepdims=True) +
                tf.reduce_sum(self.embeddings ** 2, axis=1) -
                2 * tf.matmul(inputs, self.embeddings, transpose_b=True)
        )

        encoding_indices = tf.argmin(distances, axis=1)
        quantized = tf.nn.embedding_lookup(self.embeddings, encoding_indices)

        if training:
            e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)
            q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)
            loss = q_latent_loss + self.commitment_cost * e_latent_loss
            self.add_loss(loss)

        quantized = inputs + tf.stop_gradient(quantized - inputs)
        return quantized

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            **super().get_config(),
            'num_embeddings': self.num_embeddings,
            'embedding_dim': self.embedding_dim,
            'commitment_cost': self.commitment_cost
        }


class RMSNorm(Layer):
    """Root Mean Square Layer Normalization"""

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

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), 'epsilon': self.epsilon}


class DepthwiseSeparableConv(Layer):
    """Depthwise Separable Convolution com SPECTRAL NORM"""

    def __init__(self, filters, kernel_size=3, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

    def build(self, input_shape):
        self.depthwise = DepthwiseConv1D(
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='same',
            use_bias=False
        )
        # Pointwise com Spectral Norm
        self.pointwise = SpectralConv1D(
            filters=self.filters,
            kernel_size=1,
            use_bias=False
        )
        self.norm = RMSNorm()
        super().build(input_shape)

    def call(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // self.strides, self.filters)

    def get_config(self):
        return {
            **super().get_config(),
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides
        }


class SqueezeExcitation(Layer):
    """Squeeze-and-Excitation Block com SPECTRAL NORM"""

    def __init__(self, ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = int(input_shape[-1])
        reduced = max(channels // self.ratio, 8)

        self.pool = GlobalAveragePooling1D()
        self.fc1 = SpectralDense(reduced, activation='relu')
        self.fc2 = SpectralDense(channels, activation='sigmoid')
        super().build(input_shape)

    def call(self, x):
        scale = self.pool(x)
        scale = self.fc1(scale)
        scale = self.fc2(scale)
        scale = tf.expand_dims(scale, axis=1)
        return x * scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), 'ratio': self.ratio}


class FiLM(Layer):
    """Feature-wise Linear Modulation com SPECTRAL NORM"""

    def build(self, input_shape):
        features_shape, condition_shape = input_shape
        channels = int(features_shape[-1])

        self.gamma_dense = SpectralDense(channels)
        self.beta_dense = SpectralDense(channels)
        super().build(input_shape)

    def call(self, inputs):
        features, condition = inputs
        bf = tf.shape(features)[0]
        bc = tf.shape(condition)[0]

        condition = tf.cond(
            tf.not_equal(bf, bc),
            lambda: tf.repeat(condition, repeats=bf // bc, axis=0),
            lambda: condition
        )

        gamma = self.gamma_dense(condition)
        beta = self.beta_dense(condition)

        gamma = tf.expand_dims(gamma, axis=1)
        beta = tf.expand_dims(beta, axis=1)

        return features * (1.0 + gamma) + beta


class EfficientAttention(Layer):
    """Efficient Multi-Head Attention com SPECTRAL NORM"""

    def __init__(self, num_heads=4, head_dim=32, use_spectral=True, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_model = num_heads * head_dim
        self.use_spectral = use_spectral

        import math
        self.scale_value = 1.0 / math.sqrt(float(head_dim))

    def build(self, input_shape):
        d_input = int(input_shape[-1])

        if self.use_spectral:
            self.wq = SpectralDense(self.d_model, use_bias=False)
            self.wk = SpectralDense(self.d_model, use_bias=False)
            self.wv = SpectralDense(self.d_model, use_bias=False)
            self.wo = SpectralDense(d_input, use_bias=False)
        else:
            self.wq = Dense(self.d_model, use_bias=False)
            self.wk = Dense(self.d_model, use_bias=False)
            self.wv = Dense(self.d_model, use_bias=False)
            self.wo = Dense(d_input, use_bias=False)

        super().build(input_shape)

    def call(self, x):
        batch = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = tf.reshape(q, [batch, seq_len, self.num_heads, self.head_dim])
        k = tf.reshape(k, [batch, seq_len, self.num_heads, self.head_dim])
        v = tf.reshape(v, [batch, seq_len, self.num_heads, self.head_dim])

        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        scores = tf.matmul(q, k, transpose_b=True) * self.scale_value
        weights = tf.nn.softmax(scores, axis=-1)

        attended = tf.matmul(weights, v)
        attended = tf.transpose(attended, [0, 2, 1, 3])
        attended = tf.reshape(attended, [batch, seq_len, self.d_model])

        output = self.wo(attended)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            **super().get_config(),
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'use_spectral': self.use_spectral
        }


class GLU(Layer):
    """Gated Linear Unit"""

    def call(self, x):
        half = tf.shape(x)[-1] // 2
        return x[..., :half] * tf.nn.sigmoid(x[..., half:])

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (input_shape[-1] // 2,)


class ConvTransformerBlock(Layer):
    """Hybrid Conv-Transformer Block com SPECTRAL NORM"""

    def __init__(self, filters, num_heads=4, head_dim=32, ff_ratio=2.0,
                 dropout=0.1, use_spectral_attn=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ff_ratio = ff_ratio
        self.dropout_rate = dropout
        self.use_spectral_attn = use_spectral_attn

    def build(self, input_shape):
        self.conv = DepthwiseSeparableConv(self.filters, kernel_size=3)
        self.se = SqueezeExcitation(ratio=4)
        self.attn = EfficientAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            use_spectral=self.use_spectral_attn
        )

        ff_dim = int(self.filters * self.ff_ratio)
        self.ff1 = SpectralDense(ff_dim * 2)
        self.glu = GLU()
        self.ff2 = SpectralDense(self.filters)

        self.norm1 = RMSNorm()
        self.norm2 = RMSNorm()
        self.norm3 = RMSNorm()

        self.drop1 = Dropout(self.dropout_rate)
        self.drop2 = Dropout(self.dropout_rate)
        self.drop3 = Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, x, training=None):
        conv_out = self.conv(x)
        conv_out = self.se(conv_out)
        conv_out = self.drop1(conv_out, training=training)
        x = self.norm1(x + conv_out)

        attn_out = self.attn(x)
        attn_out = self.drop2(attn_out, training=training)
        x = self.norm2(x + attn_out)

        ff_out = self.ff1(x)
        ff_out = self.glu(ff_out)
        ff_out = self.ff2(ff_out)
        ff_out = self.drop3(ff_out, training=training)
        x = self.norm3(x + ff_out)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            **super().get_config(),
            'filters': self.filters,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'ff_ratio': self.ff_ratio,
            'dropout_rate': self.dropout_rate,
            'use_spectral_attn': self.use_spectral_attn
        }


class MultiScaleFusion(Layer):
    """Multi-Scale Feature Fusion com SPECTRAL NORM"""

    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.num_scales = len(input_shape)

        self.projections = [
            SpectralDense(self.output_dim, name=f'proj_{i}')
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
        base_batch = tf.reduce_min([tf.shape(x)[0] for x in inputs])

        fused = []
        for i, (x, proj) in enumerate(zip(inputs, self.projections)):
            x = tf.reshape(x, [base_batch, -1, x.shape[-1]])
            x = tf.reduce_mean(x, axis=1)
            x = proj(x)
            fused.append(x * weights[i])

        return tf.add_n(fused)


class VanillaGenerator(Activations):
    """
    Hybrid Conv-Transformer Generator com VQ Hierárquica e SPECTRAL NORMALIZATION

    OTIMIZADO PARA WGAN-GP:
    - Spectral Norm em TODAS as camadas Dense e Conv1D críticas
    - Lipschitz constraint mais forte
    - Estabilidade máxima durante treinamento
    """

    @staticmethod
    def _safe_int(value, default):
        if isinstance(value, dict): return default
        try:
            return int(value)
        except:
            return default

    @staticmethod
    def _safe_float(value, default):
        if isinstance(value, dict): return default
        try:
            return float(value)
        except:
            return default

    @staticmethod
    def _safe_list(value, default):
        if isinstance(value, dict) or value is None: return default
        if isinstance(value, (list, tuple)): return list(value)
        return default

    def __init__(
            self,
            latent_dimension: int,
            output_shape: int,
            activation_function: Callable,
            initializer_mean: float,
            initializer_deviation: float,
            dropout_decay_rate_g: float,
            last_layer_activation: Callable,
            dataset_type: type = np.float32,
            number_samples_per_class: Optional[Dict[str, int]] = None,
            num_stages: int = 4,
            base_filters: int = 128,
            tokens_per_stage: List[int] = None,
            num_heads: int = 8,
            head_dim: int = 128,
            ff_ratio: float = 2.0,
            dropout_rate: float = 0.1,
            use_tanh_output: bool = False,
            use_mixed_precision: bool = False,
            use_spectral_attn: bool = True,
            # VQ parameters
            use_vq: bool = True,
            vq_num_embeddings: int = 512,
            vq_embedding_dim: int = 64,
            vq_commitment_cost: float = 0.25,
    ):
        if latent_dimension <= 0 or output_shape <= 0:
            raise ValueError("latent_dimension and output_shape must be > 0")

        self._latent_dim = latent_dimension
        self._output_shape = output_shape
        self._activation_fn = activation_function
        self._last_activation = last_layer_activation
        self._dropout_rate = dropout_decay_rate_g
        self._dtype = dataset_type
        self._class_info = number_samples_per_class

        self._num_stages = self._safe_int(num_stages, 3)
        self._base_filters = self._safe_int(base_filters, 64)
        self._tokens_per_stage = self._safe_list(tokens_per_stage, [16, 32, 64])

        if len(self._tokens_per_stage) != self._num_stages:
            self._tokens_per_stage = [32 * (2 ** i) for i in range(self._num_stages)]

        self._num_heads = self._safe_int(num_heads, 4)
        self._head_dim = self._safe_int(head_dim, 32)
        self._ff_ratio = self._safe_float(ff_ratio, 2.0)
        self._dropout_internal = self._safe_float(dropout_rate, 0.1)
        self._use_tanh = bool(use_tanh_output) if not isinstance(use_tanh_output, dict) else False
        self._mixed_precision = bool(use_mixed_precision) if not isinstance(use_mixed_precision, dict) else False
        self._use_spectral_attn = bool(use_spectral_attn) if not isinstance(use_spectral_attn, dict) else True

        # VQ parameters
        self._use_vq = bool(use_vq) if not isinstance(use_vq, dict) else True
        self._vq_num_embeddings = self._safe_int(vq_num_embeddings, 512)
        self._vq_embedding_dim = self._safe_int(vq_embedding_dim, 64)
        self._vq_commitment_cost = self._safe_float(vq_commitment_cost, 0.25)

        self._model = None

        print("\n" + "=" * 80)
        print("🚀 HYBRID GENERATOR - FULL SPECTRAL NORMALIZATION FOR WGAN-GP")
        print("=" * 80)
        print(f"Stages: {self._num_stages} | Tokens: {self._tokens_per_stage}")
        print(f"Filters: {self._base_filters} | Heads: {self._num_heads}x{self._head_dim}")
        print(f"🔒 Spectral Norm: ALL Dense + Conv1D layers")
        print(f"🔒 Spectral Norm in Attention: {self._use_spectral_attn}")
        if self._use_vq:
            print(f"📊 VQ Hierárquica: {self._num_stages} níveis × {self._vq_num_embeddings} codes")
        print("=" * 80 + "\n")

    def get_generator(self) -> Model:
        """Build generator com SPECTRAL NORM completa"""

        if not self._class_info:
            raise ValueError("number_samples_per_class required")

        if self._mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        num_classes = self._class_info['number_classes']

        # === INPUTS ===
        z = Input(shape=(self._latent_dim,), dtype=tf.float32, name='z')
        y = Input(shape=(num_classes,), dtype=tf.float32, name='y')

        condition = y

        # === INITIAL PROJECTION (com Spectral Norm) ===
        x = Concatenate(name='concat_initial')([z, condition])
        x = SpectralDense(self._base_filters, activation='gelu', name='init_channel_proj')(x)
        x = RMSNorm(name='init_norm')(x)
        x = Lambda(lambda t: tf.expand_dims(t, axis=1), name='init_add_token_dim')(x)

        stage_outs = []
        vq_code_prev = None

        # === STAGES COM VQ HIERÁRQUICA + SPECTRAL NORM ===
        for i in range(self._num_stages):
            n_filters = self._base_filters * (2 ** i)

            # Condicionamento hierárquico
            if self._use_vq and vq_code_prev is not None:
                condition = Concatenate(name=f's{i}_condition')([y, vq_code_prev])
            else:
                condition = y

            # Projeção de canais (Spectral Norm)
            if i > 0:
                x = SpectralDense(n_filters, activation='gelu', name=f's{i}_channel_proj')(x)

            # Upsampling temporal (Spectral Norm)
            x = SpectralConv1D(
                filters=n_filters,
                kernel_size=3,
                strides=2 if i > 0 else self._tokens_per_stage[0],
                name=f's{i}_token_upsample'
            )(x)

            # Bloco híbrido Conv-Transformer (com Spectral Norm interno)
            x = ConvTransformerBlock(
                filters=n_filters,
                num_heads=self._num_heads,
                head_dim=self._head_dim,
                ff_ratio=self._ff_ratio,
                dropout=self._dropout_internal,
                use_spectral_attn=self._use_spectral_attn,
                name=f's{i}_block'
            )(x)

            # FiLM (Spectral Norm)
            x = FiLM(name=f's{i}_film')([x, condition])

            # Pool features
            stage_features = GlobalAveragePooling1D(name=f's{i}_pool')(x)
            stage_outs.append(stage_features)

            # VQ hierárquica
            if self._use_vq:
                vq_proj = SpectralDense(
                    self._vq_embedding_dim,
                    activation='gelu',
                    name=f's{i}_vq_proj'
                )(stage_features)

                vq_code_prev = VectorQuantization(
                    num_embeddings=self._vq_num_embeddings,
                    embedding_dim=self._vq_embedding_dim,
                    commitment_cost=self._vq_commitment_cost,
                    name=f's{i}_vq'
                )(vq_proj)

        # === MULTI-SCALE FUSION (Spectral Norm) ===
        x = MultiScaleFusion(output_dim=self._base_filters * 2, name='fusion')(stage_outs)

        if self._use_vq and vq_code_prev is not None:
            x = Concatenate(name='fusion_with_final_vq')([x, vq_code_prev])

        # === OUTPUT HEAD (Spectral Norm) ===
        x = SpectralDense(self._base_filters * 2, activation='gelu', name='pre_out')(x)
        x = RMSNorm(name='pre_out_norm')(x)

        x = SpectralDense(self._output_shape, name='output')(x)

        if self._use_tanh:
            x = Activation('tanh', dtype=tf.float32, name='tanh')(x)
        elif self._last_activation:
            x = self._add_activation_layer(x, self._last_activation)

        model = Model(inputs=[z, y], outputs=x, name='HybridGenerator_SpectralNorm')
        self._model = model

        print(f"\n✅ Model built! Params: {model.count_params():,}")
        print("🔒 Spectral Normalization aplicada em TODAS as camadas críticas")
        print("✅ Otimizado para WGAN-GP com Lipschitz constraint forte\n")
        model.summary()

        return model

    def sample_latent(self, batch_size: int, seed: Optional[int] = None):
        if seed: np.random.seed(seed)
        return np.random.randn(batch_size, self._latent_dim).astype(np.float32)

    def get_model(self):
        return self._model

    @property
    def trainable_variables(self):
        return self._model.trainable_variables if self._model else []

    def get_config(self):
        return {
            'latent_dimension': self._latent_dim,
            'output_shape': self._output_shape,
            'num_stages': self._num_stages,
            'base_filters': self._base_filters,
            'tokens_per_stage': self._tokens_per_stage,
            'num_heads': self._num_heads,
            'head_dim': self._head_dim,
            'ff_ratio': self._ff_ratio,
            'dropout_rate': self._dropout_internal,
            'use_tanh_output': self._use_tanh,
            'use_spectral_attn': self._use_spectral_attn,
            'use_vq': self._use_vq,
            'vq_num_embeddings': self._vq_num_embeddings,
            'vq_embedding_dim': self._vq_embedding_dim,
            'vq_commitment_cost': self._vq_commitment_cost
        }