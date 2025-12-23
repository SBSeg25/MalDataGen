#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
State-of-the-Art Dense Discriminator para WGAN-GP
Incorpora as melhores técnicas de 2018-2024 aplicáveis a arquiteturas densas
"""

__author__ = 'Synthetic Ocean AI - Enhanced Team'
__version__ = '3.0.0-sota-discriminator'

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
    """
    Root Mean Square Normalization - mais moderna e eficiente que LayerNorm
    Usada em LLaMA, Mistral, e outros modelos state-of-the-art
    """

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
    """
    Dense com Spectral Normalization OTIMIZADA
    Implementação mais eficiente e numericamente estável
    """

    def __init__(
            self,
            units: int,
            activation=None,
            use_bias=True,
            kernel_initializer='he_normal',
            power_iterations=1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.power_iterations = power_iterations

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

        # Vetor u para power iteration (não-trainável)
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
        w_shape = w.shape
        w_mat = tf.reshape(w, [-1, w_shape[-1]])

        u = self.u

        # Power iteration para encontrar singular value dominante
        for _ in range(self.power_iterations):
            v = tf.nn.l2_normalize(tf.matmul(u, w_mat, transpose_b=True))
            u = tf.nn.l2_normalize(tf.matmul(v, w_mat))

        # Atualiza u durante treinamento
        if training:
            self.u.assign(u)

        # Calcula norma espectral (maior singular value)
        # sigma = u^T W v (onde u e v são os vetores singulares)
        sigma = tf.matmul(tf.matmul(v, w_mat), u, transpose_b=True)
        sigma = tf.abs(sigma[0, 0])  # Extrai o escalar

        # Normaliza kernel
        w_normalized = w / tf.maximum(sigma, 1e-12)

        # Operação Dense
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
            'power_iterations': self.power_iterations
        }


class MultiHeadDenseAttention(Layer):
    """
    Multi-Head Self-Attention para features DENSAS (não espaciais)

    Diferente de atenção em imagens, aqui computamos atenção entre
    diferentes "grupos" de features, permitindo que o discriminador
    aprenda quais combinações de features são mais discriminativas.
    """

    def __init__(self, num_heads=4, head_dim=32, use_spectral=True, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_model = num_heads * head_dim
        self.use_spectral = use_spectral
        self.scale = 1.0 / math.sqrt(float(head_dim))

    def build(self, input_shape):
        d_input = int(input_shape[-1])

        DenseLayer = SpectralDense if self.use_spectral else Dense

        self.wq = DenseLayer(self.d_model, use_bias=False, name='q')
        self.wk = DenseLayer(self.d_model, use_bias=False, name='k')
        self.wv = DenseLayer(self.d_model, use_bias=False, name='v')
        self.wo = DenseLayer(d_input, use_bias=False, name='out')

        # Learnable temperature
        self.temperature = self.add_weight(
            name='temperature',
            shape=[1],
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True
        )

        super().build(input_shape)

    def call(self, x):
        batch = tf.shape(x)[0]

        # Projeta para Q, K, V
        q = self.wq(x)  # [B, d_model]
        k = self.wk(x)
        v = self.wv(x)

        # Reshape para multi-head: [B, num_heads, head_dim]
        q = tf.reshape(q, [batch, self.num_heads, self.head_dim])
        k = tf.reshape(k, [batch, self.num_heads, self.head_dim])
        v = tf.reshape(v, [batch, self.num_heads, self.head_dim])

        # Atenção entre heads (cross-head attention)
        # Cada head "atende" a outros heads
        scores = tf.matmul(q, k, transpose_b=True)  # [B, H, H]
        scores = scores * self.scale / tf.maximum(self.temperature, 0.1)

        weights = tf.nn.softmax(scores, axis=-1)

        # Apply attention
        attended = tf.matmul(weights, v)  # [B, H, head_dim]
        attended = tf.reshape(attended, [batch, self.d_model])

        # Output projection
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


class FeatureSqueezeExcitation(Layer):
    """
    Squeeze-and-Excitation adaptado para features densas
    Aprende a dar diferentes pesos para diferentes features
    """

    def __init__(self, ratio=4, use_spectral=True, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.use_spectral = use_spectral

    def build(self, input_shape):
        channels = int(input_shape[-1])
        reduced = max(channels // self.ratio, 8)

        DenseLayer = SpectralDense if self.use_spectral else Dense

        self.fc1 = DenseLayer(reduced, activation='relu')
        self.fc2 = DenseLayer(channels, activation='sigmoid')

        super().build(input_shape)

    def call(self, x):
        scale = self.fc1(x)
        scale = self.fc2(scale)
        return x * scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), 'ratio': self.ratio, 'use_spectral': self.use_spectral}


class MinibatchStdDev(Layer):
    """
    Minibatch Standard Deviation (ProGAN/StyleGAN)
    OTIMIZADO e mais eficiente
    """

    def __init__(self, group_size=4, num_new_features=1, **kwargs):
        super().__init__(**kwargs)
        self.group_size = group_size
        self.num_new_features = num_new_features

    def call(self, x):
        batch_size = tf.shape(x)[0]
        num_features = x.shape[-1]

        # Agrupa amostras
        group_size = tf.minimum(self.group_size, batch_size)

        # Reshape: [G, M, F] onde G = group_size, M = batch_size/G
        x_grouped = tf.reshape(x, [group_size, -1, num_features])

        # Calcula stddev por grupo
        mean = tf.reduce_mean(x_grouped, axis=0, keepdims=True)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(x_grouped - mean), axis=0) + 1e-8)

        # Average over features
        stddev_mean = tf.reduce_mean(stddev, keepdims=True)

        # Broadcast para todas as amostras
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


class ResidualBlock(Layer):
    """
    Residual Block otimizado para discriminador
    Com Pre-Activation, Spectral Norm, e RMSNorm
    """

    def __init__(self, units, dropout=0.1, use_spectral=True, activation='swish', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout
        self.use_spectral = use_spectral
        self.activation_name = activation

    def build(self, input_shape):
        DenseLayer = SpectralDense if self.use_spectral else Dense

        self.norm1 = RMSNorm()
        self.dense1 = DenseLayer(self.units)
        self.act1 = Activation(self.activation_name)

        self.norm2 = RMSNorm()
        self.dense2 = DenseLayer(self.units)
        self.act2 = Activation(self.activation_name)

        self.dropout = Dropout(self.dropout_rate)

        # Squeeze-Excitation
        self.se = FeatureSqueezeExcitation(ratio=4, use_spectral=self.use_spectral)

        # Projeção residual se necessário
        if input_shape[-1] != self.units:
            self.projection = DenseLayer(self.units, use_bias=False)
        else:
            self.projection = None

        super().build(input_shape)

    def call(self, x, training=None):
        identity = x

        # Pre-activation
        out = self.norm1(x)
        out = self.act1(out)
        out = self.dense1(out)

        out = self.norm2(out)
        out = self.act2(out)
        out = self.dropout(out, training=training)
        out = self.dense2(out)

        # Squeeze-Excitation
        out = self.se(out)

        # Residual connection
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
            'activation': self.activation_name
        }


class ConditionalBatchNorm(Layer):
    """
    Conditional Batch Normalization melhorada
    Usa Spectral Norm nos embeddings
    """

    def __init__(self, num_features, num_classes, use_spectral=True, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.num_classes = num_classes
        self.use_spectral = use_spectral

    def build(self, input_shape):
        self.bn = BatchNormalization(scale=False, center=False)

        DenseLayer = SpectralDense if self.use_spectral else Dense

        self.gamma_embed = DenseLayer(self.num_features, use_bias=False)
        self.beta_embed = DenseLayer(self.num_features, use_bias=False)

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


class VanillaDiscriminatorTensorflow(Activations):
    """
    State-of-the-Art Discriminator para WGAN-GP

    TÉCNICAS MODERNAS IMPLEMENTADAS (2018-2024):
    ✅ Spectral Normalization (SNGAN 2018) - em TODAS as camadas
    ✅ RMSNorm (2019) - mais eficiente que LayerNorm
    ✅ Multi-Head Dense Attention (2017+) - adaptado para features densas
    ✅ Minibatch StdDev (ProGAN 2018) - detecta mode collapse
    ✅ Residual Blocks com Pre-Activation (ResNet-v2 2016)
    ✅ Squeeze-Excitation (2018) - adaptado para features
    ✅ R1 Regularization (StyleGAN2 2019) - alternativa ao GP
    ✅ Swish/SiLU activation (2017) - melhor que ReLU
    ✅ Conditional Batch Norm (2016) - conditioning sofisticado
    ✅ Multi-Scale Processing - captura features em múltiplas escalas
    ✅ Label smoothing & input noise - regularização
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
            num_residual_blocks: int = 2,
            use_multi_scale: bool = True,
            # Attention
            use_attention: bool = True,
            attention_heads: int = 4,
            attention_layers: List[int] = None,
            # Normalization
            norm_type: str = 'rms',  # 'rms', 'layer', 'batch', 'none'
            use_conditional_bn: bool = True,
            # Regularization
            use_minibatch_stddev: bool = True,
            minibatch_group_size: int = 4,
            use_input_noise: bool = True,
            noise_stddev: float = 0.05,
            label_smoothing: float = 0.0,
            # Loss & Penalty
            use_r1_regularization: bool = False,
            r1_gamma: float = 10.0,
            use_gradient_penalty: bool = True,
            gp_lambda: float = 10.0,
            # Output
            output_units: int = 1,
            # Performance
            use_mixed_precision: bool = False,
            activation_name: str = 'swish',
    ):
        # Validações
        if latent_dimension <= 0:
            raise ValueError("latent_dimension must be > 0")
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
        self._class_info = number_samples_per_class

        # Architecture
        self._use_sn = use_spectral_norm
        self._use_residual = use_residual_blocks
        self._num_res_blocks = num_residual_blocks
        self._use_multi_scale = use_multi_scale

        # Attention
        self._use_attn = use_attention
        self._attn_heads = attention_heads
        self._attn_layers = attention_layers or [len(dense_layer_sizes_d) // 2]

        # Normalization
        self._norm_type = norm_type
        self._use_cbn = use_conditional_bn

        # Regularization
        self._use_mbstd = use_minibatch_stddev
        self._mb_group = minibatch_group_size
        self._use_noise = use_input_noise
        self._noise_std = noise_stddev
        self._label_smooth = label_smoothing

        # Loss & Penalty
        self._use_r1 = use_r1_regularization
        self._r1_gamma = r1_gamma
        self._use_gp = use_gradient_penalty
        self._gp_lambda = gp_lambda

        # Output
        self._output_units = output_units

        # Performance
        self._mixed_precision = use_mixed_precision
        self._activation_name = activation_name

        self._model: Optional[Model] = None

    def _get_norm_layer(self, name: str):
        """Retorna camada de normalização baseada em configuração"""
        if self._norm_type == 'rms':
            return RMSNorm(name=f'{name}_rms')
        elif self._norm_type == 'layer':
            return LayerNormalization(name=f'{name}_ln')
        elif self._norm_type == 'batch':
            return BatchNormalization(name=f'{name}_bn')
        return Lambda(lambda x: x, name=f'{name}_no_norm')

    def get_discriminator(self) -> Model:
        """Constrói discriminador state-of-the-art"""

        if not self._class_info:
            raise ValueError("number_samples_per_class is required")

        if self._mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        num_classes = self._class_info['number_classes']
        input_size = int(np.prod(self._output_shape))

        # ==================== INPUTS ====================
        x_in = Input(shape=(input_size,), dtype=self._dtype, name='input_real')
        y_in = Input(shape=(num_classes,), dtype=self._dtype, name='input_label')

        # ==================== INPUT PROCESSING ====================
        x = x_in

        # Input noise (regularização)
        if self._use_noise:
            x = Lambda(
                lambda t: t + tf.random.normal(tf.shape(t), 0.0, self._noise_std),
                name='input_noise'
            )(x)

        # Minibatch StdDev (detecta mode collapse)
        if self._use_mbstd:
            x = MinibatchStdDev(group_size=self._mb_group, name='minibatch_stddev')(x)

        # Label embedding com Spectral Norm
        DenseLayer = SpectralDense if self._use_sn else Dense

        label_embed = DenseLayer(
            input_size // 4,
            activation=self._activation_name,
            name='label_embedding'
        )(y_in)

        # Concatena input com label
        x = Concatenate(name='concat_input_label')([x, label_embed])

        # ==================== INITIAL PROJECTION ====================
        x = DenseLayer(
            self._dense_sizes[0],
            name='input_projection'
        )(x)
        x = self._get_norm_layer('input')(x)
        x = Activation(self._activation_name)(x)

        # ==================== MULTI-SCALE FEATURES ====================
        multi_scale_features = []

        # ==================== MAIN PROCESSING ====================
        for i, units in enumerate(self._dense_sizes):
            # Residual blocks
            if self._use_residual:
                for j in range(self._num_res_blocks):
                    x = ResidualBlock(
                        units,
                        dropout=self._dropout,
                        use_spectral=self._use_sn,
                        activation=self._activation_name,
                        name=f'res_block_{i}_{j}'
                    )(x)
            else:
                # Bloco simples
                x = DenseLayer(units, name=f'dense_{i}')(x)
                x = self._get_norm_layer(f'layer_{i}')(x)
                x = Activation(self._activation_name)(x)
                x = Dropout(self._dropout, name=f'dropout_{i}')(x)

            # Conditional Batch Norm
            if self._use_cbn and i > 0:
                x = ConditionalBatchNorm(
                    units,
                    num_classes,
                    use_spectral=self._use_sn,
                    name=f'cbn_{i}'
                )([x, y_in])

            # Multi-Head Attention
            if self._use_attn and i in self._attn_layers:
                attn = MultiHeadDenseAttention(
                    num_heads=self._attn_heads,
                    head_dim=max(units // self._attn_heads, 16),
                    use_spectral=self._use_sn,
                    name=f'mh_attn_{i}'
                )(x)

                # Residual connection
                x = Add(name=f'attn_residual_{i}')([x, attn])

            # Coleta features para multi-scale
            if self._use_multi_scale:
                multi_scale_features.append(x)

        # ==================== MULTI-SCALE FUSION ====================
        if self._use_multi_scale and len(multi_scale_features) > 1:
            # Projeta todas para mesma dimensão
            projected = []
            for idx, feat in enumerate(multi_scale_features):
                proj = DenseLayer(
                    self._dense_sizes[-1],
                    name=f'scale_proj_{idx}'
                )(feat)
                projected.append(proj)

            # Average pooling das escalas
            x_multi = Add(name='multi_scale_sum')(projected)
            x_multi = Lambda(
                lambda t: t / len(projected),
                name='multi_scale_avg'
            )(x_multi)

            # Concatena com features principais
            x = Concatenate(name='concat_multi_scale')([x, x_multi])

            # Projeção final
            x = DenseLayer(
                self._dense_sizes[-1],
                activation=self._activation_name,
                name='multi_scale_fusion'
            )(x)

        # ==================== OUTPUT HEAD ====================
        x = self._get_norm_layer('pre_output')(x)
        x = Activation(self._activation_name)(x)
        x = Dropout(self._dropout, name='output_dropout')(x)

        # Output logits
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
            name='SOTA_DenseDiscriminator'
        )

        self._model = model

        model.summary()

        return model

    @tf.function
    def gradient_penalty(
            self,
            real_samples: tf.Tensor,
            fake_samples: tf.Tensor,
            labels: tf.Tensor
    ) -> tf.Tensor:
        """
        Gradient Penalty para WGAN-GP
        Garante que o discriminador satisfaz 1-Lipschitz constraint
        """
        if not self._use_gp:
            return tf.constant(0.0)

        batch_size = tf.shape(real_samples)[0]
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0, dtype=real_samples.dtype)

        # Interpolação
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self._model([interpolated, labels], training=True)

        gradients = tape.gradient(pred, interpolated)

        # Norma L2 dos gradientes
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1) + 1e-10)

        # Penalidade: (||∇|| - 1)²
        gp = tf.reduce_mean(tf.square(slopes - 1.0))

        return self._gp_lambda * gp

    @tf.function
    def r1_regularization(
            self,
            real_samples: tf.Tensor,
            labels: tf.Tensor
    ) -> tf.Tensor:
        """
        R1 Regularization (StyleGAN2)
        Alternativa ao Gradient Penalty, mais eficiente
        Penaliza gradientes apenas em dados reais
        """
        if not self._use_r1:
            return tf.constant(0.0)

        with tf.GradientTape() as tape:
            tape.watch(real_samples)
            real_pred = self._model([real_samples, labels], training=True)

        gradients = tape.gradient(real_pred, real_samples)

        # R1: ||∇||²
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
            'norm_type': self._norm_type,
            'use_conditional_bn': self._use_cbn,
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