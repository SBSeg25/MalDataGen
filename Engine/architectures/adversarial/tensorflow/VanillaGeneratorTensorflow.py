# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#
# """
# Hybrid Convolutional-Transformer Generator com Vector Quantization Global
# Versão ULTRA ESTÁVEL com VQ único após fusion multi-escala
# """
#
# __author__ = 'Synthetic Ocean AI - Enhanced Team'
# __version__ = '5.0.0-stable-global-vq'
#
# try:
#     import sys
#     import numpy as np
#     import tensorflow as tf
#     from typing import Dict, List, Optional, Callable, Tuple
#     from tensorflow.keras.layers import (
#         Layer, Dense, Input, Dropout, Concatenate, Add,
#         Lambda, LayerNormalization, Embedding, Reshape,
#         Conv1D, DepthwiseConv1D, GlobalAveragePooling1D,
#         Multiply, Activation, BatchNormalization
#     )
#     from tensorflow.keras.models import Model
#     from tensorflow.keras.initializers import GlorotUniform, Orthogonal, HeNormal
#     from tensorflow.keras import backend as K
#
#     from Engine.activations.Activations import Activations
#
# except ImportError as error:
#     print(error)
#     sys.exit(-1)
#
#
# class VectorQuantization(Layer):
#     """
#     Vector Quantization Layer - STABLE
#
#     Implementação simplificada do VQ-VAE
#     Aplica quantização vetorial em features agregadas
#     """
#
#     def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25, **kwargs):
#         super().__init__(**kwargs)
#         self.num_embeddings = num_embeddings
#         self.embedding_dim = embedding_dim
#         self.commitment_cost = commitment_cost
#
#     def build(self, input_shape):
#         # Codebook: (num_embeddings, embedding_dim)
#         self.embeddings = self.add_weight(
#             name='codebook',
#             shape=(self.num_embeddings, self.embedding_dim),
#             initializer='uniform',
#             trainable=True
#         )
#         super().build(input_shape)
#
#     def call(self, inputs, training=None):
#         # inputs: (batch, embedding_dim)
#         batch_size = tf.shape(inputs)[0]
#
#         # Distância L2 para cada embedding
#         # (batch, emb_dim) vs (num_emb, emb_dim)
#         distances = (
#                 tf.reduce_sum(inputs ** 2, axis=1, keepdims=True) +
#                 tf.reduce_sum(self.embeddings ** 2, axis=1) -
#                 2 * tf.matmul(inputs, self.embeddings, transpose_b=True)
#         )
#
#         # Encontra código mais próximo: (batch,)
#         encoding_indices = tf.argmin(distances, axis=1)
#
#         # Busca embeddings quantizados: (batch, emb_dim)
#         quantized = tf.nn.embedding_lookup(self.embeddings, encoding_indices)
#
#         if training:
#             # VQ Loss: commitment loss
#             e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)
#             q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)
#             loss = q_latent_loss + self.commitment_cost * e_latent_loss
#             self.add_loss(loss)
#
#         # Straight-through estimator
#         quantized = inputs + tf.stop_gradient(quantized - inputs)
#
#         return quantized
#
#     def compute_output_shape(self, input_shape):
#         return input_shape
#
#     def get_config(self):
#         return {
#             **super().get_config(),
#             'num_embeddings': self.num_embeddings,
#             'embedding_dim': self.embedding_dim,
#             'commitment_cost': self.commitment_cost
#         }
#
#
# class RMSNorm(Layer):
#     """Root Mean Square Layer Normalization - STABLE"""
#
#     def __init__(self, epsilon=1e-6, **kwargs):
#         super().__init__(**kwargs)
#         self.epsilon = epsilon
#
#     def build(self, input_shape):
#         self.scale = self.add_weight(
#             name='scale',
#             shape=(input_shape[-1],),
#             initializer='ones',
#             trainable=True
#         )
#         super().build(input_shape)
#
#     def call(self, x):
#         variance = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
#         x_norm = x * tf.math.rsqrt(variance + self.epsilon)
#         return x_norm * self.scale
#
#     def compute_output_shape(self, input_shape):
#         return input_shape
#
#     def get_config(self):
#         config = super().get_config()
#         config.update({'epsilon': self.epsilon})
#         return config
#
#
# class SpectralDense(Layer):
#     """Dense com Spectral Normalization - VERSÃO 100% ESTÁVEL"""
#
#     def __init__(self, units, use_bias=True, **kwargs):
#         super().__init__(**kwargs)
#         self.units = units
#         self.use_bias = use_bias
#
#     def build(self, input_shape):
#         input_dim = int(input_shape[-1])
#
#         self.kernel = self.add_weight(
#             name='kernel',
#             shape=(input_dim, self.units),
#             initializer=HeNormal(),
#             trainable=True
#         )
#
#         if self.use_bias:
#             self.bias = self.add_weight(
#                 name='bias',
#                 shape=(self.units,),
#                 initializer='zeros',
#                 trainable=True
#             )
#
#         self.u = self.add_weight(
#             name='u',
#             shape=(1, self.units),
#             initializer='random_normal',
#             trainable=False
#         )
#
#         super().build(input_shape)
#
#     def call(self, x):
#         w = self.kernel
#         v = tf.matmul(self.u, w, transpose_b=True)
#         v = tf.nn.l2_normalize(v, axis=-1)
#
#         u_new = tf.matmul(v, w)
#         u_new = tf.nn.l2_normalize(u_new, axis=-1)
#
#         sigma = tf.reduce_sum(u_new * tf.matmul(v, w))
#         w_normalized = w / (sigma + 1e-6)
#
#         self.u.assign(u_new)
#
#         output = tf.matmul(x, w_normalized)
#         if self.use_bias:
#             output = tf.nn.bias_add(output, self.bias)
#
#         return output
#
#     def compute_output_shape(self, input_shape):
#         return tuple(input_shape[:-1]) + (self.units,)
#
#     def get_config(self):
#         config = super().get_config()
#         config.update({'units': self.units, 'use_bias': self.use_bias})
#         return config
#
#
# class DepthwiseSeparableConv(Layer):
#     """Depthwise Separable Convolution - STABLE"""
#
#     def __init__(self, filters, kernel_size=3, strides=1, **kwargs):
#         super().__init__(**kwargs)
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.strides = strides
#
#     def build(self, input_shape):
#         self.depthwise = DepthwiseConv1D(
#             kernel_size=self.kernel_size,
#             strides=self.strides,
#             padding='same',
#             use_bias=False
#         )
#         self.pointwise = Conv1D(filters=self.filters, kernel_size=1, use_bias=False)
#         self.norm = RMSNorm()
#         super().build(input_shape)
#
#     def call(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         x = self.norm(x)
#         return x
#
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], input_shape[1] // self.strides, self.filters)
#
#     def get_config(self):
#         return {
#             **super().get_config(),
#             'filters': self.filters,
#             'kernel_size': self.kernel_size,
#             'strides': self.strides
#         }
#
#
# class SqueezeExcitation(Layer):
#     """Squeeze-and-Excitation Block - STABLE"""
#
#     def __init__(self, ratio=4, **kwargs):
#         super().__init__(**kwargs)
#         self.ratio = ratio
#
#     def build(self, input_shape):
#         channels = int(input_shape[-1])
#         reduced = max(channels // self.ratio, 8)
#
#         self.pool = GlobalAveragePooling1D()
#         self.fc1 = Dense(reduced, activation='relu')
#         self.fc2 = Dense(channels, activation='sigmoid')
#         super().build(input_shape)
#
#     def call(self, x):
#         scale = self.pool(x)
#         scale = self.fc1(scale)
#         scale = self.fc2(scale)
#         scale = tf.expand_dims(scale, axis=1)
#         return x * scale
#
#     def compute_output_shape(self, input_shape):
#         return input_shape
#
#     def get_config(self):
#         return {**super().get_config(), 'ratio': self.ratio}
#
#
# class FiLM(Layer):
#     """Feature-wise Linear Modulation - BATCH SAFE"""
#
#     def build(self, input_shape):
#         features_shape, condition_shape = input_shape
#         channels = int(features_shape[-1])
#
#         self.gamma_dense = Dense(channels, kernel_initializer='zeros')
#         self.beta_dense = Dense(channels, kernel_initializer='zeros')
#         super().build(input_shape)
#
#     def call(self, inputs):
#         features, condition = inputs
#         bf = tf.shape(features)[0]
#         bc = tf.shape(condition)[0]
#
#         condition = tf.cond(
#             tf.not_equal(bf, bc),
#             lambda: tf.repeat(condition, repeats=bf // bc, axis=0),
#             lambda: condition
#         )
#
#         gamma = self.gamma_dense(condition)
#         beta = self.beta_dense(condition)
#
#         gamma = tf.expand_dims(gamma, axis=1)
#         beta = tf.expand_dims(beta, axis=1)
#
#         return features * (1.0 + gamma) + beta
#
#
# class EfficientAttention(Layer):
#     """Efficient Multi-Head Attention - ULTRA STABLE"""
#
#     def __init__(self, num_heads=4, head_dim=32, **kwargs):
#         super().__init__(**kwargs)
#         self.num_heads = num_heads
#         self.head_dim = head_dim
#         self.d_model = num_heads * head_dim
#
#         import math
#         self.scale_value = 1.0 / math.sqrt(float(head_dim))
#
#     def build(self, input_shape):
#         d_input = int(input_shape[-1])
#
#         self.wq = Dense(self.d_model, use_bias=False)
#         self.wk = Dense(self.d_model, use_bias=False)
#         self.wv = Dense(self.d_model, use_bias=False)
#         self.wo = Dense(d_input, use_bias=False)
#
#         super().build(input_shape)
#
#     def call(self, x):
#         batch = tf.shape(x)[0]
#         seq_len = tf.shape(x)[1]
#
#         q = self.wq(x)
#         k = self.wk(x)
#         v = self.wv(x)
#
#         q = tf.reshape(q, [batch, seq_len, self.num_heads, self.head_dim])
#         k = tf.reshape(k, [batch, seq_len, self.num_heads, self.head_dim])
#         v = tf.reshape(v, [batch, seq_len, self.num_heads, self.head_dim])
#
#         q = tf.transpose(q, [0, 2, 1, 3])
#         k = tf.transpose(k, [0, 2, 1, 3])
#         v = tf.transpose(v, [0, 2, 1, 3])
#
#         scores = tf.matmul(q, k, transpose_b=True) * self.scale_value
#         weights = tf.nn.softmax(scores, axis=-1)
#
#         attended = tf.matmul(weights, v)
#         attended = tf.transpose(attended, [0, 2, 1, 3])
#         attended = tf.reshape(attended, [batch, seq_len, self.d_model])
#
#         output = self.wo(attended)
#         return output
#
#     def compute_output_shape(self, input_shape):
#         return input_shape
#
#     def get_config(self):
#         return {
#             **super().get_config(),
#             'num_heads': self.num_heads,
#             'head_dim': self.head_dim
#         }
#
#
# class GLU(Layer):
#     """Gated Linear Unit - STABLE"""
#
#     def call(self, x):
#         half = tf.shape(x)[-1] // 2
#         return x[..., :half] * tf.nn.sigmoid(x[..., half:])
#
#     def compute_output_shape(self, input_shape):
#         return input_shape[:-1] + (input_shape[-1] // 2,)
#
#
# class ConvTransformerBlock(Layer):
#     """Hybrid Conv-Transformer Block - ULTRA STABLE"""
#
#     def __init__(self, filters, num_heads=4, head_dim=32, ff_ratio=2.0, dropout=0.1, **kwargs):
#         super().__init__(**kwargs)
#         self.filters = filters
#         self.num_heads = num_heads
#         self.head_dim = head_dim
#         self.ff_ratio = ff_ratio
#         self.dropout_rate = dropout
#
#     def build(self, input_shape):
#         self.conv = DepthwiseSeparableConv(self.filters, kernel_size=3)
#         self.se = SqueezeExcitation(ratio=4)
#         self.attn = EfficientAttention(num_heads=self.num_heads, head_dim=self.head_dim)
#
#         ff_dim = int(self.filters * self.ff_ratio)
#         self.ff1 = Dense(ff_dim * 2)
#         self.glu = GLU()
#         self.ff2 = Dense(self.filters)
#
#         self.norm1 = RMSNorm()
#         self.norm2 = RMSNorm()
#         self.norm3 = RMSNorm()
#
#         self.drop1 = Dropout(self.dropout_rate)
#         self.drop2 = Dropout(self.dropout_rate)
#         self.drop3 = Dropout(self.dropout_rate)
#
#         super().build(input_shape)
#
#     def call(self, x, training=None):
#         conv_out = self.conv(x)
#         conv_out = self.se(conv_out)
#         conv_out = self.drop1(conv_out, training=training)
#         x = self.norm1(x + conv_out)
#
#         attn_out = self.attn(x)
#         attn_out = self.drop2(attn_out, training=training)
#         x = self.norm2(x + attn_out)
#
#         ff_out = self.ff1(x)
#         ff_out = self.glu(ff_out)
#         ff_out = self.ff2(ff_out)
#         ff_out = self.drop3(ff_out, training=training)
#         x = self.norm3(x + ff_out)
#
#         return x
#
#     def compute_output_shape(self, input_shape):
#         return input_shape
#
#     def get_config(self):
#         return {
#             **super().get_config(),
#             'filters': self.filters,
#             'num_heads': self.num_heads,
#             'head_dim': self.head_dim,
#             'ff_ratio': self.ff_ratio,
#             'dropout_rate': self.dropout_rate
#         }
#
#
# class MultiScaleFusion(Layer):
#     """Multi-Scale Feature Fusion - BATCH SAFE"""
#
#     def __init__(self, output_dim, **kwargs):
#         super().__init__(**kwargs)
#         self.output_dim = output_dim
#
#     def build(self, input_shape):
#         self.num_scales = len(input_shape)
#
#         self.projections = [
#             Dense(self.output_dim, name=f'proj_{i}')
#             for i in range(self.num_scales)
#         ]
#
#         self.scale_weights = self.add_weight(
#             name='fusion_weights',
#             shape=(self.num_scales,),
#             initializer='ones',
#             trainable=True
#         )
#         super().build(input_shape)
#
#     def call(self, inputs):
#         weights = tf.nn.softmax(self.scale_weights)
#         base_batch = tf.reduce_min([tf.shape(x)[0] for x in inputs])
#
#         fused = []
#         for i, (x, proj) in enumerate(zip(inputs, self.projections)):
#             bi = tf.shape(x)[0]
#             x = tf.reshape(x, [base_batch, -1, x.shape[-1]])
#             x = tf.reduce_mean(x, axis=1)
#             x = proj(x)
#             fused.append(x * weights[i])
#
#         return tf.add_n(fused)
#
#
# class VanillaGenerator(Activations):
#     """
#     Hybrid Conv-Transformer Generator com Vector Quantization Global
#
#     ARQUITETURA MAIS ESTÁVEL: VQ único após fusion multi-escala
#     - Sem loops complexos
#     - Um único codebook
#     - Regularização global sem interferir no treinamento dos estágios
#     """
#
#     @staticmethod
#     def _safe_int(value, default):
#         if isinstance(value, dict): return default
#         try:
#             return int(value)
#         except:
#             return default
#
#     @staticmethod
#     def _safe_float(value, default):
#         if isinstance(value, dict): return default
#         try:
#             return float(value)
#         except:
#             return default
#
#     @staticmethod
#     def _safe_list(value, default):
#         if isinstance(value, dict) or value is None: return default
#         if isinstance(value, (list, tuple)): return list(value)
#         return default
#
#     def __init__(
#             self,
#             latent_dimension: int,
#             output_shape: int,
#             activation_function: Callable,
#             initializer_mean: float,
#             initializer_deviation: float,
#             dropout_decay_rate_g: float,
#             last_layer_activation: Callable,
#             dataset_type: type = np.float32,
#             number_samples_per_class: Optional[Dict[str, int]] = None,
#             num_stages: int = 3,
#             base_filters: int = 64,
#             tokens_per_stage: List[int] = None,
#             num_heads: int = 4,
#             head_dim: int = 32,
#             ff_ratio: float = 2.0,
#             dropout_rate: float = 0.1,
#             use_tanh_output: bool = False,
#             use_mixed_precision: bool = False,
#             # VQ parameters
#             use_vq: bool = True,
#             vq_num_embeddings: int = 512,
#             vq_embedding_dim: int = 64,
#             vq_commitment_cost: float = 0.25,
#     ):
#         if latent_dimension <= 0 or output_shape <= 0:
#             raise ValueError("latent_dimension and output_shape must be > 0")
#
#         self._latent_dim = latent_dimension
#         self._output_shape = output_shape
#         self._activation_fn = activation_function
#         self._last_activation = last_layer_activation
#         self._dropout_rate = dropout_decay_rate_g
#         self._dtype = dataset_type
#         self._class_info = number_samples_per_class
#
#         self._num_stages = self._safe_int(num_stages, 3)
#         self._base_filters = self._safe_int(base_filters, 64)
#         self._tokens_per_stage = self._safe_list(tokens_per_stage, [16, 32, 64])
#
#         if len(self._tokens_per_stage) != self._num_stages:
#             self._tokens_per_stage = [32 * (2 ** i) for i in range(self._num_stages)]
#
#         self._num_heads = self._safe_int(num_heads, 4)
#         self._head_dim = self._safe_int(head_dim, 32)
#         self._ff_ratio = self._safe_float(ff_ratio, 2.0)
#         self._dropout_internal = self._safe_float(dropout_rate, 0.1)
#         self._use_tanh = bool(use_tanh_output) if not isinstance(use_tanh_output, dict) else False
#         self._mixed_precision = bool(use_mixed_precision) if not isinstance(use_mixed_precision, dict) else False
#
#         # VQ parameters
#         self._use_vq = bool(use_vq) if not isinstance(use_vq, dict) else True
#         self._vq_num_embeddings = self._safe_int(vq_num_embeddings, 512)
#         self._vq_embedding_dim = self._safe_int(vq_embedding_dim, 64)
#         self._vq_commitment_cost = self._safe_float(vq_commitment_cost, 0.25)
#
#         self._model = None
#
#         print("\n" + "=" * 70)
#         print("🚀 Hybrid Conv-Transformer Generator - VQ GLOBAL (ESTÁVEL)")
#         print("=" * 70)
#         print(f"Stages: {self._num_stages} | Tokens: {self._tokens_per_stage}")
#         print(f"Filters: {self._base_filters} | Heads: {self._num_heads}x{self._head_dim}")
#         if self._use_vq:
#             print(f"VQ Global: {self._vq_num_embeddings} codes × {self._vq_embedding_dim}D")
#             print("  ↳ Aplicada APÓS fusion multi-escala (mais estável)")
#         print("=" * 70 + "\n")
#
#     def get_generator(self) -> Model:
#         """Build the generator com VQ global após fusion"""
#
#         if not self._class_info:
#             raise ValueError("number_samples_per_class required")
#
#         if self._mixed_precision:
#             tf.keras.mixed_precision.set_global_policy('mixed_float16')
#
#         num_classes = self._class_info['number_classes']
#
#         # === INPUTS ===
#         z = Input(shape=(self._latent_dim,), dtype=tf.float32, name='z')
#         y = Input(shape=(num_classes,), dtype=tf.float32, name='y')
#
#         # === FLUXO PRINCIPAL ===
#         x = Concatenate(name='concat_main')([z, y])
#
#         # === INITIAL PROJECTION ===
#         x = Dense(self._base_filters, name='init_channel_proj')(x)
#         x = RMSNorm(name='init_norm')(x)
#         x = Activation('gelu')(x)
#
#         x = Lambda(lambda t: tf.expand_dims(t, axis=1), name='init_add_token_dim')(x)
#
#         stage_outs = []
#
#         # === STAGES (SEM VQ INTERMEDIÁRIO) ===
#         for i in range(self._num_stages):
#             n_filters = self._base_filters * (2 ** i)
#
#             # Projeção de canais
#             if i > 0:
#                 x = Dense(n_filters, name=f's{i}_channel_proj')(x)
#                 x = Activation('gelu')(x)
#
#             # Upsampling temporal
#             x = Conv1D(
#                 filters=n_filters,
#                 kernel_size=3,
#                 padding='same',
#                 strides=2 if i > 0 else self._tokens_per_stage[0],
#                 name=f's{i}_token_upsample'
#             )(x)
#
#             # Bloco híbrido Conv-Transformer
#             x = ConvTransformerBlock(
#                 filters=n_filters,
#                 num_heads=self._num_heads,
#                 head_dim=self._head_dim,
#                 ff_ratio=self._ff_ratio,
#                 dropout=self._dropout_internal,
#                 name=f's{i}_block'
#             )(x)
#
#             # FiLM com labels originais (sem VQ intermediário)
#             x = FiLM(name=f's{i}_film')([x, y])
#
#             # Coleta output do estágio
#             stage_outs.append(GlobalAveragePooling1D(name=f's{i}_pool')(x))
#
#         # === MULTI-SCALE FUSION ===
#         x = MultiScaleFusion(output_dim=self._base_filters * 2, name='fusion')(stage_outs)
#
#         # === VQ GLOBAL (APÓS FUSION) ===
#         if self._use_vq:
#             # Projeção para dimensão VQ
#             x_vq_proj = Dense(self._vq_embedding_dim, name='vq_projection')(x)
#             x_vq_proj = Activation('gelu')(x_vq_proj)
#
#             # Vector Quantization global
#             x_vq = VectorQuantization(
#                 num_embeddings=self._vq_num_embeddings,
#                 embedding_dim=self._vq_embedding_dim,
#                 commitment_cost=self._vq_commitment_cost,
#                 name='global_vq'
#             )(x_vq_proj)
#
#             # Concatena features originais + quantizadas
#             x = Concatenate(name='fusion_with_vq')([x, x_vq])
#
#         # === OUTPUT HEAD ===
#         x = Dense(self._base_filters * 2, name='pre_out')(x)
#         x = RMSNorm(name='pre_out_norm')(x)
#         x = Activation('gelu')(x)
#
#         x = Dense(self._output_shape, name='output')(x)
#
#         if self._use_tanh:
#             x = Activation('tanh', dtype=tf.float32, name='tanh')(x)
#         elif self._last_activation:
#             x = self._add_activation_layer(x, self._last_activation)
#
#         model = Model(inputs=[z, y], outputs=x, name='HybridGeneratorGlobalVQ')
#         self._model = model
#
#         print(f"\n✅ Model built! Params: {model.count_params():,}")
#         if self._use_vq:
#             print(f"📊 VQ Global aplicada após fusion (arquitetura mais estável)")
#         print()
#         model.summary()
#
#         return model
#
#     def sample_latent(self, batch_size: int, seed: Optional[int] = None):
#         if seed: np.random.seed(seed)
#         return np.random.randn(batch_size, self._latent_dim).astype(np.float32)
#
#     def get_model(self):
#         return self._model
#
#     @property
#     def trainable_variables(self):
#         return self._model.trainable_variables if self._model else []
#
#     def get_config(self):
#         return {
#             'latent_dimension': self._latent_dim,
#             'output_shape': self._output_shape,
#             'num_stages': self._num_stages,
#             'base_filters': self._base_filters,
#             'tokens_per_stage': self._tokens_per_stage,
#             'num_heads': self._num_heads,
#             'head_dim': self._head_dim,
#             'ff_ratio': self._ff_ratio,
#             'dropout_rate': self._dropout_internal,
#             'use_tanh_output': self._use_tanh,
#             'use_vq': self._use_vq,
#             'vq_num_embeddings': self._vq_num_embeddings,
#             'vq_embedding_dim': self._vq_embedding_dim,
#             'vq_commitment_cost': self._vq_commitment_cost
#         }


# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid Convolutional-Transformer Generator com Vector Quantization Hierárquica
Versão ESTÁVEL com VQ feedforward em múltiplos níveis (estilo VQ-VAE-2)
"""

__author__ = 'Synthetic Ocean AI - Enhanced Team'
__version__ = '5.1.0-stable-hierarchical-vq'

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


class VectorQuantization(Layer):
    """
    Vector Quantization Layer - STABLE

    Implementação simplificada do VQ-VAE
    Aplica quantização vetorial em features agregadas
    """

    def __init__(self, num_embeddings=128, embedding_dim=64, commitment_cost=0.25, **kwargs):
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

    def build(self, input_shape):
        # Codebook: (num_embeddings, embedding_dim)
        self.embeddings = self.add_weight(
            name='codebook',
            shape=(self.num_embeddings, self.embedding_dim),
            initializer='uniform',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        # inputs: (batch, embedding_dim)
        batch_size = tf.shape(inputs)[0]

        # Distância L2 para cada embedding
        distances = (
                tf.reduce_sum(inputs ** 2, axis=1, keepdims=True) +
                tf.reduce_sum(self.embeddings ** 2, axis=1) -
                2 * tf.matmul(inputs, self.embeddings, transpose_b=True)
        )

        # Encontra código mais próximo: (batch,)
        encoding_indices = tf.argmin(distances, axis=1)

        # Busca embeddings quantizados: (batch, emb_dim)
        quantized = tf.nn.embedding_lookup(self.embeddings, encoding_indices)

        if training:
            # VQ Loss: commitment loss
            e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)
            q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)
            loss = q_latent_loss + self.commitment_cost * e_latent_loss
            self.add_loss(loss)

        # Straight-through estimator
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
    """Root Mean Square Layer Normalization - STABLE"""

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
        config = super().get_config()
        config.update({'epsilon': self.epsilon})
        return config


class SpectralDense(Layer):
    """Dense com Spectral Normalization - VERSÃO 100% ESTÁVEL"""

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

        self.u = self.add_weight(
            name='u',
            shape=(1, self.units),
            initializer='random_normal',
            trainable=False
        )

        super().build(input_shape)

    def call(self, x):
        w = self.kernel
        v = tf.matmul(self.u, w, transpose_b=True)
        v = tf.nn.l2_normalize(v, axis=-1)

        u_new = tf.matmul(v, w)
        u_new = tf.nn.l2_normalize(u_new, axis=-1)

        sigma = tf.reduce_sum(u_new * tf.matmul(v, w))
        w_normalized = w / (sigma + 1e-6)

        self.u.assign(u_new)

        output = tf.matmul(x, w_normalized)
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)

        return output

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.units,)

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units, 'use_bias': self.use_bias})
        return config


class DepthwiseSeparableConv(Layer):
    """Depthwise Separable Convolution - STABLE"""

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
        self.pointwise = Conv1D(filters=self.filters, kernel_size=1, use_bias=False)
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
    """Squeeze-and-Excitation Block - STABLE"""

    def __init__(self, ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = int(input_shape[-1])
        reduced = max(channels // self.ratio, 8)

        self.pool = GlobalAveragePooling1D()
        self.fc1 = Dense(reduced, activation='relu')
        self.fc2 = Dense(channels, activation='sigmoid')
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
    """Feature-wise Linear Modulation - BATCH SAFE"""

    def build(self, input_shape):
        features_shape, condition_shape = input_shape
        channels = int(features_shape[-1])

        self.gamma_dense = Dense(channels, kernel_initializer='zeros')
        self.beta_dense = Dense(channels, kernel_initializer='zeros')
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
    """Efficient Multi-Head Attention - ULTRA STABLE"""

    def __init__(self, num_heads=4, head_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_model = num_heads * head_dim

        import math
        self.scale_value = 1.0 / math.sqrt(float(head_dim))

    def build(self, input_shape):
        d_input = int(input_shape[-1])

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
            'head_dim': self.head_dim
        }


class GLU(Layer):
    """Gated Linear Unit - STABLE"""

    def call(self, x):
        half = tf.shape(x)[-1] // 2
        return x[..., :half] * tf.nn.sigmoid(x[..., half:])

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (input_shape[-1] // 2,)


class ConvTransformerBlock(Layer):
    """Hybrid Conv-Transformer Block - ULTRA STABLE"""

    def __init__(self, filters, num_heads=4, head_dim=32, ff_ratio=2.0, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ff_ratio = ff_ratio
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.conv = DepthwiseSeparableConv(self.filters, kernel_size=3)
        self.se = SqueezeExcitation(ratio=4)
        self.attn = EfficientAttention(num_heads=self.num_heads, head_dim=self.head_dim)

        ff_dim = int(self.filters * self.ff_ratio)
        self.ff1 = Dense(ff_dim * 2)
        self.glu = GLU()
        self.ff2 = Dense(self.filters)

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
            'dropout_rate': self.dropout_rate
        }


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
        base_batch = tf.reduce_min([tf.shape(x)[0] for x in inputs])

        fused = []
        for i, (x, proj) in enumerate(zip(inputs, self.projections)):
            bi = tf.shape(x)[0]
            x = tf.reshape(x, [base_batch, -1, x.shape[-1]])
            x = tf.reduce_mean(x, axis=1)
            x = proj(x)
            fused.append(x * weights[i])

        return tf.add_n(fused)


class VanillaGenerator(Activations):
    """
    Hybrid Conv-Transformer Generator com VQ Hierárquica Feedforward

    ARQUITETURA ESTÁVEL inspirada em VQ-VAE-2:
    - Stage 0: processa → VQ₀ (códigos grossos)
    - Stage 1: condicionado por VQ₀ → processa → VQ₁ (códigos médios)
    - Stage 2: condicionado por VQ₁ → processa → VQ₂ (códigos finos)

    Fluxo CAUSAL sem loops circulares
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
            base_filters: int = 72,
            tokens_per_stage: List[int] = None,
            num_heads: int = 8,
            head_dim: int = 72,
            ff_ratio: float = 2.0,
            dropout_rate: float = 0.1,
            use_tanh_output: bool = False,
            use_mixed_precision: bool = False,
            # VQ parameters
            use_vq: bool = True,
            vq_num_embeddings: int = 256,
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

        # VQ parameters
        self._use_vq = bool(use_vq) if not isinstance(use_vq, dict) else True
        self._vq_num_embeddings = self._safe_int(vq_num_embeddings, 512)
        self._vq_embedding_dim = self._safe_int(vq_embedding_dim, 64)
        self._vq_commitment_cost = self._safe_float(vq_commitment_cost, 0.25)

        self._model = None

        print("\n" + "=" * 70)
        print("🚀 Hybrid Conv-Transformer - VQ HIERÁRQUICA FEEDFORWARD")
        print("=" * 70)
        print(f"Stages: {self._num_stages} | Tokens: {self._tokens_per_stage}")
        print(f"Filters: {self._base_filters} | Heads: {self._num_heads}x{self._head_dim}")
        if self._use_vq:
            print(f"VQ Hierárquica: {self._num_stages} níveis × {self._vq_num_embeddings} codes")
            print("  ↳ Arquitetura feedforward causal (VQ-VAE-2 style)")
        print("=" * 70 + "\n")

    def get_generator(self) -> Model:
        """Build generator com VQ hierárquica feedforward (ESTÁVEL)"""

        if not self._class_info:
            raise ValueError("number_samples_per_class required")

        if self._mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        num_classes = self._class_info['number_classes']

        # === INPUTS ===
        z = Input(shape=(self._latent_dim,), dtype=tf.float32, name='z')
        y = Input(shape=(num_classes,), dtype=tf.float32, name='y')

        # Condicionamento inicial: apenas labels
        condition = y

        # === INITIAL PROJECTION ===
        x = Concatenate(name='concat_initial')([z, condition])
        x = Dense(self._base_filters, name='init_channel_proj')(x)
        x = RMSNorm(name='init_norm')(x)
        x = Activation('gelu')(x)
        x = Lambda(lambda t: tf.expand_dims(t, axis=1), name='init_add_token_dim')(x)

        stage_outs = []
        vq_code_prev = None

        # === STAGES COM VQ FEEDFORWARD ===
        for i in range(self._num_stages):
            n_filters = self._base_filters * (2 ** i)

            # Atualiza condicionamento com VQ do nível anterior
            if self._use_vq and vq_code_prev is not None:
                condition = Concatenate(name=f's{i}_condition')([y, vq_code_prev])
            else:
                condition = y

            # Projeção de canais
            if i > 0:
                x = Dense(n_filters, name=f's{i}_channel_proj')(x)
                x = Activation('gelu')(x)

            # Upsampling temporal
            x = Conv1D(
                filters=n_filters,
                kernel_size=3,
                padding='same',
                strides=2 if i > 0 else self._tokens_per_stage[0],
                name=f's{i}_token_upsample'
            )(x)

            # Bloco híbrido Conv-Transformer
            x = ConvTransformerBlock(
                filters=n_filters,
                num_heads=self._num_heads,
                head_dim=self._head_dim,
                ff_ratio=self._ff_ratio,
                dropout=self._dropout_internal,
                name=f's{i}_block'
            )(x)

            # FiLM com condicionamento (inclui VQ anterior)
            x = FiLM(name=f's{i}_film')([x, condition])

            # Pool features processadas do estágio
            stage_features = GlobalAveragePooling1D(name=f's{i}_pool')(x)
            stage_outs.append(stage_features)

            # === VQ NESTE NÍVEL (após processamento) ===
            if self._use_vq:
                # Projeção para dimensão VQ
                vq_proj = Dense(
                    self._vq_embedding_dim,
                    activation='gelu',
                    name=f's{i}_vq_proj'
                )(stage_features)

                # Vector Quantization
                vq_code_prev = VectorQuantization(
                    num_embeddings=self._vq_num_embeddings,
                    embedding_dim=self._vq_embedding_dim,
                    commitment_cost=self._vq_commitment_cost,
                    name=f's{i}_vq'
                )(vq_proj)

        # === MULTI-SCALE FUSION ===
        x = MultiScaleFusion(output_dim=self._base_filters * 2, name='fusion')(stage_outs)

        # Adiciona último código VQ ao output (opcional)
        if self._use_vq and vq_code_prev is not None:
            x = Concatenate(name='fusion_with_final_vq')([x, vq_code_prev])

        # === OUTPUT HEAD ===
        x = Dense(self._base_filters * 2, name='pre_out')(x)
        x = RMSNorm(name='pre_out_norm')(x)
        x = Activation('gelu')(x)

        x = Dense(self._output_shape, name='output')(x)

        if self._use_tanh:
            x = Activation('tanh', dtype=tf.float32, name='tanh')(x)
        elif self._last_activation:
            x = self._add_activation_layer(x, self._last_activation)

        model = Model(inputs=[z, y], outputs=x, name='HybridGeneratorHierarchicalVQ')
        self._model = model

        print(f"\n✅ Model built! Params: {model.count_params():,}")
        if self._use_vq:
            print(f"📊 VQ Hierárquica Feedforward: {self._num_stages} níveis causais")
            print("   Stage i condicionado por VQ_{i-1} (estilo VQ-VAE-2)")
        print()
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
            'use_vq': self._use_vq,
            'vq_num_embeddings': self._vq_num_embeddings,
            'vq_embedding_dim': self._vq_embedding_dim,
            'vq_commitment_cost': self._vq_commitment_cost
        }