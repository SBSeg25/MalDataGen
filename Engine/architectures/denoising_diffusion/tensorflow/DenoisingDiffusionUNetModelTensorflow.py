__author__ = 'Synthetic Ocean AI - ULTRA-SOTA Team'
__version__ = '5.0.0-extreme-memory-optimization'

import sys
import math
import warnings
import tensorflow as tf
from typing import List, Dict, Optional, Tuple
from tensorflow.keras.layers import (
    Layer, Dense, Input, Dropout, Concatenate, Add,
    Lambda, Embedding, Reshape, Conv1D, DepthwiseConv1D,
    GlobalAveragePooling1D, Activation, Flatten, UpSampling1D,
    LayerNormalization, MultiHeadAttention, SeparableConv1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal, TruncatedNormal
from tensorflow.keras.mixed_precision import set_global_policy
import numpy as np

try:
    from Engine.activations.Activations import Activations
except ImportError:
    class Activations:
        def _add_activation_layer(self, x, activation):
            return Activation(activation)(x)


# ============================================================================
# GRADIENT CHECKPOINTING WRAPPER
# ============================================================================

class GradientCheckpointLayer(Layer):
    """Wrapper para gradient checkpointing - reduz memória durante backprop"""

    def __init__(self, layer, **kwargs):
        super().__init__(**kwargs)
        self.wrapped_layer = layer

    def build(self, input_shape):
        self.wrapped_layer.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        if training:
            # Durante treino, usa recompute no backward pass
            return tf.recompute_grad(lambda x: self.wrapped_layer(x, training=training))(inputs)
        return self.wrapped_layer(inputs, training=training)

    def compute_output_shape(self, input_shape):
        return self.wrapped_layer.compute_output_shape(input_shape)


# ============================================================================
# LOW-RANK ROTARY EMBEDDINGS
# ============================================================================

class LowRankRotaryEmbedding(Layer):
    def __init__(self, dim, max_seq_len=2048, rank_ratio=0.5, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        # Ensure rank is at least 2 and even for rotation
        self.rank = max(2, int(dim * rank_ratio))
        if self.rank % 2 != 0:
            self.rank += 1
        self.rank = min(self.rank, dim)

    def build(self, input_shape):
        half_rank = self.rank // 2
        inv_freq = 1.0 / (10000 ** (np.arange(0, self.rank, 2, dtype=np.float32) / self.rank))
        self.inv_freq = self.add_weight(
            name='rope_inv_freq',
            shape=(half_rank,),
            initializer=tf.keras.initializers.Constant(inv_freq),
            trainable=False,
            dtype=tf.float32
        )
        super().build(input_shape)

    def call(self, x):
        # x shape: [Batch, Seq, Heads, Head_Dim]
        shape = tf.shape(x)
        batch, seq_len, heads, head_dim = shape[0], shape[1], shape[2], shape[3]

        if self.dim < 2 or self.rank < 2:
            return x

        # Generate frequencies
        t = tf.cast(tf.range(seq_len), tf.float32)
        freqs = tf.einsum('i,j->ij', t, self.inv_freq)  # [Seq, Half_Rank]

        # Reshape freqs for broadcasting: [1, Seq, 1, Half_Rank]
        cos_emb = tf.cos(freqs)[None, :, None, :]
        sin_emb = tf.sin(freqs)[None, :, None, :]

        # Slicing the part to rotate
        x_to_rotate = x[..., :self.rank]

        # Safer rotation: split and reconstruct
        # Instead of 0::2 which is brittle with small dims, use split
        x1, x2 = tf.split(x_to_rotate, 2, axis=-1)

        x_rotated = tf.concat([
            x1 * cos_emb - x2 * sin_emb,
            x2 * cos_emb + x1 * sin_emb
        ], axis=-1)

        # Recombine with the rest of the embedding (if any)
        if self.rank < self.dim:
            x_rest = x[..., self.rank:]
            return tf.concat([x_rotated, x_rest], axis=-1)

        return x_rotated

# ============================================================================
# ULTRA-EFFICIENT GROUPED QUERY ATTENTION
# ============================================================================

class UltraEfficientGQA(Layer):
    """GQA com otimizações extremas de memória"""

    def __init__(self, d_model, num_heads=4, num_kv_heads=1, dropout=0.1,
                 chunk_size=16, low_rank_ratio=0.75, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

        # Limita heads para economizar memória
        max_heads = max(1, d_model // 8)
        self.num_heads = min(num_heads, max_heads)
        self.num_kv_heads = min(num_kv_heads, self.num_heads)

        # Garante divisibilidade
        while self.num_heads % self.num_kv_heads != 0:
            self.num_kv_heads = max(1, self.num_kv_heads - 1)

        self.head_dim = d_model // self.num_heads
        self.dropout_rate = dropout
        self.chunk_size = chunk_size
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.low_rank_ratio = low_rank_ratio

    def build(self, input_shape):
        # Usa low-rank factorization: W = U @ V^T
        # Ao invés de (d_model, d_model), usa (d_model, rank) @ (rank, d_model)
        rank = max(8, int(self.d_model * self.low_rank_ratio))

        # Q projection factorizada
        self.q_down = Dense(rank, use_bias=False, kernel_initializer=HeNormal(), name='q_down')
        self.q_up = Dense(self.d_model, use_bias=False, kernel_initializer=HeNormal(), name='q_up')

        # K, V com dimensões reduzidas
        kv_dim = self.num_kv_heads * self.head_dim
        self.k_proj = Dense(kv_dim, use_bias=False, kernel_initializer=HeNormal(), name='k_proj')
        self.v_proj = Dense(kv_dim, use_bias=False, kernel_initializer=HeNormal(), name='v_proj')

        # O projection factorizada
        self.o_down = Dense(rank, use_bias=False, kernel_initializer=HeNormal(), name='o_down')
        self.o_up = Dense(self.d_model, use_bias=False, kernel_initializer=HeNormal(), name='o_up')

        self.dropout = Dropout(self.dropout_rate)
        self.rope = LowRankRotaryEmbedding(self.head_dim, rank_ratio=0.5)

        # Build sublayers
        self.q_down.build(input_shape)
        self.q_up.build((input_shape[0], input_shape[1], rank))
        self.k_proj.build(input_shape)
        self.v_proj.build(input_shape)
        self.o_down.build((input_shape[0], input_shape[1], self.d_model))
        self.o_up.build((input_shape[0], input_shape[1], rank))

        super().build(input_shape)

    def call(self, x, context=None, training=None):
        batch = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        if context is None:
            context = x

        # Factorized projections
        q = self.q_up(self.q_down(x))
        k = self.k_proj(context)
        v = self.v_proj(context)

        # Reshape
        q = tf.reshape(q, [batch, seq_len, self.num_heads, self.head_dim])
        k = tf.reshape(k, [batch, tf.shape(context)[1], self.num_kv_heads, self.head_dim])
        v = tf.reshape(v, [batch, tf.shape(context)[1], self.num_kv_heads, self.head_dim])

        # Apply RoPE
        q = self.rope(q)
        k = self.rope(k)

        # Expand KV
        k = tf.repeat(k, self.num_queries_per_kv, axis=2)
        v = tf.repeat(v, self.num_queries_per_kv, axis=2)

        # Transpose
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        # Memory-efficient attention
        out = self._chunked_attention(q, k, v, training)

        # Reshape and project
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [batch, seq_len, self.d_model])

        # Factorized output projection
        out = self.o_up(self.o_down(out))

        return out

    def _chunked_attention(self, q, k, v, training):
        scale = tf.cast(self.head_dim, q.dtype) ** -0.5
        q_len = tf.shape(q)[2]

        num_chunks = tf.cast(tf.math.ceil(tf.cast(q_len, tf.float32) /
                                          tf.cast(self.chunk_size, tf.float32)), tf.int32)

        outputs = tf.TensorArray(
            dtype=q.dtype,
            size=num_chunks,
            dynamic_size=False,
            clear_after_read=False,
            infer_shape=True
        )

        # O segredo está em manter a estrutura idêntica (i, outputs_array)
        def body(i, outputs_array):
            start = i * self.chunk_size
            end = tf.minimum(start + self.chunk_size, q_len)
            q_chunk = q[:, :, start:end, :]

            scores = tf.matmul(q_chunk, k, transpose_b=True) * scale
            attn = tf.nn.softmax(scores, axis=-1)
            attn = self.dropout(attn, training=training)

            chunk_out = tf.matmul(attn, v)
            # RETORNO COMO TUPLA
            return (i + 1, outputs_array.write(i, chunk_out))

        def condition(i, outputs_array):
            return i < num_chunks

        # LOOP INICIANDO COM TUPLA
        _, final_outputs = tf.while_loop(
            condition,
            body,
            loop_vars=(0, outputs),  # Use tupla aqui
            maximum_iterations=num_chunks
        )

        stacked_out = final_outputs.stack()

        # Re-order and merge: [Batch, Heads, Chunks * Chunk_Size, Dim]
        # Then slice to original q_len to handle if q_len isn't perfectly divisible
        out = tf.transpose(stacked_out, [1, 2, 0, 3, 4])
        batch = tf.shape(q)[0]
        heads = tf.shape(q)[1]
        out = tf.reshape(out, [batch, heads, -1, self.head_dim])

        return out[:, :, :q_len, :]


# ============================================================================
# EFFICIENT SWIGLU WITH BOTTLENECK
# ============================================================================

class EfficientSwiGLU(Layer):
    """SwiGLU com bottleneck para reduzir memória"""

    def __init__(self, dim, expansion_factor=2.0, bottleneck_ratio=0.5, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.bottleneck_dim = max(8, int(dim * bottleneck_ratio))
        self.hidden_dim = int(self.bottleneck_dim * expansion_factor)

    def build(self, input_shape):
        # Bottleneck down
        self.down = Dense(self.bottleneck_dim, use_bias=False,
                          kernel_initializer=HeNormal(), name='down')

        # Gate and value
        self.w1 = Dense(self.hidden_dim, use_bias=False,
                        kernel_initializer=HeNormal(), name='w1')
        self.w2 = Dense(self.hidden_dim, use_bias=False,
                        kernel_initializer=HeNormal(), name='w2')

        # Up projection
        self.up = Dense(self.dim, use_bias=False,
                        kernel_initializer=HeNormal(), name='up')

        # Build
        self.down.build(input_shape)
        self.w1.build((input_shape[0], input_shape[1], self.bottleneck_dim))
        self.w2.build((input_shape[0], input_shape[1], self.bottleneck_dim))
        self.up.build((input_shape[0], input_shape[1], self.hidden_dim))

        super().build(input_shape)

    def call(self, x):
        # Bottleneck
        x_down = self.down(x)

        # SwiGLU
        gate = tf.nn.swish(self.w1(x_down))
        value = self.w2(x_down)
        hidden = gate * value

        # Up projection
        return self.up(hidden)


# ============================================================================
# SEPARABLE ADALN (MEMORY EFFICIENT)
# ============================================================================

class SeparableAdaLN(Layer):
    """AdaLN com separação de parâmetros para economia de memória"""

    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape):
        if isinstance(input_shape, list):
            x_shape, cond_shape = input_shape
        else:
            x_shape = cond_shape = input_shape

        # Ao invés de 6*self.d_model, usa bottleneck
        bottleneck_dim = max(16, self.d_model // 4)

        self.cond_down = Dense(bottleneck_dim, kernel_initializer=HeNormal(), name='cond_down')
        self.cond_up = Dense(self.d_model * 6, kernel_initializer='zeros',
                             bias_initializer='zeros', name='cond_up')
        self.norm = LayerNormalization(epsilon=1e-6, name='norm')

        self.norm.build(x_shape)
        self.cond_down.build(cond_shape)
        self.cond_up.build((cond_shape[0], bottleneck_dim))

        super().build(input_shape)

    def call(self, inputs):
        x, cond = inputs
        x_norm = self.norm(x)

        # Bottleneck conditioning
        cond_features = self.cond_down(cond)
        ada_params = self.cond_up(cond_features)

        if ada_params.shape.rank == 2:
            ada_params = tf.expand_dims(ada_params, axis=1)

        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = tf.split(
            ada_params, 6, axis=-1
        )

        x_modulated = x_norm * (1 + scale_attn) + shift_attn

        return x_modulated, gate_attn, shift_ffn, scale_ffn, gate_ffn


# ============================================================================
# EFFICIENT DIT BLOCK
# ============================================================================

class EfficientDiTBlock(Layer):
    """DiT Block otimizado para memória"""

    def __init__(self, d_model, num_heads=4, num_kv_heads=1, dropout=0.1,
                 chunk_size=16, use_checkpointing=True, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dropout_rate = dropout
        self.chunk_size = chunk_size
        self.use_checkpointing = use_checkpointing

    def build(self, input_shape):
        if isinstance(input_shape, list):
            x_shape, cond_shape = input_shape
        else:
            x_shape = input_shape
            cond_shape = (input_shape[0], self.d_model * 4)

        self.adaln = SeparableAdaLN(self.d_model, name='adaln')

        self.attn = UltraEfficientGQA(
            self.d_model,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            dropout=self.dropout_rate,
            chunk_size=self.chunk_size,
            name='attn'
        )

        self.ffn = EfficientSwiGLU(
            self.d_model,
            expansion_factor=2.0,
            bottleneck_ratio=0.5,
            name='ffn'
        )

        self.norm_ffn = LayerNormalization(epsilon=1e-6, name='norm_ffn')
        self.dropout = Dropout(self.dropout_rate)

        # Build sublayers
        self.adaln.build([x_shape, cond_shape])
        self.attn.build(x_shape)
        self.ffn.build(x_shape)
        self.norm_ffn.build(x_shape)

        super().build(input_shape)

    def call(self, inputs, training=None):
        x, cond = inputs

        # AdaLN
        x_mod, gate_attn, shift_ffn, scale_ffn, gate_ffn = self.adaln([x, cond])

        # Attention com gradient checkpointing opcional
        if self.use_checkpointing and training:
            attn_out = tf.recompute_grad(lambda inp: self.attn(inp, training=training))(x_mod)
        else:
            attn_out = self.attn(x_mod, training=training)

        attn_out = self.dropout(attn_out, training=training)
        x = x + gate_attn * attn_out

        # FFN
        x_norm = self.norm_ffn(x)
        x_mod_ffn = x_norm * (1 + scale_ffn) + shift_ffn

        if self.use_checkpointing and training:
            ffn_out = tf.recompute_grad(lambda inp: self.ffn(inp))(x_mod_ffn)
        else:
            ffn_out = self.ffn(x_mod_ffn)

        ffn_out = self.dropout(ffn_out, training=training)
        x = x + gate_ffn * ffn_out

        return x


# ============================================================================
# DEPTHWISE SEPARABLE RESIDUAL BLOCK
# ============================================================================

class DepthwiseSeparableResBlock(Layer):
    """Bloco residual com convoluções separáveis - muito mais eficiente"""

    def __init__(self, dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.dropout_rate = dropout

    def build(self, input_shape):
        if isinstance(input_shape, list):
            x_shape, cond_shape = input_shape
        else:
            x_shape = input_shape
            cond_shape = (input_shape[0], self.dim * 4)

        # Conditioning com bottleneck
        bottleneck = max(16, self.dim // 4)
        self.cond_down = Dense(bottleneck, activation='swish', name='cond_down')
        self.cond_up = Dense(self.dim * 2, name='cond_up')

        # Depthwise separable convs (muito mais eficiente que conv normal)
        self.dwconv1 = DepthwiseConv1D(kernel_size=3, padding='same',
                                       depthwise_initializer=HeNormal(), name='dwconv1')
        self.pwconv1 = Conv1D(self.dim, kernel_size=1, kernel_initializer=HeNormal(), name='pwconv1')

        self.dwconv2 = DepthwiseConv1D(kernel_size=3, padding='same',
                                       depthwise_initializer=HeNormal(), name='dwconv2')
        self.pwconv2 = Conv1D(self.dim, kernel_size=1, kernel_initializer=HeNormal(), name='pwconv2')

        self.norm1 = LayerNormalization(epsilon=1e-6, name='norm1')
        self.norm2 = LayerNormalization(epsilon=1e-6, name='norm2')
        self.dropout = Dropout(self.dropout_rate)

        # Build
        self.cond_down.build(cond_shape)
        self.cond_up.build((cond_shape[0], bottleneck))
        self.dwconv1.build(x_shape)
        self.pwconv1.build(x_shape)
        self.dwconv2.build(x_shape)
        self.pwconv2.build(x_shape)
        self.norm1.build(x_shape)
        self.norm2.build(x_shape)

        super().build(input_shape)

    def call(self, inputs, training=None):
        x, cond = inputs

        # Efficient conditioning
        cond_emb = self.cond_up(self.cond_down(cond))
        scale, shift = tf.split(cond_emb, 2, axis=-1)

        if scale.shape.rank == 2:
            scale = tf.expand_dims(scale, axis=1)
            shift = tf.expand_dims(shift, axis=1)

        # First block
        h = self.norm1(x)
        h = h * (1 + scale) + shift
        h = tf.nn.swish(h)
        h = self.dwconv1(h)
        h = self.pwconv1(h)
        h = self.dropout(h, training=training)

        # Second block
        h = self.norm2(h)
        h = tf.nn.swish(h)
        h = self.dwconv2(h)
        h = self.pwconv2(h)
        h = self.dropout(h, training=training)

        return x + h


# ============================================================================
# COMPACT TIME EMBEDDING
# ============================================================================

class CompactTimeEmbedding(Layer):
    """Time embedding com dimensão reduzida"""

    def __init__(self, dim, max_period=10000, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.max_period = max_period
        self.fourier_dim = max(8, dim // 4)

    def build(self, input_shape):
        # Usa dimensão Fourier reduzida
        half_dim = self.fourier_dim // 2
        freqs = np.exp(
            -math.log(self.max_period) *
            np.arange(0, half_dim, dtype=np.float32) / half_dim
        )
        self.freqs = self.add_weight(
            name='fourier_freqs',
            shape=(half_dim,),
            initializer=tf.keras.initializers.Constant(freqs),
            trainable=False,
            dtype=tf.float32
        )

        # Projeta para dim final
        self.proj = Dense(self.dim, name='proj')
        self.proj.build((input_shape[0], self.fourier_dim))

        super().build(input_shape)

    def call(self, timesteps):
        timesteps = tf.cast(timesteps, tf.float32)
        timesteps = tf.expand_dims(timesteps, -1)
        args = timesteps * self.freqs[None, :]
        embedding = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
        return self.proj(embedding)


# ============================================================================
# ULTRA-EFFICIENT DIFFUSION U-NET
# ============================================================================

class DenoisingDiffusionUNetModelTensorflow(Activations):
    """
    Diffusion U-Net com otimização extrema de memória.

    Características:
    - Low-rank factorization em todas as projeções pesadas
    - Depthwise separable convolutions
    - Gradient checkpointing
    - Chunked attention com chunks menores
    - Bottleneck conditioning
    - Mais camadas, matrizes menores
    """

    def __init__(
            self,
            output_shape: int = 128,
            embedding_channels: int = 1,
            list_neurons_per_level: List[int] = None,
            number_residual_blocks: int = 3,  # Aumentado
            num_heads: int = 4,
            num_kv_heads: int = 1,  # Reduzido
            dropout_rate: float = 0.1,
            intermediary_activation_function: str = 'gelu',
            last_layer_activation: str = 'linear',
            number_samples_per_class: Optional[Dict] = None,
            attention_chunk_size: int = 8,  # Reduzido
            use_gradient_checkpointing: bool = True,
            use_mixed_precision: bool = False,
            **kwargs
    ):
        # Dimensões menores por nível, mas mais níveis
        list_neurons_per_level =  [8, 16, 24]

        if use_mixed_precision:
            set_global_policy('mixed_float16')
            print("✅ Mixed precision enabled (float16/float32)")

        self._output_shape = self._adjust_output_shape(output_shape, len(list_neurons_per_level))
        self._embedding_channels = embedding_channels
        self._list_neurons = list_neurons_per_level
        self._number_residual_blocks = number_residual_blocks

        # Configuração conservadora de atenção
        min_dim = min(list_neurons_per_level)
        max_heads = max(1, min_dim // 8)

        self._num_heads = min(num_heads, max_heads)
        self._num_kv_heads = min(num_kv_heads, self._num_heads)

        while self._num_heads % self._num_kv_heads != 0:
            self._num_kv_heads = max(1, self._num_kv_heads - 1)

        self._dropout = dropout_rate
        self._intermediary_activation = intermediary_activation_function
        self._last_activation = last_layer_activation
        self._class_info = number_samples_per_class
        self._attention_chunk_size = attention_chunk_size
        self._use_checkpointing = use_gradient_checkpointing

        print(f"\n{'=' * 60}")
        print(f"🚀 Ultra-Efficient Diffusion U-Net v5.0")
        print(f"{'=' * 60}")
        print(f"Memory Optimizations:")
        print(f"  ✓ Low-rank factorization (75% rank)")
        print(f"  ✓ Depthwise separable convolutions")
        print(f"  ✓ Gradient checkpointing: {use_gradient_checkpointing}")
        print(f"  ✓ Chunk size: {attention_chunk_size}")
        print(f"  ✓ Bottleneck conditioning (4x reduction)")
        print(f"  ✓ Levels: {len(list_neurons_per_level)} (deeper network)")
        print(f"  ✓ Blocks per level: {number_residual_blocks}")
        print(f"  ✓ KV heads: {self._num_kv_heads} (shared keys/values)")
        print(f"{'=' * 60}\n")

    @staticmethod
    def _adjust_output_shape(shape: int, num_downsamples: int) -> int:
        required_multiple = 2 ** num_downsamples
        if shape % required_multiple == 0:
            return shape
        padded = math.ceil(shape / required_multiple) * required_multiple
        warnings.warn(f"output_shape {shape} adjusted to {padded}", UserWarning)
        return padded

    def build_model(self):
        """Constrói o modelo ultra-eficiente"""

        if not self._class_info or 'number_classes' not in self._class_info:
            raise ValueError("number_samples_per_class must contain 'number_classes'")

        num_classes = self._class_info['number_classes']

        # ===== INPUTS =====
        image_input = Input(
            shape=(self._output_shape, self._embedding_channels),
            name="image_input",
            dtype=tf.float32
        )
        time_input = Input(shape=(), dtype=tf.int32, name="time_input")
        label_input = Input(shape=(num_classes,), dtype=tf.float32, name="label_input")

        # ===== EMBEDDINGS COM ECONOMIA =====
        base_dim = self._list_neurons[0]

        # Separable conv inicial
        x = DepthwiseConv1D(kernel_size=7, padding='same',
                            depthwise_initializer=HeNormal(), name='initial_dwconv')(image_input)
        x = Conv1D(base_dim, kernel_size=1, kernel_initializer=HeNormal(), name='initial_pwconv')(x)

        # Compact embeddings
        time_emb = CompactTimeEmbedding(base_dim * 4, name='time_embed')(time_input)
        time_emb = Dense(base_dim * 2, activation='swish', name='time_dense')(time_emb)

        label_emb = Dense(base_dim * 2, activation='swish', name='label_dense')(label_input)

        cond = Concatenate(name='cond_concat')([time_emb, label_emb])

        # Bottleneck conditioning
        cond_bottleneck = max(16, base_dim)
        cond = Dense(cond_bottleneck, activation='swish', name='cond_bottleneck')(cond)
        cond = Dense(base_dim * 4, name='cond_expand')(cond)

        # ===== ENCODER =====
        skip_connections = []
        last_level_idx = len(self._list_neurons) - 1

        for level, dim in enumerate(self._list_neurons):
            is_deepest = (level == last_level_idx)

            for block_idx in range(self._number_residual_blocks):
                if is_deepest:
                    # Attention apenas no nível mais profundo
                    x = EfficientDiTBlock(
                        dim,
                        num_heads=self._num_heads,
                        num_kv_heads=self._num_kv_heads,
                        dropout=self._dropout,
                        chunk_size=self._attention_chunk_size,
                        use_checkpointing=self._use_checkpointing,
                        name=f'enc_attn_l{level}_b{block_idx}'
                    )([x, cond])
                else:
                    # Depthwise separable convs nos outros níveis
                    x = DepthwiseSeparableResBlock(
                        dim,
                        dropout=self._dropout,
                        name=f'enc_conv_l{level}_b{block_idx}'
                    )([x, cond])

            skip_connections.append(x)

            if level < len(self._list_neurons) - 1:
                next_dim = self._list_neurons[level + 1]
                # Separable downsample
                x = DepthwiseConv1D(kernel_size=3, strides=2, padding='same',
                                    name=f'down_dw_l{level}')(x)
                x = Conv1D(next_dim, kernel_size=1, name=f'down_pw_l{level}')(x)

        # ===== BOTTLENECK =====
        bottleneck_dim = self._list_neurons[-1]

        for i in range(self._number_residual_blocks):
            x = EfficientDiTBlock(
                bottleneck_dim,
                num_heads=self._num_heads,
                num_kv_heads=self._num_kv_heads,
                dropout=self._dropout,
                chunk_size=self._attention_chunk_size,
                use_checkpointing=self._use_checkpointing,
                name=f'bottleneck_attn_{i}'
            )([x, cond])

        # ===== DECODER =====
        for level in reversed(range(len(self._list_neurons))):
            dim = self._list_neurons[level]
            is_deepest = (level == last_level_idx)

            if level < len(self._list_neurons) - 1:
                # Upsample
                x = UpSampling1D(size=2, name=f'upsample_l{level}')(x)
                x = DepthwiseConv1D(kernel_size=3, padding='same', name=f'up_dw_l{level}')(x)
                x = Conv1D(dim, kernel_size=1, name=f'up_pw_l{level}')(x)

            # Merge skip
            skip = skip_connections.pop()
            x = Concatenate(axis=-1, name=f'concat_l{level}')([x, skip])
            x = Conv1D(dim, kernel_size=1, name=f'merge_l{level}')(x)

            # Decoder blocks
            for block_idx in range(self._number_residual_blocks):
                if is_deepest:
                    x = EfficientDiTBlock(
                        dim,
                        num_heads=self._num_heads,
                        num_kv_heads=self._num_kv_heads,
                        dropout=self._dropout,
                        chunk_size=self._attention_chunk_size,
                        use_checkpointing=self._use_checkpointing,
                        name=f'dec_attn_l{level}_b{block_idx}'
                    )([x, cond])
                else:
                    x = DepthwiseSeparableResBlock(
                        dim,
                        dropout=self._dropout,
                        name=f'dec_conv_l{level}_b{block_idx}'
                    )([x, cond])

        # ===== OUTPUT =====
        x = LayerNormalization(epsilon=1e-6, name='final_norm')(x)
        x = Conv1D(self._embedding_channels, kernel_size=1, name='final_conv')(x)

        if self._last_activation and self._last_activation != 'linear':
            x = Activation(self._last_activation, name='final_activation')(x)

        model = Model(
            inputs=[image_input, time_input, label_input],
            outputs=x,
            name='UltraEfficientDiffusionUNet'
        )

        total_params = model.count_params()
        print(f"\n{'=' * 60}")
        print(f"✅ Model Built Successfully!")
        print(f"{'=' * 60}")
        print(f"Total parameters: {total_params:,}")
        print(f"Estimated memory per forward pass: ~{self._estimate_memory(total_params):.1f} MB")
        print(f"\n💡 Memory Tips:")
        print(f"  - Use batch_size <= 8 for training")
        print(f"  - Enable gradient_checkpointing=True")
        print(f"  - Enable mixed_precision=True for 2x memory reduction")
        print(f"  - Use tf.data for efficient data loading")
        print(f"{'=' * 60}\n")
        model.summary()
        return model

    def _estimate_memory(self, params):
        """Estima uso de memória (muito aproximado)"""
        # Params em float32 = 4 bytes cada
        param_memory = (params * 4) / (1024 ** 2)  # MB
        # Ativações aproximadamente 3x params
        activation_memory = param_memory * 3
        return param_memory + activation_memory

    # Properties
    @property
    def embedding_dimension(self):
        return self._output_shape

    @property
    def embedding_channels(self):
        return self._embedding_channels

    @property
    def list_neurons_per_level(self):
        return self._list_neurons

    @property
    def number_residual_blocks(self):
        return self._number_residual_blocks

