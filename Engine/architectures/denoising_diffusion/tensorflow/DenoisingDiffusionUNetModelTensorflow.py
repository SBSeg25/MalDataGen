__author__ = 'Synthetic Ocean AI - ULTRA-SOTA Team'
__version__ = '4.0.3-memory-optimized'

import sys
import math
import warnings
import tensorflow as tf
from typing import List, Dict, Optional, Tuple
from tensorflow.keras.layers import (
    Layer, Dense, Input, Dropout, Concatenate, Add,
    Lambda, Embedding, Reshape, Conv1D, DepthwiseConv1D,
    GlobalAveragePooling1D, Activation, Flatten, UpSampling1D,
    LayerNormalization, MultiHeadAttention
)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal, TruncatedNormal
import numpy as np

try:
    from Engine.activations.Activations import Activations
except ImportError:
    class Activations:
        def _add_activation_layer(self, x, activation):
            return Activation(activation)(x)


# ============================================================================
# ULTRA-SOTA: ROTARY POSITION EMBEDDINGS (RoPE)
# ============================================================================

class RotaryPositionEmbedding(Layer):
    """Rotary Position Embeddings (RoPE)"""

    def __init__(self, dim, max_seq_len=2048, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.max_seq_len = max_seq_len

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        half_dim = self.dim // 2
        inv_freq = 1.0 / (10000 ** (np.arange(0, self.dim, 2, dtype=np.float32) / self.dim))
        self.inv_freq = self.add_weight(
            name='rope_inv_freq',
            shape=(half_dim,),
            initializer=tf.keras.initializers.Constant(inv_freq),
            trainable=False,
            dtype=tf.float32
        )
        super().build(input_shape)

    def call(self, x):
        if self.dim < 2:
            return x

        seq_len = tf.shape(x)[1]
        t = tf.cast(tf.range(seq_len), tf.float32)
        freqs = tf.einsum('i,j->ij', t, self.inv_freq)
        freqs = tf.reshape(freqs, [1, seq_len, 1, -1])

        cos_emb = tf.cos(freqs)
        sin_emb = tf.sin(freqs)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_even_rot = x_even * cos_emb - x_odd * sin_emb
        x_odd_rot = x_odd * cos_emb + x_even * sin_emb

        x_rotated = tf.stack([x_even_rot, x_odd_rot], axis=-1)
        x_rotated = tf.reshape(x_rotated, tf.shape(x))

        return x_rotated


# ============================================================================
# ULTRA-SOTA: GROUPED QUERY ATTENTION (GQA)
# ============================================================================
class GroupedQueryAttention(Layer):
    """Grouped Query Attention (GQA) - Memory-Efficient Version"""

    def __init__(self, d_model, num_heads=8, num_kv_heads=2, dropout=0.1,
                 chunk_size=64, use_memory_efficient=True, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

        max_heads = d_model // 2
        if max_heads < 1:
            max_heads = 1

        self.num_heads = min(num_heads, max_heads)
        self.num_kv_heads = min(num_kv_heads, self.num_heads)
        self.head_dim = d_model // self.num_heads
        self.dropout_rate = dropout
        self.chunk_size = chunk_size

        assert d_model % self.num_heads == 0
        assert self.num_heads % self.num_kv_heads == 0

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.d_model)

    def build(self, input_shape):
        self.q_proj = Dense(self.d_model, use_bias=False, kernel_initializer=HeNormal())
        self.k_proj = Dense(self.num_kv_heads * self.head_dim, use_bias=False, kernel_initializer=HeNormal())
        self.v_proj = Dense(self.num_kv_heads * self.head_dim, use_bias=False, kernel_initializer=HeNormal())
        self.o_proj = Dense(self.d_model, use_bias=False, kernel_initializer=HeNormal())

        self.dropout = Dropout(self.dropout_rate)
        self.rope = RotaryPositionEmbedding(self.head_dim)

        super().build(input_shape)

    def call(self, x, context=None, training=None):
        original_batch = tf.shape(x)[0]
        original_seq = tf.shape(x)[1]

        if context is None:
            context = x

        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)

        q = tf.reshape(q, [original_batch, original_seq, self.num_heads, self.head_dim])
        k = tf.reshape(k, [tf.shape(context)[0], tf.shape(context)[1], self.num_kv_heads, self.head_dim])
        v = tf.reshape(v, [tf.shape(context)[0], tf.shape(context)[1], self.num_kv_heads, self.head_dim])

        q = self.rope(q)
        k = self.rope(k)

        k = tf.repeat(k, self.num_queries_per_kv, axis=2)
        v = tf.repeat(v, self.num_queries_per_kv, axis=2)

        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        out = self._memory_efficient_attention(q, k, v, training)

        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [original_batch, original_seq, self.num_heads * self.head_dim])

        out = self.o_proj(out)
        out = tf.ensure_shape(out, [None, None, self.d_model])

        return out

    def _memory_efficient_attention(self, q, k, v, training):
        scale = tf.cast(self.head_dim, q.dtype) ** -0.5
        batch = tf.shape(q)[0]
        heads = tf.shape(q)[1]
        seq_len = tf.shape(q)[2]
        head_dim = tf.shape(q)[3]

        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        padded_len = num_chunks * self.chunk_size
        padding_needed = padded_len - seq_len

        q_padded = tf.pad(q, [[0, 0], [0, 0], [0, padding_needed], [0, 0]], constant_values=0)
        q_chunked = tf.reshape(q_padded, [batch, heads, num_chunks, self.chunk_size, head_dim])

        def process_chunk(chunk_idx):
            q_chunk = q_chunked[:, :, chunk_idx, :, :]
            scores = tf.matmul(q_chunk, k, transpose_b=True) * scale
            attn_weights = tf.nn.softmax(scores, axis=-1)
            attn_weights = self.dropout(attn_weights, training=training)
            out_chunk = tf.matmul(attn_weights, v)
            return out_chunk

        outputs = tf.map_fn(
            process_chunk,
            tf.range(num_chunks),
            fn_output_signature=tf.TensorSpec(shape=[None, None, self.chunk_size, None], dtype=q.dtype),
            parallel_iterations=1
        )

        outputs = tf.transpose(outputs, [1, 2, 0, 3, 4])
        out = tf.reshape(outputs, [batch, heads, padded_len, head_dim])
        out = out[:, :, :seq_len, :]

        return out


# ============================================================================
# ULTRA-SOTA: SwiGLU ACTIVATION
# ============================================================================

class SwiGLU(Layer):
    """SwiGLU Activation"""

    def __init__(self, dim, hidden_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.hidden_dim = hidden_dim or int(dim * 8 / 3)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.dim)

    def build(self, input_shape):
        self.w1 = Dense(self.hidden_dim, use_bias=False, kernel_initializer=HeNormal())
        self.w2 = Dense(self.hidden_dim, use_bias=False, kernel_initializer=HeNormal())
        self.w3 = Dense(self.dim, use_bias=False, kernel_initializer=HeNormal())
        super().build(input_shape)

    def call(self, x):
        gate = tf.nn.swish(self.w1(x))
        value = self.w2(x)
        return self.w3(gate * value)


# ============================================================================
# ULTRA-SOTA: ADALN-ZERO
# ============================================================================

class AdaLNZero(Layer):
    """Adaptive Layer Norm with Zero Initialization"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        x_shape, cond_shape = input_shape
        batch, seq_len, dim = x_shape
        return [
            (batch, seq_len, dim),
            (batch, 1, dim),
            (batch, 1, dim),
            (batch, 1, dim),
            (batch, 1, dim)
        ]

    def build(self, input_shape):
        features_shape, condition_shape = input_shape
        dim = features_shape[-1]

        self.ada_proj = Dense(
            dim * 6,
            kernel_initializer='zeros',
            bias_initializer='zeros'
        )

        self.norm = LayerNormalization(epsilon=1e-6)
        super().build(input_shape)

    def call(self, inputs):
        x, cond = inputs
        x_norm = self.norm(x)
        ada_params = self.ada_proj(cond)
        ada_params = tf.expand_dims(ada_params, axis=1)

        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = tf.split(
            ada_params, 6, axis=-1
        )

        x_modulated = x_norm * (1 + scale_attn) + shift_attn

        return x_modulated, gate_attn, shift_ffn, scale_ffn, gate_ffn


# ============================================================================
# ULTRA-SOTA: DiT BLOCK
# ============================================================================

class DiTBlock(Layer):
    """Diffusion Transformer Block"""

    def __init__(self, d_model, num_heads=8, num_kv_heads=2, mlp_ratio=4, dropout=0.1,
                 chunk_size=128, use_memory_efficient=True, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout
        self.chunk_size = chunk_size
        self.use_memory_efficient = use_memory_efficient

    def compute_output_shape(self, input_shape):
        x_shape, cond_shape = input_shape
        return x_shape

    def build(self, input_shape):
        self.adaln = AdaLNZero()

        self.attn = GroupedQueryAttention(
            self.d_model,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            dropout=self.dropout_rate,
            chunk_size=self.chunk_size,
            use_memory_efficient=self.use_memory_efficient
        )

        self.ffn = SwiGLU(
            self.d_model,
            hidden_dim=int(self.d_model * self.mlp_ratio)
        )

        self.norm_ffn = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, inputs, training=None):
        x, cond = inputs

        x_mod, gate_attn, shift_ffn, scale_ffn, gate_ffn = self.adaln([x, cond])

        attn_out = self.attn(x_mod, training=training)
        attn_out = self.dropout(attn_out, training=training)

        x = tf.ensure_shape(x, [None, None, self.d_model])
        attn_out = tf.ensure_shape(attn_out, [None, None, self.d_model])
        gate_attn = tf.ensure_shape(gate_attn, [None, 1, self.d_model])

        x = x + gate_attn * attn_out

        x_norm = self.norm_ffn(x)
        x_mod_ffn = x_norm * (1 + scale_ffn) + shift_ffn

        ffn_out = self.ffn(x_mod_ffn)
        ffn_out = self.dropout(ffn_out, training=training)

        ffn_out = tf.ensure_shape(ffn_out, [None, None, self.d_model])
        gate_ffn = tf.ensure_shape(gate_ffn, [None, 1, self.d_model])

        x = x + gate_ffn * ffn_out

        return x


# ============================================================================
# ULTRA-SOTA: CONVOLUTIONAL BLOCKS
# ============================================================================

class ConvNextBlock(Layer):
    """ConvNeXt Block - Usado nos níveis sem atenção"""

    def __init__(self, dim, mlp_ratio=4, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        self.dwconv = DepthwiseConv1D(
            kernel_size=7,
            padding='same',
            depthwise_initializer=HeNormal()
        )

        self.norm = LayerNormalization(epsilon=1e-6)

        hidden_dim = int(self.dim * self.mlp_ratio)
        self.pwconv1 = Dense(hidden_dim, kernel_initializer=TruncatedNormal(stddev=0.02))
        self.pwconv2 = Dense(self.dim, kernel_initializer=TruncatedNormal(stddev=0.02))

        self.dropout = Dropout(self.dropout_rate)

        self.gamma = self.add_weight(
            name='layer_scale',
            shape=(self.dim,),
            initializer=tf.keras.initializers.Constant(1e-6),
            trainable=True
        )

        super().build(input_shape)

    def call(self, x, training=None):
        shortcut = x

        x = self.dwconv(x)
        x = self.norm(x)

        x = self.pwconv1(x)
        x = tf.nn.gelu(x)
        x = self.pwconv2(x)

        x = self.gamma * x
        x = self.dropout(x, training=training)

        return shortcut + x


class ConvResidualBlock(Layer):
    """Bloco Residual Convolucional Simples com Conditioning"""

    def __init__(self, dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.dropout_rate = dropout

    def compute_output_shape(self, input_shape):
        x_shape, cond_shape = input_shape
        return x_shape

    def build(self, input_shape):
        # Conditioning projection
        self.cond_proj = Dense(self.dim * 2, activation='swish')

        # Convolutional layers
        self.conv1 = Conv1D(self.dim, kernel_size=3, padding='same', kernel_initializer=HeNormal())
        self.conv2 = Conv1D(self.dim, kernel_size=3, padding='same', kernel_initializer=HeNormal())

        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)

        self.dropout = Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, inputs, training=None):
        x, cond = inputs

        # Process conditioning
        cond_emb = self.cond_proj(cond)
        scale, shift = tf.split(cond_emb, 2, axis=-1)
        scale = tf.expand_dims(scale, axis=1)
        shift = tf.expand_dims(shift, axis=1)

        # First conv block
        h = self.norm1(x)
        h = h * (1 + scale) + shift
        h = tf.nn.swish(h)
        h = self.conv1(h)
        h = self.dropout(h, training=training)

        # Second conv block
        h = self.norm2(h)
        h = tf.nn.swish(h)
        h = self.conv2(h)
        h = self.dropout(h, training=training)

        return x + h


# ============================================================================
# HELPER: FOURIER TIME EMBEDDING
# ============================================================================

class FourierTimeEmbedding(Layer):
    """Fourier Time Embeddings"""

    def __init__(self, dim, max_period=10000, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.max_period = max_period

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dim)

    def build(self, input_shape):
        half_dim = self.dim // 2
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
        self.proj = Dense(self.dim)
        super().build(input_shape)

    def call(self, timesteps):
        timesteps = tf.cast(timesteps, tf.float32)
        timesteps = tf.expand_dims(timesteps, -1)
        args = timesteps * self.freqs[None, :]
        embedding = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
        return self.proj(embedding)


# ============================================================================
# ULTRA-SOTA: U-NET MEMORY-OPTIMIZED
# ============================================================================

class DenoisingDiffusionUNetModelTensorflow(Activations):
    """
    Ultra State-of-the-Art Diffusion U-Net - Memory Optimized

    ✅ ATENÇÃO APENAS NO ÚLTIMO NÍVEL
    - Níveis iniciais/intermediários: ConvResidualBlocks (mais rápidos, menos memória)
    - Último nível: DiTBlocks com atenção completa (melhor qualidade)
    - Bottleneck: DiTBlocks com atenção completa

    Isso reduz drasticamente o uso de memória mantendo alta qualidade!
    """

    def __init__(
            self,
            output_shape: int = 128,
            embedding_channels: int = 1,
            list_neurons_per_level: List[int] = None,
            number_residual_blocks: int = 2,
            normalization_groups: int = 1,
            num_heads: int = 4,
            head_dim: int = 16,
            dropout_rate: float = 0.1,
            se_ratio: int = 4,
            intermediary_activation_function: str = 'gelu',
            intermediary_activation_alpha: float = 0.05,
            last_layer_activation: str = 'linear',
            number_samples_per_class: Optional[Dict] = None,
            num_kv_heads: int = 2,
            attention_chunk_size: int = 32,
            use_memory_efficient_attention: bool = True,
            **kwargs
    ):

        list_neurons_per_level =  [4, 8, 16, 32, 64, 96]

        if not isinstance(output_shape, int) or output_shape <= 0:
            raise ValueError("output_shape must be a positive integer")

        if not isinstance(embedding_channels, int) or embedding_channels <= 0:
            raise ValueError("embedding_channels must be a positive integer")

        if not isinstance(list_neurons_per_level, list) or not all(
                isinstance(n, int) and n > 0 for n in list_neurons_per_level):
            raise ValueError("list_neurons_per_level must be a list of positive integers")

        if not isinstance(number_residual_blocks, int) or number_residual_blocks <= 0:
            raise ValueError("number_residual_blocks must be a positive integer")

        self._output_shape = self._adjust_output_shape(output_shape, len(list_neurons_per_level))
        self._embedding_channels = embedding_channels
        self._list_neurons = list_neurons_per_level
        self._number_residual_blocks = number_residual_blocks
        self._normalization_groups = normalization_groups

        # Configuração de atenção
        min_dim = min(list_neurons_per_level)
        max_possible_heads = min_dim // 4

        if max_possible_heads < 1:
            max_possible_heads = 1
            warnings.warn(
                f"Smallest dimension ({min_dim}) too small for multi-head attention. Using single head.",
                UserWarning
            )

        self._num_heads = min(num_heads, max_possible_heads)
        self._num_kv_heads = min(num_kv_heads, self._num_heads)

        while self._num_heads % self._num_kv_heads != 0:
            self._num_kv_heads -= 1
        if self._num_kv_heads < 1:
            self._num_kv_heads = 1

        self._head_dim = head_dim
        self._dropout = dropout_rate
        self._se_ratio = se_ratio
        self._intermediary_activation = intermediary_activation_function
        self._intermediary_activation_alpha = intermediary_activation_alpha
        self._last_activation = last_layer_activation
        self._class_info = number_samples_per_class

        self._num_transformer_blocks = number_residual_blocks
        self._attention_chunk_size = attention_chunk_size
        self._use_memory_efficient = use_memory_efficient_attention

        print("\n" + "=" * 80)
        print("🚀 ULTRA-SOTA DIFFUSION U-NET v4.0.3 (MEMORY-OPTIMIZED)")
        print("=" * 80)
        print(f"✅ ATENÇÃO APENAS NO ÚLTIMO NÍVEL + BOTTLENECK")
        print(f"   • Níveis 0-{len(list_neurons_per_level) - 2}: ConvResidualBlocks (rápido, baixa memória)")
        print(f"   • Nível {len(list_neurons_per_level) - 1}: DiTBlocks com atenção (alta qualidade)")
        print(f"   • Bottleneck: DiTBlocks com atenção completa")
        print(f"GQA: Query heads={self._num_heads}, KV heads={self._num_kv_heads}")
        print(f"Chunk size: {attention_chunk_size} (memory-efficient: {use_memory_efficient_attention})")
        print(f"Neurons per level: {list_neurons_per_level}")
        print("=" * 80 + "\n")

    @staticmethod
    def _adjust_output_shape(shape: int, num_downsamples: int) -> int:
        required_multiple = 2 ** num_downsamples

        if shape % required_multiple == 0:
            return shape

        padded = math.ceil(shape / required_multiple) * required_multiple
        warnings.warn(
            f"output_shape {shape} adjusted to {padded}",
            UserWarning
        )
        return padded

    def build_model(self):
        """Constrói o modelo memory-optimized"""

        if self._class_info is None:
            raise ValueError("number_samples_per_class is required")

        if not isinstance(self._class_info, dict):
            raise ValueError("number_samples_per_class must be a dictionary")

        if 'number_classes' not in self._class_info:
            raise ValueError("number_samples_per_class must contain 'number_classes' key")

        num_classes = self._class_info['number_classes']

        # ===== INPUTS =====
        image_input = Input(
            shape=(self._output_shape, self._embedding_channels),
            name="image_input"
        )
        time_input = Input(shape=(), dtype=tf.int32, name="time_input")
        label_input = Input(shape=(num_classes,), dtype=tf.float32, name="label_input")

        # ===== EMBEDDINGS =====
        base_dim = self._list_neurons[0]

        x = Conv1D(base_dim, kernel_size=3, padding='same', kernel_initializer=HeNormal())(image_input)

        time_emb = FourierTimeEmbedding(base_dim * 4)(time_input)
        time_emb = Dense(base_dim * 4, activation='swish')(time_emb)
        time_emb = Dense(base_dim * 4)(time_emb)

        label_emb = Dense(base_dim * 2, activation='swish')(label_input)
        label_emb = Dense(base_dim * 2)(label_emb)

        cond = Concatenate()([time_emb, label_emb])
        cond = Dense(base_dim * 4, activation='swish')(cond)

        # ===== ENCODER =====
        skip_connections = []
        last_level_idx = len(self._list_neurons) - 1

        for level, dim in enumerate(self._list_neurons):
            is_last_level = (level == last_level_idx)

            for block_idx in range(self._number_residual_blocks):
                if is_last_level:
                    # ✅ ÚLTIMO NÍVEL: Usa atenção completa
                    x = DiTBlock(
                        dim,
                        num_heads=self._num_heads,
                        num_kv_heads=self._num_kv_heads,
                        dropout=self._dropout,
                        chunk_size=self._attention_chunk_size,
                        use_memory_efficient=self._use_memory_efficient,
                        name=f'enc_attention_l{level}_b{block_idx}'
                    )([x, cond])
                else:
                    # ✅ OUTROS NÍVEIS: Usa apenas convoluções
                    x = ConvResidualBlock(
                        dim,
                        dropout=self._dropout,
                        name=f'enc_conv_l{level}_b{block_idx}'
                    )([x, cond])

            skip_connections.append(x)

            # Downsample
            if level < len(self._list_neurons) - 1:
                next_dim = self._list_neurons[level + 1]
                x = Conv1D(next_dim, kernel_size=3, strides=2, padding='same')(x)

        # ===== BOTTLENECK =====
        bottleneck_dim = self._list_neurons[-1]

        print(f"📍 Bottleneck: {self._number_residual_blocks} DiTBlocks com atenção")
        for i in range(self._number_residual_blocks):
            x = DiTBlock(
                bottleneck_dim,
                num_heads=self._num_heads,
                num_kv_heads=self._num_kv_heads,
                dropout=self._dropout,
                chunk_size=self._attention_chunk_size,
                use_memory_efficient=self._use_memory_efficient,
                name=f'bottleneck_attention_{i}'
            )([x, cond])

        # ===== DECODER =====
        for level in reversed(range(len(self._list_neurons))):
            dim = self._list_neurons[level]
            is_last_level = (level == last_level_idx)

            # Upsample
            if level < len(self._list_neurons) - 1:
                x = UpSampling1D(size=2)(x)
                x = Conv1D(dim, kernel_size=3, padding='same')(x)

            # Skip connection
            skip = skip_connections.pop()
            x = Concatenate(axis=-1)([x, skip])
            x = Conv1D(dim, kernel_size=1)(x)

            # Blocos
            for block_idx in range(self._number_residual_blocks):
                if is_last_level:
                    # ✅ ÚLTIMO NÍVEL: Usa atenção completa
                    x = DiTBlock(
                        dim,
                        num_heads=self._num_heads,
                        num_kv_heads=self._num_kv_heads,
                        dropout=self._dropout,
                        chunk_size=self._attention_chunk_size,
                        use_memory_efficient=self._use_memory_efficient,
                        name=f'dec_attention_l{level}_b{block_idx}'
                    )([x, cond])
                else:
                    # ✅ OUTROS NÍVEIS: Usa apenas convoluções
                    x = ConvResidualBlock(
                        dim,
                        dropout=self._dropout,
                        name=f'dec_conv_l{level}_b{block_idx}'
                    )([x, cond])

        # ===== OUTPUT =====
        final_norm = LayerNormalization(epsilon=1e-6, name='final_norm')
        x = final_norm(x)

        x = Conv1D(self._embedding_channels, kernel_size=3, padding='same', name='final_conv')(x)

        if self._last_activation and self._last_activation != 'linear':
            x = Activation(self._last_activation, name='final_activation')(x)

        model = Model(
            inputs=[image_input, time_input, label_input],
            outputs=x,
            name='MemoryOptimizedDiffusionUNet'
        )

        print(f"✅ Model built successfully!")
        print(f"   Total parameters: {model.count_params():,}")
        print(
            f"   Memory optimization: ~{(len(self._list_neurons) - 1) / len(self._list_neurons) * 100:.0f}% reduction in attention ops")
        model.summary()
        return model

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
    def last_layer_activation(self):
        return self._last_activation

    @property
    def number_residual_blocks(self):
        return self._number_residual_blocks

    @property
    def normalization_groups(self):
        return self._normalization_groups

    @property
    def number_samples_per_class(self):
        return self._class_info