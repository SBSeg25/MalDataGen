import math

from Engine.layers.tensorflow.RMSNorm import RMSNorm

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
    import tensorflow
    from Engine.activations.Activations import Activations

except ImportError as error:
    print(error)
    sys.exit(-1)


class GroupedQueryAttention(Layer):
    """GQA with aggressive memory optimization"""

    def __init__(self, d_model, num_heads=4, num_kv_heads=1, dropout=0.1,
                 chunk_size=8, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

        max_heads = max(d_model // 8, 1)  # More aggressive head reduction
        self.num_heads = min(num_heads, max_heads)
        self.num_kv_heads = min(num_kv_heads, self.num_heads)
        self.head_dim = d_model // self.num_heads
        self.dropout_rate = dropout
        self.chunk_size = chunk_size  # Smaller chunks = less memory

        assert d_model % self.num_heads == 0
        assert self.num_heads % self.num_kv_heads == 0

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.d_model)

    def build(self, input_shape):
        # Shared projections to reduce parameters
        self.q_proj = Dense(self.d_model, use_bias=False, kernel_initializer=HeNormal())
        self.k_proj = Dense(self.num_kv_heads * self.head_dim, use_bias=False, kernel_initializer=HeNormal())
        self.v_proj = Dense(self.num_kv_heads * self.head_dim, use_bias=False, kernel_initializer=HeNormal())
        self.o_proj = Dense(self.d_model, use_bias=False, kernel_initializer=HeNormal())

        self.dropout = Dropout(self.dropout_rate)
        self.rope = RotaryPositionEmbedding(self.head_dim)

        super().build(input_shape)

    def call(self, x, context=None, training=None):
        batch = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        if context is None:
            context = x

        # Projections
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)

        # Reshape
        q = tf.reshape(q, [batch, seq_len, self.num_heads, self.head_dim])
        k = tf.reshape(k, [batch, tf.shape(context)[1], self.num_kv_heads, self.head_dim])
        v = tf.reshape(v, [batch, tf.shape(context)[1], self.num_kv_heads, self.head_dim])

        # RoPE
        q = self.rope(q)
        k = self.rope(k)

        # GQA: Repeat KV
        k = tf.repeat(k, self.num_queries_per_kv, axis=2)
        v = tf.repeat(v, self.num_queries_per_kv, axis=2)

        # Transpose
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        # Memory-efficient attention (gradient checkpointing disabled for stability)
        out = self._chunked_attention(q, k, v, training)

        # Merge heads
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [batch, seq_len, self.d_model])

        # Output projection
        out = self.o_proj(out)
        return out

    def _chunked_attention(self, q, k, v, training):
        """Process attention in small chunks to minimize memory"""
        scale = tf.cast(self.head_dim, q.dtype) ** -0.5
        batch = tf.shape(q)[0]
        heads = tf.shape(q)[1]
        seq_len = tf.shape(q)[2]
        head_dim = tf.shape(q)[3]

        # Calculate chunks
        num_chunks = tf.cast(tf.math.ceil(tf.cast(seq_len, tf.float32) / tf.cast(self.chunk_size, tf.float32)),
                             tf.int32)
        padded_len = num_chunks * self.chunk_size
        pad_needed = padded_len - seq_len

        # Pad queries
        q = tf.pad(q, [[0, 0], [0, 0], [0, pad_needed], [0, 0]])
        q_chunked = tf.reshape(q, [batch, heads, num_chunks, self.chunk_size, head_dim])

        # Process chunks using map_fn (graph-compatible)
        def process_chunk(i):
            q_chunk = q_chunked[:, :, i, :, :]

            # Attention computation
            scores = tf.matmul(q_chunk, k, transpose_b=True) * scale
            attn = tf.nn.softmax(scores, axis=-1)
            attn = self.dropout(attn, training=training)

            out_chunk = tf.matmul(attn, v)
            return out_chunk

        # Map over all chunks
        outputs = tf.map_fn(
            process_chunk,
            tf.range(num_chunks),
            fn_output_signature=tf.TensorSpec(shape=[None, None, self.chunk_size, None], dtype=q.dtype),
            parallel_iterations=1  # Sequential for memory efficiency
        )  # Shape: [num_chunks, batch, heads, chunk_size, head_dim]

        # Transpose to [batch, heads, num_chunks, chunk_size, head_dim]
        outputs = tf.transpose(outputs, [1, 2, 0, 3, 4])

        # Reshape to [batch, heads, padded_len, head_dim]
        out = tf.reshape(outputs, [batch, heads, padded_len, head_dim])

        # Remove padding
        out = out[:, :, :seq_len, :]

        return out
