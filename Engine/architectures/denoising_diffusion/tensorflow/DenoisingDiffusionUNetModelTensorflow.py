__author__ = 'Synthetic Ocean AI - ULTRA-SOTA Team'
__version__ = '6.0.1-dit-linear-attention-eca-FIXED'

import sys
import math
import warnings
import tensorflow as tf
from typing import List, Dict, Optional, Tuple
from tensorflow.keras.layers import (
    Layer, Dense, Input, Dropout, Concatenate, Add,
    Lambda, Embedding, Reshape, Conv1D, DepthwiseConv1D,
    GlobalAveragePooling1D, Activation, Flatten,
    LayerNormalization, MultiHeadAttention,
    Multiply, GlobalMaxPooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal, TruncatedNormal, Constant
from tensorflow.keras.mixed_precision import set_global_policy
import numpy as np

try:
    from Engine.activations.Activations import Activations
except ImportError:
    class Activations:
        def _add_activation_layer(self, x, activation):
            return Activation(activation)(x)


# ============================================================================
# EFFICIENT CHANNEL ATTENTION (ECA)
# ============================================================================

class ECAModule(Layer):
    """
    Efficient Channel Attention - Atenção em canais com kernel 1D adaptativo.
    """

    def __init__(self, gamma=2, b=1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.b = b

    def build(self, input_shape):
        channels = input_shape[-1]
        t = int(abs((math.log2(channels) + self.b) / self.gamma))
        self.kernel_size = max(3, t if t % 2 else t + 1)

        self.conv = Conv1D(
            1,
            kernel_size=self.kernel_size,
            padding='same',
            use_bias=False,
            name='eca_conv'
        )
        super().build(input_shape)

    def call(self, x):
        gap = tf.reduce_mean(x, axis=1, keepdims=True)
        attention = self.conv(gap)
        attention = tf.nn.sigmoid(attention)
        return x * attention


# ============================================================================
# LINEAR ATTENTION (O(N) complexity) - FIXED
# ============================================================================

class LinearAttention(Layer):
    """
    Linear Attention com complexidade O(N) ao invés de O(N²).
    Usa kernel feature map para evitar materializar a matriz de atenção completa.

    CORREÇÕES:
    - Softplus ao invés de elu+1 (mais estável)
    - Epsilon maior na normalização (1e-3 vs 1e-6)
    - Dupla proteção contra divisão por zero
    """

    def __init__(self, dim, num_heads=8, dropout=0.1, eps=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout_rate = dropout
        self.eps = eps  # Epsilon maior para estabilidade

    def build(self, input_shape):
        self.q_proj = Dense(self.dim, use_bias=True, name='q_proj')  # Bias ajuda
        self.k_proj = Dense(self.dim, use_bias=True, name='k_proj')
        self.v_proj = Dense(self.dim, use_bias=True, name='v_proj')
        self.out_proj = Dense(self.dim, use_bias=True, name='out_proj')
        self.dropout = Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, x, training=None):
        batch, seq_len, _ = tf.shape(x)[0], tf.shape(x)[1], x.shape[-1]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = tf.reshape(q, [batch, seq_len, self.num_heads, self.head_dim])
        k = tf.reshape(k, [batch, seq_len, self.num_heads, self.head_dim])
        v = tf.reshape(v, [batch, seq_len, self.num_heads, self.head_dim])

        # FIX: Softplus é mais estável que elu+1
        q = tf.nn.softplus(q)
        k = tf.nn.softplus(k)

        # Linear attention: Q(K^TV)
        kv = tf.einsum('bthd,bthe->bhde', k, v)
        k_sum = tf.reduce_sum(k, axis=1, keepdims=True)

        out = tf.einsum('bthd,bhde->bthe', q, kv)
        normalizer = tf.einsum('bthd,bihd->bth', q, k_sum)

        # FIX: Epsilon maior + dupla proteção
        normalizer = tf.maximum(normalizer, self.eps)
        out = out / (tf.expand_dims(normalizer, axis=-1) + 1e-6)

        out = tf.reshape(out, [batch, seq_len, self.dim])
        out = self.out_proj(out)
        out = self.dropout(out, training=training)

        return out


# ============================================================================
# CROSS ATTENTION LINEAR - FIXED
# ============================================================================

class LinearCrossAttention(Layer):
    """Cross Attention Linear para condicionamento eficiente."""

    def __init__(self, dim, num_heads=8, dropout=0.1, eps=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout_rate = dropout
        self.eps = eps

    def build(self, input_shape):
        if isinstance(input_shape, list):
            x_shape, context_shape = input_shape
        else:
            x_shape = context_shape = input_shape

        self.q_proj = Dense(self.dim, use_bias=True, name='q_proj')
        self.k_proj = Dense(self.dim, use_bias=True, name='k_proj')
        self.v_proj = Dense(self.dim, use_bias=True, name='v_proj')
        self.out_proj = Dense(self.dim, use_bias=True, name='out_proj')
        self.dropout = Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, inputs, training=None):
        x, context = inputs

        batch = tf.shape(x)[0]
        x_len = tf.shape(x)[1]
        ctx_len = tf.shape(context)[1]

        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)

        q = tf.reshape(q, [batch, x_len, self.num_heads, self.head_dim])
        k = tf.reshape(k, [batch, ctx_len, self.num_heads, self.head_dim])
        v = tf.reshape(v, [batch, ctx_len, self.num_heads, self.head_dim])

        # FIX: Softplus
        q = tf.nn.softplus(q)
        k = tf.nn.softplus(k)

        kv = tf.einsum('bthd,bthe->bhde', k, v)
        k_sum = tf.reduce_sum(k, axis=1, keepdims=True)

        out = tf.einsum('bthd,bhde->bthe', q, kv)
        normalizer = tf.einsum('bthd,bihd->bth', q, k_sum)

        # FIX: Normalização estável
        normalizer = tf.maximum(normalizer, self.eps)
        out = out / (tf.expand_dims(normalizer, axis=-1) + 1e-6)

        out = tf.reshape(out, [batch, x_len, self.dim])
        out = self.out_proj(out)
        out = self.dropout(out, training=training)

        return out


# ============================================================================
# GEGLU ACTIVATION
# ============================================================================

class GeGLU(Layer):
    """GeGLU: GELU Gated Linear Unit"""

    def __init__(self, dim, expansion_factor=4.0, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.hidden_dim = int(dim * expansion_factor)
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.proj = Dense(self.hidden_dim * 2, use_bias=True, name='proj')
        self.out_proj = Dense(self.dim, use_bias=True, name='out_proj')
        self.dropout = Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, x, training=None):
        x = self.proj(x)
        gate, value = tf.split(x, 2, axis=-1)
        gate = tf.nn.gelu(gate, approximate=True)
        hidden = gate * value
        out = self.out_proj(hidden)
        out = self.dropout(out, training=training)
        return out


# ============================================================================
# ADAPTIVE LAYER NORMALIZATION (AdaLN)
# ============================================================================

class AdaptiveLayerNorm(Layer):
    """
    Adaptive Layer Normalization - modula a normalização com condicionamento.
    Usado em DiT para injetar informação temporal e de classe.
    """

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape):
        if isinstance(input_shape, list):
            x_shape, cond_shape = input_shape
        else:
            x_shape = cond_shape = input_shape

        self.norm = LayerNormalization(epsilon=1e-6, name='norm')

        # Projeção do condicionamento para scale e shift
        self.ada_proj = Dense(self.dim * 2, kernel_initializer='zeros', name='ada_proj')

        self.norm.build(x_shape)
        self.ada_proj.build(cond_shape)
        super().build(input_shape)

    def call(self, inputs):
        x, cond = inputs

        # Normaliza
        x_norm = self.norm(x)

        # Obtém parâmetros adaptativos
        ada_params = self.ada_proj(cond)

        # Expande para broadcast se necessário
        if ada_params.shape.rank == 2:
            ada_params = tf.expand_dims(ada_params, axis=1)

        scale, shift = tf.split(ada_params, 2, axis=-1)

        # Aplica modulação
        return x_norm * (1 + scale) + shift


# ============================================================================
# ADALN-ZERO (DiT-style initialization) - FIXED
# ============================================================================

class AdaLNZero(Layer):
    """
    AdaLN-Zero: Adaptive Layer Norm com gate inicializado próximo de zero.
    Permite que o modelo comece sem modificar o skip connection.

    CORREÇÃO: Inicialização muito pequena mas não exatamente zero.
    """

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape):
        if isinstance(input_shape, list):
            x_shape, cond_shape = input_shape
        else:
            x_shape = cond_shape = input_shape

        self.norm = LayerNormalization(epsilon=1e-6, name='norm')

        # FIX: Inicialização pequena mas não zero
        # 6 parâmetros: scale, shift, gate para self-attn e ffn
        self.ada_proj = Dense(
            self.dim * 6,
            kernel_initializer=TruncatedNormal(stddev=1e-5),  # Muito pequeno
            bias_initializer=Constant(1e-6),  # Evita zero absoluto
            name='ada_proj'
        )

        self.norm.build(x_shape)
        self.ada_proj.build(cond_shape)
        super().build(input_shape)

    def call(self, inputs):
        x, cond = inputs

        ada_params = self.ada_proj(cond)

        if ada_params.shape.rank == 2:
            ada_params = tf.expand_dims(ada_params, axis=1)

        # 6 parâmetros: scale_attn, shift_attn, gate_attn, scale_ffn, shift_ffn, gate_ffn
        scale_attn, shift_attn, gate_attn, scale_ffn, shift_ffn, gate_ffn = tf.split(
            ada_params, 6, axis=-1
        )

        x_norm = self.norm(x)

        return x_norm, scale_attn, shift_attn, gate_attn, scale_ffn, shift_ffn, gate_ffn


# ============================================================================
# DIT BLOCK (Core Building Block) - FIXED
# ============================================================================

class DiTBlock(Layer):
    """
    DiT Block com:
    - AdaLN-Zero para condicionamento
    - Linear Self-Attention
    - Linear Cross-Attention (opcional)
    - ECA Module
    - GeGLU FFN

    CORREÇÕES:
    - Fallback norm criada no build
    - Tensores zeros com shape correto
    """

    def __init__(self, dim, num_heads=8, dropout=0.1, expansion=4.0,
                 use_cross_attn=True, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.expansion = expansion
        self.use_cross_attn = use_cross_attn

    def build(self, input_shape):
        # Parse input shapes
        if isinstance(input_shape, list):
            if len(input_shape) >= 3:
                x_shape, cond_shape, context_shape = input_shape[0], input_shape[1], input_shape[2]
            elif len(input_shape) == 2:
                x_shape, cond_shape = input_shape[0], input_shape[1]
                context_shape = None
            else:
                x_shape = input_shape[0]
                cond_shape = None
                context_shape = None
        else:
            x_shape = input_shape
            cond_shape = None
            context_shape = None

        # FIX: Cria fallback norm no build
        self.fallback_norm = LayerNormalization(epsilon=1e-6, name='fallback_norm')
        self.fallback_norm.build(x_shape)

        # AdaLN-Zero
        if cond_shape is not None:
            self.adaln = AdaLNZero(self.dim, name='adaln')
            self.adaln.build([x_shape, cond_shape])

        # Self-attention
        self.self_attn = LinearAttention(
            self.dim,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
            name='self_attn'
        )
        self.self_attn.build(x_shape)

        # Cross-attention
        if self.use_cross_attn and context_shape is not None:
            self.cross_attn = LinearCrossAttention(
                self.dim,
                num_heads=self.num_heads,
                dropout=self.dropout_rate,
                name='cross_attn'
            )
            self.cross_attn.build([x_shape, context_shape])

        # Layer norms
        self.norm_ffn = LayerNormalization(epsilon=1e-6, name='norm_ffn')
        self.norm_ffn.build(x_shape)

        # ECA
        self.eca = ECAModule(name='eca')
        self.eca.build(x_shape)

        # FFN
        self.ffn = GeGLU(
            self.dim,
            expansion_factor=self.expansion,
            dropout=self.dropout_rate,
            name='ffn'
        )
        self.ffn.build(x_shape)

        super().build(input_shape)

    def call(self, inputs, training=None):
        # Parse inputs
        if isinstance(inputs, list):
            if len(inputs) >= 3:
                x, cond, context = inputs[0], inputs[1], inputs[2]
            elif len(inputs) == 2:
                x, cond = inputs[0], inputs[1]
                context = None
            else:
                x = inputs[0]
                cond = None
                context = None
        else:
            x = inputs
            cond = None
            context = None

        # AdaLN-Zero
        if cond is not None and hasattr(self, 'adaln'):
            x_norm, scale_attn, shift_attn, gate_attn, scale_ffn, shift_ffn, gate_ffn = \
                self.adaln([x, cond])
        else:
            # FIX: Usa layer criada no build
            x_norm = self.fallback_norm(x)
            # FIX: Inicializa tensores zeros com shape correto
            scale_attn = tf.zeros_like(x_norm)
            shift_attn = tf.zeros_like(x_norm)
            gate_attn = tf.zeros_like(x_norm)
            scale_ffn = tf.zeros_like(x_norm)
            shift_ffn = tf.zeros_like(x_norm)
            gate_ffn = tf.zeros_like(x_norm)

        # Modulate antes do self-attention
        x_mod = x_norm * (1 + scale_attn) + shift_attn

        # Self-attention
        attn_out = self.self_attn(x_mod, training=training)
        x = x + gate_attn * attn_out

        # Cross-attention (se disponível)
        if self.use_cross_attn and context is not None and hasattr(self, 'cross_attn'):
            cross_out = self.cross_attn([x, context], training=training)
            x = x + cross_out

        # ECA
        x = self.eca(x)

        # FFN com modulação
        x_norm_ffn = self.norm_ffn(x)
        x_mod_ffn = x_norm_ffn * (1 + scale_ffn) + shift_ffn

        ffn_out = self.ffn(x_mod_ffn, training=training)
        x = x + gate_ffn * ffn_out

        return x

    def compute_output_shape(self, input_shape):
        # Retorna a forma do primeiro input (x)
        if isinstance(input_shape, list):
            return input_shape[0]
        return input_shape


# ============================================================================
# PATCHIFY LAYER
# ============================================================================

class PatchEmbedding1D(Layer):
    """
    Converte sequência 1D em patches e projeta para dimensão embedding.
    Similar ao ViT, mas para dados 1D.
    """

    def __init__(self, patch_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim

    def build(self, input_shape):
        # Conv1D para extrair patches
        self.proj = Conv1D(
            self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding='valid',
            name='patch_proj'
        )

        self.norm = LayerNormalization(epsilon=1e-6, name='patch_norm')

        self.proj.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        # [B, T, C] -> [B, T//patch_size, embed_dim]
        x = self.proj(x)
        x = self.norm(x)
        return x

    def compute_num_patches(self, seq_len):
        return seq_len // self.patch_size


# ============================================================================
# UNPATCHIFY LAYER - FIXED
# ============================================================================

class UnpatchEmbedding1D(Layer):
    """
    Converte patches de volta para sequência original.

    CORREÇÃO: Inicialização adequada ao invés de zeros!
    """

    def __init__(self, patch_size, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.out_channels = out_channels

    def build(self, input_shape):
        embed_dim = input_shape[-1]

        # FIX: Inicialização adequada ao invés de zeros
        # Zeros faz o modelo produzir sempre saída zero no início!
        self.proj = Dense(
            self.patch_size * self.out_channels,
            kernel_initializer=TruncatedNormal(stddev=0.02),
            bias_initializer='zeros',
            name='unpatch_proj'
        )

        self.proj.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        batch = tf.shape(x)[0]
        num_patches = tf.shape(x)[1]

        # [B, num_patches, embed_dim] -> [B, num_patches, patch_size * out_channels]
        x = self.proj(x)

        # Reshape to [B, num_patches * patch_size, out_channels]
        x = tf.reshape(x, [batch, num_patches * self.patch_size, self.out_channels])

        return x


# ============================================================================
# POSITIONAL EMBEDDING (Learnable)
# ============================================================================

class LearnablePositionalEmbedding(Layer):
    """Positional embedding aprendido para patches."""

    def __init__(self, max_len, dim, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.dim = dim

    def build(self, input_shape):
        self.pos_embed = self.add_weight(
            name='pos_embed',
            shape=(1, self.max_len, self.dim),
            initializer=TruncatedNormal(stddev=0.02),
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        # Slice apenas o necessário
        pos = self.pos_embed[:, :seq_len, :]
        return x + pos


# ============================================================================
# TIME EMBEDDING
# ============================================================================

class SinusoidalTimeEmbedding(Layer):
    """Sinusoidal time embedding (DiT-style)."""

    def __init__(self, dim, max_period=10000, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.max_period = max_period

    def build(self, input_shape):
        half_dim = self.dim // 2
        freqs = np.exp(
            -math.log(self.max_period) *
            np.arange(0, half_dim, dtype=np.float32) / half_dim
        )
        self.freqs = self.add_weight(
            name='freqs',
            shape=(half_dim,),
            initializer=Constant(freqs),
            trainable=False,
            dtype=tf.float32
        )
        super().build(input_shape)

    def call(self, timesteps):
        timesteps = tf.cast(timesteps, tf.float32)
        timesteps = tf.expand_dims(timesteps, -1)
        args = timesteps * self.freqs[None, :]
        embedding = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
        return embedding


# ============================================================================
# CONTEXT GENERATOR
# ============================================================================

class ContextGenerator(Layer):
    """
    Gera contexto sequencial para cross-attention.
    Expande tokens aprendidos para o batch size.
    """

    def __init__(self, seq_len, dim, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.dim = dim

    def build(self, input_shape):
        # Tokens de contexto aprendidos
        self.context_tokens = self.add_weight(
            name='context_tokens',
            shape=(1, self.seq_len, self.dim),
            initializer=TruncatedNormal(stddev=0.02),
            trainable=True
        )

        # Projeção para adicionar informação de condicionamento
        self.cond_proj = Dense(self.dim, name='cond_proj')

        super().build(input_shape)

    def call(self, inputs):
        # inputs = conditioning embedding [B, dim]
        batch = tf.shape(inputs)[0]

        # Expande tokens para batch
        context = tf.tile(self.context_tokens, [batch, 1, 1])

        # Adiciona informação de condicionamento
        cond_expanded = tf.expand_dims(inputs, axis=1)
        cond_proj = self.cond_proj(cond_expanded)

        return context + cond_proj


# ============================================================================
# EFFICIENT DiT MODEL - FIXED
# ============================================================================

class DenoisingDiffusionUNetModelTensorflow(Activations):
    """
    Efficient Diffusion Transformer (DiT) com Linear Attention.

    Baseado na arquitetura DiT mas com otimizações:
    - Linear Attention O(N) ao invés de O(N²)
    - ECA modules para atenção em canais
    - Cross-attention para condicionamento rico
    - AdaLN-Zero para modulação adaptativa
    - GeGLU para FFN

    VERSÃO CORRIGIDA:
    - Unpatchify inicializado corretamente (não zeros)
    - Linear Attention com softplus (mais estável)
    - Normalização robusta
    - Skip connection direto
    - Condicionamento concatenado
    """

    def __init__(
            self,
            output_shape: int = 128,
            embedding_channels: int = 1,
            patch_size: int = 2,
            hidden_dim: int = 1024,
            depth: int = 20,
            num_heads: int = 16,
            dropout_rate: float = 0.1,
            ffn_expansion: float = 4.5,
            number_samples_per_class: Optional[Dict] = None,
            use_cross_attention: bool = True,
            context_seq_len: int = 32,
            use_mixed_precision: bool = False,
            last_layer_activation: str = 'linear',
            **kwargs
    ):
        if use_mixed_precision:
            set_global_policy('mixed_float16')
            print("✅ Mixed precision enabled (float16/float32)")

        self._output_shape = output_shape
        self._embedding_channels = embedding_channels
        self._patch_size = patch_size
        self._hidden_dim = hidden_dim
        self._depth = depth
        self._num_heads = num_heads
        self._dropout = dropout_rate
        self._ffn_expansion = ffn_expansion
        self._class_info = number_samples_per_class
        self._use_cross_attention = use_cross_attention
        self._context_seq_len = context_seq_len
        self._last_activation = last_layer_activation

        # Calcula número de patches
        self._num_patches = output_shape // patch_size

        print(f"\n{'=' * 70}")
        print(f"🚀 Efficient DiT (Diffusion Transformer) v6.0.1 - FIXED")
        print(f"{'=' * 70}")
        print(f"✨ Architecture Type: Pure Transformer (No U-Net)")
        print(f"\n📊 Configuration:")
        print(f"  • Input shape: {output_shape} → {self._num_patches} patches (size {patch_size})")
        print(f"  • Hidden dimension: {hidden_dim}")
        print(f"  • Transformer depth: {depth} blocks")
        print(f"  • Attention heads: {num_heads}")
        print(f"  • FFN expansion: {ffn_expansion}x")
        print(f"  • Cross-attention: {use_cross_attention}")
        print(f"  • Context length: {context_seq_len}")
        print(f"\n⚡ Optimizations:")
        print(f"  ✓ Linear Attention: O(N) complexity")
        print(f"  ✓ AdaLN-Zero: Adaptive conditioning")
        print(f"  ✓ ECA: Efficient channel attention")
        print(f"  ✓ GeGLU: Advanced activation")
        print(f"  ✓ Skip Connection: Direct path")
        print(f"\n🔧 FIXES Applied:")
        print(f"  ✓ Unpatchify: TruncatedNormal init (not zeros!)")
        print(f"  ✓ Linear Attention: Softplus feature map")
        print(f"  ✓ Normalization: Epsilon 1e-3 (more stable)")
        print(f"  ✓ DiTBlock: Proper fallback norm")
        print(f"  ✓ Conditioning: Concatenate + merge")
        print(f"  ✓ AdaLN-Zero: Small init (not exactly zero)")
        print(f"{'=' * 70}\n")

    def build_model(self):
        """Constrói o modelo DiT - VERSÃO CORRIGIDA."""

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

        # ===== PATCHIFY =====
        x = PatchEmbedding1D(
            patch_size=self._patch_size,
            embed_dim=self._hidden_dim,
            name='patch_embed'
        )(image_input)

        # FIX: Salva para skip connection
        x_init = x

        # ===== POSITIONAL EMBEDDING =====
        x = LearnablePositionalEmbedding(
            max_len=self._num_patches,
            dim=self._hidden_dim,
            name='pos_embed'
        )(x)

        # ===== TIME EMBEDDING =====
        time_emb = SinusoidalTimeEmbedding(
            dim=self._hidden_dim,
            name='time_embed'
        )(time_input)

        # MLP para processar time embedding
        time_emb = Dense(self._hidden_dim * 2, name='time_mlp1')(time_emb)
        time_emb = LayerNormalization(epsilon=1e-6)(time_emb)
        time_emb = Activation('gelu')(time_emb)

        time_emb = Dense(self._hidden_dim * 4, name='time_mlp2')(time_emb)
        time_emb = LayerNormalization(epsilon=1e-6)(time_emb)
        time_emb = Activation('gelu')(time_emb)

        time_emb = Dense(self._hidden_dim * 4, name='time_mlp3')(time_emb)

        # ===== LABEL EMBEDDING =====
        label_emb = Dense(self._hidden_dim, name='label_mlp1')(label_input)
        label_emb = LayerNormalization(epsilon=1e-6)(label_emb)
        label_emb = Activation('gelu')(label_emb)

        label_emb = Dense(self._hidden_dim * 2, name='label_mlp2')(label_emb)
        label_emb = LayerNormalization(epsilon=1e-6)(label_emb)
        label_emb = Activation('gelu')(label_emb)

        label_emb = Dense(self._hidden_dim * 4, name='label_mlp3')(label_emb)

        # ===== COMBINE CONDITIONING =====
        # FIX: Concatenar ao invés de somar (escalas diferentes)
        cond = Concatenate(name='cond_concat')([time_emb, label_emb])
        cond = Dense(self._hidden_dim * 4, name='cond_merge')(cond)
        cond = LayerNormalization(epsilon=1e-6)(cond)
        cond = Activation('gelu')(cond)
        cond = Dense(self._hidden_dim * 4, name='cond_final')(cond)

        # ===== CONTEXT FOR CROSS-ATTENTION =====
        if self._use_cross_attention:
            context = ContextGenerator(
                seq_len=self._context_seq_len,
                dim=self._hidden_dim,
                name='context_generator'
            )(cond)
        else:
            context = None

        # ===== TRANSFORMER BLOCKS =====
        for i in range(self._depth):
            if self._use_cross_attention and context is not None:
                x = DiTBlock(
                    dim=self._hidden_dim,
                    num_heads=self._num_heads,
                    dropout=self._dropout,
                    expansion=self._ffn_expansion,
                    use_cross_attn=True,
                    name=f'dit_block_{i}'
                )([x, cond, context])
            else:
                x = DiTBlock(
                    dim=self._hidden_dim,
                    num_heads=self._num_heads,
                    dropout=self._dropout,
                    expansion=self._ffn_expansion,
                    use_cross_attn=False,
                    name=f'dit_block_{i}'
                )([x, cond])

        # ===== FINAL LAYER NORM =====
        x = LayerNormalization(epsilon=1e-6, name='final_norm')(x)

        # FIX: SKIP CONNECTION - Caminho direto da entrada
        skip_proj = Dense(
            self._hidden_dim,
            kernel_initializer='zeros',
            name='skip_connection'
        )
        x = x + skip_proj(x_init)

        # ===== FINAL PROCESSING =====
        # FIX: Processamento com normalizações
        x = Dense(self._hidden_dim * 2, name='pre_unpatch_1')(x)
        x = LayerNormalization(epsilon=1e-6, name='norm_pre_1')(x)
        x = Activation('gelu')(x)

        x = Dense(self._hidden_dim, name='pre_unpatch_2')(x)
        x = LayerNormalization(epsilon=1e-6, name='norm_pre_2')(x)
        x = Activation('gelu')(x)

        # ===== UNPATCHIFY =====
        x = UnpatchEmbedding1D(
            patch_size=self._patch_size,
            out_channels=self._embedding_channels,
            name='unpatch'
        )(x)

        if self._last_activation and self._last_activation != 'linear':
            x = Activation(self._last_activation, name='final_activation')(x)

        # ===== BUILD MODEL =====
        model = Model(
            inputs=[image_input, time_input, label_input],
            outputs=x,
            name='EfficientDiT_Fixed'
        )

        # ===== COUNT PARAMETERS =====
        total_params = model.count_params()

        print(f"\n{'=' * 70}")
        print(f"✅ DiT Model Built Successfully (FIXED VERSION)!")
        print(f"{'=' * 70}")
        print(f"📊 Statistics:")
        print(f"  • Total parameters: {total_params:,} (~{total_params / 1e6:.1f}M)")
        print(f"  • Model size: ~{total_params * 4 / 1e6:.1f} MB (float32)")
        print(f"  • Model size: ~{total_params * 2 / 1e6:.1f} MB (float16)")
        print(f"  • Transformer blocks: {self._depth}")
        print(f"  • Hidden dimension: {self._hidden_dim}")
        print(f"  • Number of patches: {self._num_patches}")
        print(f"  • Parameters per block: ~{total_params / self._depth / 1e6:.1f}M")
        print(f"\n🔧 Applied Fixes:")
        print(f"  ✓ Unpatchify initialization: TruncatedNormal (not zeros!)")
        print(f"  ✓ Linear Attention: Softplus feature map")
        print(f"  ✓ Normalization: Stable with eps=1e-3")
        print(f"  ✓ Skip connection: Direct path from input")
        print(f"  ✓ Conditioning: Concatenate + merge")
        print(f"  ✓ AdaLN gates: Small init (1e-5)")
        print(f"  ✓ Extra LayerNorms: Before unpatch")
        print(f"  ✓ Bias in projections: Better convergence")
        print(f"\n🎯 Training Settings:")
        print(f"  • Data normalization: [-1, 1] or [0, 1]")
        print(f"  • Batch size: 2-8")
        print(f"  • Learning rate: 1e-4 (start)")
        print(f"  • LR schedule: Cosine with warmup")
        print(f"  • Optimizer: AdamW (weight_decay=0.01)")
        print(f"  • Gradient clipping: 1.0")
        print(f"  • EMA: decay=0.9999")
        print(f"\n⚠️  Critical:")
        print(f"  • Check data normalization FIRST")
        print(f"  • Monitor loss - should decrease from epoch 1")
        print(f"  • If NaN: reduce LR to 5e-5")
        print(f"  • Use mixed precision for speed")
        print(f"{'=' * 70}\n")

        return model

    # Properties
    @property
    def embedding_dimension(self):
        return self._output_shape

    @property
    def embedding_channels(self):
        return self._embedding_channels

    @property
    def hidden_dim(self):
        return self._hidden_dim

    @property
    def depth(self):
        return self._depth