"""
Hybrid Convolutional-Transformer Generator com Vector Quantization Hierárquica
Versão ULTRA-MODERNA + ULTRA-EFICIENTE: RVQ + GQA + RoPE + ECA + FiLM + GeGLU
"""

__author__ = 'Synthetic Ocean AI - Enhanced Team'
__version__ = '7.0.0-ultra-efficient'

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


class ResidualVectorQuantization(Layer):
    """
    Residual Vector Quantization (RVQ) - SUPERIOR AO VQ SIMPLES

    Usado em: SoundStream, EnCodec, Descript Audio Codec

    Vantagens sobre VQ tradicional:
    - Múltiplos níveis de quantização (refinamento progressivo)
    - Melhor reconstrução (cada nível captura resíduos)
    - Codebook usage mais eficiente
    - Qualidade superior com mesmo bitrate
    """

    def __init__(self, num_embeddings=256, embedding_dim=64, num_quantizers=3,
                 commitment_cost=0.25, **kwargs):
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_quantizers = num_quantizers
        self.commitment_cost = commitment_cost

    def build(self, input_shape):
        # Múltiplos codebooks (um por quantizer)
        self.codebooks = []
        for i in range(self.num_quantizers):
            codebook = self.add_weight(
                name=f'codebook_{i}',
                shape=(self.num_embeddings, self.embedding_dim),
                initializer='uniform',
                trainable=True
            )
            self.codebooks.append(codebook)
        super().build(input_shape)

    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]

        residual = inputs
        quantized_out = 0.0
        total_loss = 0.0

        # Quantização em cascata
        for i, codebook in enumerate(self.codebooks):
            # Distância L2 para cada embedding
            distances = (
                tf.reduce_sum(residual ** 2, axis=1, keepdims=True) +
                tf.reduce_sum(codebook ** 2, axis=1) -
                2 * tf.matmul(residual, codebook, transpose_b=True)
            )

            # Código mais próximo
            encoding_indices = tf.argmin(distances, axis=1)
            quantized = tf.nn.embedding_lookup(codebook, encoding_indices)

            if training:
                # VQ Loss para este nível
                e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - residual) ** 2)
                q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(residual)) ** 2)
                loss = q_latent_loss + self.commitment_cost * e_latent_loss
                total_loss += loss

            # Straight-through estimator
            quantized = residual + tf.stop_gradient(quantized - residual)

            # Acumula quantização
            quantized_out = quantized_out + quantized

            # Calcula resíduo para próximo nível
            residual = residual - tf.stop_gradient(quantized)

        if training:
            self.add_loss(total_loss / self.num_quantizers)

        return quantized_out

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            **super().get_config(),
            'num_embeddings': self.num_embeddings,
            'embedding_dim': self.embedding_dim,
            'num_quantizers': self.num_quantizers,
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
        config = super().get_config()
        config.update({'epsilon': self.epsilon})
        return config


class Mish(Layer):
    """
    Mish Activation - ULTRA MODERNA

    f(x) = x * tanh(softplus(x))

    Propriedades superiores:
    - Suave e contínua até segunda derivada
    - Não-monotônica (permite auto-regularização)
    - Superior a Swish/SiLU em muitas tarefas
    - Usado em YOLOv4, EfficientDet
    """

    def call(self, x):
        return x * tf.nn.tanh(tf.nn.softplus(x))

    def compute_output_shape(self, input_shape):
        return input_shape


class GeGLU(Layer):
    """
    GELU Gated Linear Unit - STATE-OF-THE-ART

    Usado em PaLM, PaLM-2, e outros LLMs de última geração.
    """

    def call(self, x):
        half = tf.shape(x)[-1] // 2
        gate = x[..., :half]
        value = x[..., half:]
        return tf.nn.gelu(gate, approximate=False) * value

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (input_shape[-1] // 2,)


class RoPE(Layer):
    """
    Rotary Position Embedding - SOTA para sequências

    Usado em: LLaMA, PaLM, GPT-NeoX, Mistral, etc.
    """

    def __init__(self, dim, max_seq_len=2048, base=10000, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

    def build(self, input_shape):
        # Precomputa frequências
        inv_freq = 1.0 / (self.base ** (tf.range(0, self.dim, 2, dtype=tf.float32) / self.dim))
        self.inv_freq = tf.Variable(
            initial_value=inv_freq,
            trainable=False,
            name='inv_freq'
        )
        super().build(input_shape)

    def call(self, x, seq_dim=1):
        # x shape: [batch, num_heads, seq_len, head_dim] quando seq_dim=2
        seq_len = tf.shape(x)[seq_dim]

        # Posições: [0, 1, 2, ..., seq_len-1]
        t = tf.cast(tf.range(seq_len), tf.float32)

        # Frequências: [seq_len, dim/2]
        freqs = tf.einsum('i,j->ij', t, self.inv_freq)

        # Concatena sin e cos: [seq_len, dim]
        emb = tf.concat([freqs, freqs], axis=-1)

        # Aplica rotação
        cos = tf.cos(emb)  # [seq_len, dim]
        sin = tf.sin(emb)  # [seq_len, dim]

        # Reshape baseado em seq_dim
        # Para x: [batch, num_heads, seq_len, head_dim], queremos [1, 1, seq_len, head_dim]
        shape = [1] * len(x.shape)
        shape[seq_dim] = seq_len
        shape[-1] = self.dim

        cos = tf.reshape(cos, shape)
        sin = tf.reshape(sin, shape)

        # Rotaciona x
        x_rot = self._rotate_half(x)
        return x * cos + x_rot * sin

    def _rotate_half(self, x):
        """Rotaciona metade das dimensões"""
        x1, x2 = tf.split(x, 2, axis=-1)
        return tf.concat([-x2, x1], axis=-1)

    def get_config(self):
        return {
            **super().get_config(),
            'dim': self.dim,
            'max_seq_len': self.max_seq_len,
            'base': self.base
        }


class EfficientChannelAttention(Layer):
    """
    Efficient Channel Attention (ECA) - SUPERIOR AO SE

    Proposto em "ECA-Net: Efficient Channel Attention for Deep CNNs"

    Vantagens sobre Squeeze-and-Excitation:
    - Apenas 3 parâmetros vs centenas no SE
    - Convolução 1D adaptativa vs FC layers
    - Complexidade O(k) vs O(C²/r)
    """

    def __init__(self, gamma=2, b=1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.b = b

    def build(self, input_shape):
        channels = int(input_shape[-1])

        # Calcula kernel size adaptativo
        t = int(abs((np.log2(channels) / self.gamma) + (self.b / self.gamma)))
        k = t if t % 2 else t + 1  # Garante kernel ímpar
        k = max(k, 3)  # Mínimo de 3

        self.kernel_size = k
        self.channels = channels

        # Global Average Pooling
        self.pool = GlobalAveragePooling1D()

        # Convolução 1D adaptativa
        self.conv = Conv1D(
            filters=1,
            kernel_size=self.kernel_size,
            padding='same',
            use_bias=False
        )

        super().build(input_shape)

    def call(self, x):
        # Squeeze: Global pooling -> (batch, channels)
        y = self.pool(x)

        # Reshape para conv1d: (batch, 1, channels)
        y = tf.expand_dims(y, axis=1)

        # Convolução adaptativa 1D
        y = self.conv(y)  # (batch, 1, 1)

        # Sigmoid activation
        y = tf.nn.sigmoid(y)

        # Broadcast e multiply
        return x * y

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            **super().get_config(),
            'gamma': self.gamma,
            'b': self.b
        }


class FiLMResidual(Layer):
    """
    Feature-wise Linear Modulation with Residual Connection

    Versão melhorada do FiLM tradicional com skip connection.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        features_shape, condition_shape = input_shape
        channels = int(features_shape[-1])

        # Layer normalization
        self.norm = LayerNormalization(epsilon=self.epsilon)

        # Projeção da condição para 2 * channels (scale, shift)
        self.condition_proj = Dense(
            channels * 2,
            kernel_initializer='glorot_uniform',
            name='film_condition_proj'
        )

        super().build(input_shape)

    def call(self, inputs):
        features, condition = inputs

        # Batch size matching
        bf = tf.shape(features)[0]
        bc = tf.shape(condition)[0]

        condition = tf.cond(
            tf.not_equal(bf, bc),
            lambda: tf.repeat(condition, repeats=bf // bc, axis=0),
            lambda: condition
        )

        # Normaliza features
        normalized = self.norm(features)

        # Projeta condição para scale e shift
        params = self.condition_proj(condition)  # (batch, 2 * channels)
        params = tf.expand_dims(params, axis=1)  # (batch, 1, 2 * channels)

        channels = tf.shape(features)[-1]

        # Split em 2 parâmetros
        scale = params[..., :channels]        # γ (gamma)
        shift = params[..., channels:]        # β (beta)

        # FiLM com residual: x + γ * norm(x) + β
        modulated = scale * normalized + shift
        output = features + modulated  # Residual connection

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        return {
            **super().get_config(),
            'epsilon': self.epsilon
        }


class DepthwiseSeparableConv(Layer):
    """Depthwise Separable Convolution com Mish"""

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
        self.activation = Mish()
        super().build(input_shape)

    def call(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.activation(x)
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


class GroupedQueryAttention(Layer):
    """
    Grouped Query Attention (GQA) - ULTRA EFICIENTE

    Usado em: LLaMA 2, Mistral, Code LLaMA

    Vantagens sobre MHA:
    - 40-50% menos parâmetros em K/V
    - 2-3x mais rápido na inferência
    - Mantém qualidade similar ao MHA
    """

    def __init__(self, num_heads=8, num_kv_heads=2, head_dim=32, use_rope=True, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.d_model = num_heads * head_dim
        self.use_rope = use_rope

        # Quantos Q heads por KV head
        self.num_queries_per_kv = num_heads // num_kv_heads

        import math
        self.scale_value = 1.0 / math.sqrt(float(head_dim))

    def build(self, input_shape):
        d_input = int(input_shape[-1])

        # Q tem todos os heads
        self.wq = Dense(self.d_model, use_bias=False, name='wq')

        # K e V têm menos heads (economia de memória!)
        self.wk = Dense(self.num_kv_heads * self.head_dim, use_bias=False, name='wk')
        self.wv = Dense(self.num_kv_heads * self.head_dim, use_bias=False, name='wv')

        self.wo = Dense(d_input, use_bias=False, name='wo')

        # RoPE para encoding posicional
        if self.use_rope:
            self.rope = RoPE(dim=self.head_dim)

        # QK Normalization (estabiliza atenção)
        self.q_norm = RMSNorm()
        self.k_norm = RMSNorm()

        super().build(input_shape)

    def call(self, x):
        batch = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # Projeta Q, K, V
        q = self.wq(x)  # [B, L, num_heads * head_dim]
        k = self.wk(x)  # [B, L, num_kv_heads * head_dim]
        v = self.wv(x)  # [B, L, num_kv_heads * head_dim]

        # Reshape Q
        q = tf.reshape(q, [batch, seq_len, self.num_heads, self.head_dim])
        q = tf.transpose(q, [0, 2, 1, 3])  # [B, num_heads, L, head_dim]

        # Reshape K, V
        k = tf.reshape(k, [batch, seq_len, self.num_kv_heads, self.head_dim])
        v = tf.reshape(v, [batch, seq_len, self.num_kv_heads, self.head_dim])

        k = tf.transpose(k, [0, 2, 1, 3])  # [B, num_kv_heads, L, head_dim]
        v = tf.transpose(v, [0, 2, 1, 3])  # [B, num_kv_heads, L, head_dim]

        # Aplica RoPE
        if self.use_rope:
            q = self.rope(q, seq_dim=2)
            k = self.rope(k, seq_dim=2)

        # QK Normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Expande K e V para todos os Q heads (repeat)
        k = tf.repeat(k, repeats=self.num_queries_per_kv, axis=1)
        v = tf.repeat(v, repeats=self.num_queries_per_kv, axis=1)

        # Scaled dot-product attention
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
            'num_kv_heads': self.num_kv_heads,
            'head_dim': self.head_dim,
            'use_rope': self.use_rope
        }


class ConvTransformerBlock(Layer):
    """
    Hybrid Conv-Transformer Block - ULTRA MODERN + ULTRA EFICIENTE

    Arquitetura state-of-the-art otimizada:
    - Convolution + ECA: local patterns com 90% menos params que SE
    - GQA: global dependencies com 40% menos params
    - RoPE: encoding posicional sem params extras
    - GeGLU FFN: usado em PaLM-2
    - Mish activation: superior a SiLU/Swish
    """

    def __init__(self, filters, num_heads=8, num_kv_heads=2, head_dim=32,
                 ff_ratio=2.0, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.ff_ratio = ff_ratio
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.conv = DepthwiseSeparableConv(self.filters, kernel_size=3)
        self.eca = EfficientChannelAttention()
        self.attn = GroupedQueryAttention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            use_rope=True
        )

        # GeGLU FFN: precisa de 2x dimensão para split
        ff_dim = int(self.filters * self.ff_ratio)
        self.ff1 = Dense(ff_dim * 2)
        self.geglu = GeGLU()
        self.ff2 = Dense(self.filters)

        self.norm1 = RMSNorm()
        self.norm2 = RMSNorm()
        self.norm3 = RMSNorm()

        self.drop1 = Dropout(self.dropout_rate)
        self.drop2 = Dropout(self.dropout_rate)
        self.drop3 = Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, x, training=None):
        # Convolutional path com ECA
        conv_out = self.conv(x)
        conv_out = self.eca(conv_out)
        conv_out = self.drop1(conv_out, training=training)
        x = self.norm1(x + conv_out)

        # Attention path
        attn_out = self.attn(x)
        attn_out = self.drop2(attn_out, training=training)
        x = self.norm2(x + attn_out)

        # GeGLU FFN path
        ff_out = self.ff1(x)
        ff_out = self.geglu(ff_out)
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
            'num_kv_heads': self.num_kv_heads,
            'head_dim': self.head_dim,
            'ff_ratio': self.ff_ratio,
            'dropout_rate': self.dropout_rate
        }


class MultiScaleFusion(Layer):
    """Multi-Scale Feature Fusion"""

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

    def get_config(self):
        return {
            **super().get_config(),
            'output_dim': self.output_dim
        }


class VanillaGenerator(Activations):
    """
    Hybrid Conv-Transformer Generator - ARQUITETURA ULTRA-MODERNA + EFICIENTE

    Features state-of-the-art:
    - RVQ Hierárquica (Residual VQ - usado em EnCodec/SoundStream)
    - FiLM Residual (mais eficiente que AdaLN-Zero)
    - GeGLU FFN (usado em PaLM-2)
    - GQA (Grouped Query Attention - 40% menos params)
    - RoPE (Rotary Position Embedding)
    - ECA attention (superior a SE, 10x menos params)
    - Mish activation (SOTA)

    Economia de memória:
    - GQA: 40-50% menos parâmetros em K/V
    - ECA: 90% menos parâmetros que SE
    - RoPE: sem embeddings aprendidos extras
    - RVQ: melhor qualidade com mesmo bitrate
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
            num_stages: int = 1,
            base_filters: int = 64,
            tokens_per_stage: List[int] = None,
            num_heads: int = 8,
            num_kv_heads: int = 2,
            head_dim: int = 32,
            ff_ratio: float = 2.0,
            dropout_rate: float = 0.1,
            use_tanh_output: bool = False,
            use_mixed_precision: bool = False,
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
        self._tokens_per_stage = [2, 4, 8]
        if len(self._tokens_per_stage) != self._num_stages:
            self._tokens_per_stage = [16 * (2 ** i) for i in range(self._num_stages)]

        self._num_heads = self._safe_int(num_heads, 8)
        self._num_kv_heads = self._safe_int(num_kv_heads, 2)
        self._head_dim = self._safe_int(head_dim, 32)
        self._ff_ratio = self._safe_float(ff_ratio, 2.0)
        self._dropout_internal = self._safe_float(dropout_rate, 0.1)
        self._use_tanh = bool(use_tanh_output) if not isinstance(use_tanh_output, dict) else False
        self._mixed_precision = bool(use_mixed_precision) if not isinstance(use_mixed_precision, dict) else False

        self._use_vq = bool(use_vq) if not isinstance(use_vq, dict) else True
        self._vq_num_embeddings = self._safe_int(vq_num_embeddings, 256)
        self._vq_embedding_dim = self._safe_int(vq_embedding_dim, 64)
        self._vq_commitment_cost = self._safe_float(vq_commitment_cost, 0.25)

        self._model = None

    def get_generator(self) -> Model:

        if not self._class_info:
            raise ValueError("number_samples_per_class required")

        if self._mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        num_classes = self._class_info['number_classes']

        z = Input(shape=(self._latent_dim,), dtype=tf.float32, name='z')
        y = Input(shape=(num_classes,), dtype=tf.float32, name='y')

        condition = y

        x = Concatenate(name='concat_initial')([z, condition])
        x = Dense(self._base_filters, name='init_channel_proj')(x)
        x = RMSNorm(name='init_norm')(x)
        x = Mish(name='init_mish')(x)

        # Projeta para tokens iniciais
        tokens_init = self._tokens_per_stage[0]
        x = Dense(tokens_init * self._base_filters, name='init_token_proj')(x)
        x = Reshape((tokens_init, self._base_filters), name='init_reshape')(x)

        stage_outs = []
        vq_code_prev = None

        for i in range(self._num_stages):

            n_filters = self._base_filters * (2 ** i)

            if self._use_vq and vq_code_prev is not None:
                condition = Concatenate(name=f's{i}_condition')([y, vq_code_prev])
            else:
                condition = y

            if i > 0:
                # Upsample tokens: dobra o número de tokens
                x = Lambda(lambda t: tf.repeat(t, repeats=2, axis=1),
                          name=f's{i}_token_repeat')(x)
                x = Conv1D(filters=n_filters, kernel_size=3, padding='same',
                          name=f's{i}_channel_proj')(x)
                x = Mish(name=f's{i}_mish')(x)

            x = ConvTransformerBlock(
                filters=n_filters,
                num_heads=self._num_heads,
                num_kv_heads=self._num_kv_heads,
                head_dim=self._head_dim,
                ff_ratio=self._ff_ratio,
                dropout=self._dropout_internal,
                name=f's{i}_block'
            )(x)

            # FiLM Residual modulation
            x = FiLMResidual(name=f's{i}_film')([x, condition])

            stage_features = GlobalAveragePooling1D(name=f's{i}_pool')(x)
            stage_outs.append(stage_features)

            if self._use_vq:
                vq_proj = Dense(self._vq_embedding_dim, name=f's{i}_vq_proj')(stage_features)
                vq_proj = Mish(name=f's{i}_vq_mish')(vq_proj)

                vq_code_prev = ResidualVectorQuantization(
                    num_embeddings=self._vq_num_embeddings,
                    embedding_dim=self._vq_embedding_dim,
                    num_quantizers=3,  # RVQ com 3 níveis
                    commitment_cost=self._vq_commitment_cost,
                    name=f's{i}_rvq'
                )(vq_proj)

        x = MultiScaleFusion(output_dim=self._base_filters * 2, name='fusion')(stage_outs)

        if self._use_vq and vq_code_prev is not None:
            x = Concatenate(name='fusion_with_final_vq')([x, vq_code_prev])

        x = Dense(self._base_filters * 2, name='pre_out')(x)
        x = RMSNorm(name='pre_out_norm')(x)
        x = Mish(name='pre_out_mish')(x)

        x = Dense(self._output_shape, name='output')(x)

        if self._use_tanh:
            x = Activation('tanh', dtype=tf.float32, name='tanh')(x)
        elif self._last_activation:
            x = self._add_activation_layer(x, self._last_activation)

        model = Model(inputs=[z, y], outputs=x, name='HybridGenerator_GQA_RVQ_ECA_FiLM')
        self._model = model

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
            'num_kv_heads': self._num_kv_heads,
            'head_dim': self._head_dim,
            'ff_ratio': self._ff_ratio,
            'dropout_rate': self._dropout_internal,
            'use_tanh_output': self._use_tanh,
            'use_vq': self._use_vq,
            'vq_num_embeddings': self._vq_num_embeddings,
            'vq_embedding_dim': self._vq_embedding_dim,
            'vq_commitment_cost': self._vq_commitment_cost
        }