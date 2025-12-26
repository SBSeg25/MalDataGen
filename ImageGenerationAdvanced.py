import os
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt


# =====================================================================
# CONFIGURAÇÃO
# =====================================================================
def setup_environment():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ {len(gpus)} GPU(s) disponível(is)")
    else:
        print("⚠ Nenhuma GPU encontrada")

    mixed_precision.set_global_policy('mixed_float16')
    print("✓ Mixed precision (FP16) habilitado")


# =====================================================================
# COMPONENTES TRANSFORMER
# =====================================================================
class RMSNorm(layers.Layer):
    def __init__(self, dim, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.eps = eps

    def build(self, input_shape):
        self.scale = self.add_weight(
            shape=(self.dim,), initializer="ones", name="scale"
        )

    def call(self, x):
        rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.eps)
        return x / rms * self.scale


class RotaryEmbedding(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        inv_freq = 1.0 / (10000 ** (tf.range(0, dim // 2, dtype=tf.float32) / (dim // 2)))
        self.inv_freq = inv_freq

    def call(self, seq_len):
        t = tf.range(seq_len, dtype=tf.float32)
        freqs = tf.einsum("i,j->ij", t, self.inv_freq)
        return tf.cos(freqs), tf.sin(freqs)

    @staticmethod
    def apply_rotary_emb(q, cos, sin):
        cos = tf.cast(cos, q.dtype)
        sin = tf.cast(sin, q.dtype)
        cos, sin = cos[None, None, :, :], sin[None, None, :, :]
        q1, q2 = tf.split(q, 2, axis=-1)
        return tf.concat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1)


class TransformerAttention(layers.Layer):
    def __init__(self, dim, num_heads, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.rope = RotaryEmbedding(self.head_dim)
        self.qkv = layers.Dense(dim * 3, use_bias=False)
        self.proj = layers.Dense(dim)
        self.dropout = layers.Dropout(dropout)

    def call(self, x, training=False):
        B = tf.shape(x)[0]
        N = tf.shape(x)[1]

        qkv = tf.reshape(self.qkv(x), [B, N, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        cos, sin = self.rope(N)
        q = RotaryEmbedding.apply_rotary_emb(q, cos, sin)
        k = RotaryEmbedding.apply_rotary_emb(k, cos, sin)

        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn, training=training)

        out = tf.matmul(attn, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [B, N, self.dim])

        return self.proj(out)


class SwiGLU(layers.Layer):
    def __init__(self, dim, mlp_ratio=4.0, **kwargs):
        super().__init__(**kwargs)
        hidden = int(dim * mlp_ratio)
        self.fc1 = layers.Dense(hidden * 2)
        self.fc2 = layers.Dense(dim)

    def call(self, x):
        x_proj = self.fc1(x)
        x, gate = tf.split(x_proj, 2, axis=-1)
        return self.fc2(x * tf.nn.silu(gate))


class TransformerBlock(layers.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = RMSNorm(dim)
        self.attn = TransformerAttention(dim, num_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_ratio)

    def call(self, x, training=False):
        x = x + self.attn(self.norm1(x), training=training)
        x = x + self.mlp(self.norm2(x))
        return x


class LocalCrossAttention(layers.Layer):
    """Cross-attention local entre tokens vizinhos."""

    def __init__(self, dim, num_heads=4, window_size=8, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = layers.Dense(dim * 3, use_bias=False)
        self.proj = layers.Dense(dim)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False):
        B, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        # Normalização pré-atenção
        x_norm = self.norm(x)

        # QKV
        qkv = self.qkv(x_norm)
        qkv = tf.reshape(qkv, [B, N, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention local (windowed)
        attn = tf.matmul(q, k, transpose_b=True) * self.scale

        # Criar máscara de janela local
        positions = tf.range(N, dtype=tf.float32)
        pos_diff = tf.abs(positions[:, None] - positions[None, :])
        mask = tf.cast(pos_diff <= self.window_size, tf.float32)
        mask = mask[None, None, :, :]  # [1, 1, N, N]

        # Aplicar máscara
        attn = tf.where(mask > 0, attn, -1e9)
        attn = tf.nn.softmax(attn, axis=-1)

        # Computar saída
        out = tf.matmul(attn, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [B, N, C])

        return x + self.proj(out)  # Residual


class HierarchicalCrossAttention(layers.Layer):
    """Cross-attention entre níveis hierárquicos."""

    def __init__(self, dim, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = layers.Dense(dim, use_bias=False)
        self.kv_proj = layers.Dense(dim * 2, use_bias=False)
        self.proj = layers.Dense(dim)
        self.norm_q = layers.LayerNormalization(epsilon=1e-6)
        self.norm_kv = layers.LayerNormalization(epsilon=1e-6)

        self.kv_dim_proj = None

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, query, key_value, training=False):
        B = tf.shape(query)[0]
        N_q = tf.shape(query)[1]
        N_kv = tf.shape(key_value)[1]

        if key_value.shape[-1] != self.dim:
            if self.kv_dim_proj is None:
                self.kv_dim_proj = layers.Dense(self.dim)
            key_value = self.kv_dim_proj(key_value)

        q_norm = self.norm_q(query)
        kv_norm = self.norm_kv(key_value)

        q = self.q_proj(q_norm)
        kv = self.kv_proj(kv_norm)

        q = tf.reshape(q, [B, N_q, self.num_heads, self.head_dim])
        q = tf.transpose(q, [0, 2, 1, 3])

        kv = tf.reshape(kv, [B, N_kv, 2, self.num_heads, self.head_dim])
        kv = tf.transpose(kv, [2, 0, 3, 1, 4])
        k, v = kv[0], kv[1]

        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)

        out = tf.matmul(attn, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [B, N_q, self.dim])

        return query + self.proj(out)


class HierarchicalVAEEncoder(layers.Layer):
    """Encoder hierárquico com cross-attention entre níveis e local."""

    def __init__(self, input_dim=12288, latent_dim=128, hidden_dims=[512, 384, 256],
                 num_heads=8, depth_per_level=2, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.num_levels = len(hidden_dims)
        self.num_heads = num_heads
        self.depth_per_level = depth_per_level

        self.num_tokens, self.token_dim = self._compute_token_config(input_dim)

        self.input_proj = layers.Dense(hidden_dims[0])
        self.input_norm = layers.LayerNormalization(epsilon=1e-6)

        self.pos_embed_init = self.add_weight(
            shape=(1, self.num_tokens, hidden_dims[0]),
            initializer='zeros',
            trainable=True,
            name='pos_embed_init'
        )

        self.level_blocks = []
        self.level_norms = []
        self.level_projs = []
        self.pos_embeds = []

        self.hierarchical_cross_attns = []
        self.local_cross_attns = []

        current_seq_len = self.num_tokens
        for i, dim in enumerate(hidden_dims):
            blocks = [
                TransformerBlock(dim, num_heads, dropout=0.1)
                for _ in range(depth_per_level)
            ]
            self.level_blocks.append(blocks)

            self.local_cross_attns.append(
                LocalCrossAttention(dim, num_heads=4, window_size=8)
            )

            if i > 0:
                self.hierarchical_cross_attns.append(
                    HierarchicalCrossAttention(dim, num_heads=num_heads)
                )
            else:
                self.hierarchical_cross_attns.append(None)

            self.level_norms.append(layers.LayerNormalization(epsilon=1e-6))

            if i < len(hidden_dims) - 1:
                current_seq_len = current_seq_len // 2
                self.level_projs.append(layers.Dense(hidden_dims[i + 1]))

                pos_embed = self.add_weight(
                    shape=(1, current_seq_len, hidden_dims[i + 1]),
                    initializer='zeros',
                    trainable=True,
                    name=f'pos_embed_level_{i + 1}'
                )
                self.pos_embeds.append(pos_embed)
            else:
                self.level_projs.append(None)
                self.pos_embeds.append(None)

        self.attn_pool = layers.Dense(1)
        self.final_norm = layers.LayerNormalization(epsilon=1e-6)
        self.fc_mu = layers.Dense(latent_dim, dtype='float32')
        self.fc_logvar = layers.Dense(latent_dim, dtype='float32')

    def _compute_token_config(self, input_dim):
        sqrt_dim = int(np.sqrt(input_dim))

        for power in range(4, 12):
            num_tokens = 2 ** power
            if input_dim % num_tokens == 0:
                token_dim = input_dim // num_tokens
                if 32 <= token_dim <= 128:
                    return num_tokens, token_dim

        best_num_tokens = sqrt_dim
        for divisor in range(sqrt_dim, 16, -1):
            if input_dim % divisor == 0:
                best_num_tokens = divisor
                break

        return best_num_tokens, input_dim // best_num_tokens

    def _attention_pool(self, x):
        attn_weights = self.attn_pool(x)
        attn_weights = tf.nn.softmax(attn_weights, axis=1)
        pooled = tf.reduce_sum(x * attn_weights, axis=1)
        return pooled

    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        x = tf.cast(x, tf.float16)

        x = tf.reshape(x, [batch_size, self.num_tokens, self.token_dim])
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = x + tf.cast(self.pos_embed_init, x.dtype)

        level_features = []

        for i, (blocks, local_cross, hier_cross, norm, proj, pos_embed) in enumerate(
                zip(self.level_blocks, self.local_cross_attns, self.hierarchical_cross_attns,
                    self.level_norms, self.level_projs, self.pos_embeds)
        ):
            residual = x
            for block in blocks:
                x = block(x, training=training)

            if x.shape == residual.shape:
                x = x + residual

            x = local_cross(x, training=training)

            if hier_cross is not None and len(level_features) > 0:
                x = hier_cross(x, level_features[-1], training=training)

            x = norm(x)

            level_features.append(x)

            if i < len(self.level_blocks) - 1:
                seq_len = tf.shape(x)[1]
                new_seq_len = seq_len // 2

                x_reshaped = tf.reshape(x, [batch_size, new_seq_len, 2, self.hidden_dims[i]])
                x_avg = tf.reduce_mean(x_reshaped, axis=2)
                x_max = tf.reduce_max(x_reshaped, axis=2)
                x = (x_avg + x_max) / 2.0

                if proj is not None:
                    x = proj(x)
                    if pos_embed is not None:
                        x = x + tf.cast(pos_embed, x.dtype)

        x = self.final_norm(x)
        x = self._attention_pool(x)
        x = tf.cast(x, tf.float32)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class HierarchicalVAEDecoder(layers.Layer):
    """Decoder hierárquico SEM skip connections com encoder."""

    def __init__(self, output_dim=12288, hidden_dims=[256, 384, 512],
                 num_heads=8, depth_per_level=2, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.num_levels = len(hidden_dims)
        self.num_heads = num_heads
        self.depth_per_level = depth_per_level

        self.final_tokens, self.token_dim = self._compute_token_config(output_dim)
        self.initial_tokens = self.final_tokens // (2 ** (len(hidden_dims) - 1))

        self.fc_init = layers.Dense(hidden_dims[0] * self.initial_tokens)
        self.init_norm = layers.LayerNormalization(epsilon=1e-6)

        self.pos_embed_init = self.add_weight(
            shape=(1, self.initial_tokens, hidden_dims[0]),
            initializer='zeros',
            trainable=True,
            name='pos_embed_init'
        )

        self.level_blocks = []
        self.level_norms = []
        self.level_projs = []
        self.upsample_convs = []
        self.pos_embeds = []

        # Apenas cross-attention hierárquica e local (SEM skip connections)
        self.hierarchical_cross_attns = []
        self.local_cross_attns = []

        current_seq_len = self.initial_tokens
        for i, dim in enumerate(hidden_dims):
            blocks = [
                TransformerBlock(dim, num_heads, dropout=0.1)
                for _ in range(depth_per_level)
            ]
            self.level_blocks.append(blocks)

            self.local_cross_attns.append(
                LocalCrossAttention(dim, num_heads=4, window_size=8)
            )

            if i > 0:
                self.hierarchical_cross_attns.append(
                    HierarchicalCrossAttention(dim, num_heads=num_heads)
                )
            else:
                self.hierarchical_cross_attns.append(None)

            self.level_norms.append(layers.LayerNormalization(epsilon=1e-6))

            if i < len(hidden_dims) - 1:
                current_seq_len = current_seq_len * 2

                self.upsample_convs.append(
                    layers.Conv1D(hidden_dims[i + 1], kernel_size=3, padding='same')
                )
                self.level_projs.append(layers.Dense(hidden_dims[i + 1]))

                pos_embed = self.add_weight(
                    shape=(1, current_seq_len, hidden_dims[i + 1]),
                    initializer='zeros',
                    trainable=True,
                    name=f'pos_embed_level_{i + 1}'
                )
                self.pos_embeds.append(pos_embed)
            else:
                self.upsample_convs.append(None)
                self.level_projs.append(None)
                self.pos_embeds.append(None)

        self.final_norm = layers.LayerNormalization(epsilon=1e-6)
        self.final_refine = layers.Conv1D(self.hidden_dims[-1], kernel_size=5, padding='same')
        self.upsample_final = layers.Dense(self.token_dim)

    def _compute_token_config(self, output_dim):
        sqrt_dim = int(np.sqrt(output_dim))

        for power in range(4, 12):
            num_tokens = 2 ** power
            if output_dim % num_tokens == 0:
                token_dim = output_dim // num_tokens
                if 32 <= token_dim <= 128:
                    return num_tokens, token_dim

        best_num_tokens = sqrt_dim
        for divisor in range(sqrt_dim, 16, -1):
            if output_dim % divisor == 0:
                best_num_tokens = divisor
                break

        return best_num_tokens, output_dim // best_num_tokens

    def _smooth_upsample(self, x, conv_layer):
        x = tf.repeat(x, 2, axis=1)
        if conv_layer is not None:
            x = conv_layer(x)
        return x

    def call(self, z, training=False):
        """
        Decoder INDEPENDENTE - não recebe encoder features.
        z: latent vector [B, latent_dim]
        """
        batch_size = tf.shape(z)[0]
        z = tf.cast(z, tf.float16)

        x = self.fc_init(z)
        x = tf.reshape(x, [batch_size, self.initial_tokens, self.hidden_dims[0]])
        x = self.init_norm(x)
        x = x + tf.cast(self.pos_embed_init, x.dtype)

        decoder_features = []

        for i, (blocks, local_cross, hier_cross, norm, proj, conv, pos_embed) in enumerate(
                zip(self.level_blocks, self.local_cross_attns, self.hierarchical_cross_attns,
                    self.level_norms, self.level_projs, self.upsample_convs, self.pos_embeds)
        ):
            residual = x
            for block in blocks:
                x = block(x, training=training)

            if x.shape == residual.shape:
                x = x + residual

            x = local_cross(x, training=training)

            # Apenas cross-attention hierárquica dentro do decoder
            if hier_cross is not None and len(decoder_features) > 0:
                x = hier_cross(x, decoder_features[-1], training=training)

            x = norm(x)

            decoder_features.append(x)

            if i < len(self.level_blocks) - 1:
                x = self._smooth_upsample(x, conv)

                if proj is not None:
                    x = proj(x)
                    if pos_embed is not None:
                        x = x + tf.cast(pos_embed, x.dtype)

        current_seq_len = tf.shape(x)[1]
        if current_seq_len < self.final_tokens:
            num_repeats = self.final_tokens // current_seq_len
            x = tf.repeat(x, num_repeats, axis=1)

        x = self.final_norm(x)
        x = self.final_refine(x)
        x = self.upsample_final(x)

        x = tf.reshape(x, [batch_size, self.final_tokens * self.token_dim])
        x = tf.cast(x, tf.float32)

        return x


class HierarchicalVAE(keras.Model):
    """VAE Transformer Hierárquico SEM skip connections encoder-decoder."""

    def __init__(self, input_dim, latent_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = HierarchicalVAEEncoder(latent_dim=latent_dim)
        self.decoder = HierarchicalVAEDecoder(output_dim=input_dim)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def reparameterize(self, mu, logvar):
        eps = tf.random.normal(shape=tf.shape(mu), dtype=tf.float32)
        z = mu + tf.exp(0.5 * logvar) * eps
        return z

    def encode(self, x):
        """Encode input para espaço latente."""
        x = tf.cast(x, tf.float32)
        mu, logvar = self.encoder(x, training=False)
        return self.reparameterize(mu, logvar)

    def decode(self, z):
        """Decode do espaço latente - SEM encoder features."""
        z = tf.cast(z, tf.float32)
        return self.decoder(z, training=False)

    def train_step(self, data):
        x = data
        x = tf.cast(x, tf.float32)
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, self.input_dim])

        with tf.GradientTape() as tape:
            # Encode (retorna apenas mu e logvar)
            mu, logvar = self.encoder(x, training=True)
            z = self.reparameterize(mu, logvar)

            # Decode (independente, sem encoder features)
            x_recon = self.decoder(z, training=True)
            x_recon = tf.reshape(x_recon, [batch_size, self.input_dim])

            # Losses
            recon_loss = tf.reduce_mean(tf.square(x - x_recon))
            kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
            total_loss = recon_loss + 0.0001 * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


# =====================================================================
# GAUSSIAN DIFFUSION (DDPM)
# =====================================================================
class GaussianDiffusion:
    """Gaussian diffusion para vetores unidimensionais."""

    def __init__(self, beta_start=1e-4, beta_end=0.02, timesteps=1000,
                 clip_min=-1.0, clip_max=1.0):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.clip_min = clip_min
        self.clip_max = clip_max

        betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)
        self.num_timesteps = int(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.betas = tf.constant(betas, dtype=tf.float32)
        self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf.float32)
        self.alphas_cumprod_prev = tf.constant(alphas_cumprod_prev, dtype=tf.float32)

        self.sqrt_alphas_cumprod = tf.constant(
            np.sqrt(alphas_cumprod), dtype=tf.float32
        )
        self.sqrt_one_minus_alphas_cumprod = tf.constant(
            np.sqrt(1.0 - alphas_cumprod), dtype=tf.float32
        )
        self.log_one_minus_alphas_cumprod = tf.constant(
            np.log(1.0 - alphas_cumprod), dtype=tf.float32
        )
        self.sqrt_recip_alphas_cumprod = tf.constant(
            np.sqrt(1.0 / alphas_cumprod), dtype=tf.float32
        )
        self.sqrt_recipm1_alphas_cumprod = tf.constant(
            np.sqrt(1.0 / alphas_cumprod - 1), dtype=tf.float32
        )

        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = tf.constant(posterior_variance, dtype=tf.float32)
        self.posterior_log_variance_clipped = tf.constant(
            np.log(np.maximum(posterior_variance, 1e-20)), dtype=tf.float32
        )
        self.posterior_mean_coef1 = tf.constant(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            dtype=tf.float32,
        )
        self.posterior_mean_coef2 = tf.constant(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
            dtype=tf.float32,
        )

    def _extract(self, a, t, x_shape):
        batch_size = x_shape[0]
        out = tf.gather(a, t)
        return tf.reshape(out, [batch_size, 1])

    def q_sample(self, x_start, t, noise):
        x_start_shape = tf.shape(x_start)
        return (
                self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
                + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        x_t_shape = tf.shape(x_t)
        return (
                self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape) * x_t
                - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t_shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        x_t_shape = tf.shape(x_t)
        posterior_mean = (
                self._extract(self.posterior_mean_coef1, t, x_t_shape) * x_start
                + self._extract(self.posterior_mean_coef2, t, x_t_shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t_shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t_shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, pred_noise, x, t, clip_denoised=True):
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        if clip_denoised:
            x_recon = tf.clip_by_value(x_recon, self.clip_min, self.clip_max)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, pred_noise, x, t, clip_denoised=True):
        model_mean, _, model_log_variance = self.p_mean_variance(
            pred_noise, x=x, t=t, clip_denoised=clip_denoised
        )
        noise = tf.random.normal(shape=tf.shape(x), dtype=x.dtype)
        nonzero_mask = tf.reshape(
            1 - tf.cast(tf.equal(t, 0), tf.float32), [tf.shape(x)[0], 1]
        )
        return model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise


# =====================================================================
# DiT COMPONENTS (para difusão no espaço latente)
# =====================================================================
class AdaLNZero(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.linear1 = layers.Dense(dim, activation="silu")
        self.linear2 = layers.Dense(6 * dim, kernel_initializer="zeros")

    def call(self, cond):
        return self.linear2(self.linear1(cond))


class DiTBlock(layers.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = RMSNorm(dim)
        self.attn = TransformerAttention(dim, num_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_ratio)
        self.ada = AdaLNZero(dim)

    def call(self, x, cond):
        cond = tf.cast(cond, tf.float32)
        mods = self.ada(cond)
        mods = tf.cast(mods, x.dtype)

        s1, sc1, g1, s2, sc2, g2 = tf.split(mods, 6, axis=-1)

        h = self.norm1(x)
        h = h * (1 + sc1[:, None, :]) + s1[:, None, :]
        x = x + g1[:, None, :] * self.attn(h)

        h = self.norm2(x)
        h = h * (1 + sc2[:, None, :]) + s2[:, None, :]
        return x + g2[:, None, :] * self.mlp(h)


class TimestepEmbedding(layers.Layer):
    def __init__(self, dim, max_period=10000, **kwargs):
        super().__init__(**kwargs)
        half_dim = dim // 2
        freqs = tf.exp(
            -math.log(max_period) * tf.range(0, half_dim, dtype=tf.float32) / half_dim
        )
        self.freqs = tf.constant(freqs, dtype=tf.float32)

    def call(self, timesteps):
        timesteps = tf.cast(timesteps, tf.float32)
        args = timesteps[:, None] * self.freqs[None, :]
        embedding = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
        return embedding


class LatentDiTNetwork(keras.Model):
    """DiT para difusão no espaço latente (128D)."""

    def __init__(self, latent_dim, num_classes, hidden_dim=384, depth=6, num_heads=6, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.latent_proj = layers.Dense(hidden_dim)
        self.class_token = self.add_weight(
            shape=(1, 1, hidden_dim), initializer="zeros", trainable=True, name="class_token"
        )

        self.t_emb = TimestepEmbedding(hidden_dim)
        self.t_mlp = layers.Dense(hidden_dim, activation="silu")
        self.label_emb = layers.Embedding(num_classes + 1, hidden_dim)

        self.blocks = [DiTBlock(hidden_dim, num_heads) for _ in range(depth)]

        self.out_norm = RMSNorm(hidden_dim)
        self.out_ada_linear1 = layers.Dense(hidden_dim, activation="silu")
        self.out_ada_linear2 = layers.Dense(2 * hidden_dim, kernel_initializer="zeros")
        self.out_proj = layers.Dense(latent_dim, kernel_initializer="zeros")
        self.output_cast = layers.Activation('linear', dtype='float32')

    def call(self, inputs, training=False):
        latents, timesteps, labels = inputs
        B = tf.shape(latents)[0]

        x = self.latent_proj(latents)
        x = x[:, None, :]

        ct = tf.tile(self.class_token, [B, 1, 1])
        ct = tf.cast(ct, x.dtype)
        x = tf.concat([ct, x], axis=1)

        t_emb = self.t_emb(timesteps)
        t_emb = self.t_mlp(t_emb)
        label_emb = self.label_emb(labels)
        cond = t_emb + label_emb

        for block in self.blocks:
            x = block(x, cond)

        mods = self.out_ada_linear2(self.out_ada_linear1(cond))
        mods = tf.cast(mods, x.dtype)
        shift, scale = tf.split(mods, 2, axis=-1)

        x = self.out_norm(x)
        x = x * (1 + scale[:, None, :]) + shift[:, None, :]
        x = x[:, 1:, :]
        x = self.out_proj(x[:, 0, :])

        return self.output_cast(x)


# =====================================================================
# MODELO COMPLETO: VAE + LATENT DIFFUSION
# =====================================================================
class LatentDiffusionModel(keras.Model):
    def __init__(self, img_shape, num_classes, latent_dim=128, vae=None, **kwargs):
        super().__init__(**kwargs)
        self.img_shape = img_shape
        self.vector_dim = np.prod(img_shape)
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        self.vae = vae
        self.vae.trainable = False

        self.network = LatentDiTNetwork(latent_dim, num_classes, hidden_dim=384, depth=6, num_heads=6)
        self.ema_network = LatentDiTNetwork(latent_dim, num_classes, hidden_dim=384, depth=6, num_heads=6)

        self.diffusion = GaussianDiffusion(timesteps=1000, clip_min=-3.0, clip_max=3.0)
        self.ema_decay = 0.9999
        self._initialized = False

    def _initialize_weights(self):
        if self._initialized:
            return

        print("  • Criando pesos com forward pass dummy...")
        dummy_latent = tf.zeros((2, self.latent_dim), dtype=tf.float32)
        dummy_t = tf.zeros((2,), dtype=tf.int32)
        dummy_l = tf.zeros((2,), dtype=tf.int64)

        _ = self.network([dummy_latent, dummy_t, dummy_l])
        _ = self.ema_network([dummy_latent, dummy_t, dummy_l])

        print("  • Copiando pesos para rede EMA...")
        self.ema_network.set_weights(self.network.get_weights())
        self._initialized = True
        print("  ✓ Inicialização completa!")

    def train_step(self, data):
        if not self._initialized:
            raise RuntimeError("Modelo não inicializado! Chame _initialize_weights() antes do fit().")

        images, labels = data
        batch_size = tf.shape(images)[0]
        vectors = tf.reshape(images, [batch_size, -1])

        latents = self.vae.encode(vectors)

        t = tf.random.uniform([batch_size], 0, self.diffusion.num_timesteps, dtype=tf.int32)

        noise = tf.random.normal(tf.shape(latents), dtype=latents.dtype)
        noisy_latents = self.diffusion.q_sample(latents, t, noise)

        with tf.GradientTape() as tape:
            pred_noise = self.network([noisy_latents, t, labels], training=True)

            pred_f32 = tf.cast(pred_noise, tf.float32)
            noise_f32 = tf.cast(noise, tf.float32)
            loss = tf.reduce_mean(tf.square(pred_f32 - noise_f32))

        grads = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_weights))

        for w, ema_w in zip(self.network.weights, self.ema_network.weights):
            ema_w.assign(self.ema_decay * ema_w + (1 - self.ema_decay) * w)

        return {"loss": loss}

    @tf.function
    def sample(self, batch_size, labels, guidance_scale=1.5, num_steps=None):
        if num_steps is None:
            num_steps = self.diffusion.num_timesteps

        z = tf.random.normal([batch_size, self.latent_dim], dtype=tf.float32)
        null_labels = tf.fill([batch_size], self.num_classes)

        for i in tf.range(num_steps - 1, -1, -1):
            t = tf.fill([batch_size], i)

            pred_noise_cond = self.ema_network([z, t, labels], training=False)
            pred_noise_cond = tf.cast(pred_noise_cond, tf.float32)

            if guidance_scale > 1.0:
                pred_noise_uncond = self.ema_network([z, t, null_labels], training=False)
                pred_noise_uncond = tf.cast(pred_noise_uncond, tf.float32)
                pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)
            else:
                pred_noise = pred_noise_cond

            z = self.diffusion.p_sample(pred_noise, z, t, clip_denoised=True)

        x = self.vae.decode(z)
        return x


# =====================================================================
# CALLBACKS
# =====================================================================
class VAEReconstructionCallback(keras.callbacks.Callback):
    def __init__(self, test_data, img_shape, every_n_epochs=10):
        super().__init__()
        self.test_data = test_data[:16]
        self.img_shape = img_shape
        self.every_n_epochs = every_n_epochs
        self.output_dir = "vae_reconstructions"
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n_epochs == 0:
            print(f"\nGerando reconstruções VAE na época {epoch + 1}...")

            test_vectors = tf.reshape(self.test_data, [len(self.test_data), -1])

            z = self.model.encode(test_vectors)
            recon_vectors = self.model.decode(z)

            originals = self.test_data
            reconstructions = recon_vectors.numpy().reshape(-1, *self.img_shape)

            originals = np.clip((originals + 1) / 2, 0, 1)
            reconstructions = np.clip((reconstructions + 1) / 2, 0, 1)

            fig, axes = plt.subplots(4, 8, figsize=(16, 8))
            fig.suptitle(f"VAE Reconstruções (SEM Skip Connections) - Época {epoch + 1}", fontsize=16)

            for i in range(16):
                axes[i // 4, (i % 4) * 2].imshow(originals[i])
                axes[i // 4, (i % 4) * 2].axis("off")
                if i == 0:
                    axes[i // 4, (i % 4) * 2].set_title("Original")

                axes[i // 4, (i % 4) * 2 + 1].imshow(reconstructions[i])
                axes[i // 4, (i % 4) * 2 + 1].axis("off")
                if i == 0:
                    axes[i // 4, (i % 4) * 2 + 1].set_title("Reconstrução")

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"epoch_{epoch + 1:04d}.png"), dpi=150)
            plt.close()


class LatentDiffusionCallback(keras.callbacks.Callback):
    def __init__(self, img_shape, num_classes, samples_per_class=5, every_n_epochs=10, guidance_scale=1.5):
        super().__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.every_n_epochs = every_n_epochs
        self.guidance_scale = guidance_scale
        self.output_dir = "samples_latent_diffusion"
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n_epochs == 0:
            print(f"\nGerando amostras de difusão latente na época {epoch + 1}...")
            all_samples = []
            for c in range(self.num_classes):
                labels = tf.fill([self.samples_per_class], c)
                vec_samples = self.model.sample(
                    self.samples_per_class,
                    labels=labels,
                    guidance_scale=self.guidance_scale
                )
                all_samples.append(vec_samples.numpy())

            all_vectors = np.concatenate(all_samples, axis=0)
            imgs = all_vectors.reshape(-1, *self.img_shape)
            imgs = np.clip((imgs + 1) / 2, 0, 1)

            fig, axes = plt.subplots(self.samples_per_class, self.num_classes,
                                     figsize=(self.num_classes * 2, self.samples_per_class * 2))
            fig.suptitle(f"Difusão Latente - Época {epoch + 1}", fontsize=16)
            for i in range(self.samples_per_class):
                for j in range(self.num_classes):
                    ax = axes[i, j] if self.samples_per_class > 1 else axes[j]
                    ax.imshow(imgs[i * self.num_classes + j])
                    ax.axis("off")
                    if i == 0:
                        ax.set_title(f"Classe {j}")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"epoch_{epoch + 1:04d}.png"), dpi=150)
            plt.close()


# =====================================================================
# MAIN
# =====================================================================
def main():
    setup_environment()

    IMG_SIZE = 64
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
    VECTOR_DIM = np.prod(IMG_SHAPE)
    LATENT_DIM = 128
    N_CLASSES = 10
    VAE_BATCH_SIZE = 64
    DIFFUSION_BATCH_SIZE = 64
    VAE_EPOCHS = 1
    DIFFUSION_EPOCHS = 100

    # Carregar dados
    from PIL import Image
    dataset_dir = "./50k"

    if not os.path.exists(dataset_dir):
        print(f"⚠ Dataset não encontrado em {dataset_dir}. Usando dados sintéticos.")
        np.random.seed(42)
        x = np.random.uniform(-1, 1, (1000, IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)
        y = np.random.randint(0, N_CLASSES, 1000)
        print(f"✓ Geradas {len(x)} imagens sintéticas para teste")
    else:
        files = sorted(os.listdir(dataset_dir))[:24000]
        imgs = []
        for f in files:
            try:
                img_path = os.path.join(dataset_dir, f)
                img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                img_array = (np.array(img, dtype=np.float32) / 127.5) - 1.0
                imgs.append(img_array)
            except Exception as e:
                print(f"Erro ao carregar {f}: {e}")
                continue

        x = np.array(imgs, dtype=np.float32)
        y = np.random.randint(0, N_CLASSES, len(x))
        print(f"✓ Carregadas {len(x)} imagens reais do diretório {dataset_dir}")

    # ==================================================================
    # FASE 1: TREINAR VAE (SEM SKIP CONNECTIONS)
    # ==================================================================
    print("\n" + "=" * 70)
    print("FASE 1: TREINAMENTO DO VAE (SEM SKIP CONNECTIONS)")
    print("=" * 70)

    encoder_dims = [512, 512, 512]
    decoder_dims = [512, 512, 512]

    vae = HierarchicalVAE(
        input_dim=VECTOR_DIM,
        latent_dim=LATENT_DIM
    )
    vae.encoder = HierarchicalVAEEncoder(
        input_dim=VECTOR_DIM,
        latent_dim=LATENT_DIM,
        hidden_dims=encoder_dims,
        num_heads=8,
        depth_per_level=2
    )
    vae.decoder = HierarchicalVAEDecoder(
        output_dim=VECTOR_DIM,
        hidden_dims=decoder_dims,
        num_heads=8,
        depth_per_level=2
    )

    vae.compile(optimizer=keras.optimizers.AdamW(1e-4, weight_decay=0.01))

    x_flat = tf.reshape(x, [len(x), -1])

    vae_callback = VAEReconstructionCallback(
        test_data=x[:16],
        img_shape=IMG_SHAPE,
        every_n_epochs=20
    )

    print("\nTestando forward pass do VAE...")
    test_batch = x_flat[:2]
    try:
        mu, logvar = vae.encoder(test_batch, training=False)
        print(f"  ✓ Encoder: {test_batch.shape} -> mu: {mu.shape}, logvar: {logvar.shape}")
        z = vae.reparameterize(mu, logvar)
        print(f"  ✓ Reparameterize: {mu.shape} -> z: {z.shape}")
        recon = vae.decoder(z, training=False)
        print(f"  ✓ Decoder: {z.shape} -> recon: {recon.shape}")
        print("  ✓ VAE forward pass OK!\n")
    except Exception as e:
        print(f"  ✗ Erro no forward pass: {e}")
        import traceback
        traceback.print_exc()
        raise

    print(f"\nTreinando VAE por {VAE_EPOCHS} épocas...")
    vae.fit(
        tf.data.Dataset.from_tensor_slices(x_flat).shuffle(24000).batch(VAE_BATCH_SIZE).prefetch(tf.data.AUTOTUNE),
        epochs=VAE_EPOCHS,
        callbacks=[vae_callback]
    )

    print("\n✓ VAE treinado! Salvando modelo...")
    vae.save_weights("vae_no_skip.weights.h5")
    print("✓ VAE salvo em vae_no_skip.weights.h5")

    # ==================================================================
    # FASE 2: TREINAR DIFUSÃO NO ESPAÇO LATENTE
    # ==================================================================
    print("\n" + "=" * 70)
    print("FASE 2: TREINAMENTO DA DIFUSÃO NO ESPAÇO LATENTE")
    print("=" * 70)

    ldm = LatentDiffusionModel(IMG_SHAPE, N_CLASSES, latent_dim=LATENT_DIM, vae=vae)
    ldm.compile(optimizer=keras.optimizers.AdamW(1e-4, weight_decay=0.01))

    print("Inicializando pesos do modelo de difusão...")
    ldm._initialize_weights()
    print("✓ Modelo de difusão criado e inicializado!")

    diffusion_callback = LatentDiffusionCallback(
        img_shape=IMG_SHAPE,
        num_classes=N_CLASSES,
        samples_per_class=4,
        every_n_epochs=10,
        guidance_scale=1.5
    )

    print(f"\nIniciando treinamento de difusão: {len(x)} amostras, {DIFFUSION_EPOCHS} épocas")
    ldm.fit(
        tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10000).batch(DIFFUSION_BATCH_SIZE).prefetch(
            tf.data.AUTOTUNE),
        epochs=DIFFUSION_EPOCHS,
        callbacks=[diffusion_callback]
    )

    # ==================================================================
    # GERAR AMOSTRAS FINAIS
    # ==================================================================
    print("\n" + "=" * 70)
    print("GERANDO AMOSTRAS FINAIS")
    print("=" * 70)

    final_samples = []
    for c in range(N_CLASSES):
        labels = tf.fill([10], c)
        samples = ldm.sample(10, labels=labels, guidance_scale=1.5)
        final_samples.append(samples.numpy())

    final_samples = np.concatenate(final_samples, axis=0)
    final_imgs = final_samples.reshape(-1, *IMG_SHAPE)
    final_imgs = np.clip((final_imgs + 1) / 2, 0, 1)

    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    fig.suptitle("Amostras Finais - Difusão Latente (VAE sem Skip Connections)", fontsize=20)
    for i in range(10):
        for j in range(10):
            axes[i, j].imshow(final_imgs[i * 10 + j])
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(f"C{j}", fontsize=10)
    plt.tight_layout()
    plt.savefig("final_samples_no_skip.png", dpi=200)
    plt.close()

    np.save("final_samples_no_skip.npy", final_samples)
    print("✓ Amostras finais salvas!")
    print("✓ Treinamento completo!")
    print("\nArquivos gerados:")
    print("  • vae_no_skip.weights.h5")
    print("  • vae_reconstructions/")
    print("  • samples_latent_diffusion/")
    print("  • final_samples_no_skip.png")
    print("  • final_samples_no_skip.npy")


if __name__ == "__main__":
    main()