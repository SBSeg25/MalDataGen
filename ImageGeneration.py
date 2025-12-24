#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
State-of-the-Art Diffusion Transformer (DiT) Implementation
Incorporando as melhores práticas de 2024-2025:
- EDM noise schedule
- RMSNorm + SwiGLU
- Flash Attention
- DPM-Solver++ sampling
- Classifier-Free Guidance
- Mixed Precision Training
- Cosine Decay + Warmup
"""

import os
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# =====================================================================
# CONFIGURAÇÃO DE GPU E PRECISÃO
# =====================================================================

def setup_environment():
    """Configurar GPU e mixed precision."""
    # GPU setup
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ {len(gpus)} GPU(s) disponível(is)")
        except RuntimeError as e:
            print(f"⚠ Erro ao configurar GPU: {e}")
    else:
        print("⚠ Nenhuma GPU encontrada - usando CPU")

    # Mixed precision (FP16)
    policy = keras.mixed_precision.Policy('mixed_float16')
    keras.mixed_precision.set_global_policy(policy)
    print("✓ Mixed precision (FP16) habilitado")


# =====================================================================
# EDM NOISE SCHEDULE (substitui Gaussian Diffusion)
# =====================================================================

class EDMNoiseSchedule:
    """
    Elucidating the Design Space of Diffusion Models (EDM)
    Superior ao beta schedule linear tradicional.
    """

    def __init__(
            self,
            sigma_min=0.002,
            sigma_max=80.0,
            rho=7.0,
            num_steps=1000,
            sigma_data=0.5
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.num_steps = num_steps
        self.sigma_data = sigma_data

        # EDM schedule - distribui sigmas de forma mais eficiente
        ramp = np.linspace(0, 1, num_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        self.sigmas = tf.constant(sigmas, dtype=tf.float32)

    def get_scalings(self, sigma):
        """EDM preconditioning - melhora estabilidade do treinamento."""
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / tf.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_in = 1 / tf.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_noise = 0.25 * tf.math.log(sigma)
        return c_skip, c_out, c_in, c_noise

    def add_noise(self, x, t):
        """Adiciona ruído aos dados no timestep t."""
        sigma = tf.gather(self.sigmas, t)
        sigma = tf.reshape(sigma, [-1, 1, 1, 1])
        noise = tf.random.normal(tf.shape(x), dtype=x.dtype)
        return x + sigma * noise, noise, sigma


# =====================================================================
# RMSNORM (substitui LayerNorm - mais eficiente)
# =====================================================================

class RMSNorm(layers.Layer):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.dim = dim

    def build(self, input_shape):
        self.scale = self.add_weight(
            shape=(self.dim,),
            initializer="ones",
            trainable=True,
            name="scale"
        )

    def call(self, x):
        rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.eps)
        return x / rms * self.scale


# =====================================================================
# ROTARY POSITION EMBEDDINGS (RoPE)
# =====================================================================

class RotaryEmbedding(layers.Layer):
    """Rotary Position Embeddings - melhor que embeddings fixos."""

    def __init__(self, dim, max_seq_len=2048, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

        # Pre-compute frequencies
        inv_freq = 1.0 / (10000 ** (tf.range(0, dim, 2, dtype=tf.float32) / dim))
        self.inv_freq = tf.constant(inv_freq)

    def call(self, seq_len):
        t = tf.range(seq_len, dtype=tf.float32)
        freqs = tf.einsum('i,j->ij', t, self.inv_freq)
        emb = tf.concat([freqs, freqs], axis=-1)
        return tf.cos(emb), tf.sin(emb)

    @staticmethod
    def apply_rotary_emb(x, cos, sin):
        """Aplica rotary embeddings."""
        x1, x2 = tf.split(x, 2, axis=-1)
        return tf.concat([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin
        ], axis=-1)


# =====================================================================
# SWIGLU FFN (substitui GELU)
# =====================================================================

class SwiGLU(layers.Layer):
    """SwiGLU activation - superior ao GELU padrão."""

    def __init__(self, dim, hidden_dim, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.w1 = layers.Dense(hidden_dim, use_bias=False)
        self.w2 = layers.Dense(hidden_dim, use_bias=False)
        self.w3 = layers.Dense(dim, use_bias=False)
        self.dropout = layers.Dropout(dropout)

    def call(self, x, training=False):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = tf.nn.silu(x1) * x2  # SwiGLU
        return self.dropout(self.w3(hidden), training=training)


# =====================================================================
# FLASH ATTENTION (memória eficiente)
# =====================================================================

class FlashAttention(layers.Layer):
    """
    Memory-efficient attention implementation.
    Nota: Flash Attention real requer kernels CUDA customizados.
    Esta é uma implementação aproximada eficiente.
    """

    def __init__(self, dim, num_heads, dropout=0.0, qk_norm=True, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qk_norm = qk_norm

        self.qkv = layers.Dense(dim * 3, use_bias=False)
        self.proj = layers.Dense(dim)
        self.dropout = layers.Dropout(dropout)

        # QK-Norm para estabilidade
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def call(self, x, rope_cos=None, rope_sin=None, training=False):
        B, N, C = tf.shape(x)[0], tf.shape(x)[1], x.shape[2]

        # QKV projection
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [B, N, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE if provided
        if rope_cos is not None and rope_sin is not None:
            q = RotaryEmbedding.apply_rotary_emb(q, rope_cos, rope_sin)
            k = RotaryEmbedding.apply_rotary_emb(k, rope_cos, rope_sin)

        # QK-Norm
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Scaled dot-product attention
        attn = (q @ tf.transpose(k, [0, 1, 3, 2])) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn, training=training)

        # Combine heads
        x = attn @ v
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [B, N, C])

        return self.proj(x)


# =====================================================================
# ADALN-ZERO (inicialização zero para estabilidade)
# =====================================================================

class AdaLNZero(layers.Layer):
    """Adaptive Layer Norm com Zero Initialization."""

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape):
        # Zero initialization para gates - crucial para estabilidade!
        self.modulation = keras.Sequential([
            layers.Dense(dim, activation="silu"),
            layers.Dense(6 * dim, kernel_initializer="zeros")
        ])

    def call(self, conditioning):
        return self.modulation(conditioning)


# =====================================================================
# SOTA DIT BLOCK
# =====================================================================

class SOTADiTBlock(layers.Layer):
    """
    State-of-the-Art DiT Block:
    - RMSNorm
    - Flash Attention com RoPE
    - SwiGLU FFN
    - AdaLN-Zero
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads

        # Pre-normalization
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        # Attention
        self.attn = FlashAttention(dim, num_heads, dropout, qk_norm=True)

        # FFN
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio), dropout)

        # AdaLN-Zero
        self.adaln = AdaLNZero(dim)

    def call(self, x, conditioning, rope_cos=None, rope_sin=None, training=False):
        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            tf.split(self.adaln(conditioning), 6, axis=-1)

        # Self-attention with adaptive norm
        norm_x = self.norm1(x)
        norm_x = norm_x * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        x = x + gate_msa[:, None, :] * self.attn(norm_x, rope_cos, rope_sin, training=training)

        # MLP with adaptive norm
        norm_x = self.norm2(x)
        norm_x = norm_x * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        x = x + gate_mlp[:, None, :] * self.mlp(norm_x, training=training)

        return x


# =====================================================================
# PATCH EMBEDDING
# =====================================================================

class PatchEmbedding(layers.Layer):
    """Convert image to sequence of patch embeddings."""

    def __init__(self, patch_size, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.projection = layers.Dense(hidden_dim)

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return self.projection(patches)


# =====================================================================
# TIMESTEP EMBEDDING COM FOURIER FEATURES
# =====================================================================

class TimestepEmbedding(layers.Layer):
    """Fourier feature timestep embeddings."""

    def __init__(self, dim, max_period=10000, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.max_period = max_period
        half_dim = dim // 2

        # Pre-compute frequencies
        freqs = tf.exp(
            -math.log(max_period) * tf.range(0, half_dim, dtype=tf.float32) / half_dim
        )
        self.freqs = tf.constant(freqs)

    def call(self, timesteps):
        # Fourier features
        args = tf.cast(timesteps, tf.float32)[:, None] * self.freqs[None, :]
        embedding = tf.concat([tf.cos(args), tf.sin(args)], axis=-1)

        # Additional MLP
        return embedding


# =====================================================================
# LABEL EMBEDDING COM DROPOUT PARA CFG
# =====================================================================

class LabelEmbedding(layers.Layer):
    """Label embeddings com suporte a Classifier-Free Guidance."""

    def __init__(self, num_classes, hidden_dim, dropout_prob=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob

    def build(self, input_shape):
        # +1 para unconditional (null class)
        self.embeddings = self.add_weight(
            shape=(self.num_classes + 1, self.hidden_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="label_embeddings",
        )

    def call(self, labels, training=False):
        # Durante treinamento, aplica dropout para CFG
        if training:
            # 10% das vezes, usa null class
            mask = tf.random.uniform([tf.shape(labels)[0]]) < self.dropout_prob
            labels = tf.where(mask, self.num_classes, labels)

        return tf.nn.embedding_lookup(self.embeddings, labels)


# =====================================================================
# FINAL LAYER
# =====================================================================

class FinalLayer(layers.Layer):
    """Final layer com AdaLN-Zero."""

    def __init__(self, patch_size, out_channels, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim

        self.norm = RMSNorm(hidden_dim)
        self.linear = layers.Dense(
            patch_size * patch_size * out_channels,
            kernel_initializer="zeros"  # Zero init
        )
        self.adaLN_modulation = keras.Sequential([
            layers.Dense(hidden_dim, activation="silu"),
            layers.Dense(2 * hidden_dim, kernel_initializer="zeros"),
        ])

    def call(self, x, conditioning):
        params = self.adaLN_modulation(conditioning)
        shift, scale = tf.split(params, 2, axis=-1)

        x = self.norm(x)
        x = x * (1 + scale[:, None, :]) + shift[:, None, :]
        x = self.linear(x)
        return x


# =====================================================================
# BUILD SOTA DIT MODEL
# =====================================================================

def build_sota_dit(
        img_size,
        img_channels,
        patch_size,
        num_classes,
        hidden_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout_rate=0.1,
):
    """Build State-of-the-Art DiT model."""

    # Inputs
    image_input = layers.Input(shape=(img_size, img_size, img_channels), name="images")
    time_input = layers.Input(shape=(), dtype=tf.float32, name="timesteps")
    label_input = layers.Input(shape=(), dtype=tf.int64, name="labels")

    # Patch embedding
    x = PatchEmbedding(patch_size, hidden_dim)(image_input)
    num_patches = (img_size // patch_size) ** 2

    # RoPE setup
    rope = RotaryEmbedding(hidden_dim // num_heads)
    rope_cos, rope_sin = rope(num_patches)

    # Timestep and label embeddings
    t_emb = TimestepEmbedding(hidden_dim)(time_input)
    t_emb = layers.Dense(hidden_dim, activation="silu")(t_emb)

    y_emb = LabelEmbedding(num_classes, hidden_dim, dropout_prob=0.1)(
        label_input, training=True
    )

    # Combine conditioning
    conditioning = t_emb + y_emb

    # Transformer blocks
    for _ in range(depth):
        x = SOTADiTBlock(hidden_dim, num_heads, mlp_ratio, dropout_rate)(
            x, conditioning, rope_cos, rope_sin
        )

    # Final layer
    x = FinalLayer(patch_size, img_channels, hidden_dim)(x, conditioning)

    # Reshape to image
    x = layers.Reshape((img_size // patch_size, img_size // patch_size,
                        patch_size * patch_size * img_channels))(x)

    # Unpatchify
    batch_size = tf.shape(image_input)[0]
    x = tf.reshape(x, [
        batch_size,
        img_size // patch_size,
        img_size // patch_size,
        patch_size,
        patch_size,
        img_channels
    ])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [batch_size, img_size, img_size, img_channels])

    return keras.Model([image_input, time_input, label_input], x, name="SOTA_DiT")


# =====================================================================
# DPM-SOLVER++ SAMPLER (muito mais rápido que DDPM)
# =====================================================================

class DPMSolverPlusPlus:
    """
    DPM-Solver++: Fast ODE solver para difusão.
    20-50x mais rápido que DDPM com qualidade similar.
    """

    def __init__(self, noise_schedule):
        self.noise_schedule = noise_schedule

    def sample(
            self,
            model,
            shape,
            labels,
            num_steps=20,
            guidance_scale=1.5,
            order=2
    ):
        """
        Sample usando DPM-Solver++ de segunda ordem.

        Args:
            model: Modelo de difusão
            shape: Shape das imagens
            labels: Labels para geração
            num_steps: Número de steps (20-50 é suficiente!)
            guidance_scale: Escala de CFG
            order: Ordem do solver (1, 2 ou 3)
        """
        batch_size = shape[0]

        # Start from noise
        x = tf.random.normal(shape, dtype=tf.float32)

        # Timestep schedule (Karras et al. schedule)
        timesteps = self._get_karras_schedule(num_steps)

        # Unconditional labels para CFG
        null_labels = tf.fill([batch_size], self.noise_schedule.num_steps)

        # Second-order DPM-Solver++
        x_prev = None

        for i, t in enumerate(timesteps):
            tt = tf.fill([batch_size], t)
            sigma = self.noise_schedule.sigmas[int(t)]
            sigma_tensor = tf.fill([batch_size, 1, 1, 1], sigma)

            # Model prediction com EDM preconditioning
            c_skip, c_out, c_in, c_noise = self.noise_schedule.get_scalings(sigma)

            # Conditional prediction
            x_in = x * c_in
            pred_cond = model([x_in, c_noise * tt, labels], training=False)

            # CFG
            if guidance_scale > 1.0:
                pred_uncond = model([x_in, c_noise * tt, null_labels], training=False)
                pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            else:
                pred = pred_cond

            # EDM denoiser
            denoised = c_skip * x + c_out * pred

            # DPM-Solver++ update
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                sigma_next = self.noise_schedule.sigmas[int(t_next)]

                if order == 1 or i == 0:
                    # First-order update
                    x = denoised + (sigma_next / sigma) * (x - denoised)
                else:
                    # Second-order update
                    h = sigma_next - sigma
                    h_prev = sigma - self.noise_schedule.sigmas[int(timesteps[i - 1])]
                    r = h_prev / h

                    x = denoised + (sigma_next / sigma) * (x - denoised) + \
                        (h / (2 * h_prev)) * (denoised - x_prev)

                x_prev = denoised
            else:
                x = denoised

        return x

    def _get_karras_schedule(self, num_steps):
        """Karras et al. noise schedule - distribui steps de forma ótima."""
        sigmas = self.noise_schedule.sigmas.numpy()

        # Inverte para ir de high -> low noise
        sigma_min = sigmas.min()
        sigma_max = sigmas.max()

        rho = 7.0  # Parâmetro de curvatura
        ramp = np.linspace(0, 1, num_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho

        # Converte sigmas para indices de timestep
        timesteps = []
        for sigma in sigmas:
            idx = np.argmin(np.abs(self.noise_schedule.sigmas.numpy() - sigma))
            timesteps.append(idx)

        return timesteps


# =====================================================================
# COSINE DECAY WITH WARMUP
# =====================================================================

class CosineDecayWithWarmup(keras.optimizers.schedules.LearningRateSchedule):
    """Cosine decay com warmup - usado em todos os modelos SOTA."""

    def __init__(
            self,
            initial_lr,
            decay_steps,
            warmup_steps,
            alpha=0.0,
            name=None
    ):
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.alpha = alpha
        self.name = name

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)

        # Warmup
        warmup_lr = self.initial_lr * step / warmup_steps

        # Cosine decay
        completed = (step - warmup_steps) / (decay_steps - warmup_steps)
        completed = tf.clip_by_value(completed, 0.0, 1.0)
        cosine_decayed = 0.5 * (1.0 + tf.cos(tf.constant(math.pi) * completed))
        decayed_lr = self.alpha + (self.initial_lr - self.alpha) * cosine_decayed

        return tf.where(step < warmup_steps, warmup_lr, decayed_lr)


# =====================================================================
# CALLBACKS
# =====================================================================

class GenerateImagesCallback(keras.callbacks.Callback):
    """Callback para gerar imagens durante treinamento."""

    def __init__(
            self,
            output_dir,
            num_classes,
            sampler,
            samples_per_class=5,
            every_n_epochs=10,
            guidance_scale=1.5
    ):
        super().__init__()
        self.output_dir = output_dir
        self.num_classes = num_classes
        self.sampler = sampler
        self.samples_per_class = samples_per_class
        self.every_n_epochs = every_n_epochs
        self.guidance_scale = guidance_scale

        self.samples_dir = os.path.join(output_dir, "training_samples")
        os.makedirs(self.samples_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.every_n_epochs == 0 or epoch == 0:
            print(f"\n{'=' * 70}")
            print(f"📸 Gerando amostras na época {epoch + 1}...")
            print(f"{'=' * 70}")

            all_samples = []

            for class_idx in range(self.num_classes):
                labels = tf.fill([self.samples_per_class], class_idx)

                shape = (
                    self.samples_per_class,
                    self.model.img_size,
                    self.model.img_size,
                    self.model.img_channels
                )

                samples = self.sampler.sample(
                    self.model.ema_network,
                    shape,
                    labels,
                    num_steps=20,  # DPM-Solver++ com apenas 20 steps!
                    guidance_scale=self.guidance_scale
                )

                # Normaliza para [0, 1]
                samples = samples.numpy()
                samples = np.clip(samples, -3, 3)  # Clip outliers
                samples = (samples - samples.min()) / (samples.max() - samples.min() + 1e-8)

                all_samples.append(samples)

            all_samples = np.concatenate(all_samples, axis=0)

            save_path = os.path.join(
                self.samples_dir,
                f"epoch_{epoch + 1:04d}.png"
            )

            self._visualize_and_save(all_samples, epoch + 1, save_path)
            print(f"✓ Amostras salvas: {save_path}")
            print(f"{'=' * 70}\n")

    def _visualize_and_save(self, samples, epoch, save_path):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            np.save(save_path.replace('.png', '.npy'), samples)
            return

        n_samples = len(samples)
        cols = self.num_classes
        rows = self.samples_per_class

        fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
        fig.suptitle(f'Época {epoch} - DPM-Solver++ (20 steps)',
                     fontsize=16, fontweight='bold')

        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)

        idx = 0
        for i in range(rows):
            for j in range(cols):
                img = samples[idx]
                img = np.clip(img, 0, 1)

                if img.shape[-1] == 3:
                    axes[i, j].imshow(img)
                else:
                    axes[i, j].imshow(img.squeeze(), cmap="gray")

                if i == 0:
                    axes[i, j].set_title(f'Classe {j}', fontsize=10)

                axes[i, j].axis("off")
                idx += 1

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# =====================================================================
# SOTA DIFFUSION TRANSFORMER
# =====================================================================

class SOTADiffusionTransformer(keras.Model):
    """
    State-of-the-Art Diffusion Transformer com:
    - EDM noise schedule
    - RMSNorm + SwiGLU
    - Flash Attention
    - DPM-Solver++ sampling
    - Classifier-Free Guidance
    - EMA adaptativo
    """

    def __init__(
            self,
            img_size,
            img_channels,
            patch_size,
            num_classes,
            hidden_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            dropout_rate=0.1,
            sigma_min=0.002,
            sigma_max=80.0,
            num_steps=1000,
            ema_decay=0.9999,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.img_size = img_size
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.ema_decay = ema_decay

        # Build networks
        self.network = build_sota_dit(
            img_size, img_channels, patch_size, num_classes,
            hidden_dim, depth, num_heads, mlp_ratio, dropout_rate
        )

        self.ema_network = build_sota_dit(
            img_size, img_channels, patch_size, num_classes,
            hidden_dim, depth, num_heads, mlp_ratio, dropout_rate
        )

        # Initialize EMA
        self.ema_network.set_weights(self.network.get_weights())

        # EDM noise schedule
        self.noise_schedule = EDMNoiseSchedule(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            num_steps=num_steps
        )

        # DPM-Solver++ sampler
        self.sampler = DPMSolverPlusPlus(self.noise_schedule)

    def train_step(self, data):
        # Unpack data
        if isinstance(data, tuple):
            if len(data) == 2:
                x, y = data
                if isinstance(x, tuple):
                    images, labels = x
                else:
                    images = x
                    labels = y
            else:
                images, labels = data[0]
        else:
            images, labels = data

        batch_size = tf.shape(images)[0]

        # Sample timesteps uniformemente
        t = tf.random.uniform(
            minval=0, maxval=self.num_steps, shape=(batch_size,), dtype=tf.int64
        )

        with tf.GradientTape() as tape:
            # Add noise com EDM
            noisy_images, noise, sigma = self.noise_schedule.add_noise(images, t)

            # EDM preconditioning
            c_skip, c_out, c_in, c_noise = self.noise_schedule.get_scalings(sigma)

            # Model prediction
            model_input = noisy_images * tf.reshape(c_in, [-1, 1, 1, 1])
            pred = self.network([model_input, c_noise[:, 0, 0, 0], labels], training=True)

            # EDM target
            target = (images - c_skip * noisy_images) / c_out
            target = tf.reshape(target, tf.shape(pred))

            # Loss (MSE)
            loss = tf.reduce_mean(tf.square(pred - target))

            # Loss scaling para mixed precision
            loss = self.optimizer.get_scaled_loss(loss)

        # Update weights
        gradients = tape.gradient(loss, self.network.trainable_weights)
        gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        # Update EMA
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema_decay * ema_weight + (1 - self.ema_decay) * weight)

        return {"loss": loss}

    def test_step(self, data):
        """Validation step."""
        if isinstance(data, tuple):
            if len(data) == 2:
                x, y = data
                if isinstance(x, tuple):
                    images, labels = x
                else:
                    images = x
                    labels = y
            else:
                images, labels = data[0]
        else:
            images, labels = data

        batch_size = tf.shape(images)[0]
        t = tf.random.uniform(
            minval=0, maxval=self.num_steps, shape=(batch_size,), dtype=tf.int64
        )

        noisy_images, noise, sigma = self.noise_schedule.add_noise(images, t)
        c_skip, c_out, c_in, c_noise = self.noise_schedule.get_scalings(sigma)

        model_input = noisy_images * tf.reshape(c_in, [-1, 1, 1, 1])
        pred = self.network([model_input, c_noise[:, 0, 0, 0], labels], training=False)

        target = (images - c_skip * noisy_images) / c_out
        target = tf.reshape(target, tf.shape(pred))

        loss = tf.reduce_mean(tf.square(pred - target))

        return {"loss": loss}

    def fit_model(
            self,
            x_real_samples,
            y_real_samples,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            output_dir="./output_sota_diffusion",
            save_images_every=10,
            verbose=1
    ):
        """Train the SOTA diffusion model."""
        print(f"\n{'=' * 70}")
        print(f"🚀 Iniciando Treinamento SOTA")
        print(f"{'=' * 70}")
        print(f"  • Arquitetura: RMSNorm + SwiGLU + Flash Attention")
        print(f"  • Noise Schedule: EDM")
        print(f"  • Sampler: DPM-Solver++ (20 steps)")
        print(f"  • Guidance: Classifier-Free (CFG)")
        print(f"  • Precision: Mixed FP16")
        print(f"  • Épocas: {epochs}")
        print(f"  • Batch size: {batch_size}")
        print(f"{'=' * 70}\n")

        # Normalize images
        x_normalized = x_real_samples.astype(np.float32)

        # Split train/val
        num_samples = len(x_normalized)
        num_val = int(num_samples * validation_split)
        num_train = num_samples - num_val

        indices = np.random.permutation(num_samples)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]

        x_train = x_normalized[train_indices]
        y_train = y_real_samples[train_indices]
        x_val = x_normalized[val_indices]
        y_val = y_real_samples[val_indices]

        # Create datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Learning rate schedule
        total_steps = epochs * (num_train // batch_size)
        warmup_steps = min(5000, total_steps // 10)

        lr_schedule = CosineDecayWithWarmup(
            initial_lr=1e-4,
            decay_steps=total_steps,
            warmup_steps=warmup_steps,
            alpha=1e-6
        )

        # Optimizer com gradient clipping
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=0.01,
            clipnorm=1.0,
            beta_1=0.9,
            beta_2=0.999
        )

        # Mixed precision setup
        optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)

        self.compile(optimizer=optimizer)

        # Callbacks
        callbacks = [
            GenerateImagesCallback(
                output_dir=output_dir,
                num_classes=self.num_classes,
                sampler=self.sampler,
                samples_per_class=5,
                every_n_epochs=save_images_every,
                guidance_scale=1.5
            )
        ]

        # Train
        history = self.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )

        return history

    def save_model(self, filepath):
        """Save model weights."""
        self.network.save_weights(filepath)
        print(f"✓ Modelo salvo: {filepath}")

    def load_model(self, filepath):
        """Load model weights."""
        self.network.load_weights(filepath)
        self.ema_network.set_weights(self.network.get_weights())
        print(f"✓ Modelo carregado: {filepath}")

    def get_samples(
            self,
            samples_per_class,
            num_sampling_steps=20,
            guidance_scale=1.5,
            verbose=True
    ):
        """
        Generate synthetic samples usando DPM-Solver++.

        Args:
            samples_per_class: Dict com formato {0: n0, 1: n1, ...}
            num_sampling_steps: Steps para DPM-Solver++ (20-50 recomendado)
            guidance_scale: CFG scale (1.0 = sem guidance, 1.5-7.5 típico)
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"🎨 Gerando Amostras com DPM-Solver++")
            print(f"  • Sampling steps: {num_sampling_steps} (vs 1000 DDPM!)")
            print(f"  • Guidance scale: {guidance_scale}")
            print(f"{'=' * 70}")

        all_samples = []
        classes_dict = samples_per_class.get("classes", samples_per_class)

        for class_idx, num_samples in classes_dict.items():
            if verbose:
                print(f"  • Classe {class_idx}: {num_samples} amostras...", end=" ")

            labels = tf.fill([num_samples], class_idx)
            shape = (num_samples, self.img_size, self.img_size, self.img_channels)

            samples = self.sampler.sample(
                self.ema_network,
                shape,
                labels,
                num_steps=num_sampling_steps,
                guidance_scale=guidance_scale
            )

            # Normalize
            samples = samples.numpy()
            samples = np.clip(samples, -3, 3)
            samples = (samples - samples.min()) / (samples.max() - samples.min() + 1e-8)

            all_samples.append(samples)

            if verbose:
                print("✓")

        all_samples = np.concatenate(all_samples, axis=0)

        if verbose:
            print(f"\n  ✓ Total gerado: {len(all_samples)} amostras")
            print(f"{'=' * 70}\n")

        return all_samples


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def load_dataset(dataset_dir, image_size, max_samples, n_classes):
    """Carrega dataset."""
    from PIL import Image

    print(f"\n{'=' * 70}")
    print(f"📂 Carregando Dataset")
    print(f"{'=' * 70}")

    x_samples = []
    image_files = sorted(os.listdir(dataset_dir))

    for i, img_name in enumerate(image_files):
        if i % 1000 == 0:
            print(f"  Processando: {i}/{len(image_files)}", end='\r')

        try:
            img_path = os.path.join(dataset_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            img = img.resize(image_size, Image.LANCZOS)
            img_array = np.asarray(img, dtype=np.float32) / 255.0
            x_samples.append(img_array)
        except Exception as e:
            continue

        if len(x_samples) >= max_samples:
            break

    x_samples = np.array(x_samples, dtype=np.float32)
    np.random.seed(42)
    y_samples = np.random.randint(0, n_classes, size=len(x_samples))

    print(f"\n✓ Dataset carregado: {len(x_samples)} imagens")
    print(f"{'=' * 70}\n")

    return x_samples, y_samples


def main():
    """Pipeline principal SOTA."""

    # Setup environment
    setup_environment()

    # =====================
    # CONFIGURAÇÕES
    # =====================

    IMAGE_SIZE = (64, 64)
    DATASET_DIR = "./50k"
    MAX_SAMPLES = 12000
    N_CLASSES = 10

    # Arquitetura (ajuste conforme GPU)
    PATCH_SIZE = 4
    HIDDEN_DIM = 384  # 384, 768, 1024
    DEPTH = 12  # 6, 12, 24
    NUM_HEADS = 6  # 6, 12, 16
    DROPOUT_RATE = 0.1
    MLP_RATIO = 4.0

    # EDM noise schedule
    SIGMA_MIN = 0.002
    SIGMA_MAX = 80.0
    NUM_STEPS = 1000

    # Training
    BATCH_SIZE = 128
    EPOCHS = 100
    EMA_DECAY = 0.9999
    VALIDATION_SPLIT = 0.1

    # Sampling
    SAMPLING_STEPS = 20  # DPM-Solver++ - muito mais rápido!
    GUIDANCE_SCALE = 1.5

    OUTPUT_DIR = "./output_sota_diffusion"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"🚀 STATE-OF-THE-ART DIFFUSION TRANSFORMER")
    print(f"{'=' * 70}")
    print(f"  ✓ EDM noise schedule")
    print(f"  ✓ RMSNorm + SwiGLU + Flash Attention")
    print(f"  ✓ DPM-Solver++ sampling (20 steps vs 1000!)")
    print(f"  ✓ Classifier-Free Guidance")
    print(f"  ✓ Mixed Precision FP16")
    print(f"  ✓ Cosine Decay + Warmup")
    print(f"{'=' * 70}\n")

    # Load data
    x_real, y_real = load_dataset(DATASET_DIR, IMAGE_SIZE, MAX_SAMPLES, N_CLASSES)

    # Create model
    print(f"{'=' * 70}")
    print(f"🏗️ Criando Modelo SOTA")
    print(f"{'=' * 70}")

    model = SOTADiffusionTransformer(
        img_size=IMAGE_SIZE[0],
        img_channels=3,
        patch_size=PATCH_SIZE,
        num_classes=N_CLASSES,
        hidden_dim=HIDDEN_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        dropout_rate=DROPOUT_RATE,
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        num_steps=NUM_STEPS,
        ema_decay=EMA_DECAY
    )

    # Train
    model.fit_model(
        x_real_samples=x_real,
        y_real_samples=y_real,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        output_dir=OUTPUT_DIR,
        save_images_every=10,
        verbose=1
    )

    # Save
    model_path = os.path.join(OUTPUT_DIR, "sota_diffusion_weights.h5")
    model.save_model(model_path)

    # Generate samples
    print(f"\n{'=' * 70}")
    print(f"🎨 Gerando Amostras Finais")
    print(f"{'=' * 70}")

    samples_per_class = {i: 10 for i in range(N_CLASSES)}

    synthetic_samples = model.get_samples(
        samples_per_class=samples_per_class,
        num_sampling_steps=SAMPLING_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        verbose=True
    )

    # Save
    samples_path = os.path.join(OUTPUT_DIR, "sota_synthetic_samples.npy")
    np.save(samples_path, synthetic_samples)

    print(f"\n{'=' * 70}")
    print(f"✅ PIPELINE SOTA CONCLUÍDO!")
    print(f"{'=' * 70}")
    print(f"  • Modelo: {model_path}")
    print(f"  • Amostras: {samples_path}")
    print(f"  • Visualizações: {OUTPUT_DIR}/training_samples/")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()