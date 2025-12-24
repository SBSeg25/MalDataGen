#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diffusion Transformer (DiT) Implementation
Based on "Scalable Diffusion Models with Transformers" paper
"""

import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# =====================================================================
# GAUSSIAN DIFFUSION UTILITIES
# =====================================================================

class GaussianDiffusion:
    """
    Implements the forward and reverse diffusion processes.
    """

    def __init__(
            self,
            beta_start=1e-4,
            beta_end=0.02,
            timesteps=1000,
            clip_min=-1.0,
            clip_max=1.0,
    ):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Linear variance schedule
        betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)
        self.num_timesteps = int(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        # Convert to TensorFlow constants
        self.betas = tf.constant(betas, dtype=tf.float32)
        self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf.float32)
        self.alphas_cumprod_prev = tf.constant(alphas_cumprod_prev, dtype=tf.float32)

        # Diffusion calculations
        self.sqrt_alphas_cumprod = tf.constant(
            np.sqrt(alphas_cumprod), dtype=tf.float32
        )
        self.sqrt_one_minus_alphas_cumprod = tf.constant(
            np.sqrt(1.0 - alphas_cumprod), dtype=tf.float32
        )
        self.sqrt_recip_alphas_cumprod = tf.constant(
            np.sqrt(1.0 / alphas_cumprod), dtype=tf.float32
        )
        self.sqrt_recipm1_alphas_cumprod = tf.constant(
            np.sqrt(1.0 / alphas_cumprod - 1), dtype=tf.float32
        )

        # Posterior calculations
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
        """Extract coefficients at specified timesteps."""
        batch_size = x_shape[0]
        out = tf.gather(a, t)
        return tf.reshape(out, [batch_size, 1, 1, 1])

    def q_sample(self, x_start, t, noise):
        """Forward diffusion: add noise to data."""
        x_start_shape = tf.shape(x_start)
        return (
                self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
                + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        """Predict x_0 from x_t and noise."""
        x_t_shape = tf.shape(x_t)
        return (
                self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape) * x_t
                - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t_shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """Compute posterior q(x_{t-1} | x_t, x_0)."""
        x_t_shape = tf.shape(x_t)
        posterior_mean = (
                self._extract(self.posterior_mean_coef1, t, x_t_shape) * x_start
                + self._extract(self.posterior_mean_coef2, t, x_t_shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t_shape)
        posterior_log_variance = self._extract(
            self.posterior_log_variance_clipped, t, x_t_shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(self, pred_noise, x, t, clip_denoised=True):
        """Compute mean and variance for reverse process."""
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        if clip_denoised:
            x_recon = tf.clip_by_value(x_recon, self.clip_min, self.clip_max)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, pred_noise, x, t, clip_denoised=True):
        """Sample from reverse process."""
        model_mean, _, model_log_variance = self.p_mean_variance(
            pred_noise, x=x, t=t, clip_denoised=clip_denoised
        )
        noise = tf.random.normal(shape=tf.shape(x), dtype=x.dtype)

        # No noise when t == 0
        nonzero_mask = tf.reshape(
            1 - tf.cast(tf.equal(t, 0), tf.float32), [tf.shape(x)[0], 1, 1, 1]
        )
        return model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise


# =====================================================================
# TRANSFORMER COMPONENTS
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


class TimestepEmbedding(layers.Layer):
    """Sinusoidal timestep embeddings."""

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2

    def call(self, timesteps):
        emb = math.log(10000) / (self.half_dim - 1)
        emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -emb)
        emb = tf.cast(timesteps, dtype=tf.float32)[:, None] * emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb


class LabelEmbedding(layers.Layer):
    """Learnable label embeddings."""

    def __init__(self, num_classes, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.num_classes, self.hidden_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="label_embeddings",
        )

    def call(self, labels):
        return tf.nn.embedding_lookup(self.embeddings, labels)


class DiTBlock(layers.Layer):
    """
    Diffusion Transformer Block with adaptive layer norm.
    """

    def __init__(self, hidden_dim, num_heads, ffn_expansion=4.0, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ffn_expansion = ffn_expansion
        self.dropout_rate = dropout_rate

        # Adaptive Layer Norm
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        # Attention
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_dim // num_heads,
            dropout=dropout_rate,
        )

        # FFN
        self.ffn = keras.Sequential([
            layers.Dense(int(hidden_dim * ffn_expansion), activation="gelu"),
            layers.Dropout(dropout_rate),
            layers.Dense(hidden_dim),
            layers.Dropout(dropout_rate),
        ])

        # Adaptive modulation (scale and shift)
        self.adaLN_modulation = keras.Sequential([
            layers.Dense(hidden_dim, activation="silu"),
            layers.Dense(6 * hidden_dim),  # 6 params: scale & shift for norm1, norm2, gate
        ])

    def call(self, x, conditioning, training=False):
        # Get adaptive parameters
        params = self.adaLN_modulation(conditioning)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = tf.split(
            params, 6, axis=-1
        )

        # Self-attention with adaptive norm
        norm_x = self.norm1(x)
        norm_x = norm_x * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        attn_out = self.attention(norm_x, norm_x, training=training)
        x = x + gate_msa[:, None, :] * attn_out

        # FFN with adaptive norm
        norm_x = self.norm2(x)
        norm_x = norm_x * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        ffn_out = self.ffn(norm_x, training=training)
        x = x + gate_mlp[:, None, :] * ffn_out

        return x


class FinalLayer(layers.Layer):
    """Final layer to predict noise."""

    def __init__(self, patch_size, out_channels, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim

        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.linear = layers.Dense(patch_size * patch_size * out_channels)
        self.adaLN_modulation = keras.Sequential([
            layers.Dense(hidden_dim, activation="silu"),
            layers.Dense(2 * hidden_dim),
        ])

    def call(self, x, conditioning):
        # Adaptive modulation
        params = self.adaLN_modulation(conditioning)
        shift, scale = tf.split(params, 2, axis=-1)

        x = self.norm(x)
        x = x * (1 + scale[:, None, :]) + shift[:, None, :]
        x = self.linear(x)
        return x


# =====================================================================
# DIFFUSION TRANSFORMER (DiT) MODEL
# =====================================================================

def build_dit_model(
        img_size,
        img_channels,
        patch_size,
        num_classes,
        hidden_dim=768,
        depth=12,
        num_heads=12,
        ffn_expansion=4.0,
        dropout_rate=0.1,
):
    """
    Build Diffusion Transformer model.

    Args:
        img_size: Image size (assumes square images)
        img_channels: Number of image channels
        patch_size: Size of patches
        num_classes: Number of classes for conditioning
        hidden_dim: Hidden dimension size
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        ffn_expansion: FFN expansion factor
        dropout_rate: Dropout rate

    Returns:
        Keras Model
    """

    # Inputs
    image_input = layers.Input(shape=(img_size, img_size, img_channels), name="images")
    time_input = layers.Input(shape=(), dtype=tf.int64, name="timesteps")
    label_input = layers.Input(shape=(), dtype=tf.int64, name="labels")

    # Patch embedding
    x = PatchEmbedding(patch_size, hidden_dim)(image_input)
    num_patches = (img_size // patch_size) ** 2

    # Positional embedding
    pos_embedding = layers.Embedding(
        input_dim=num_patches,
        output_dim=hidden_dim,
        name="pos_embedding"
    )(tf.range(num_patches))
    x = x + pos_embedding[None, :, :]

    # Timestep and label embeddings
    t_emb = TimestepEmbedding(hidden_dim)(time_input)
    t_emb = layers.Dense(hidden_dim, activation="silu")(t_emb)

    y_emb = LabelEmbedding(num_classes, hidden_dim)(label_input)

    # Combine conditioning
    conditioning = t_emb + y_emb

    # Transformer blocks
    for _ in range(depth):
        x = DiTBlock(hidden_dim, num_heads, ffn_expansion, dropout_rate)(
            x, conditioning
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

    return keras.Model([image_input, time_input, label_input], x, name="DiT")


# =====================================================================
# DIFFUSION MODEL WRAPPER
# =====================================================================

class DiffusionTransformer(keras.Model):
    """
    Complete Diffusion Transformer Model with training and sampling.
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
            ffn_expansion=4.0,
            dropout_rate=0.1,
            timesteps=1000,
            beta_start=1e-4,
            beta_end=0.02,
            ema_decay=0.9999,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.img_size = img_size
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.timesteps = timesteps
        self.ema_decay = ema_decay

        # Build networks
        self.network = build_dit_model(
            img_size, img_channels, patch_size, num_classes,
            hidden_dim, depth, num_heads, ffn_expansion, dropout_rate
        )

        self.ema_network = build_dit_model(
            img_size, img_channels, patch_size, num_classes,
            hidden_dim, depth, num_heads, ffn_expansion, dropout_rate
        )

        # Initialize EMA with same weights
        self.ema_network.set_weights(self.network.get_weights())

        # Diffusion utilities
        self.gdf_util = GaussianDiffusion(
            beta_start=beta_start,
            beta_end=beta_end,
            timesteps=timesteps,
        )

    def train_step(self, data):
        images, labels = data
        batch_size = tf.shape(images)[0]

        # Sample timesteps
        t = tf.random.uniform(
            minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64
        )

        with tf.GradientTape() as tape:
            # Sample noise
            noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)

            # Diffuse images
            images_t = self.gdf_util.q_sample(images, t, noise)

            # Predict noise
            pred_noise = self.network([images_t, t, labels], training=True)

            # Compute loss
            loss = self.compiled_loss(noise, pred_noise)

        # Update weights
        gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        # Update EMA
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema_decay * ema_weight + (1 - self.ema_decay) * weight)

        return {"loss": loss}

    @tf.function
    def generate_images(self, num_images, labels, guidance_scale=1.0):
        """
        Generate images using the trained model.

        Args:
            num_images: Number of images to generate
            labels: Class labels for generation
            guidance_scale: Classifier-free guidance scale

        Returns:
            Generated images
        """
        # Start from noise
        samples = tf.random.normal(
            shape=(num_images, self.img_size, self.img_size, self.img_channels),
            dtype=tf.float32
        )

        # Reverse diffusion
        for t in reversed(range(0, self.timesteps)):
            tt = tf.cast(tf.fill([num_images], t), dtype=tf.int64)

            # Predict noise
            pred_noise = self.ema_network([samples, tt, labels], training=False)

            # Apply guidance if scale > 1
            if guidance_scale > 1.0:
                # Unconditional prediction (use class num_classes as null label)
                uncond_labels = tf.fill([num_images], self.num_classes)
                uncond_noise = self.ema_network([samples, tt, uncond_labels], training=False)

                # Guided prediction
                pred_noise = uncond_noise + guidance_scale * (pred_noise - uncond_noise)

            # Denoise
            samples = self.gdf_util.p_sample(pred_noise, samples, tt, clip_denoised=True)

        return samples

    def call(self, inputs, training=False):
        """Forward pass."""
        images, timesteps, labels = inputs
        return self.network([images, timesteps, labels], training=training)


# =====================================================================
# USAGE EXAMPLE
# =====================================================================

import os
import numpy as np
from PIL import Image
import tensorflow as tf


# Configurar GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ {len(gpus)} GPU(s) disponível(is)")
    except RuntimeError as e:
        print(f"⚠ Erro ao configurar GPU: {e}")


# Verificar matplotlib
try:
    import matplotlib.pyplot as plt

    matplotlib_available = True
except ImportError:
    matplotlib_available = False
    print("⚠️ Matplotlib não disponível - visualizações desabilitadas")

# =====================
# CONFIGURAÇÕES
# =====================

# Dataset
IMAGE_SIZE = (16, 16)
INPUT_SHAPE = (16, 16, 3)
DATASET_DIR = "./50k"
MAX_SAMPLES = 12000
N_CLASSES = 10

# Modelo
PATCH_SIZE = 2
HIDDEN_DIM = 512  # Começar com menor para treinar mais rápido
DEPTH = 12  # Profundidade moderada
NUM_HEADS = 8
DROPOUT_RATE = 0.1
FFN_EXPANSION = 4.0
USE_CROSS_ATTENTION = True
CONTEXT_SEQ_LEN = 32

# Diffusion
BETA_START = 1e-4
BETA_END = 0.02
TIME_STEPS = 1000
CLIP_MIN = -1.0
CLIP_MAX = 1.0

# Treinamento
BATCH_SIZE = 4  # Ajuste conforme sua GPU
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EMA_DECAY = 0.9999
VALIDATION_SPLIT = 0.1

# Mixed Precision
USE_MIXED_PRECISION = False  # Desabilitar por enquanto devido a bugs

# Output
OUTPUT_DIR = "./output_diffusion"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================
# CARREGAR DADOS
# =====================

def load_dataset(dataset_dir, image_size, max_samples, n_classes):
    """Carrega e prepara o dataset."""
    print(f"\n{'=' * 70}")
    print(f"📂 Carregando Dataset")
    print(f"{'=' * 70}")

    x_samples = []
    image_files = sorted(os.listdir(dataset_dir))

    print(f"  • Diretório: {dataset_dir}")
    print(f"  • Tamanho alvo: {image_size}")
    print(f"  • Limite: {max_samples} imagens")

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
            print(f"\n⚠ Erro ao carregar {img_name}: {e}")
            continue

        if len(x_samples) >= max_samples:
            break

    x_samples = np.array(x_samples, dtype=np.float32)

    # Gera labels aleatórios (substitua por labels reais se disponível)
    np.random.seed(42)
    y_samples = np.random.randint(0, n_classes, size=len(x_samples))

    print(f"\n\n✓ Dataset carregado:")
    print(f"  • Total de imagens: {len(x_samples)}")
    print(f"  • Shape: {x_samples.shape}")
    print(f"  • Range: [{x_samples.min():.3f}, {x_samples.max():.3f}]")
    print(f"  • Classes: {n_classes}")
    print(f"  • Distribuição por classe:")

    for i in range(n_classes):
        count = np.sum(y_samples == i)
        print(f"    Classe {i}: {count} amostras ({count / len(y_samples) * 100:.1f}%)")

    print(f"{'=' * 70}\n")

    return x_samples, y_samples


# =====================
# VISUALIZAÇÃO
# =====================

def visualize_samples(samples, title, save_path, n_samples=36):
    """Visualiza e salva amostras."""
    if not matplotlib_available:
        print("⚠ Matplotlib não disponível - pulando visualização")
        return

    n = min(n_samples, samples.shape[0])
    cols = 6
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 2 * rows))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Flatten axes se necessário
    if rows == 1:
        axes = axes.reshape(1, -1)

    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx < n:
                img = samples[idx]

                # Garante que está no range correto
                img = np.clip(img, 0, 1)

                if img.shape[-1] == 3:  # RGB
                    axes[i, j].imshow(img)
                else:  # Grayscale
                    axes[i, j].imshow(img.squeeze(), cmap="gray")

                axes[i, j].axis("off")
                idx += 1
            else:
                axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Visualização salva: {save_path}")


# =====================
# MAIN
# =====================

def main():
    print(f"\n{'=' * 70}")
    print(f"🚀 DIFFUSION MODEL WITH DiT - TRAINING PIPELINE")
    print(f"{'=' * 70}")

    # Carrega dados
    x_real, y_real = load_dataset(
        DATASET_DIR,
        IMAGE_SIZE,
        MAX_SAMPLES,
        N_CLASSES
    )

    # Visualiza amostras reais
    visualize_samples(
        x_real[:36],
        "Real Samples from Dataset",
        os.path.join(OUTPUT_DIR, "real_samples.png")
    )

    # =====================
    # CRIA MODELO
    # =====================

    print(f"\n{'=' * 70}")
    print(f"🏗️ Criando Modelo de Difusão")
    print(f"{'=' * 70}")

    model = DenoisingDiffusion(
        number_classes=N_CLASSES,
        beta_start=BETA_START,
        beta_end=BETA_END,
        time_steps=TIME_STEPS,
        clip_min=CLIP_MIN,
        clip_max=CLIP_MAX,
        patch_size=PATCH_SIZE,
        hidden_dim=HIDDEN_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        ffn_expansion=FFN_EXPANSION,
        use_cross_attention=USE_CROSS_ATTENTION,
        context_seq_len=CONTEXT_SEQ_LEN,
        use_mixed_precision=USE_MIXED_PRECISION,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        ema_decay=EMA_DECAY
    )

    # =====================
    # TREINAMENTO
    # =====================

    model.fit_model(
        input_shape=INPUT_SHAPE,
        x_real_samples=x_real,
        y_real_samples=y_real,
        flatten=True,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=1
    )

    # Salva modelo
    model_path = os.path.join(OUTPUT_DIR, "diffusion_model_weights.h5")
    model.save_model(model_path)

    # =====================
    # GERAÇÃO DE AMOSTRAS
    # =====================

    print(f"\n{'=' * 70}")
    print(f"🎨 Gerando Amostras Sintéticas")
    print(f"{'=' * 70}")

    # Define quantas amostras gerar por classe
    samples_per_class = {
        "number_classes": N_CLASSES,
        "classes": {i: 5 for i in range(N_CLASSES)}  # 5 amostras por classe
    }

    # Gera amostras
    synthetic_samples = model.get_samples(
        samples_per_class=samples_per_class,
        guidance_scale=1.0,
        verbose=True
    )

    # Salva arrays
    samples_path = os.path.join(OUTPUT_DIR, "synthetic_samples.npy")
    np.save(samples_path, synthetic_samples)
    print(f"  ✓ Arrays salvos: {samples_path}")

    # Visualiza
    visualize_samples(
        synthetic_samples,
        "Generated Synthetic Samples",
        os.path.join(OUTPUT_DIR, "synthetic_samples.png"),
        n_samples=min(len(synthetic_samples), 50)
    )

    # =====================
    # COMPARAÇÃO
    # =====================

    if matplotlib_available and len(synthetic_samples) >= 18:
        print(f"\n{'=' * 70}")
        print(f"📊 Criando Comparação Real vs Sintético")
        print(f"{'=' * 70}")

        fig, axes = plt.subplots(2, 9, figsize=(18, 4))
        fig.suptitle("Comparação: Real (top) vs Sintético (bottom)", fontsize=14, fontweight='bold')

        for i in range(9):
            # Real
            axes[0, i].imshow(x_real[i])
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel('Real', fontsize=12, fontweight='bold')

            # Sintético
            axes[1, i].imshow(synthetic_samples[i])
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel('Sintético', fontsize=12, fontweight='bold')

        plt.tight_layout()
        comparison_path = os.path.join(OUTPUT_DIR, "comparison.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Comparação salva: {comparison_path}")

    # =====================
    # ESTATÍSTICAS FINAIS
    # =====================

    print(f"\n{'=' * 70}")
    print(f"📈 ESTATÍSTICAS FINAIS")
    print(f"{'=' * 70}")
    print(f"\n📊 Dataset Real:")
    print(f"  • Amostras: {len(x_real)}")
    print(f"  • Shape: {x_real.shape}")
    print(f"  • Média: {x_real.mean():.4f}")
    print(f"  • Std: {x_real.std():.4f}")

    print(f"\n🎨 Dataset Sintético:")
    print(f"  • Amostras: {len(synthetic_samples)}")
    print(f"  • Shape: {synthetic_samples.shape}")
    print(f"  • Média: {synthetic_samples.mean():.4f}")
    print(f"  • Std: {synthetic_samples.std():.4f}")

    print(f"\n📁 Arquivos Salvos:")
    print(f"  • Modelo: {model_path}")
    print(f"  • Amostras: {samples_path}")
    print(f"  • Visualizações: {OUTPUT_DIR}/*.png")

    print(f"\n{'=' * 70}")
    print(f"✅ PIPELINE CONCLUÍDO COM SUCESSO!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()