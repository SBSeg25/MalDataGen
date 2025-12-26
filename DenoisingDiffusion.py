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


# =====================================================================
# EDM NOISE SCHEDULE
# =====================================================================
class EDMNoiseSchedule:
    def __init__(self, num_steps=1000, sigma_data=0.5):
        ramp = np.linspace(0, 1, num_steps)
        sigmas = (80.0 ** (1 / 7) + ramp * (0.002 ** (1 / 7) - 80.0 ** (1 / 7))) ** 7
        self.sigmas = tf.constant(sigmas, dtype=tf.float32)
        self.num_steps = num_steps
        self.sigma_data = sigma_data

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / tf.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_in = 1 / tf.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_noise = 0.25 * tf.math.log(sigma + 1e-8)
        return c_skip, c_out, c_in, c_noise

    def add_noise(self, x, t):
        sigma = tf.gather(self.sigmas, t)
        sigma = tf.reshape(sigma, [-1] + [1] * (len(x.shape) - 1))
        noise = tf.random.normal(tf.shape(x), dtype=x.dtype)
        return x + sigma * noise, noise, sigma


# =====================================================================
# COMPONENTES BÁSICOS DO TRANSFORMER
# =====================================================================
class RMSNorm(layers.Layer):
    def __init__(self, dim, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.eps = eps

    def build(self, input_shape):
        self.scale = self.add_weight(shape=(self.dim,), initializer="ones", name="scale")
        super().build(input_shape)

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


class FlashAttention(layers.Layer):
    def __init__(self, dim, num_heads, dropout=0.05, **kwargs):
        super().__init__(**kwargs)
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.rope = RotaryEmbedding(self.head_dim)
        self.qkv = layers.Dense(3 * dim, use_bias=False)
        self.proj = layers.Dense(dim)
        self.dropout = layers.Dropout(dropout)

    def call(self, x, training=False):
        B, N, _ = tf.shape(x)[0], tf.shape(x)[1], x.shape[2]

        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [B, N, 3, self.num_heads, self.head_dim])
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


# =====================================================================
# CROSS-ATTENTION HIERÁRQUICA
# =====================================================================
class HierarchicalCrossAttention(layers.Layer):
    """Cross-attention entre diferentes escalas hierárquicas"""

    def __init__(self, dim, num_heads, dropout=0.05, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = layers.Dense(dim, use_bias=False)
        self.kv_proj = layers.Dense(2 * dim, use_bias=False)
        self.proj = layers.Dense(dim)
        self.dropout = layers.Dropout(dropout)

    def call(self, x, context, training=False):
        """
        x: query (escala fina)
        context: key/value (escala grossa)
        """
        B, N, _ = tf.shape(x)[0], tf.shape(x)[1], x.shape[2]
        _, M, _ = tf.shape(context)[0], tf.shape(context)[1], context.shape[2]

        # Query da escala atual
        q = self.q_proj(x)
        q = tf.reshape(q, [B, N, self.num_heads, self.head_dim])
        q = tf.transpose(q, [0, 2, 1, 3])

        # Key e Value da escala grossa
        kv = self.kv_proj(context)
        kv = tf.reshape(kv, [B, M, 2, self.num_heads, self.head_dim])
        kv = tf.transpose(kv, [2, 0, 3, 1, 4])
        k, v = kv[0], kv[1]

        # Cross-attention
        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn, training=training)

        out = tf.matmul(attn, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [B, N, self.dim])

        return self.proj(out)


class SwiGLU(layers.Layer):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.05, **kwargs):
        super().__init__(**kwargs)
        hidden = int(dim * mlp_ratio)
        self.fc1 = layers.Dense(hidden * 2)
        self.fc2 = layers.Dense(dim)
        self.dropout = layers.Dropout(dropout)

    def call(self, x, training=False):
        x_proj = self.fc1(x)
        x, gate = tf.split(x_proj, 2, axis=-1)
        return self.fc2(self.dropout(x * tf.nn.silu(gate), training=training))


class AdaLNZero(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.linear1 = layers.Dense(dim, activation="silu")
        self.linear2 = layers.Dense(6 * dim, kernel_initializer="zeros")

    def call(self, cond):
        return self.linear2(self.linear1(cond))


# =====================================================================
# BLOCO HIERÁRQUICO DO TRANSFORMER
# =====================================================================
class HierarchicalDiTBlock(layers.Layer):
    """Bloco DiT com cross-attention hierárquica"""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.05,
                 has_cross_attn=False, **kwargs):
        super().__init__(**kwargs)
        self.has_cross_attn = has_cross_attn

        # Self-attention
        self.norm1 = RMSNorm(dim)
        self.attn = FlashAttention(dim, num_heads, dropout)

        # Cross-attention (se houver escala mais grossa)
        if has_cross_attn:
            self.norm_cross = RMSNorm(dim)
            self.cross_attn = HierarchicalCrossAttention(dim, num_heads, dropout)

        # MLP
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_ratio, dropout)

        # Adaptive Layer Norm
        self.ada = AdaLNZero(dim)

    def call(self, x, cond, coarse_features=None, training=False):
        cond = tf.cast(cond, tf.float32)
        mods = self.ada(cond)
        mods = tf.cast(mods, x.dtype)
        s1, sc1, g1, s2, sc2, g2 = tf.split(mods, 6, axis=-1)

        # Self-attention
        h = self.norm1(x)
        h = h * (1 + sc1[:, None, :]) + s1[:, None, :]
        x = x + g1[:, None, :] * self.attn(h, training=training)

        # Cross-attention com escala mais grossa (se disponível)
        if self.has_cross_attn and coarse_features is not None:
            h_cross = self.norm_cross(x)
            x = x + self.cross_attn(h_cross, coarse_features, training=training)

        # MLP
        h = self.norm2(x)
        h = h * (1 + sc2[:, None, :]) + s2[:, None, :]
        x = x + g2[:, None, :] * self.mlp(h, training=training)

        return x


class TimestepEmbedding(layers.Layer):
    def __init__(self, dim, max_period=10000, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        half_dim = dim // 2
        freqs = tf.exp(-math.log(max_period) * tf.range(0, half_dim, dtype=tf.float32) / half_dim)
        self.freqs = tf.constant(freqs, dtype=tf.float32)

    def call(self, timesteps):
        timesteps = tf.cast(timesteps, tf.float32)
        args = timesteps[:, None] * self.freqs[None, :]
        return tf.concat([tf.sin(args), tf.cos(args)], axis=-1)


# =====================================================================
# MODELO COM TRANSFORMERS HIERÁRQUICOS
# =====================================================================
class HierarchicalDiTNetwork(keras.Model):
    def __init__(self, vector_dim, num_classes, hidden_dim=768,
                 depth_per_scale=[4, 6, 6], num_heads=12,
                 label_dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.vector_dim = vector_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.label_dropout = label_dropout

        # Configuração hierárquica OTIMIZADA: [coarse, medium, fine]
        # Coarse: 8 patches, Medium: 16 patches, Fine: 64 patches
        self.patch_sizes = [vector_dim // 8, vector_dim // 16, vector_dim // 64]
        self.num_patches = [8, 16, 64]

        print(f"\n🔷 ARQUITETURA HIERÁRQUICA:")
        print(f"  Escala Coarse:  {self.num_patches[0]} patches x {self.patch_sizes[0]} dims")
        print(f"  Escala Medium:  {self.num_patches[1]} patches x {self.patch_sizes[1]} dims")
        print(f"  Escala Fine:    {self.num_patches[2]} patches x {self.patch_sizes[2]} dims")

        # Projeções para cada escala
        self.coarse_proj = layers.Dense(hidden_dim)
        self.medium_proj = layers.Dense(hidden_dim)
        self.fine_proj = layers.Dense(hidden_dim)

        # Class tokens
        self.coarse_token = self.add_weight(shape=(1, 1, hidden_dim),
                                            initializer="zeros", trainable=True)
        self.medium_token = self.add_weight(shape=(1, 1, hidden_dim),
                                            initializer="zeros", trainable=True)
        self.fine_token = self.add_weight(shape=(1, 1, hidden_dim),
                                          initializer="zeros", trainable=True)

        # Embeddings condicionais
        self.t_emb = TimestepEmbedding(hidden_dim)
        self.t_mlp = layers.Dense(hidden_dim, activation="silu")
        self.label_emb = layers.Embedding(num_classes + 1, hidden_dim)

        # Blocos hierárquicos
        # Escala Coarse (sem cross-attention)
        self.coarse_blocks = [
            HierarchicalDiTBlock(hidden_dim, num_heads, has_cross_attn=False)
            for _ in range(depth_per_scale[0])
        ]

        # Escala Medium (com cross-attention para coarse)
        self.medium_blocks = [
            HierarchicalDiTBlock(hidden_dim, num_heads, has_cross_attn=True)
            for _ in range(depth_per_scale[1])
        ]

        # Escala Fine (com cross-attention para medium)
        self.fine_blocks = [
            HierarchicalDiTBlock(hidden_dim, num_heads, has_cross_attn=True)
            for _ in range(depth_per_scale[2])
        ]

        # Upsampling entre escalas
        self.coarse_to_medium = layers.Dense(hidden_dim)
        self.medium_to_fine = layers.Dense(hidden_dim)

        # Saída
        self.out_norm = RMSNorm(hidden_dim)
        self.out_ada_linear1 = layers.Dense(hidden_dim, activation="silu")
        self.out_ada_linear2 = layers.Dense(2 * hidden_dim, kernel_initializer="zeros")
        self.out_proj = layers.Dense(self.patch_sizes[2], kernel_initializer="zeros")
        self.output_cast = layers.Activation('linear', dtype='float32')

    def call(self, inputs, training=False):
        vectors, timesteps, labels = inputs
        B = tf.shape(vectors)[0]

        # === ESCALA COARSE ===
        coarse_patches = tf.reshape(vectors, [B, self.num_patches[0], self.patch_sizes[0]])
        x_coarse = self.coarse_proj(coarse_patches)
        ct_coarse = tf.tile(self.coarse_token, [B, 1, 1])
        ct_coarse = tf.cast(ct_coarse, x_coarse.dtype)
        x_coarse = tf.concat([ct_coarse, x_coarse], axis=1)

        # Embedding condicional
        t_emb = self.t_emb(timesteps)
        t_emb = self.t_mlp(t_emb)
        if training and self.label_dropout > 0:
            mask = tf.random.uniform([B]) > self.label_dropout
            labels = tf.where(mask, labels, self.num_classes)
        label_emb = self.label_emb(labels)
        cond = t_emb + label_emb

        # Processa escala coarse
        for block in self.coarse_blocks:
            x_coarse = block(x_coarse, cond, training=training)

        coarse_features = x_coarse[:, 1:, :]  # Remove class token

        # === ESCALA MEDIUM ===
        medium_patches = tf.reshape(vectors, [B, self.num_patches[1], self.patch_sizes[1]])
        x_medium = self.medium_proj(medium_patches)
        ct_medium = tf.tile(self.medium_token, [B, 1, 1])
        ct_medium = tf.cast(ct_medium, x_medium.dtype)
        x_medium = tf.concat([ct_medium, x_medium], axis=1)

        # Upsample coarse features para guiar medium
        coarse_upsampled = self.coarse_to_medium(coarse_features)

        # Processa escala medium com cross-attention para coarse
        for block in self.medium_blocks:
            x_medium = block(x_medium, cond, coarse_upsampled, training=training)

        medium_features = x_medium[:, 1:, :]

        # === ESCALA FINE ===
        fine_patches = tf.reshape(vectors, [B, self.num_patches[2], self.patch_sizes[2]])
        x_fine = self.fine_proj(fine_patches)
        ct_fine = tf.tile(self.fine_token, [B, 1, 1])
        ct_fine = tf.cast(ct_fine, x_fine.dtype)
        x_fine = tf.concat([ct_fine, x_fine], axis=1)

        # Upsample medium features para guiar fine
        medium_upsampled = self.medium_to_fine(medium_features)

        # Processa escala fine com cross-attention para medium
        for block in self.fine_blocks:
            x_fine = block(x_fine, cond, medium_upsampled, training=training)

        # === SAÍDA ===
        mods = self.out_ada_linear2(self.out_ada_linear1(cond))
        mods = tf.cast(mods, x_fine.dtype)
        shift, scale = tf.split(mods, 2, axis=-1)

        x_fine = self.out_norm(x_fine[:, 1:, :])  # Remove class token
        x_fine = x_fine * (1 + scale[:, None, :]) + shift[:, None, :]
        x_fine = self.out_proj(x_fine)

        # Reconstrói vetor completo
        output = tf.reshape(x_fine, [B, self.vector_dim])
        return self.output_cast(output)


# =====================================================================
# CALLBACK
# =====================================================================
class ImageGenerationCallback(keras.callbacks.Callback):
    def __init__(self, img_shape, num_classes, samples_per_class=5,
                 every_n_epochs=10, guidance_scale=3.0):
        super().__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.every_n_epochs = every_n_epochs
        self.guidance_scale = guidance_scale
        self.output_dir = "samples_hierarchical"
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n_epochs == 0:
            print(f"\n{'=' * 60}")
            print(f"Gerando amostras na época {epoch + 1} | Loss: {logs.get('loss', 0):.6f}")
            all_samples = []
            for c in range(self.num_classes):
                labels = tf.fill([self.samples_per_class], c)
                vec_samples = self.model.sampler.sample(
                    self.model.ema_network,
                    shape=(self.samples_per_class, np.prod(self.img_shape)),
                    labels=labels, num_classes=self.num_classes,
                    guidance_scale=self.guidance_scale
                )
                all_samples.append(vec_samples.numpy())
            all_vectors = np.concatenate(all_samples, axis=0)
            print(f"  Range: [{all_vectors.min():.3f}, {all_vectors.max():.3f}] | Mean: {all_vectors.mean():.3f}")
            imgs = all_vectors.reshape(-1, *self.img_shape)
            imgs = np.clip((imgs + 1) / 2, 0, 1)
            fig, axes = plt.subplots(self.samples_per_class, self.num_classes,
                                     figsize=(self.num_classes * 2, self.samples_per_class * 2))
            fig.suptitle(f"Época {epoch + 1} (Transformers Hierárquicos)", fontsize=16)
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
            print(f"✓ Salvo em {self.output_dir}")
            print(f"{'=' * 60}\n")


# =====================================================================
# DPM-SOLVER++
# =====================================================================
class DPMSolverPP:
    def __init__(self, noise_schedule):
        self.ns = noise_schedule

    def sample(self, model, shape, labels, num_steps=100, guidance_scale=3.0, num_classes=10):
        x = tf.random.normal(shape, dtype=tf.float32)
        sigmas = self.ns.sigmas.numpy()
        steps = np.linspace(len(sigmas) - 1, 0, num_steps, dtype=int)
        null_labels = tf.fill([shape[0]], num_classes)
        for i, t_idx in enumerate(steps):
            sigma = float(sigmas[t_idx])
            c_skip, c_out, c_in, c_noise = self.ns.get_scalings(sigma)
            x_in = x * c_in
            t_input = tf.fill([shape[0]], c_noise)
            pred_cond = model([x_in, t_input, labels], training=False)
            pred_cond = tf.cast(pred_cond, tf.float32)
            if guidance_scale > 1.0:
                pred_uncond = model([x_in, t_input, null_labels], training=False)
                pred_uncond = tf.cast(pred_uncond, tf.float32)
                pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            else:
                pred = pred_cond
            denoised = c_skip * x + c_out * pred
            if i < len(steps) - 1:
                sigma_next = float(sigmas[steps[i + 1]])
                x = denoised + (sigma_next / sigma) * (x - denoised)
            else:
                x = denoised
        return x


# =====================================================================
# MODELO PRINCIPAL
# =====================================================================
class HierarchicalDiT(keras.Model):
    def __init__(self, img_shape, num_classes, hidden_dim=768,
                 depth_per_scale=[4, 6, 6], num_heads=12,
                 label_dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.img_shape = img_shape
        self.vector_dim = np.prod(img_shape)
        self.num_classes = num_classes
        self.network = HierarchicalDiTNetwork(
            self.vector_dim, self.num_classes, hidden_dim,
            depth_per_scale, num_heads, label_dropout
        )
        self.ema_network = HierarchicalDiTNetwork(
            self.vector_dim, self.num_classes, hidden_dim,
            depth_per_scale, num_heads, label_dropout
        )
        self.ns = EDMNoiseSchedule()
        self.sampler = DPMSolverPP(self.ns)
        self.ema_decay = 0.9999
        self._initialized = False

    def _initialize_weights(self):
        if self._initialized:
            return
        print("  • Inicializando pesos...")
        dummy = (tf.zeros((2, self.vector_dim)), tf.zeros((2,)), tf.zeros((2,), dtype=tf.int64))
        _ = self.network(dummy)
        _ = self.ema_network(dummy)
        self.ema_network.set_weights(self.network.get_weights())
        self._initialized = True
        print("  ✓ Pesos inicializados!")

    def train_step(self, data):
        if not self._initialized:
            raise RuntimeError("Modelo não inicializado!")
        images, labels = data
        batch_size = tf.shape(images)[0]
        vectors = tf.reshape(images, [batch_size, -1])
        t = tf.random.uniform([batch_size], 0, self.ns.num_steps, dtype=tf.int32)
        noisy, _, sigma = self.ns.add_noise(vectors, t)
        c_skip, c_out, c_in, c_noise = self.ns.get_scalings(sigma)
        c_noise_scalar = tf.reshape(c_noise, [batch_size])
        with tf.GradientTape() as tape:
            pred = self.network([noisy * c_in, c_noise_scalar, labels], training=True)
            target = (vectors - c_skip * noisy) / c_out
            loss = tf.reduce_mean(tf.square(tf.cast(pred, tf.float32) - tf.cast(target, tf.float32)))
        grads = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_weights))
        for w, ema_w in zip(self.network.weights, self.ema_network.weights):
            ema_w.assign(self.ema_decay * ema_w + (1 - self.ema_decay) * w)
        return {"loss": loss}


# =====================================================================
# MAIN
# =====================================================================
def main():
    setup_environment()

    IMG_SIZE = 64
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
    N_CLASSES = 10
    BATCH_SIZE = 32
    EPOCHS = 1000

    from PIL import Image
    dataset_dir = "./256x256/faces"

    if not os.path.exists(dataset_dir):
        print(f"⚠ Dataset não encontrado. Usando dados sintéticos.")
        np.random.seed(42)
        x = np.random.uniform(-1, 1, (1000, IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)
        y = np.random.randint(0, N_CLASSES, 1000)
    else:
        files = sorted(os.listdir(dataset_dir))[:36000]
        imgs = []
        for f in files:
            try:
                img = Image.open(os.path.join(dataset_dir, f)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                imgs.append((np.array(img, dtype=np.float32) / 127.5) - 1.0)
            except:
                continue
        x = np.array(imgs, dtype=np.float32)

        # ===== LABELS RANDÔMICOS =====
        print(f"\n🎲 Gerando labels randômicos...")
        np.random.seed(42)  # Para reprodutibilidade
        y = np.random.randint(0, N_CLASSES, len(x))

        # Mostra distribuição
        unique, counts = np.unique(y, return_counts=True)
        print(f"✓ Labels gerados aleatoriamente:")
        print(f"  Distribuição dos labels:")
        for c, count in zip(unique, counts):
            print(f"    Classe {c}: {count} imagens ({count / len(y) * 100:.1f}%)")

        print(f"✓ Dataset: {len(x)} imagens com {N_CLASSES} classes randômicas")

    print("\n🔷 Criando modelo DiT HIERÁRQUICO...")
    model = HierarchicalDiT(
        IMG_SHAPE, N_CLASSES,
        hidden_dim=320,
        depth_per_scale=[4, 4, 4],
        num_heads=8,
        label_dropout=0.1
    )
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=0.01))
    model._initialize_weights()

    callback = ImageGenerationCallback(IMG_SHAPE, N_CLASSES, every_n_epochs=10, guidance_scale=3.0)

    print(f"\n{'=' * 60}")
    print(f"🚀 INICIANDO TREINAMENTO (TRANSFORMERS HIERÁRQUICOS)")
    print(f"  Dataset: {len(x)} imagens")
    print(f"  Classes: {N_CLASSES} (labels RANDÔMICOS)")
    print(f"  Escalas: Coarse (8) → Medium (16) → Fine (64)")
    print(f"  Épocas: {EPOCHS}")
    print(f"{'=' * 60}\n")

    def create_dataset(images, labels, batch_size, shuffle=True):
        def generator():
            indices = np.arange(len(images))
            while True:
                if shuffle:
                    np.random.shuffle(indices)
                for i in indices:
                    yield images[i], labels[i]

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=IMG_SHAPE, dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )

        if shuffle:
            dataset = dataset.shuffle(10000, reshuffle_each_iteration=True)

        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    print(f"\nCriando dataset eficiente com {len(x)} imagens...")
    train_dataset = create_dataset(x, y, BATCH_SIZE, shuffle=True)
    print("✓ Dataset criado com sucesso!")

    model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=len(x) // BATCH_SIZE,
        callbacks=[callback]
    )

    print("\n✓ Treinamento concluído!")


if __name__ == "__main__":
    main()