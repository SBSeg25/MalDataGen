
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

    # Mixed precision: computação em FP16, pesos em FP32
    # Conditioning sempre em FP32 para estabilidade numérica
    mixed_precision.set_global_policy('mixed_float16')
    print("✓ Mixed precision (FP16) habilitado")


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
# COMPONENTES DO TRANSFORMER
# =====================================================================
class RMSNorm(layers.Layer):
    def __init__(self, dim, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.eps = eps
        self.scale = None

    def build(self, input_shape):
        self.scale = self.add_weight(
            shape=(self.dim,),
            initializer="ones",
            name="scale"
        )
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
    def __init__(self, dim, num_heads, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_rate = dropout
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

        attn = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) * self.scale, axis=-1)
        attn = self.dropout(attn, training=training)

        out = tf.matmul(attn, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [B, N, self.dim])
        return self.proj(out)


class SwiGLU(layers.Layer):
    def __init__(self, dim, mlp_ratio=4.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        hidden = int(dim * mlp_ratio)
        self.fc1 = layers.Dense(hidden * 2)
        self.fc2 = layers.Dense(dim)

    def call(self, x):
        x_proj = self.fc1(x)
        x, gate = tf.split(x_proj, 2, axis=-1)
        return self.fc2(x * tf.nn.silu(gate))


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
        self.dim = dim
        self.norm1 = RMSNorm(dim)
        self.attn = FlashAttention(dim, num_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_ratio)
        self.ada = AdaLNZero(dim)

    def call(self, x, cond):
        cond = tf.cast(cond, tf.float32)
        mods = self.ada(cond)
        mods = tf.cast(mods, x.dtype)

        s1, sc1, g1, s2, sc2, g2 = tf.split(mods, 6, axis=-1)

        # Attention block
        h = self.norm1(x)
        h = h * (1 + sc1[:, None, :]) + s1[:, None, :]
        x = x + g1[:, None, :] * self.attn(h)

        # MLP block
        h = self.norm2(x)
        h = h * (1 + sc2[:, None, :]) + s2[:, None, :]
        return x + g2[:, None, :] * self.mlp(h)


class TimestepEmbedding(layers.Layer):
    def __init__(self, dim, max_period=10000, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
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


# =====================================================================
# MODELO SUBCLASSED PARA VETORES
# =====================================================================
class DiTNetwork(keras.Model):
    def __init__(self, vector_dim, num_classes, hidden_dim=768, depth=12, num_heads=12, **kwargs):
        super().__init__(**kwargs)
        self.vector_dim = vector_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.depth = depth

        # Camadas de entrada
        self.vec_proj = layers.Dense(hidden_dim)
        self.class_token = self.add_weight(
            shape=(1, 1, hidden_dim),
            initializer="zeros",
            trainable=True,
            name="class_token"
        )

        # Embeddings
        self.t_emb = TimestepEmbedding(hidden_dim)
        self.t_mlp = layers.Dense(hidden_dim, activation="silu")
        self.label_emb = layers.Embedding(num_classes + 1, hidden_dim)

        # Blocos transformer
        self.blocks = [DiTBlock(hidden_dim, num_heads) for _ in range(depth)]

        # Camada de saída
        self.out_norm = RMSNorm(hidden_dim)
        self.out_ada_linear1 = layers.Dense(hidden_dim, activation="silu")
        self.out_ada_linear2 = layers.Dense(2 * hidden_dim, kernel_initializer="zeros")
        self.out_proj = layers.Dense(vector_dim, kernel_initializer="zeros")

        # Camada final para converter para float32
        self.output_cast = layers.Activation('linear', dtype='float32')

    def call(self, inputs, training=False):
        vectors, timesteps, labels = inputs
        B = tf.shape(vectors)[0]

        x = self.vec_proj(vectors)
        x = x[:, None, :]

        ct = tf.tile(self.class_token, [B, 1, 1])
        ct = tf.cast(ct, x.dtype)
        x = tf.concat([ct, x], axis=1)

        # === CORREÇÃO AQUI ===
        t_emb = self.t_emb(timesteps)  # mantém float16
        t_emb = self.t_mlp(t_emb)  # mantém float16
        label_emb = self.label_emb(labels)  # já vem em float16 com mixed precision
        cond = t_emb + label_emb  # soma em float16 → OK!

        # Remova todos os casts para float32 aqui
        # cond = tf.cast(t_emb + label_emb, x.dtype)  ← REMOVA esta linha

        for block in self.blocks:
            x = block(x, cond)

        # Camada de saída — aqui pode manter o cast para float32 se quiser estabilidade
        mods = self.out_ada_linear2(self.out_ada_linear1(cond))  # cond já em float16
        mods = tf.cast(mods, x.dtype)
        shift, scale = tf.split(mods, 2, axis=-1)

        x = self.out_norm(x)
        x = x * (1 + scale[:, None, :]) + shift[:, None, :]
        x = x[:, 1:, :]
        x = self.out_proj(x[:, 0, :])

        return self.output_cast(x)

# =====================================================================
# CALLBACK
# =====================================================================
class ImageGenerationCallback(keras.callbacks.Callback):
    def __init__(self, img_shape, num_classes, samples_per_class=5, every_n_epochs=10, guidance_scale=1.5):
        super().__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.every_n_epochs = every_n_epochs
        self.guidance_scale = guidance_scale
        self.output_dir = "samples_flattened"
        os.makedirs(self.output_dir, exist_ok=True)

    def _vector_to_image(self, vectors):
        imgs = vectors.reshape(-1, *self.img_shape)
        imgs = np.clip((imgs + 1) / 2, 0, 1)
        return imgs

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n_epochs == 0:
            print(f"\nGerando amostras na época {epoch + 1}...")
            all_samples = []
            for c in range(self.num_classes):
                labels = tf.fill([self.samples_per_class], c)
                vec_samples = self.model.sampler.sample(
                    self.model.ema_network,
                    shape=(self.samples_per_class, np.prod(self.img_shape)),
                    labels=labels,
                    num_classes=self.num_classes,
                    guidance_scale=self.guidance_scale
                )
                all_samples.append(vec_samples.numpy())
            all_vectors = np.concatenate(all_samples, axis=0)
            imgs = self._vector_to_image(all_vectors)

            fig, axes = plt.subplots(self.samples_per_class, self.num_classes,
                                     figsize=(self.num_classes * 2, self.samples_per_class * 2))
            fig.suptitle(f"Época {epoch + 1}", fontsize=16)
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
            print(f"✓ Amostras salvas em {self.output_dir}")


# =====================================================================
# DPM-SOLVER++
# =====================================================================
class DPMSolverPP:
    def __init__(self, noise_schedule):
        self.ns = noise_schedule

    def sample(self, model, shape, labels, num_steps=20, guidance_scale=1.5, num_classes=10):
        x = tf.random.normal(shape, dtype=tf.float32)
        sigmas = self.ns.sigmas.numpy()  # aqui pode manter .numpy() porque é um tensor grande
        steps = np.linspace(len(sigmas) - 1, 0, num_steps, dtype=int)

        null_labels = tf.fill([shape[0]], num_classes)

        for i, t_idx in enumerate(steps):  # renomeei para t_idx para evitar confusão
            sigma = float(sigmas[t_idx])
            c_skip, c_out, c_in, c_noise = self.ns.get_scalings(sigma)

            # Agora c_skip, c_out, c_in, c_noise já são floats Python!
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
class FlattenedDiT(keras.Model):
    def __init__(self, img_shape, num_classes, hidden_dim=512, depth=8, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.img_shape = img_shape
        self.vector_dim = np.prod(img_shape)
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.num_heads = num_heads

        # Criar networks
        self.network = DiTNetwork(
            self.vector_dim, self.num_classes, self.hidden_dim, self.depth, self.num_heads
        )
        self.ema_network = DiTNetwork(
            self.vector_dim, self.num_classes, self.hidden_dim, self.depth, self.num_heads
        )

        self.ns = EDMNoiseSchedule()
        self.sampler = DPMSolverPP(self.ns)
        self.ema_decay = 0.9999
        self._initialized = False

    def _initialize_weights(self):
        """Inicializa os pesos das redes com uma passada dummy (deve rodar em modo eager)"""
        if self._initialized:
            return

        print("  • Criando pesos com forward pass dummy...")
        # Usar float32 para inicialização para evitar problemas de tipo
        dummy_vec = tf.zeros((2, self.vector_dim), dtype=tf.float32)
        dummy_t = tf.zeros((2,), dtype=tf.float32)
        dummy_l = tf.zeros((2,), dtype=tf.int64)

        # Forward pass para criar os pesos (deve ser executado em modo eager)
        _ = self.network([dummy_vec, dummy_t, dummy_l])
        _ = self.ema_network([dummy_vec, dummy_t, dummy_l])

        print("  • Copiando pesos para rede EMA...")
        # Copiar pesos para EMA
        self.ema_network.set_weights(self.network.get_weights())

        self._initialized = True
        print("  ✓ Inicialização completa!")

    def train_step(self, data):
        # Verificação de segurança
        if not self._initialized:
            raise RuntimeError("Modelo não foi inicializado! Chame model._initialize_weights() antes do fit().")

        images, labels = data
        batch_size = tf.shape(images)[0]
        vectors = tf.reshape(images, [batch_size, -1])

        t = tf.random.uniform([batch_size], 0, self.ns.num_steps, dtype=tf.int32)
        noisy, _, sigma = self.ns.add_noise(vectors, t)
        c_skip, c_out, c_in, c_noise = self.ns.get_scalings(sigma)

        # Reshape para remover dimensões extras
        c_noise_scalar = tf.reshape(c_noise, [batch_size])

        with tf.GradientTape() as tape:
            pred = self.network([noisy * c_in, c_noise_scalar, labels], training=True)
            target = (vectors - c_skip * noisy) / c_out

            # Loss sempre em float32 para estabilidade
            pred_f32 = tf.cast(pred, tf.float32)
            target_f32 = tf.cast(target, tf.float32)
            loss = tf.reduce_mean(tf.square(pred_f32 - target_f32))

        grads = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_weights))

        # Atualizar EMA
        for w, ema_w in zip(self.network.weights, self.ema_network.weights):
            ema_w.assign(self.ema_decay * ema_w + (1 - self.ema_decay) * w)

        return {"loss": loss}

    def get_samples(self, num_samples_per_class, num_steps=20, guidance_scale=1.5):
        all_samples = []
        for c in range(self.num_classes):
            labels = tf.fill([num_samples_per_class], c)
            shape = (num_samples_per_class, self.vector_dim)
            samples = self.sampler.sample(
                self.ema_network, shape, labels, num_steps, guidance_scale, self.num_classes
            )
            all_samples.append(samples.numpy())
        return np.concatenate(all_samples, axis=0)


# =====================================================================
# MAIN
# =====================================================================
def main():
    setup_environment()

    IMG_SIZE = 64
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
    N_CLASSES = 10
    BATCH_SIZE = 16
    EPOCHS = 10

    # Carregamento dos dados
    from PIL import Image
    dataset_dir = "./50k"

    if not os.path.exists(dataset_dir):
        print(f"⚠ Dataset não encontrado em {dataset_dir}. Usando dados sintéticos.")
        np.random.seed(42)
        x = np.random.uniform(-1, 1, (1000, IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)
        y = np.random.randint(0, N_CLASSES, 1000)
        print(f"✓ Geradas {len(x)} imagens sintéticas para teste")
    else:
        files = sorted(os.listdir(dataset_dir))[:12000]
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

    # Criar modelo
    print("\nCriando modelo...")
    model = FlattenedDiT(IMG_SHAPE, N_CLASSES, hidden_dim=512, depth=8, num_heads=8)
    model.compile(optimizer=keras.optimizers.AdamW(1e-4, weight_decay=0.01))

    # IMPORTANTE: Inicializar pesos ANTES do treinamento
    print("Inicializando pesos do modelo...")
    model._initialize_weights()
    print("✓ Modelo criado e inicializado!")

    # Callback
    callback = ImageGenerationCallback(
        img_shape=IMG_SHAPE,
        num_classes=N_CLASSES,
        samples_per_class=4,
        every_n_epochs=5,
        guidance_scale=1.5
    )

    print(f"\nIniciando treinamento: {len(x)} amostras, {EPOCHS} épocas")
    model.fit(
        tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE),
        epochs=EPOCHS,
        callbacks=[callback]
    )

    # Geração final
    print("\nGerando amostras finais...")
    final_samples = model.get_samples(10)
    final_imgs = final_samples.reshape(-1, *IMG_SHAPE)
    final_imgs = np.clip((final_imgs + 1) / 2, 0, 1)
    np.save("final_samples.npy", final_samples)
    print("✓ Amostras finais salvas em final_samples.npy")
    print("✓ Treinamento concluído! Verifique a pasta 'samples_flattened'")


if __name__ == "__main__":
    main()
