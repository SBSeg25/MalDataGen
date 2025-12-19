import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal


import os
import numpy as np
from PIL import Image
import tensorflow as tf

os.environ["ML_FRAMEWORK"] = "tensorflow"

from Engine.architectures.adversarial.AdversarialModel import AdversarialModel
from Engine.models.Adversarial import Adversarial

from tensorflow.keras.layers import (
    Input, Dense, Flatten, Dropout, Concatenate, Conv2D, Reshape,
    LeakyReLU, Layer, Conv2DTranspose, Add, Multiply,
    GlobalAveragePooling2D, Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# =====================
# Configurações
# =====================
IMAGE_SIZE = (64, 64)
INPUT_SHAPE = (64, 64, 3)
DATASET_DIR = "./50k"
MAX_SAMPLES = 8400
N_CLASSES = 64
BATCH_LIMIT = 8400
LATENT_DIMENSION = 64
CODEBOOK_SIZE = N_CLASSES
CODEBOOK_DIM = 64

OUTPUT_DIR = "./output_mnist"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================
# Carregar Dados
# =====================
x_real_samples = []
image_files = sorted(os.listdir(DATASET_DIR))

print("📂 Carregando imagens...")
for img_name in image_files:
    try:
        img = Image.open(os.path.join(DATASET_DIR, img_name)).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        img = np.asarray(img, dtype=np.float32) / 255.0
        x_real_samples.append(img)
    except Exception:
        continue
    if len(x_real_samples) >= MAX_SAMPLES:
        break

x_real_samples = np.array(x_real_samples, dtype=np.float32)
print(f"✓ {len(x_real_samples)} imagens carregadas")

class VectorQuantizer(Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25,
                 ema_decay=0.99, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.ema_decay = ema_decay
        self.epsilon = epsilon

    def build(self, input_shape):
        initializer = tf.keras.initializers.RandomUniform(
            -1.0 / self.num_embeddings,
             1.0 / self.num_embeddings
        )
        self.embeddings = self.add_weight(
            shape=(self.num_embeddings, self.embedding_dim),
            initializer=initializer,
            trainable=True,
            name="embeddings"
        )
        self.ema_cluster_size = self.add_weight(
            shape=(self.num_embeddings,),
            initializer="zeros",
            trainable=False
        )
        self.ema_w = self.add_weight(
            shape=(self.num_embeddings, self.embedding_dim),
            initializer=initializer,
            trainable=False
        )

    def call(self, x, training=None):
        flat_x = tf.reshape(x, [-1, self.embedding_dim])

        flat_x = tf.nn.l2_normalize(flat_x, axis=-1)
        emb = tf.nn.l2_normalize(self.embeddings, axis=-1)

        distances = -tf.matmul(flat_x, emb, transpose_b=True)
        encoding_indices = tf.argmin(distances, axis=1)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)

        quantized = tf.nn.embedding_lookup(self.embeddings, encoding_indices)
        quantized = tf.reshape(quantized, tf.shape(x))

        e_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        q_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(q_loss + self.beta * e_loss)

        if training:
            enc_sum = tf.reduce_sum(encodings, axis=0)
            self.ema_cluster_size.assign(
                self.ema_decay * self.ema_cluster_size + (1 - self.ema_decay) * enc_sum
            )

            dw = tf.matmul(encodings, flat_x, transpose_a=True)
            self.ema_w.assign(self.ema_decay * self.ema_w + (1 - self.ema_decay) * dw)

            n = tf.reduce_sum(self.ema_cluster_size)
            cluster_size = (
                (self.ema_cluster_size + self.epsilon) /
                (n + self.num_embeddings * self.epsilon) * n
            )
            self.embeddings.assign(self.ema_w / tf.reshape(cluster_size, [-1, 1]))

        quantized = x + tf.stop_gradient(quantized - x)
        spatial_indices = tf.reshape(
            encoding_indices,
            [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]]
        )
        return quantized, spatial_indices

class GroupNormalization(Layer):
    def __init__(self, groups=8, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups
        self.epsilon = epsilon

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels % self.groups != 0:
            raise ValueError(
                f"Channels ({channels}) must be divisible by groups ({self.groups})"
            )

        self.gamma = self.add_weight(
            shape=(channels,),
            initializer="ones",
            trainable=True,
            name="gamma"
        )
        self.beta = self.add_weight(
            shape=(channels,),
            initializer="zeros",
            trainable=True,
            name="beta"
        )

    def call(self, x):
        input_shape = tf.shape(x)
        N = input_shape[0]
        H = input_shape[1]
        W = input_shape[2]
        C = x.shape[-1]  # ⚠️ valor estático, seguro aqui

        x = tf.reshape(
            x,
            [N, H, W, self.groups, C // self.groups]
        )

        mean, var = tf.nn.moments(x, axes=[1, 2, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.epsilon)

        x = tf.reshape(x, [N, H, W, C])
        return x * self.gamma + self.beta

class ResidualBlock(Layer):
    def __init__(self, filters, dropout=0.1):
        super().__init__()
        self.filters = filters
        self.dropout = dropout

    def build(self, shape):
        self.gn1 = GroupNormalization(min(32, self.filters))
        self.gn2 = GroupNormalization(min(32, self.filters))
        self.conv1 = Conv2D(self.filters, 3, padding="same")
        self.conv2 = Conv2D(self.filters, 3, padding="same")
        self.drop = Dropout(self.dropout)
        self.proj = Conv2D(self.filters, 1, padding="same") if shape[-1] != self.filters else None

    def call(self, x, training=False):
        r = x
        x = self.gn1(x)
        x = tf.nn.swish(x)
        x = self.conv1(x)
        x = self.gn2(x)
        x = tf.nn.swish(x)
        x = self.drop(x, training=training)
        x = self.conv2(x)
        if self.proj:
            r = self.proj(r)
        return x + r

class SelfAttention(Layer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def build(self, input_shape):
        self.q = Conv2D(self.channels // 8, 1)
        self.k = Conv2D(self.channels // 8, 1)
        self.v = Conv2D(self.channels, 1)
        self.out = Conv2D(self.channels, 1)
        self.gamma = self.add_weight(
            shape=(),
            initializer="zeros",
            trainable=True
        )

    def call(self, x):
        # Shapes dinâmicos
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]

        # Shape estático (canal)
        channels = x.shape[-1]

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = tf.reshape(q, [batch_size, height * width, -1])
        k = tf.reshape(k, [batch_size, height * width, -1])
        v = tf.reshape(v, [batch_size, height * width, channels])

        scale = tf.sqrt(tf.cast(tf.shape(q)[-1], tf.float32))
        attn = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / scale)

        out = tf.matmul(attn, v)
        out = tf.reshape(out, [batch_size, height, width, channels])

        return x + self.gamma * self.out(out)
def downsample(x, f, attn=False):
    x = Conv2D(f, 4, strides=2, padding="same", activation="swish")(x)
    x = ResidualBlock(f)(x)
    if attn:
        x = SelfAttention(f)(x)
    return x

def upsample(x, f):
    x = Conv2DTranspose(f, 4, strides=2, padding="same", activation="swish")(x)
    x = ResidualBlock(f)(x)
    return x

def build_vqvae():
    inp = Input(INPUT_SHAPE)
    x = Conv2D(16, 3, padding="same")(inp)
    x = downsample(x, 32)
    x = downsample(x, 64)
    x = downsample(x, 256, attn=True)
    x = ResidualBlock(128)(x)
    x = SelfAttention(128)(x)
    x = GroupNormalization()(x)
    x = Activation("swish")(x)
    z = Conv2D(CODEBOOK_DIM, 1, padding="same")(x)

    quant = VectorQuantizer(CODEBOOK_SIZE, CODEBOOK_DIM)
    zq, idx = quant(z)

    x = Conv2D(128, 3, padding="same")(zq)
    x = ResidualBlock(128)(x)
    x = upsample(x, 64)
    x = upsample(x, 32)
    x = upsample(x, 16)
    x = GroupNormalization()(x)
    x = Activation("swish")(x)
    out = Conv2D(3, 1, activation="sigmoid")(x)

    return Model(inp, out), Model(inp, idx)

vqvae, vqvae_indices = build_vqvae()
vqvae.compile(
    optimizer=tf.keras.optimizers.Adam(5e-4),
    loss="mse",
    metrics=["mae"]
)

print("✓ VQ-VAE pronto")

vqvae.fit(
    x_real_samples,
    x_real_samples,
    epochs=50,
    batch_size=128,
    validation_split=0.15,
    callbacks=[
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(patience=10)
    ]
)





















class MyClass(AdversarialModel):

    @staticmethod
    def get_discriminator():
        image_shape = (32, 32, 3)
        number_classes = 64
        last_layer_activation = "sigmoid"
        dense_layer_sizes = (256, 128)
        dropout_rate = 0.3
        initializer_mean = 0.0
        initializer_deviation = 0.02
        dataset_type = None
        """
        Build and return a 2D convolutional discriminator model.

        Input format: image_shape (64x64x3)
        """

        if dataset_type is None:
            import numpy as np
            dataset_type = np.float32

        initialization = RandomNormal(mean=initializer_mean, stddev=initializer_deviation)
        image_input = Input(shape=image_shape, dtype=dataset_type, name="image_input")
        label_input = Input(shape=(number_classes,), dtype=dataset_type, name="label_input")
        label_embedding = Dense(image_shape[0] * image_shape[1], kernel_initializer=initialization)(label_input)
        label_embedding = LeakyReLU(alpha=0.2)(label_embedding)
        label_embedding = Reshape((image_shape[0], image_shape[1], 1))(label_embedding)

        x = Concatenate(axis=-1)([image_input, label_embedding])

        x = Conv2D(
            32,
            kernel_size=4,
            strides=2,
            padding="same",
            kernel_initializer=initialization
        )(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout_rate)(x)

        x = Conv2D(
            64,
            kernel_size=4,
            strides=2,
            padding="same",
            kernel_initializer=initialization
        )(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout_rate)(x)

        x = Conv2D(
            128,
            kernel_size=4,
            strides=2,
            padding="same",
            kernel_initializer=initialization
        )(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout_rate)(x)

        x = Conv2D(
            256,
            kernel_size=4,
            strides=2,
            padding="same",
            kernel_initializer=initialization
        )(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout_rate)(x)

        # ------------------------------------------------------------------
        # Classification head
        # ------------------------------------------------------------------
        x = Flatten()(x)

        for units in dense_layer_sizes:
            x = Dense(units, kernel_initializer=initialization)(x)
            x = Dropout(dropout_rate)(x)
            x = LeakyReLU(alpha=0.2)(x)

        x = Dense(1, kernel_initializer=initialization)(x)

        if last_layer_activation == "sigmoid":
            from tensorflow.keras.activations import sigmoid
            validity = sigmoid(x)
        else:
            validity = x

        # ------------------------------------------------------------------
        # Final model
        # ------------------------------------------------------------------
        model = Model(
            inputs=[image_input, label_input],
            outputs=validity,
            name="Discriminator_64x64x3"
        )

        return model

    @staticmethod
    def get_generator():
        """
        Builds and returns a 2D convolutional generator model.

        Output format: 64x64x3
        """
        latent_dimension =  LATENT_DIMENSION
        number_classes =  64
        activation_function="relu"
        last_layer_activation="tanh"
        initializer_mean=0.0
        initializer_deviation=0.02
        dataset_type=None

        from tensorflow.keras.layers import (
            Input, Dense, Concatenate, Reshape,
            Conv2DTranspose, BatchNormalization
        )
        from tensorflow.keras.models import Model
        from tensorflow.keras.initializers import RandomNormal
        from tensorflow.keras.layers import ReLU, LeakyReLU
        import numpy as np

        if dataset_type is None:
            dataset_type = np.float32

        initialization = RandomNormal(
            mean=initializer_mean,
            stddev=initializer_deviation
        )

        # ------------------------------------------------------------------
        # Inputs
        # ------------------------------------------------------------------
        latent_input = Input(
            shape=(latent_dimension,),
            dtype=dataset_type,
            name="latent_input"
        )

        label_input = Input(
            shape=(number_classes,),
            dtype=dataset_type,
            name="label_input"
        )

        # ------------------------------------------------------------------
        # Conditional latent embedding
        # ------------------------------------------------------------------
        x = Concatenate()([latent_input, label_input])

        x = Dense(
            4 * 4 * 256,
            kernel_initializer=initialization
        )(x)

        if activation_function == "leaky_relu":
            x = LeakyReLU(alpha=0.2)(x)
        else:
            x = ReLU()(x)

        x = Reshape((4, 4, 256))(x)

        # ------------------------------------------------------------------
        # Upsampling blocks: 4 → 8 → 16 → 32 → 64
        # ------------------------------------------------------------------
        x = Conv2DTranspose(
            128, kernel_size=4, strides=2, padding="same",
            kernel_initializer=initialization
        )(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2DTranspose(
            64, kernel_size=4, strides=2, padding="same",
            kernel_initializer=initialization
        )(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2DTranspose(
            32, kernel_size=4, strides=2, padding="same",
            kernel_initializer=initialization
        )(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # ------------------------------------------------------------------
        # Output layer (64x64x3)
        # ------------------------------------------------------------------
        generator_output = Conv2DTranspose(
            3,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=initialization
        )(x)

        if last_layer_activation == "tanh":
            from tensorflow.keras.activations import tanh
            generator_output = tanh(generator_output)
        elif last_layer_activation == "sigmoid":
            from tensorflow.keras.activations import sigmoid
            generator_output = sigmoid(generator_output)

        generator_model = Model(
            inputs=[latent_input, label_input],
            outputs=generator_output,
            name="Generator_64x64x3"
        )

        return generator_model





































print("✓ Treinamento do VQ-VAE concluído")

# =====================
# Extrair Índices do Codebook
# =====================
print("\n🎯 Extraindo reconstruções e índices do codebook...")

# Reconstruções
reconstructed = vqvae.predict(
    x_real_samples,
    verbose=0,
    batch_size=64
)

# Índices espaciais do codebook
indices_spatial = vqvae_indices.predict(
    x_real_samples,
    verbose=0,
    batch_size=64
)

print(f"  Shape dos índices espaciais: {indices_spatial.shape}")
print(f"  Shape dos índices espaciais: {indices_spatial.shape}")

cluster_labels = []
for i in range(len(indices_spatial)):
    flat_indices = indices_spatial[i].flatten()
    unique_vals, counts = np.unique(flat_indices, return_counts=True)
    most_common_idx = unique_vals[np.argmax(counts)]
    cluster_labels.append(most_common_idx)

cluster_labels = np.array(cluster_labels, dtype=np.int32)

print(f"✓ Índices extraídos: {cluster_labels.shape}")
print(f"  Distribuição dos códigos:")
unique, counts = np.unique(cluster_labels, return_counts=True)
for code_id, count in zip(unique, counts):
    print(f"    Código {code_id}: {count} imagens ({count / len(cluster_labels) * 100:.1f}%)")

print(f"\n  Códigos utilizados: {len(unique)}/{CODEBOOK_SIZE}")
if len(unique) < CODEBOOK_SIZE:
    print(f"  ⚠️ Atenção: {CODEBOOK_SIZE - len(unique)} códigos não foram usados")

y_real_samples = cluster_labels

# =====================
# Visualização
# =====================
try:
    import matplotlib.pyplot as plt

    print("\n📊 Visualizando agrupamento por códigos do codebook...")
    fig, axes = plt.subplots(N_CLASSES, 10, figsize=(20, 2 * N_CLASSES))
    fig.suptitle('Amostras por Código do Codebook (VQ-VAE SOTA)', fontsize=16, fontweight='bold')

    for code_id in range(N_CLASSES):
        code_indices = np.where(cluster_labels == code_id)[0]

        if len(code_indices) > 0:
            samples_to_show = np.random.choice(
                code_indices,
                min(10, len(code_indices)),
                replace=False
            )

            for idx, sample_idx in enumerate(samples_to_show):
                axes[code_id, idx].imshow(x_real_samples[sample_idx])
                axes[code_id, idx].axis('off')
                if idx == 0:
                    axes[code_id, idx].set_title(
                        f'Código {code_id} ({len(code_indices)})',
                        fontsize=10,
                        fontweight='bold'
                    )
        else:
            for idx in range(10):
                axes[code_id, idx].axis('off')
                if idx == 0:
                    axes[code_id, idx].text(
                        0.5, 0.5,
                        f'Código {code_id}\n(não usado)',
                        ha='center',
                        va='center',
                        fontsize=9
                    )

    plt.tight_layout()
    viz_path = os.path.join(OUTPUT_DIR, "codebook_visualization.png")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Visualização salva: {viz_path}")

    # Reconstruções
    n_examples = 8
    fig, axes = plt.subplots(2, n_examples, figsize=(16, 4))
    fig.suptitle('Original vs Reconstruído (VQ-VAE SOTA)', fontsize=14, fontweight='bold')

    for i in range(n_examples):
        axes[0, i].imshow(x_real_samples[i])
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12)

        axes[1, i].imshow(reconstructed[i])
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Reconstruído', fontsize=12)

    plt.tight_layout()
    recon_path = os.path.join(OUTPUT_DIR, "vqvae_reconstruction.png")
    plt.savefig(recon_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Reconstruções salvas: {recon_path}")

    # Histograma de perda
    reconstruction_loss = np.mean(np.square(x_real_samples - reconstructed), axis=(1, 2, 3))

    plt.figure(figsize=(10, 5))
    plt.hist(reconstruction_loss, bins=50, edgecolor='black')
    plt.title('Distribuição da Perda de Reconstrução', fontsize=14, fontweight='bold')
    plt.xlabel('MSE')
    plt.ylabel('Frequência')
    plt.grid(alpha=0.3)
    loss_path = os.path.join(OUTPUT_DIR, "reconstruction_loss_distribution.png")
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Histograma salvo: {loss_path}")

    matplotlib_available = True
except ImportError:
    matplotlib_available = False
    print("⚠️ Matplotlib não disponível - pulando visualização")

# =====================
# GAN (mantido do código original)
# =====================
number_samples_per_class = {
    "number_classes": N_CLASSES,
    "classes": {i: 8 for i in range(N_CLASSES)}
}

models = {
    "adversarial": Adversarial(
        number_classes=N_CLASSES,
        number_samples_per_class=number_samples_per_class,
        model=MyClass
    ),
}

for model_name, model in models.items():
    print(f"\n{'=' * 60}")
    print(f"🔧 Modelo: {model_name}")
    print(f"{'=' * 60}")

    print(f"⏳ Treinando {model_name}...")
    model.fit_model(
        input_shape=INPUT_SHAPE,
        x_real_samples=x_real_samples,
        y_real_samples=y_real_samples,
        flatten=False
    )
    print(f"✓ Treinamento concluído")

    print(f"🎨 Gerando amostras sintéticas...")
    synthetic_samples = model.get_samples(number_samples_per_class)
    print(f"✓ Geradas {synthetic_samples.shape[0]} amostras")

    if matplotlib_available:
        n = min(36, synthetic_samples.shape[0])
        cols = 6
        rows = n // cols

        fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
        fig.suptitle(f"Amostras Sintéticas - {model_name} (VQ-VAE SOTA)", fontsize=16, fontweight='bold')

        idx = 0
        for i in range(rows):
            for j in range(cols):
                axes[i, j].imshow(synthetic_samples[idx].squeeze(), cmap="gray")
                axes[i, j].axis("off")
                idx += 1

        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f"{model_name}_samples.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"💾 Imagem salva: {output_path}")

    samples_path = os.path.join(OUTPUT_DIR, f"{model_name}_samples.npy")
    np.save(samples_path, synthetic_samples)
    print(f"💾 Arrays salvos: {samples_path}")

print(f"\n{'=' * 60}")
print(f"✅ Processamento completo!")
print(f"📁 Resultados salvos em: {OUTPUT_DIR}")
print(f"🔖 Codebook treinado com {CODEBOOK_SIZE} vetores")
print(f"📊 Arquitetura: VQ-VAE Estado da Arte")
print(f"   - Blocos Residuais com Group Normalization")
print(f"   - Self-Attention Multi-Escala")
print(f"   - Channel Attention (Squeeze-Excitation)")
print(f"   - Total parâmetros: {vqvae.count_params():,}")
print(f"{'=' * 60}")