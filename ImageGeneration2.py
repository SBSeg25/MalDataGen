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