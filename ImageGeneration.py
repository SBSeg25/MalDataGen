import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal

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

# Verificar matplotlib
try:
    import matplotlib.pyplot as plt

    matplotlib_available = True
except ImportError:
    matplotlib_available = False
    print("⚠️ Matplotlib não disponível - visualizações desabilitadas")

# =====================
# Configurações
# =====================
IMAGE_SIZE = (32, 32)
INPUT_SHAPE = (32, 32, 3)
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
print(f"✓ Carregadas {len(x_real_samples)} imagens")

# =====================
# Gerar Rótulos Aleatórios (0-63)
# =====================
np.random.seed(42)  # Para reprodutibilidade
y_real_samples_tuples = np.random.randint(0, N_CLASSES, size=len(x_real_samples))
print(f"✓ Gerados {len(y_real_samples_tuples)} rótulos aleatórios (0-{N_CLASSES - 1})")

number_samples_per_class = {
    "number_classes": N_CLASSES,
    "classes": {i: len(y_real_samples_tuples[y_real_samples_tuples == i]) for i in range(N_CLASSES)}
}

print(f"\n📊 Distribuição de classes:")
for i in range(min(10, N_CLASSES)):  # Mostrar primeiras 10 classes
    print(f"  Classe {i}: {number_samples_per_class['classes'][i]} amostras")
if N_CLASSES > 10:
    print(f"  ... (total de {N_CLASSES} classes)")

models = {
    "adversarial": Adversarial(
        number_classes=N_CLASSES,
        number_samples_per_class=number_samples_per_class,
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
        y_real_samples=y_real_samples_tuples,
        flatten=True
    )
    print(f"✓ Treinamento concluído")

    print(f"🎨 Gerando amostras sintéticas...")

    # Definir quantas amostras gerar por classe
    samples_per_class_dict = {
        "number_classes": N_CLASSES,
        "classes": {i: 8 for i in range(N_CLASSES)}  # 8 amostras por classe
    }

    synthetic_samples = model.get_samples(samples_per_class_dict)
    print(f"✓ Geradas {synthetic_samples.shape[0]} amostras")

    if matplotlib_available:
        n = min(36, synthetic_samples.shape[0])
        cols = 6
        rows = n // cols

        fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
        fig.suptitle(f"Amostras Sintéticas - {model_name}", fontsize=16, fontweight='bold')

        idx = 0
        for i in range(rows):
            for j in range(cols):
                # Ajustar visualização para imagens RGB
                img = synthetic_samples[idx]
                if img.shape[-1] == 3:  # RGB
                    axes[i, j].imshow(img)
                else:  # Grayscale
                    axes[i, j].imshow(img.squeeze(), cmap="gray")
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

print("\n" + "=" * 60)
print("✅ Pipeline concluído com sucesso!")
print("=" * 60)