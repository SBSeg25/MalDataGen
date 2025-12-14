#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exemplo Condicional: Geração de Imagens CIFAR-10
Dataset: CIFAR-10 (torchvision)
Ideal para validar VAEs / GANs condicionais em imagens RGB
"""

import os
import numpy as np

from Engine.Models.WassersteinGP import WassersteinGP

os.environ["ML_FRAMEWORK"] = "tensorflow"

from Engine.Models.Adversarial import Adversarial
from Engine.Models.Autoencoder import Autoencoder
from Engine.Models.VariationalAutoencoder import VariationalAutoencoder

# =====================
# Configurações CIFAR-10
# =====================

IMAGE_SIZE = (16, 16)
INPUT_SHAPE = (16, 16, 3)

N_CLASSES = 10
BATCH_LIMIT = 2200

try:
    from torchvision.datasets import CIFAR10
except ImportError:
    raise ImportError("Instale torchvision: pip install torchvision")

from PIL import Image

# =====================
# Carregamento Dataset
# =====================

dataset = CIFAR10(
    root="./data",
    train=True,
    download=True
)

x_real_samples = []
y_real_samples = []

for img, label in dataset:
    # CIFAR-10 vem como PIL.Image em RGB
    img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE)

    img = np.asarray(img, dtype=np.float32) / 255.0  # normalização [0,1]

    x_real_samples.append(img)
    y_real_samples.append(int(label))

    if len(x_real_samples) >= BATCH_LIMIT:
        break

x_real_samples = np.array(x_real_samples, dtype=np.float32)
y_real_samples = np.array(y_real_samples, dtype=np.int32)

# =====================
# Modelo
# =====================

model = WassersteinGP(number_classes=N_CLASSES)

model.fit_model(
    input_shape=INPUT_SHAPE,
    x_real_samples=x_real_samples,
    y_real_samples=y_real_samples
)

# =====================
# Geração Condicional
# =====================

number_samples_per_class = {
    "number_classes": N_CLASSES,
    "classes": {i: 8 for i in range(N_CLASSES)}
}

synthetic_samples = model.get_samples(number_samples_per_class)

# =====================
# Salvar Imagens
# =====================

# Criar diretório de saída
output_dir = "./output_cifar10"
os.makedirs(output_dir, exist_ok=True)

try:
    import matplotlib.pyplot as plt

    # Salvar grade de imagens
    n = min(36, synthetic_samples.shape[0])
    cols = 6
    rows = n // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    fig.suptitle("CIFAR-10 Sintético (Modelo Condicional)", fontsize=16)

    idx = 0
    for i in range(rows):
        for j in range(cols):
            axes[i, j].imshow(synthetic_samples[idx])
            axes[i, j].axis("off")
            idx += 1

    plt.tight_layout()

    # Salvar a grade completa
    grid_path = os.path.join(output_dir, "cifar10_grid.png")
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Grade salva em: {grid_path}")

    # Salvar imagens individuais
    for idx, sample in enumerate(synthetic_samples):
        img_array = (sample * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_array, mode='RGB')
        img_path = os.path.join(output_dir, f"cifar10_sample_{idx:03d}.png")
        img_pil.save(img_path)

    print(f"✓ {len(synthetic_samples)} imagens individuais salvas em: {output_dir}")
    print(f"✓ Processo concluído!")

except ImportError:
    print("⚠️ Matplotlib não disponível")