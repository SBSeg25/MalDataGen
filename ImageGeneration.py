#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exemplo Condicional: Geração de Imagens MNIST
Dataset: MNIST (torchvision)
Ideal para validar VAEs condicionais
"""

import os
import numpy as np

from Engine.Models.Autoencoder import Autoencoder
from Engine.Models.VariationalAutoencoder import VariationalAutoencoder
os.environ["ML_FRAMEWORK"] = "tensorflow"


# =====================
# Configurações MNIST
# =====================

IMAGE_SIZE = (28, 28)
INPUT_SHAPE = (28, 28, 1)

N_CLASSES = 10
BATCH_LIMIT = 20000


try:
    from torchvision.datasets import MNIST
except ImportError:
    raise ImportError("Instale torchvision: pip install torchvision")

from PIL import Image


# =====================
# Carregamento Dataset
# =====================

dataset = MNIST(
    root="./data",
    train=True,
    download=True
)

x_real_samples = []
y_real_samples = []

for img, label in dataset:
    # MNIST vem como PIL.Image em grayscale
    img = img.convert("L")  # garante 1 canal
    img = img.resize(IMAGE_SIZE)

    img = np.asarray(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)  # (28, 28, 1)

    x_real_samples.append(img)
    y_real_samples.append(int(label))

    if len(x_real_samples) >= BATCH_LIMIT:
        break

x_real_samples = np.array(x_real_samples, dtype=np.float32)
y_real_samples = np.array(y_real_samples, dtype=np.int32)


# =====================
# Modelo
# =====================

model = Autoencoder(number_classes=N_CLASSES)

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
# Visualização
# =====================

try:
    import matplotlib.pyplot as plt

    n = min(25, synthetic_samples.shape[0])
    cols = 5
    rows = n // cols

    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    fig.suptitle("MNIST Sintético (VAE Condicional)", fontsize=16)

    idx = 0
    for i in range(rows):
        for j in range(cols):
            axes[i, j].imshow(
                synthetic_samples[idx].squeeze(),
                cmap="gray"
            )
            axes[i, j].axis("off")
            idx += 1

    plt.tight_layout()
    plt.show()
    print("✓ Visualização concluída")

except ImportError:
    print("⚠️ Matplotlib não disponível")
