#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exemplo Condicional: Geração de Imagens 128x128x3
Dataset: CIFAR-10 (torchvision)
Estável, pequeno e ideal para Latent Diffusion
"""

import os
import numpy as np
os.environ["ML_FRAMEWORK"] = "pytorch"
from Engine.Models.VariationalAutoencoder import VariationalAutoencoder





IMAGE_SIZE = (32, 32)
INPUT_SHAPE = (32, 32, 3)

N_CLASSES = 10
BATCH_LIMIT = 10000


try:
    from torchvision.datasets import CIFAR10
except ImportError:
    raise ImportError("Instale torchvision: pip install torchvision")

from PIL import Image

dataset = CIFAR10(
    root="./data",
    train=True,
    download=True
)

x_real_samples = []
y_real_samples = []

for img, label in dataset:
    # CIFAR-10 já vem como PIL.Image
    img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img = np.asarray(img, dtype=np.float32) / 255.0

    x_real_samples.append(img)
    y_real_samples.append(int(label))

    if len(x_real_samples) >= BATCH_LIMIT:
        break

x_real_samples = np.array(x_real_samples, dtype=np.float32)
y_real_samples = np.array(y_real_samples, dtype=np.int32)


model = VariationalAutoencoder(number_classes=N_CLASSES)


model.fit_model(
    input_shape=INPUT_SHAPE,
    x_real_samples=x_real_samples,
    y_real_samples=y_real_samples
)

number_samples_per_class = {
    "number_classes": N_CLASSES,
    "classes": {i: 8 for i in range(N_CLASSES)}
}

synthetic_samples = model.get_samples(number_samples_per_class)


try:
    import matplotlib.pyplot as plt

    n = min(25, synthetic_samples.shape[0])
    cols = 5
    rows = n // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    fig.suptitle("CIFAR-10 Sintético (128x128)", fontsize=16)

    idx = 0
    for i in range(rows):
        for j in range(cols):
            axes[i, j].imshow(synthetic_samples[idx])
            axes[i, j].axis("off")
            idx += 1

    plt.tight_layout()
    plt.show()
    print("✓ Visualização concluída")

except ImportError:
    print("⚠️ Matplotlib não disponível")
