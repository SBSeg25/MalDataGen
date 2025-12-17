#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exemplo Condicional: Geração de Imagens MNIST - Todos os Algoritmos
Dataset: MNIST (torchvision)
Testa todos os modelos disponíveis e salva resultados
"""

import os
import numpy as np

os.environ["ML_FRAMEWORK"] = "tensorflow"

from Engine.models.QuantizedVAE import QuantizedVAE
from Engine.models.LatentDiffusion import LatentDiffusion
from Engine.models.Wasserstein import Wasserstein
from Engine.models.WassersteinGP import WassersteinGP
from Engine.models.Adversarial import Adversarial
from Engine.models.Autoencoder import Autoencoder
from Engine.models.VariationalAutoencoder import VariationalAutoencoder
from Engine.models.DenoisingDiffusion import DenoisingDiffusion

# =====================
# Configurações MNIST
# =====================

IMAGE_SIZE = (16, 16)
INPUT_SHAPE = (16, 16, 1)

N_CLASSES = 10
BATCH_LIMIT = 4800

OUTPUT_DIR = "./output_mnist"
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    from torchvision.datasets import MNIST
except ImportError:
    raise ImportError("Instale torchvision: pip install torchvision")

from PIL import Image

# =====================
# Carregamento Dataset
# =====================

print("📦 Carregando dataset MNIST...")

dataset = MNIST(
    root="./data",
    train=True,
    download=True
)

x_real_samples = []
y_real_samples = []

for img, label in dataset:
    img = img.convert("L")
    img = img.resize(IMAGE_SIZE)

    img = np.asarray(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)

    x_real_samples.append(img)
    y_real_samples.append(int(label))

    if len(x_real_samples) >= BATCH_LIMIT:
        break

x_real_samples = np.array(x_real_samples, dtype=np.float32)
y_real_samples = np.array(y_real_samples, dtype=np.int32)

print(f"✓ Dataset carregado: {x_real_samples.shape}")

# =====================
# Definição dos Modelos
# =====================

models = {
    "latent_diffusion": LatentDiffusion(number_classes=N_CLASSES),
    #"quantized_vae": QuantizedVAE(number_classes=N_CLASSES),
    #"denoising_diffusion": DenoisingDiffusion(number_classes=N_CLASSES),
    #"autoencoder": Autoencoder(number_classes=N_CLASSES),
    # "adversarial": adversarial(number_classes=N_CLASSES),
    #"wasserstein": Wasserstein(number_classes=N_CLASSES),
    #"wasserstein_gp": WassersteinGP(number_classes=N_CLASSES),
    #"variational_autoencoder": VariationalAutoencoder(number_classes=N_CLASSES)
}

# =====================
# Configuração de Geração
# =====================

number_samples_per_class = {
    "number_classes": N_CLASSES,
    "classes": {i: 8 for i in range(N_CLASSES)}
}

# =====================
# Treinamento e Geração
# =====================

try:
    import matplotlib.pyplot as plt

    matplotlib_available = True
except ImportError:
    matplotlib_available = False
    print("⚠️ Matplotlib não disponível - pulando visualização")

for model_name, model in models.items():
    print(f"\n{'=' * 60}")
    print(f"🔧 Modelo: {model_name}")
    print(f"{'=' * 60}")

    try:
        # Treinamento
        print(f"⏳ Treinando {model_name}...")
        model.fit_model(
            input_shape=INPUT_SHAPE,
            x_real_samples=x_real_samples,
            y_real_samples=y_real_samples
        )
        print(f"✓ Treinamento concluído")

        # Geração
        print(f"🎨 Gerando amostras sintéticas...")
        synthetic_samples = model.get_samples(number_samples_per_class)
        print(f"✓ Geradas {synthetic_samples.shape[0]} amostras")

        # Visualização e Salvamento
        if matplotlib_available:
            n = min(36, synthetic_samples.shape[0])
            cols = 6
            rows = n // cols

            fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
            fig.suptitle(f"MNIST Sintético - {model_name}", fontsize=16, fontweight='bold')

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

            # Salvar imagem
            output_path = os.path.join(OUTPUT_DIR, f"{model_name}_samples.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"💾 Imagem salva: {output_path}")

        # Salvar amostras como arrays NumPy
        samples_path = os.path.join(OUTPUT_DIR, f"{model_name}_samples.npy")
        np.save(samples_path, synthetic_samples)
        print(f"💾 Arrays salvos: {samples_path}")

    except Exception as e:
        print(f"❌ Erro ao processar {model_name}: {str(e)}")
        continue

print(f"\n{'=' * 60}")
print(f"✅ Processamento completo!")
print(f"📁 Resultados salvos em: {OUTPUT_DIR}")
print(f"{'=' * 60}")








# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#
# """
# Exemplo Condicional: Geração de Imagens MNIST
# Dataset: MNIST (torchvision)
# Ideal para validar VAEs condicionais
# """
#
# import os
# import numpy as np
# os.environ["ML_FRAMEWORK"] = "pytorch"
# from Engine.models.latent_diffusion import latent_diffusion
# from Engine.models.wasserstein import wasserstein
# from Engine.models.wasserstein_gp import wasserstein_gp
#
#
# from Engine.models.adversarial import adversarial
# from Engine.models.autoencoder import autoencoder
# from Engine.models.variational_autoencoder import variational_autoencoder
#
#
#
# # =====================
# # Configurações MNIST
# # =====================
#
# IMAGE_SIZE = (16, 16)
# INPUT_SHAPE = (16, 16, 1)
#
# N_CLASSES = 10
# BATCH_LIMIT = 1200
#
#
# try:
#     from torchvision.datasets import MNIST
# except ImportError:
#     raise ImportError("Instale torchvision: pip install torchvision")
#
# from PIL import Image
#
#
# # =====================
# # Carregamento Dataset
# # =====================
#
# dataset = MNIST(
#     root="./data",
#     train=True,
#     download=True
# )
#
# x_real_samples = []
# y_real_samples = []
#
# for img, label in dataset:
#     # MNIST vem como PIL.Image em grayscale
#     img = img.convert("L")  # garante 1 canal
#     img = img.resize(IMAGE_SIZE)
#
#     img = np.asarray(img, dtype=np.float32) / 255.0
#     img = np.expand_dims(img, axis=-1)  # (28, 28, 1)
#
#     x_real_samples.append(img)
#     y_real_samples.append(int(label))
#
#     if len(x_real_samples) >= BATCH_LIMIT:
#         break
#
# x_real_samples = np.array(x_real_samples, dtype=np.float32)
# y_real_samples = np.array(y_real_samples, dtype=np.int32)
#
#
# # =====================
# # Modelo
# # =====================
#
# model = latent_diffusion(number_classes=N_CLASSES)
#
# model.fit_model(
#     input_shape=INPUT_SHAPE,
#     x_real_samples=x_real_samples,
#     y_real_samples=y_real_samples
# )
#
#
# # =====================
# # Geração Condicional
# # =====================
#
# number_samples_per_class = {
#     "number_classes": N_CLASSES,
#     "classes": {i: 8 for i in range(N_CLASSES)}
# }
#
# synthetic_samples = model.get_samples(number_samples_per_class)
#
#
# # =====================
# # Visualização
# # =====================
#
# try:
#     import matplotlib.pyplot as plt
#
#     n = min(36, synthetic_samples.shape[0])
#     cols = 6
#     rows = n // cols
#
#     fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
#     fig.suptitle("MNIST Sintético (VAE Condicional)", fontsize=16)
#
#     idx = 0
#     for i in range(rows):
#         for j in range(cols):
#             axes[i, j].imshow(
#                 synthetic_samples[idx].squeeze(),
#                 cmap="gray"
#             )
#             axes[i, j].axis("off")
#             idx += 1
#
#     plt.tight_layout()
#     plt.show()
#     print("✓ Visualização concluída")
#
# except ImportError:
#     print("⚠️ Matplotlib não disponível")
#
#
#
# # #!/usr/bin/env python3
# # # -*- coding: utf-8 -*-
# #
# # """
# # Exemplo Condicional: Geração de Imagens CIFAR-10
# # Dataset: CIFAR-10 (torchvision)
# # Ideal para validar VAEs / GANs condicionais em imagens RGB
# # """
#
# # import os
# # import numpy as np
# # os.environ["ML_FRAMEWORK"] = "pytorch"
# #
# # from Engine.models.adversarial import adversarial
# # from Engine.models.autoencoder import autoencoder
# # from Engine.models.variational_autoencoder import variational_autoencoder
# #
# #
# # # =====================
# # # Configurações CIFAR-10
# # # =====================
# #
# # IMAGE_SIZE = (32, 32)
# # INPUT_SHAPE = (32, 32, 3)
# #
# # N_CLASSES = 10
# # BATCH_LIMIT = 4200
# #
# #
# # try:
# #     from torchvision.datasets import CIFAR10
# # except ImportError:
# #     raise ImportError("Instale torchvision: pip install torchvision")
# #
# # from PIL import Image
# #
# #
# # # =====================
# # # Carregamento Dataset
# # # =====================
# #
# # dataset = CIFAR10(
# #     root="./data",
# #     train=True,
# #     download=True
# # )
# #
# # x_real_samples = []
# # y_real_samples = []
# #
# # for img, label in dataset:
# #     # CIFAR-10 vem como PIL.Image em RGB
# #     img = img.convert("RGB")
# #     img = img.resize(IMAGE_SIZE)
# #
# #     img = np.asarray(img, dtype=np.float32) / 255.0  # normalização [0,1]
# #
# #     x_real_samples.append(img)
# #     y_real_samples.append(int(label))
# #
# #     if len(x_real_samples) >= BATCH_LIMIT:
# #         break
# #
# # x_real_samples = np.array(x_real_samples, dtype=np.float32)
# # y_real_samples = np.array(y_real_samples, dtype=np.int32)
# #
# #
# # # =====================
# # # Modelo
# # # =====================
# #
# # model = latent_diffusion(number_classes=N_CLASSES)
# #
# # model.fit_model(
# #     input_shape=INPUT_SHAPE,
# #     x_real_samples=x_real_samples,
# #     y_real_samples=y_real_samples
# # )
# #
# #
# # # =====================
# # # Geração Condicional
# # # =====================
# #
# # number_samples_per_class = {
# #     "number_classes": N_CLASSES,
# #     "classes": {i: 8 for i in range(N_CLASSES)}
# # }
# #
# # synthetic_samples = model.get_samples(number_samples_per_class)
# #
# #
# # # =====================
# # # Visualização
# # # =====================
# #
# # try:
# #     import matplotlib.pyplot as plt
# #
# #     n = min(36, synthetic_samples.shape[0])
# #     cols = 6
# #     rows = n // cols
# #
# #     fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
# #     fig.suptitle("CIFAR-10 Sintético (Modelo Condicional)", fontsize=16)
# #
# #     idx = 0
# #     for i in range(rows):
# #         for j in range(cols):
# #             axes[i, j].imshow(synthetic_samples[idx])
# #             axes[i, j].axis("off")
# #             idx += 1
# #
# #     plt.tight_layout()
# #     plt.show()
# #     print("✓ Visualização concluída")
# #
# # except ImportError:
# #     print("⚠️ Matplotlib não disponível")
