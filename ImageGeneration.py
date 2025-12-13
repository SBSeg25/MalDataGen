#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exemplo Condicional: Autoencoder gerando Cubos e Esferas 3D
Demonstra geração condicional de formas geométricas 3D
"""

import numpy as np

from Engine.Models.Adversarial import Adversarial
from Engine.Models.Autoencoder import Autoencoder
import os

from Engine.Models.VariationalAutoencoder import VariationalAutoencoder
from Engine.Models.Wasserstein import Wasserstein

os.environ["ML_FRAMEWORK"] = "pytorch"

# ========================================================================
# FUNÇÕES PARA GERAR FORMAS GEOMÉTRICAS 3D
# ========================================================================

def generate_cube(size, grid_size, noise_level=0.05):
    """
    Gera um cubo 3D no centro do volume

    Args:
        size: Tamanho do cubo (edge length)
        grid_size: Tamanho do grid 3D (depth, height, width)
        noise_level: Nível de ruído a adicionar
    """
    volume = np.zeros(grid_size, dtype=np.float32)
    center = grid_size[0] // 2
    half_size = size // 2

    # Criar cubo
    start = center - half_size
    end = center + half_size
    volume[start:end, start:end, start:end] = 1.0

    # Adicionar ruído
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, grid_size)
        volume = np.clip(volume + noise, 0, 1)

    return volume.astype(np.float32)


def generate_sphere(radius, grid_size, noise_level=0.05):
    """
    Gera uma esfera 3D no centro do volume

    Args:
        radius: Raio da esfera
        grid_size: Tamanho do grid 3D (depth, height, width)
        noise_level: Nível de ruído a adicionar
    """
    volume = np.zeros(grid_size, dtype=np.float32)
    center = grid_size[0] // 2

    # Criar coordenadas
    z, y, x = np.ogrid[:grid_size[0], :grid_size[1], :grid_size[2]]

    # Calcular distância do centro
    distance = np.sqrt((z - center) ** 2 + (y - center) ** 2 + (x - center) ** 2)

    # Criar esfera
    volume[distance <= radius] = 1.0

    # Adicionar ruído
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, grid_size)
        volume = np.clip(volume + noise, 0, 1)

    return volume.astype(np.float32)


def generate_pyramid(base_size, height, grid_size, noise_level=0.05):
    """
    Gera uma pirâmide 3D (base quadrada) no centro do volume

    Args:
        base_size: Tamanho da base da pirâmide
        height: Altura da pirâmide
        grid_size: Tamanho do grid 3D (depth, height, width)
        noise_level: Nível de ruído a adicionar
    """
    volume = np.zeros(grid_size, dtype=np.float32)
    center_x = grid_size[2] // 2
    center_y = grid_size[1] // 2
    base_z = grid_size[0] // 2 - height // 2  # Base da pirâmide

    # Construir pirâmide camada por camada (de baixo para cima)
    for z in range(height):
        # Calcular o tamanho da camada atual (diminui conforme sobe)
        current_size = int(base_size * (1 - z / height))
        if current_size <= 0:
            break

        half_size = current_size // 2
        z_pos = base_z + z

        # Preencher a camada quadrada
        if z_pos >= 0 and z_pos < grid_size[0]:
            y_start = max(0, center_y - half_size)
            y_end = min(grid_size[1], center_y + half_size)
            x_start = max(0, center_x - half_size)
            x_end = min(grid_size[2], center_x + half_size)

            volume[z_pos, y_start:y_end, x_start:x_end] = 1.0

    # Adicionar ruído
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, grid_size)
        volume = np.clip(volume + noise, 0, 1)

    return volume.astype(np.float32)


# ========================================================================
# CONFIGURAÇÃO DOS DADOS
# ========================================================================
N_SAMPLES = 300  # 200 cubos + 200 esferas + 200 pirâmides
DEPTH = 16
HEIGHT = 16
WIDTH = 16
N_CLASSES = 3  # 0 = Cubo, 1 = Esfera, 2 = Pirâmide

CUBE_SIZE_RANGE = (6, 10)  # Tamanho variável dos cubos
SPHERE_RADIUS_RANGE = (3, 5)  # Raio variável das esferas
PYRAMID_BASE_RANGE = (8, 12)  # Tamanho da base da pirâmide
PYRAMID_HEIGHT_RANGE = (6, 10)  # Altura da pirâmide
NOISE_LEVEL = 0.03  # Ruído para variação

print("=" * 70)
print("EXEMPLO: AUTOENCODER CONDICIONAL - CUBOS, ESFERAS E PIRÂMIDES 3D")
print("=" * 70)

# ========================================================================
# GERAR DADOS DE TREINAMENTO: CUBOS, ESFERAS E PIRÂMIDES
# ========================================================================
print("\n📦 Gerando dados de treinamento...")

x_real_samples = []
y_real_samples = []

# Gerar cubos (Classe 0)
print(f"  → Gerando {N_SAMPLES // 3} cubos...")
for i in range(N_SAMPLES // 3):
    cube_size = np.random.randint(CUBE_SIZE_RANGE[0], CUBE_SIZE_RANGE[1])
    cube = generate_cube(cube_size, (DEPTH, HEIGHT, WIDTH), NOISE_LEVEL)
    x_real_samples.append(cube)
    y_real_samples.append(0)

# Gerar esferas (Classe 1)
print(f"  → Gerando {N_SAMPLES // 3} esferas...")
for i in range(N_SAMPLES // 3):
    radius = np.random.uniform(SPHERE_RADIUS_RANGE[0], SPHERE_RADIUS_RANGE[1])
    sphere = generate_sphere(radius, (DEPTH, HEIGHT, WIDTH), NOISE_LEVEL)
    x_real_samples.append(sphere)
    y_real_samples.append(1)

# Gerar pirâmides (Classe 2)
print(f"  → Gerando {N_SAMPLES // 3} pirâmides...")
for i in range(N_SAMPLES // 3):
    base_size = np.random.randint(PYRAMID_BASE_RANGE[0], PYRAMID_BASE_RANGE[1])
    pyr_height = np.random.randint(PYRAMID_HEIGHT_RANGE[0], PYRAMID_HEIGHT_RANGE[1])
    pyramid = generate_pyramid(base_size, pyr_height, (DEPTH, HEIGHT, WIDTH), NOISE_LEVEL)
    x_real_samples.append(pyramid)
    y_real_samples.append(2)

# Converter para arrays numpy
x_real_samples = np.array(x_real_samples, dtype=np.float32)
y_real_samples = np.array(y_real_samples, dtype=np.int32)

# Embaralhar os dados
shuffle_indices = np.random.permutation(N_SAMPLES)
x_real_samples = x_real_samples[shuffle_indices]
y_real_samples = y_real_samples[shuffle_indices]

print(f"\n✓ Dados gerados:")
print(f"  - Shape dos dados: {x_real_samples.shape}")
print(f"  - Shape dos labels: {y_real_samples.shape}")
print(f"  - Distribuição:")
print(f"    • Cubos: {np.sum(y_real_samples == 0)}")
print(f"    • Esferas: {np.sum(y_real_samples == 1)}")
print(f"    • Pirâmides: {np.sum(y_real_samples == 2)}")
print(f"  - Range de valores: [{x_real_samples.min():.3f}, {x_real_samples.max():.3f}]")

# Mostrar exemplos
print(f"\n📊 Estatísticas dos volumes:")
print(f"  - Média de voxels ativos (cubos): {x_real_samples[y_real_samples == 0].mean():.3f}")
print(f"  - Média de voxels ativos (esferas): {x_real_samples[y_real_samples == 1].mean():.3f}")
print(f"  - Média de voxels ativos (pirâmides): {x_real_samples[y_real_samples == 2].mean():.3f}")

# ========================================================================
# INICIALIZAR E TREINAR O AUTOENCODER
# ========================================================================
print("\n" + "=" * 70)
print("🧠 TREINANDO O AUTOENCODER CONDICIONAL")
print("=" * 70)

input_shape = (DEPTH, HEIGHT, WIDTH)

print("\nInicializando autoencoder...")
autoencoder = VariationalAutoencoder(number_classes=3)

print("\n🚀 Treinando modelo com dados geométricos...")
print("   (Isso pode levar alguns minutos...)")

autoencoder.fit_model(
    input_shape=input_shape,
    x_real_samples=x_real_samples,
    y_real_samples=y_real_samples
)

print("\n✓ Treinamento concluído!")

# ========================================================================
# GERAR AMOSTRAS SINTÉTICAS
# ========================================================================
print("\n" + "=" * 70)
print("🎨 GERANDO AMOSTRAS SINTÉTICAS DE CUBOS, ESFERAS E PIRÂMIDES")
print("=" * 70)

number_samples_per_class = {
    "number_classes": N_CLASSES,
    "classes": {
        0: 25,  # 25 cubos sintéticos
        1: 25,  # 25 esferas sintéticas
        2: 25  # 25 pirâmides sintéticas
    }
}

print(f"\n🎲 Gerando 75 amostras sintéticas...")
print(f"   → 25 cubos (classe 0)")
print(f"   → 25 esferas (classe 1)")
print(f"   → 25 pirâmides (classe 2)")

synthetic_samples = autoencoder.get_samples(number_samples_per_class)
print(synthetic_samples.shape)
print(f"\n✓ Amostras geradas com sucesso!")
print(f"  - Shape das amostras: {synthetic_samples.shape}")
print(f"  - Range de valores: [{synthetic_samples.min():.3f}, {synthetic_samples.max():.3f}]")

# ========================================================================
# ANÁLISE DAS AMOSTRAS GERADAS
# ========================================================================
print("\n" + "=" * 70)
print("📈 ANÁLISE DAS AMOSTRAS GERADAS")
print("=" * 70)

# Separar formas sintéticas
synthetic_cubes = synthetic_samples[:25]
synthetic_spheres = synthetic_samples[25:50]
synthetic_pyramids = synthetic_samples[50:]

print(f"\n🔲 Cubos sintéticos:")
print(f"  - Shape: {synthetic_cubes.shape}")
print(f"  - Média de ativação: {synthetic_cubes.mean():.3f}")
print(f"  - Voxels ativos (>0.5): {(synthetic_cubes > 0.5).sum() / (25 * DEPTH * HEIGHT * WIDTH):.2%}")

print(f"\n⚪ Esferas sintéticas:")
print(f"  - Shape: {synthetic_spheres.shape}")
print(f"  - Média de ativação: {synthetic_spheres.mean():.3f}")
print(f"  - Voxels ativos (>0.5): {(synthetic_spheres > 0.5).sum() / (25 * DEPTH * HEIGHT * WIDTH):.2%}")

print(f"\n🔺 Pirâmides sintéticas:")
print(f"  - Shape: {synthetic_pyramids.shape}")
print(f"  - Média de ativação: {synthetic_pyramids.mean():.3f}")
print(f"  - Voxels ativos (>0.5): {(synthetic_pyramids > 0.5).sum() / (25 * DEPTH * HEIGHT * WIDTH):.2%}")

# ========================================================================
# VISUALIZAÇÃO OPCIONAL (Se matplotlib estiver disponível)
# ========================================================================
print("\n" + "=" * 70)
print("🖼️  VISUALIZAÇÃO (OPCIONAL)")
print("=" * 70)

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D


    def plot_3d_volume(volume, title="Volume 3D", threshold=0.5):
        """Plota um volume 3D"""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Encontrar voxels ativos
        voxels = volume > threshold
        ax.voxels(voxels, facecolors='cyan', edgecolor='k', alpha=0.7)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        plt.tight_layout()
        return fig


    print("\n📊 Gerando visualizações...")

    # Visualizar exemplos originais
    fig1 = plot_3d_volume(x_real_samples[y_real_samples == 0][0], "Cubo Original (Treinamento)")
    fig2 = plot_3d_volume(x_real_samples[y_real_samples == 1][0], "Esfera Original (Treinamento)")
    fig3 = plot_3d_volume(x_real_samples[y_real_samples == 2][0], "Pirâmide Original (Treinamento)")

    # Visualizar amostras geradas
    fig4 = plot_3d_volume(synthetic_cubes[0], "Cubo Sintético (Gerado)")
    fig5 = plot_3d_volume(synthetic_spheres[0], "Esfera Sintética (Gerada)")
    fig6 = plot_3d_volume(synthetic_pyramids[0], "Pirâmide Sintética (Gerada)")

    plt.show()
    print("✓ Visualizações criadas! Feche as janelas para continuar.")

except ImportError:
    print("\n⚠️  Matplotlib não disponível. Pulando visualização.")
    print("   Para visualizar, instale: pip install matplotlib")

# ========================================================================
# RESUMO
# ========================================================================
print("\n" + "=" * 70)
print("✅ EXEMPLO CONCLUÍDO COM SUCESSO!")
print("=" * 70)

print("\n🎯 O que foi demonstrado:")
print("  ✓ Geração de dados geométricos 3D (cubos, esferas e pirâmides)")
print("  ✓ Treinamento condicional do Autoencoder com 3 classes")
print("  ✓ Geração controlada de formas específicas")
print("  ✓ O modelo aprendeu a distinguir e gerar as 3 formas geométricas")
print("  ✓ Processo totalmente automático com dados N-dimensionais")

print("\n💡 Próximos passos:")
print("  → Experimente ajustar os parâmetros do Autoencoder")
print("  → Adicione mais classes (cilindros, toros, cones, etc)")
print("  → Aumente a resolução (32x32x32) para formas mais detalhadas")
print("  → Use as amostras sintéticas para data augmentation")
print("  → Explore o espaço latente para interpolar entre formas")
print("  → Teste com formas compostas (cubo + esfera, etc)")

print("\n" + "=" * 70)