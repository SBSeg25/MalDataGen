#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exemplo simples para testar o Autoencoder com dados aleatórios
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from Engine.Models.Adversarial import Adversarial
from Engine.Models.Autoencoder import Autoencoder

N_SAMPLES = 1000
N_FEATURES = 320
N_CLASSES = 2

np.random.seed(42)

x_real_samples = np.random.randn(N_SAMPLES, N_FEATURES)

time_component = np.sin(np.linspace(0, 4*np.pi, N_SAMPLES))
x_real_samples += time_component[:, np.newaxis] * 0.5

scaler = MinMaxScaler(feature_range=(0, 1))
x_real_samples = scaler.fit_transform(x_real_samples)

x_real_samples = x_real_samples.astype(np.float32)

y_real_samples = np.random.randint(0, N_CLASSES, size=N_SAMPLES)

input_shape = N_FEATURES



autoencoder_algorithm = Adversarial()

autoencoder_algorithm.fit_model(input_shape, x_real_samples, y_real_samples)

number_samples_per_class = {
    "number_classes": N_CLASSES,
    0: 100,  # 100 amostras da classe 0
    1: 100   # 100 amostras da classe 1
}
synthetic_samples = autoencoder_algorithm.get_samples(number_samples_per_class)
