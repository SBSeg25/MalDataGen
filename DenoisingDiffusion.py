#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim - Enhanced by Claude'
__version__ = '6.0.0-complete'

import os
import sys
import numpy as np
import tensorflow as tf
from typing import Dict, Optional, Tuple
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm

from Engine.algorithms.denoising_diffusion.tensorflow.GaussianDenoisingDiffusionTensorflow import \
    GaussianDenoisingDiffusionTensorflow
from Engine.architectures.denoising_diffusion.tensorflow.DenoisingDiffusionUNetModelTensorflow import \
    DenoisingDiffusionUNetModelTensorflow


# Imports dos seus módulos

class EMACallback(Callback):
    """Exponential Moving Average callback para estabilizar treinamento."""

    def __init__(self, model, decay=0.9999):
        super().__init__()
        self.model_to_track = model
        self.decay = decay
        self.ema_weights = None

    def on_train_begin(self, logs=None):
        # Inicializa EMA com pesos do modelo
        self.ema_weights = [tf.Variable(w, trainable=False) for w in self.model_to_track.weights]

    def on_batch_end(self, batch, logs=None):
        # Atualiza EMA após cada batch
        if self.ema_weights is not None:
            for ema_w, model_w in zip(self.ema_weights, self.model_to_track.weights):
                ema_w.assign(self.decay * ema_w + (1 - self.decay) * model_w)

    def apply_ema_weights(self):
        """Aplica pesos EMA ao modelo."""
        if self.ema_weights is not None:
            for ema_w, model_w in zip(self.ema_weights, self.model_to_track.weights):
                model_w.assign(ema_w)


class DenoisingDiffusion:
    """
    Wrapper completo para Diffusion Model com DiT.

    Integra:
    - GaussianDenoisingDiffusionTensorflow: Processo de difusão
    - DenoisingDiffusionUNetModelTensorflow: Modelo DiT
    - Treinamento com EMA e otimizações
    - Geração de amostras sintéticas
    """

    def __init__(
            self,
            number_classes: int = 10,
            beta_start: float = 1e-4,
            beta_end: float = 0.02,
            time_steps: int = 1000,
            clip_min: float = -1.0,
            clip_max: float = 1.0,
            patch_size: int = 2,
            hidden_dim: int = 512,
            depth: int = 12,
            num_heads: int = 8,
            dropout_rate: float = 0.1,
            ffn_expansion: float = 4.0,
            use_cross_attention: bool = True,
            context_seq_len: int = 32,
            use_mixed_precision: bool = True,
            learning_rate: float = 1e-4,
            weight_decay: float = 0.01,
            ema_decay: float = 0.9999,
            **kwargs
    ):
        """
        Inicializa o modelo de difusão.

        Args:
            number_classes: Número de classes para condicionamento
            beta_start: Beta inicial do schedule de ruído
            beta_end: Beta final do schedule de ruído
            time_steps: Número de passos de difusão
            clip_min/clip_max: Limites de clipping
            patch_size: Tamanho dos patches
            hidden_dim: Dimensão oculta do transformer
            depth: Número de blocos DiT
            num_heads: Número de cabeças de atenção
            dropout_rate: Taxa de dropout
            ffn_expansion: Fator de expansão do FFN
            use_cross_attention: Usar cross-attention
            context_seq_len: Comprimento da sequência de contexto
            use_mixed_precision: Usar mixed precision
            learning_rate: Taxa de aprendizado
            weight_decay: Weight decay para AdamW
            ema_decay: Decay para EMA
        """

        self.number_classes = number_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ema_decay = ema_decay
        self.time_steps = time_steps

        # Inicializa processo de difusão
        self.diffusion_process = GaussianDenoisingDiffusionTensorflow(
            beta_start=beta_start,
            beta_end=beta_end,
            time_steps=time_steps,
            clip_min=clip_min,
            clip_max=clip_max
        )

        # Parâmetros do modelo
        self.model_params = {
            'patch_size': patch_size,
            'hidden_dim': hidden_dim,
            'depth': depth,
            'num_heads': num_heads,
            'dropout_rate': dropout_rate,
            'ffn_expansion': ffn_expansion,
            'use_cross_attention': use_cross_attention,
            'context_seq_len': context_seq_len,
            'use_mixed_precision': use_mixed_precision,
        }

        self.model = None
        self.ema_callback = None
        self.is_trained = False

        print(f"\n{'=' * 70}")
        print(f"🎯 DenoisingDiffusion Initialized")
        print(f"{'=' * 70}")
        print(f"📊 Configuration:")
        print(f"  • Number of classes: {number_classes}")
        print(f"  • Diffusion steps: {time_steps}")
        print(f"  • Hidden dimension: {hidden_dim}")
        print(f"  • Depth: {depth} blocks")
        print(f"  • Learning rate: {learning_rate}")
        print(f"  • Weight decay: {weight_decay}")
        print(f"  • EMA decay: {ema_decay}")
        print(f"{'=' * 70}\n")

    def _normalize_data(self, x: np.ndarray) -> np.ndarray:
        """Normaliza dados para [-1, 1]."""
        # Assume que entrada está em [0, 1]
        return x * 2.0 - 1.0

    def _denormalize_data(self, x: np.ndarray) -> np.ndarray:
        """Denormaliza dados de [-1, 1] para [0, 1]."""
        return (x + 1.0) / 2.0

    def _flatten_images(self, x: np.ndarray) -> np.ndarray:
        """Converte imagens [B, H, W, C] para sequência [B, H*W, C]."""
        batch, h, w, c = x.shape
        return x.reshape(batch, h * w, c)

    def _unflatten_images(self, x: np.ndarray, target_shape: Tuple) -> np.ndarray:
        """Converte sequência [B, H*W, C] para imagens [B, H, W, C]."""
        batch = x.shape[0]
        h, w, c = target_shape
        return x.reshape(batch, h, w, c)

    def _prepare_labels(self, y: np.ndarray) -> np.ndarray:
        """Converte labels para one-hot encoding."""
        if len(y.shape) == 1:
            return tf.keras.utils.to_categorical(y, self.number_classes)
        return y

    @tf.function
    def _train_step(self, x_batch, y_batch):
        batch_size = tf.shape(x_batch)[0]

        # Amostra timesteps aleatórios
        t = tf.random.uniform(
            shape=[batch_size],
            minval=0,
            maxval=self.time_steps,
            dtype=tf.int32
        )

        # Gera ruído
        noise = tf.random.normal(shape=tf.shape(x_batch))

        # Aplica difusão forward
        x_noisy = self.diffusion_process.q_sample(x_batch, t, noise)

        with tf.GradientTape() as tape:
            # Prediz o ruído (o modelo deve aprender a prever 'noise')
            predicted_noise = self.model([x_noisy, t, y_batch], training=True)

            # Loss MSE com proteção contra NaN
            loss = tf.reduce_mean(tf.math.squared_difference(noise, predicted_noise))

            # Se usares Mixed Precision, escala a loss aqui
            if self.model_params.get('use_mixed_precision'):
                scaled_loss = self.optimizer.get_scaled_loss(loss)

        # Cálculo de gradientes
        trainable_vars = self.model.trainable_variables
        if self.model_params.get('use_mixed_precision'):
            scaled_gradients = tape.gradient(scaled_loss, trainable_vars)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(loss, trainable_vars)

        # FIX: Clipping de gradiente global para evitar explosão em Transformers
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return loss

    def fit_model(
            self,
            input_shape: Tuple,
            x_real_samples: np.ndarray,
            y_real_samples: np.ndarray,
            flatten: bool = True,
            epochs: int = 100,
            batch_size: int = 4,
            validation_split: float = 0.1,
            verbose: int = 1,
            **kwargs
    ):
        """
        Treina o modelo de difusão.

        Args:
            input_shape: Shape da entrada (H, W, C)
            x_real_samples: Amostras reais [N, H, W, C]
            y_real_samples: Labels [N] ou [N, num_classes]
            flatten: Se True, converte para sequência 1D
            epochs: Número de épocas
            batch_size: Tamanho do batch
            validation_split: Fração para validação
            verbose: Nível de verbosidade
        """

        print(f"\n{'=' * 70}")
        print(f"🚀 Starting Training")
        print(f"{'=' * 70}")

        # Normaliza dados
        x_train = self._normalize_data(x_real_samples)

        # Prepara labels
        y_train = self._prepare_labels(y_real_samples)

        # Flatten se necessário
        original_shape = input_shape
        if flatten:
            h, w, c = input_shape
            x_train = self._flatten_images(x_train)
            output_shape = h * w
            embedding_channels = c
            print(f"✓ Flattened: {input_shape} -> ({output_shape}, {embedding_channels})")
        else:
            output_shape = input_shape[0]
            embedding_channels = input_shape[-1]

        # Constrói modelo se não existir
        if self.model is None:
            number_samples_per_class = {
                'number_classes': self.number_classes,
                'classes': {i: np.sum(y_real_samples == i) for i in range(self.number_classes)}
            }

            model_builder = DenoisingDiffusionUNetModelTensorflow(
                output_shape=output_shape,
                embedding_channels=embedding_channels,
                number_samples_per_class=number_samples_per_class,
                **self.model_params
            )

            self.model = model_builder.build_model()

            # Otimizador
            self.optimizer = AdamW(
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                clipnorm=1.0
            )

            # EMA callback
            self.ema_callback = EMACallback(self.model, decay=self.ema_decay)
            self.ema_callback.on_train_begin()

        # Split treino/validação
        n_samples = len(x_train)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val

        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        x_train_split = x_train[train_indices]
        y_train_split = y_train[train_indices]
        x_val = x_train[val_indices] if n_val > 0 else None
        y_val = y_train[val_indices] if n_val > 0 else None

        print(f"\n📊 Dataset:")
        print(f"  • Training samples: {n_train}")
        print(f"  • Validation samples: {n_val}")
        print(f"  • Batch size: {batch_size}")
        print(f"  • Steps per epoch: {n_train // batch_size}")

        # Treinamento
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10

        for epoch in range(epochs):
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'=' * 70}")

            # Shuffle
            shuffle_indices = np.random.permutation(n_train)
            x_train_epoch = x_train_split[shuffle_indices]
            y_train_epoch = y_train_split[shuffle_indices]

            # Training
            train_losses = []
            steps = n_train // batch_size

            pbar = tqdm(range(steps), desc="Training", ncols=100)
            for step in pbar:
                start_idx = step * batch_size
                end_idx = start_idx + batch_size

                x_batch = x_train_epoch[start_idx:end_idx]
                y_batch = y_train_epoch[start_idx:end_idx]

                # Converte para tensors
                x_batch = tf.constant(x_batch, dtype=tf.float32)
                y_batch = tf.constant(y_batch, dtype=tf.float32)

                # Train step
                loss = self._train_step(x_batch, y_batch)
                train_losses.append(float(loss))

                # Update EMA
                self.ema_callback.on_batch_end(step)

                # Update progress bar
                pbar.set_postfix({'loss': f'{np.mean(train_losses[-10:]):.4f}'})

            avg_train_loss = np.mean(train_losses)

            # Validation
            if x_val is not None:
                val_losses = []
                val_steps = n_val // batch_size

                for step in range(val_steps):
                    start_idx = step * batch_size
                    end_idx = start_idx + batch_size

                    x_batch = tf.constant(x_val[start_idx:end_idx], dtype=tf.float32)
                    y_batch = tf.constant(y_val[start_idx:end_idx], dtype=tf.float32)

                    # Validation step (sem gradientes)
                    batch_size_val = tf.shape(x_batch)[0]
                    t = tf.random.uniform([batch_size_val], 0, self.time_steps, dtype=tf.int32)
                    noise = tf.random.normal(shape=tf.shape(x_batch))
                    x_noisy = self.diffusion_process.q_sample(x_batch, t, noise)

                    predicted_noise = self.model([x_noisy, t, y_batch], training=False)
                    loss = tf.reduce_mean(tf.square(noise - predicted_noise))
                    val_losses.append(float(loss))

                avg_val_loss = np.mean(val_losses)

                print(f"\n📈 Results:")
                print(f"  • Train Loss: {avg_train_loss:.4f}")
                print(f"  • Val Loss: {avg_val_loss:.4f}")

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    print(f"  ✓ Best model updated!")
                else:
                    patience_counter += 1
                    print(f"  ⚠ No improvement ({patience_counter}/{patience})")

                if patience_counter >= patience:
                    print(f"\n⏹ Early stopping triggered!")
                    break
            else:
                print(f"\n📈 Train Loss: {avg_train_loss:.4f}")

        # Aplica pesos EMA ao modelo final
        self.ema_callback.apply_ema_weights()

        self.is_trained = True
        self.original_shape = original_shape
        self.flatten_mode = flatten

        print(f"\n{'=' * 70}")
        print(f"✅ Training Completed!")
        print(f"{'=' * 70}\n")

    @tf.function
    def _denoise_step(self, x, t, y):
        """Um passo de denoising."""
        predicted_noise = self.model([x, t, y], training=False)
        x_denoised = self.diffusion_process.p_sample(predicted_noise, x, t, clip_denoised=True)
        return x_denoised

    def get_samples(
            self,
            samples_per_class: Dict,
            guidance_scale: float = 1.0,
            verbose: bool = True
    ) -> np.ndarray:
        """
        Gera amostras sintéticas.

        Args:
            samples_per_class: Dict com 'number_classes' e 'classes'
            guidance_scale: Escala de guidance (1.0 = sem guidance)
            verbose: Mostrar progresso

        Returns:
            Array de amostras sintéticas [N, H, W, C]
        """

        if not self.is_trained:
            raise ValueError("Model must be trained before generating samples!")

        print(f"\n{'=' * 70}")
        print(f"🎨 Generating Synthetic Samples")
        print(f"{'=' * 70}")

        # Prepara labels
        all_samples = []
        all_labels = []

        for class_idx, n_samples in samples_per_class['classes'].items():
            all_labels.extend([class_idx] * n_samples)

        n_total = len(all_labels)
        y_generate = self._prepare_labels(np.array(all_labels))

        print(f"  • Total samples to generate: {n_total}")
        print(f"  • Diffusion steps: {self.time_steps}")
        print(f"  • Guidance scale: {guidance_scale}")

        # Shape do ruído inicial
        if self.flatten_mode:
            h, w, c = self.original_shape
            noise_shape = (n_total, h * w, c)
        else:
            noise_shape = (n_total,) + self.original_shape

        # Inicia com ruído puro
        x = tf.random.normal(noise_shape, dtype=tf.float32)
        y = tf.constant(y_generate, dtype=tf.float32)

        # Denoise progressivamente
        if verbose:
            timesteps = tqdm(reversed(range(self.time_steps)), desc="Denoising", total=self.time_steps)
        else:
            timesteps = reversed(range(self.time_steps))

        for t_step in timesteps:
            t = tf.constant([t_step] * n_total, dtype=tf.int32)
            x = self._denoise_step(x, t, y)

        # Converte para numpy e denormaliza
        samples = x.numpy()
        samples = self._denormalize_data(samples)
        samples = np.clip(samples, 0, 1)

        # Unflatten se necessário
        if self.flatten_mode:
            samples = self._unflatten_images(samples, self.original_shape)

        print(f"\n✅ Generated {samples.shape[0]} samples")
        print(f"  • Shape: {samples.shape}")
        print(f"  • Range: [{samples.min():.3f}, {samples.max():.3f}]")
        print(f"{'=' * 70}\n")

        return samples

    def save_model(self, path: str):
        """Salva o modelo treinado."""
        if self.model is not None:
            self.model.save_weights(path)
            print(f"✓ Model saved to {path}")

    def load_model(self, path: str):
        """Carrega um modelo salvo."""
        if self.model is not None:
            self.model.load_weights(path)
            self.is_trained = True
            print(f"✓ Model loaded from {path}")
        else:
            raise ValueError("Model must be built before loading weights!")