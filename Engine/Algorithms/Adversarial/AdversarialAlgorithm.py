#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{2}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/06'
__credits__ = ['Synthetic Ocean AI']

# MIT License
#
# Copyright (c) 2025 Synthetic Ocean AI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

try:
    import os
    import sys
    import numpy
    import json
    import logging
    from pathlib import Path
    from abc import ABC, abstractmethod
    from typing import Any, Dict, Optional, Tuple

    # Detecta o framework a partir da variável de ambiente
    ML_FRAMEWORK = os.getenv('ML_FRAMEWORK', 'tensorflow').lower()

    # Importações condicionais
    tf = None
    torch = None
    nn = None
    Model = None
    to_categorical = None
    model_from_json = None

    if ML_FRAMEWORK == 'tensorflow':
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.utils import to_categorical
            from tensorflow.keras.models import model_from_json
        except ImportError:
            raise ImportError("TensorFlow not found. Please install: pip install tensorflow")

    elif ML_FRAMEWORK == 'pytorch':
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("PyTorch not found. Please install: pip install torch")

    else:
        raise ValueError(f"Unsupported ML_FRAMEWORK: {ML_FRAMEWORK}. Use 'tensorflow' or 'pytorch'")

except ImportError as error:
    logging.error(error)
    sys.exit(-1)


class BaseAdversarialImplementation(ABC):
    """
    Classe abstrata base para implementações específicas de framework do algoritmo adversarial.
    """

    @abstractmethod
    def train_step(self, batch: Tuple) -> Dict[str, float]:
        """Executa um passo de treinamento."""
        pass

    @abstractmethod
    def generate_samples(self, latent_noise: Any, labels: Any) -> Any:
        """Gera amostras sintéticas."""
        pass

    @abstractmethod
    def save_models(self, path: str, discriminator_name: str, generator_name: str) -> None:
        """Salva os modelos."""
        pass

    @abstractmethod
    def load_models(self, path: str, discriminator_name: str, generator_name: str) -> None:
        """Carrega os modelos."""
        pass


if ML_FRAMEWORK == 'tensorflow':
    class TensorFlowAdversarialImpl(BaseAdversarialImplementation):
        """
        Implementação do algoritmo adversarial usando TensorFlow.
        """

        def __init__(self, parent):
            self.parent = parent

        @tf.function
        def train_step(self, batch: Tuple) -> Dict[str, float]:
            """
            Executa um passo de treinamento para gerador e discriminador (TensorFlow).
            """
            real_feature, real_samples_label = batch
            batch_size = tf.shape(real_feature)[0]
            real_samples_label = tf.expand_dims(real_samples_label, axis=-1)

            # Gera amostras sintéticas
            latent_space = tf.random.normal(shape=(batch_size, self.parent._latent_dimension))
            synthetic_feature = self.parent._generator([latent_space, real_samples_label], training=False)

            # Treina discriminador
            with tf.GradientTape() as discriminator_gradient:
                label_predicted_real = self.parent._discriminator([real_feature, real_samples_label], training=True)
                label_predicted_synthetic = self.parent._discriminator([synthetic_feature, real_samples_label],
                                                                       training=True)
                label_predicted_all_samples = tf.concat([label_predicted_real, label_predicted_synthetic], axis=0)

                tensor_labels_predicted = tf.concat([
                    tf.zeros_like(label_predicted_real),
                    tf.ones_like(label_predicted_synthetic)
                ], axis=0)

                # Label smoothing
                smooth_tensor_real_data = 0.15 * tf.random.uniform(tf.shape(label_predicted_real))
                smooth_tensor_synthetic_data = -0.15 * tf.random.uniform(tf.shape(label_predicted_synthetic))
                tensor_labels_predicted += tf.concat([smooth_tensor_real_data, smooth_tensor_synthetic_data], axis=0)

                loss_value = self.parent._loss_discriminator(tensor_labels_predicted, label_predicted_all_samples)

            gradient_tape_loss = discriminator_gradient.gradient(loss_value,
                                                                 self.parent._discriminator.trainable_variables)
            self.parent._optimizer_discriminator.apply_gradients(
                zip(gradient_tape_loss, self.parent._discriminator.trainable_variables))

            # Treina gerador
            with tf.GradientTape() as generator_gradient:
                latent_space = tf.random.normal(shape=(batch_size, self.parent._latent_dimension))
                synthetic_feature = self.parent._generator([latent_space, real_samples_label], training=True)
                predicted_labels = self.parent._discriminator([synthetic_feature, real_samples_label], training=False)
                total_loss_g = self.parent._loss_generator(tf.zeros_like(predicted_labels), predicted_labels)

            gradient_tape_loss = generator_gradient.gradient(total_loss_g, self.parent._generator.trainable_variables)
            self.parent._optimizer_generator.apply_gradients(
                zip(gradient_tape_loss, self.parent._generator.trainable_variables))

            return {"loss_d": loss_value, "loss_g": total_loss_g}

        def generate_samples(self, latent_noise: Any, labels: Any) -> Any:
            """Gera amostras usando o gerador TensorFlow."""
            return self.parent._generator.predict([latent_noise, labels], verbose=0)

        def save_models(self, path: str, discriminator_name: str, generator_name: str) -> None:
            """Salva modelos TensorFlow em formato JSON + H5."""
            discriminator_model_json = self.parent._discriminator.to_json()
            with open(discriminator_name + ".json", "w") as json_file:
                json_file.write(discriminator_model_json)
            self.parent._discriminator.save_weights(discriminator_name + ".h5")

            generator_model_json = self.parent._generator.to_json()
            with open(generator_name + ".json", "w") as json_file:
                json_file.write(generator_model_json)
            self.parent._generator.save_weights(generator_name + ".h5")

        def load_models(self, path: str, discriminator_name: str, generator_name: str) -> None:
            """Carrega modelos TensorFlow de formato JSON + H5."""
            with open(discriminator_name + ".json", 'r') as json_file:
                discriminator_model_json = json_file.read()
            self.parent._discriminator = model_from_json(discriminator_model_json)
            self.parent._discriminator.load_weights(discriminator_name + ".h5")

            with open(generator_name + ".json", 'r') as json_file:
                generator_model_json = json_file.read()
            self.parent._generator = model_from_json(generator_model_json)
            self.parent._generator.load_weights(generator_name + ".h5")

if ML_FRAMEWORK == 'pytorch':
    class PyTorchAdversarialImpl(BaseAdversarialImplementation):
        """
        Implementação do algoritmo adversarial usando PyTorch.
        """

        def __init__(self, parent):
            self.parent = parent

        def train_step(self, batch: Tuple) -> Dict[str, float]:
            """
            Executa um passo de treinamento para gerador e discriminador (PyTorch).
            """
            real_feature, real_samples_label = batch
            batch_size = real_feature.shape[0]

            # Expande dimensões dos labels se necessário
            if len(real_samples_label.shape) == 1:
                real_samples_label = real_samples_label.unsqueeze(-1)

            # Gera amostras sintéticas
            latent_space = torch.randn(batch_size, self.parent._latent_dimension, device=real_feature.device)
            with torch.no_grad():
                synthetic_feature = self.parent._generator(latent_space, real_samples_label)

            # Treina discriminador
            self.parent._optimizer_discriminator.zero_grad()

            label_predicted_real = self.parent._discriminator(real_feature, real_samples_label)
            label_predicted_synthetic = self.parent._discriminator(synthetic_feature.detach(), real_samples_label)

            # Concatena predições
            label_predicted_all = torch.cat([label_predicted_real, label_predicted_synthetic], dim=0)

            # Cria labels (real=0, fake=1)
            tensor_labels = torch.cat([
                torch.zeros_like(label_predicted_real),
                torch.ones_like(label_predicted_synthetic)
            ], dim=0)

            # Label smoothing
            smooth_real = 0.15 * torch.rand_like(label_predicted_real)
            smooth_synthetic = -0.15 * torch.rand_like(label_predicted_synthetic)
            tensor_labels += torch.cat([smooth_real, smooth_synthetic], dim=0)

            loss_d = self.parent._loss_discriminator(label_predicted_all, tensor_labels)
            loss_d.backward()
            self.parent._optimizer_discriminator.step()

            # Treina gerador
            self.parent._optimizer_generator.zero_grad()

            latent_space = torch.randn(batch_size, self.parent._latent_dimension, device=real_feature.device)
            synthetic_feature = self.parent._generator(latent_space, real_samples_label)

            predicted_labels = self.parent._discriminator(synthetic_feature, real_samples_label)
            loss_g = self.parent._loss_generator(predicted_labels, torch.zeros_like(predicted_labels))

            loss_g.backward()
            self.parent._optimizer_generator.step()

            return {"loss_d": loss_d.item(), "loss_g": loss_g.item()}

        def generate_samples(self, latent_noise: Any, labels: Any) -> Any:
            """Gera amostras usando o gerador PyTorch."""
            self.parent._generator.eval()
            with torch.no_grad():
                if isinstance(latent_noise, numpy.ndarray):
                    latent_noise = torch.from_numpy(latent_noise).float()
                if isinstance(labels, numpy.ndarray):
                    labels = torch.from_numpy(labels).float()
                samples = self.parent._generator(latent_noise, labels)
                return samples.cpu().numpy()

        def save_models(self, path: str, discriminator_name: str, generator_name: str) -> None:
            """Salva modelos PyTorch em formato .pth."""
            torch.save({
                'model_state_dict': self.parent._discriminator.state_dict(),
            }, discriminator_name + ".pth")

            torch.save({
                'model_state_dict': self.parent._generator.state_dict(),
            }, generator_name + ".pth")

        def load_models(self, path: str, discriminator_name: str, generator_name: str) -> None:
            """Carrega modelos PyTorch de formato .pth."""
            discriminator_checkpoint = torch.load(discriminator_name + ".pth")
            self.parent._discriminator.load_state_dict(discriminator_checkpoint['model_state_dict'])

            generator_checkpoint = torch.load(generator_name + ".pth")
            self.parent._generator.load_state_dict(generator_checkpoint['model_state_dict'])


class AdversarialAlgorithm:
    """
    AdversarialAlgorithm

    Implementa um algoritmo de treinamento adversarial, tipicamente usado em Generative Adversarial Networks (GANs).
    Suporta tanto TensorFlow quanto PyTorch através da variável de ambiente ML_FRAMEWORK.

    Esta classe realiza treinamento adversarial utilizando um gerador e um discriminador,
    otimizando o gerador para produzir dados realistas enquanto treina o discriminador para
    diferenciar entre dados reais e falsos.

    Variável de Ambiente:
        ML_FRAMEWORK: Define o framework a ser usado ('tensorflow' ou 'pytorch').
                     Padrão: 'tensorflow'

    Attributes:
        generator_model: O modelo gerador.
        discriminator_model: O modelo discriminador.
        latent_dimension (int): Dimensionalidade do espaço latente.
        loss_generator: Função de perda para o gerador.
        loss_discriminator: Função de perda para o discriminador.
        file_name_discriminator (str): Nome do arquivo para salvar o discriminador.
        file_name_generator (str): Nome do arquivo para salvar o gerador.
        models_saved_path (str): Caminho onde os modelos serão salvos.
        latent_mean_distribution (float): Média da distribuição de ruído latente.
        latent_stander_deviation (float): Desvio padrão da distribuição de ruído latente.
        smoothing_rate (float): Taxa de suavização aplicada aos labels do discriminador.

    Example:
        >>> import os
        >>> os.environ['ML_FRAMEWORK'] = 'tensorflow'  # ou 'pytorch'
        >>> adversarial_algorithm = AdversarialAlgorithm(
        ...     generator_model=generator_model,
        ...     discriminator_model=discriminator_model,
        ...     latent_dimension=100,
        ...     loss_generator=loss_fn,
        ...     loss_discriminator=loss_fn,
        ...     file_name_discriminator="discriminator",
        ...     file_name_generator="generator",
        ...     models_saved_path="./models/",
        ...     latent_mean_distribution=0.0,
        ...     latent_stander_deviation=1.0,
        ...     smoothing_rate=0.1
        ... )
        >>> adversarial_algorithm.compile(
        ...     optimizer_generator=optimizer_g,
        ...     optimizer_discriminator=optimizer_d
        ... )
    """

    def __init__(self, generator_model,
                 discriminator_model,
                 latent_dimension,
                 loss_generator,
                 loss_discriminator,
                 file_name_discriminator,
                 file_name_generator,
                 models_saved_path,
                 latent_mean_distribution,
                 latent_stander_deviation,
                 smoothing_rate,
                 *args,
                 **kwargs):
        """
        Inicializa o algoritmo adversarial com gerador, discriminador e outras configurações.

        Args:
            generator_model: O modelo gerador.
            discriminator_model: O modelo discriminador.
            latent_dimension (int): Dimensão do espaço latente.
            loss_generator: Função de perda do gerador.
            loss_discriminator: Função de perda do discriminador.
            file_name_discriminator (str): Nome do arquivo para o discriminador.
            file_name_generator (str): Nome do arquivo para o gerador.
            models_saved_path (str): Caminho para salvar modelos.
            latent_mean_distribution (float): Média da distribuição de ruído latente.
            latent_stander_deviation (float): Desvio padrão do ruído latente.
            smoothing_rate (float): Taxa de suavização de labels.
        """
        # Validações
        if latent_dimension <= 0:
            raise ValueError("Latent dimension must be greater than 0.")

        if not isinstance(file_name_discriminator, str) or not file_name_discriminator:
            raise ValueError("Discriminator file name must be a non-empty string.")

        if not isinstance(file_name_generator, str) or not file_name_generator:
            raise ValueError("Generator file name must be a non-empty string.")

        if not isinstance(models_saved_path, str) or not models_saved_path:
            raise ValueError("Models saved path must be a non-empty string.")

        if not isinstance(latent_mean_distribution, (int, float)):
            raise TypeError("Latent mean distribution must be a number.")

        if not isinstance(latent_stander_deviation, (int, float)):
            raise TypeError("Latent standard deviation must be a number.")

        if latent_stander_deviation <= 0:
            raise ValueError("Latent standard deviation must be greater than 0.")

        if not (0.0 <= smoothing_rate <= 1.0):
            raise ValueError("Smoothing rate must be between 0 and 1.")

        self._generator = generator_model
        self._discriminator = discriminator_model
        self._latent_dimension = latent_dimension
        self._optimizer_generator = None
        self._optimizer_discriminator = None
        self._loss_generator = loss_generator
        self._loss_discriminator = loss_discriminator
        self._smoothing_rate = smoothing_rate
        self._latent_mean_distribution = latent_mean_distribution
        self._latent_stander_deviation = latent_stander_deviation
        self._file_name_discriminator = file_name_discriminator
        self._file_name_generator = file_name_generator
        self._models_saved_path = models_saved_path
        self._framework = ML_FRAMEWORK

        # Seleciona implementação baseada no framework
        if self._framework == 'tensorflow':
            self._implementation = TensorFlowAdversarialImpl(self)
        elif self._framework == 'pytorch':
            self._implementation = PyTorchAdversarialImpl(self)
        else:
            raise ValueError(f"Unsupported framework: {self._framework}")

    @property
    def framework(self) -> str:
        """Retorna o framework sendo utilizado."""
        return self._framework

    def compile(self, optimizer_generator, optimizer_discriminator, loss_generator=None, loss_discriminator=None, *args,
                **kwargs):
        """
        Compila o algoritmo adversarial definindo otimizadores e funções de perda.

        Args:
            optimizer_generator: Otimizador para o gerador.
            optimizer_discriminator: Otimizador para o discriminador.
            loss_generator: Função de perda do gerador (opcional).
            loss_discriminator: Função de perda do discriminador (opcional).
        """
        self._optimizer_generator = optimizer_generator
        self._optimizer_discriminator = optimizer_discriminator

        if loss_generator is not None:
            self._loss_generator = loss_generator
        if loss_discriminator is not None:
            self._loss_discriminator = loss_discriminator

    def train_step(self, batch: Tuple) -> Dict[str, float]:
        """
        Executa um passo de treinamento para gerador e discriminador.

        Args:
            batch (tuple): Tupla contendo features reais e labels correspondentes.

        Returns:
            dict: Dicionário contendo os valores de perda para gerador e discriminador.
        """
        return self._implementation.train_step(batch)

    def get_samples(self, number_samples_per_class: Dict[str, Any]) -> Dict[int, numpy.ndarray]:
        """
        Gera amostras de dados sintéticos para cada classe especificada usando o gerador treinado.

        Args:
            number_samples_per_class (dict):
                Dicionário especificando o número de amostras sintéticas a gerar por classe.
                Estrutura esperada:
                {
                    "classes": {class_label: number_of_samples, ...},
                    "number_classes": total_number_of_classes
                }

        Returns:
            dict: Dicionário onde cada chave é um label de classe e o valor é um array de amostras geradas.
        """
        generated_data = {}

        for label_class, number_instances in number_samples_per_class["classes"].items():
            # Cria labels one-hot encoded
            if ML_FRAMEWORK == 'tensorflow':
                label_samples_generated = to_categorical(
                    [label_class] * number_instances,
                    num_classes=number_samples_per_class["number_classes"]
                )
            else:  # PyTorch
                label_samples_generated = numpy.zeros((number_instances, number_samples_per_class["number_classes"]))
                label_samples_generated[:, label_class] = 1

            # Gera vetores de ruído latente
            latent_noise = numpy.random.normal(
                self._latent_mean_distribution,
                self._latent_stander_deviation,
                (number_instances, self._latent_dimension)
            )

            # Usa o gerador para produzir amostras sintéticas
            generated_samples = self._implementation.generate_samples(latent_noise, label_samples_generated)

            # Arredonda valores gerados
            generated_samples = numpy.rint(generated_samples)

            generated_data[label_class] = generated_samples

        return generated_data

    def save_model(self, path_output: str, k_fold: int) -> None:
        """
        Salva os modelos gerador e discriminador.

        Args:
            path_output (str): Caminho base para salvar os modelos.
            k_fold (int): Número do fold atual.
        """
        try:
            logging.info(f"Starting to save Adversarial Model for fold {k_fold}...")

            path_directory = os.path.join(path_output, self._models_saved_path)
            Path(path_directory).mkdir(parents=True, exist_ok=True)
            logging.info(f"Created/verified directory at: {path_directory}")

            discriminator_file_name = self._file_name_discriminator + "_" + str(k_fold)
            generator_file_name = self._file_name_generator + "_" + str(k_fold)

            path_model = os.path.join(path_directory, "fold_" + str(k_fold + 1))
            Path(path_model).mkdir(parents=True, exist_ok=True)
            logging.info(f"Created/verified fold directory at: {path_model}")

            discriminator_file_name = os.path.join(path_model, discriminator_file_name)
            generator_file_name = os.path.join(path_model, generator_file_name)

            logging.info("Saving models...")
            self._implementation.save_models(path_model, discriminator_file_name, generator_file_name)
            logging.info(f"Models saved successfully at: {path_model}")

        except FileExistsError:
            logging.error("Model file already exists. Aborting.")
            exit(-1)
        except Exception as e:
            logging.error(f"An error occurred while saving the models: {e}")
            exit(-1)

    def load_models(self, path_output: str, k_fold: int) -> None:
        """
        Carrega os modelos gerador e discriminador.

        Args:
            path_output (str): Caminho base de onde carregar os modelos.
            k_fold (int): Número do fold.
        """
        try:
            logging.info(f"Loading Adversarial Model for fold {k_fold + 1}...")

            path_directory = os.path.join(path_output, self._models_saved_path)
            discriminator_file_name = self._file_name_discriminator + "_" + str(k_fold + 1)
            generator_file_name = self._file_name_generator + "_" + str(k_fold + 1)

            discriminator_file_name = os.path.join(path_directory, discriminator_file_name)
            generator_file_name = os.path.join(path_directory, generator_file_name)

            logging.info("Loading models...")
            self._implementation.load_models(path_directory, discriminator_file_name, generator_file_name)
            logging.info("Models loaded successfully")

        except FileNotFoundError:
            logging.error("Model file not found. Please provide an existing and valid model.")
            exit(-1)
        except Exception as e:
            logging.error(f"An error occurred while loading the models: {e}")
            exit(-1)

    # Setters
    def set_generator(self, generator):
        self._generator = generator

    def set_discriminator(self, discriminator):
        self._discriminator = discriminator

    def set_latent_dimension(self, latent_dimension):
        self._latent_dimension = latent_dimension

    def set_optimizer_generator(self, optimizer_generator):
        self._optimizer_generator = optimizer_generator

    def set_optimizer_discriminator(self, optimizer_discriminator):
        self._optimizer_discriminator = optimizer_discriminator

    def set_loss_generator(self, loss_generator):
        self._loss_generator = loss_generator

    def set_loss_discriminator(self, loss_discriminator):
        self._loss_discriminator = loss_discriminator
