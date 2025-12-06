#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com'
__version__ = '{1}.{0}.{2}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/06'
__credits__ = ['Kayuã Oleques']

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
    import json
    import numpy
    from typing import Any, Dict, Tuple

    # Detecta o framework a partir da variável de ambiente
    ML_FRAMEWORK = os.getenv('ML_FRAMEWORK', 'tensorflow').lower()

    # Importações condicionais
    tf = None
    torch = None
    nn = None
    Model = None
    Mean = None
    to_categorical = None

    if ML_FRAMEWORK == 'tensorflow':
        try:
            import tensorflow as tf
            from tensorflow.keras.metrics import Mean
            from tensorflow.keras.models import Model
            from tensorflow.keras.utils import to_categorical
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
    print(error)
    sys.exit(-1)

if ML_FRAMEWORK == 'tensorflow':
    class AutoencoderAlgorithm(Model):
        """
        AutoencoderAlgorithm (TensorFlow/Keras)

        Implementação de AutoEncoder usando TensorFlow/Keras, herdando de tf.keras.Model.
        Fornece métodos para treinamento, geração de dados sintéticos, salvamento e carregamento.

        Args:
            encoder_model: A parte encoder do AutoEncoder.
            decoder_model: A parte decoder do AutoEncoder.
            loss_function: A função de perda para treinamento.
            file_name_encoder (str): O nome do arquivo para salvar o encoder.
            file_name_decoder (str): O nome do arquivo para salvar o decoder.
            models_saved_path (str): O caminho para salvar os modelos.
            latent_mean_distribution (float): Média da distribuição do espaço latente.
            latent_stander_deviation (float): Desvio padrão da distribuição do espaço latente.
            latent_dimension (int): A dimensionalidade do espaço latente.

        Example:
            >>> import os
            >>> os.environ['ML_FRAMEWORK'] = 'tensorflow'
            >>> autoencoder = AutoencoderAlgorithm(
            ...     encoder_model=encoder_model,
            ...     decoder_model=decoder_model,
            ...     loss_function=loss_fn,
            ...     file_name_encoder="encoder_model",
            ...     file_name_decoder="decoder_model",
            ...     models_saved_path="./autoencoder_models/",
            ...     latent_mean_distribution=0.0,
            ...     latent_stander_deviation=1.0,
            ...     latent_dimension=64
            ... )
            >>> autoencoder.compile(optimizer=optimizer)
            >>> autoencoder.fit(train_dataset, epochs=50)
        """

        def __init__(self,
                     encoder_model,
                     decoder_model,
                     loss_function,
                     file_name_encoder,
                     file_name_decoder,
                     models_saved_path,
                     latent_mean_distribution,
                     latent_stander_deviation,
                     latent_dimension):

            super().__init__()

            # Validações
            if not isinstance(encoder_model, tf.keras.Model):
                raise TypeError("encoder_model must be a tf.keras.Model instance.")

            if not isinstance(decoder_model, tf.keras.Model):
                raise TypeError("decoder_model must be a tf.keras.Model instance.")

            if not isinstance(file_name_encoder, str) or not file_name_encoder:
                raise ValueError("file_name_encoder must be a non-empty string.")

            if not isinstance(file_name_decoder, str) or not file_name_decoder:
                raise ValueError("file_name_decoder must be a non-empty string.")

            if not isinstance(models_saved_path, str) or not models_saved_path:
                raise ValueError("models_saved_path must be a non-empty string.")

            if not isinstance(latent_mean_distribution, (int, float)):
                raise TypeError("latent_mean_distribution must be a number.")

            if not isinstance(latent_stander_deviation, (int, float)):
                raise TypeError("latent_stander_deviation must be a number.")

            if latent_stander_deviation <= 0:
                raise ValueError("latent_stander_deviation must be greater than 0.")

            if not isinstance(latent_dimension, int) or latent_dimension <= 0:
                raise ValueError("latent_dimension must be a positive integer.")

            # Inicializa modelos
            self._encoder = encoder_model
            self._decoder = decoder_model
            self._loss_function = loss_function
            self._total_loss_tracker = Mean(name="loss")
            self._latent_mean_distribution = latent_mean_distribution
            self._latent_stander_deviation = latent_stander_deviation
            self._latent_dimension = latent_dimension
            self._file_name_encoder = file_name_encoder
            self._file_name_decoder = file_name_decoder
            self._models_saved_path = models_saved_path

            # Modelo combinado encoder-decoder
            self._encoder_decoder_model = Model(self._encoder.input, self._decoder(self._encoder.output))

        @property
        def framework(self) -> str:
            """Retorna o framework sendo utilizado."""
            return 'tensorflow'

        @tf.function
        def train_step(self, batch):
            """
            Executa um passo de treinamento para o AutoEncoder.

            Args:
                batch: Batch de dados de entrada.

            Returns:
                dict: Dicionário contendo o valor da perda.
            """
            batch_x, batch_y = batch

            with tf.GradientTape() as gradient_ae:
                # Forward pass
                reconstructed_data = self._encoder_decoder_model(batch_x, training=True)
                # Calcula perda
                update_gradient_loss = tf.reduce_mean(tf.square(batch_y - reconstructed_data))

            # Calcula e aplica gradientes
            gradient_update = gradient_ae.gradient(update_gradient_loss,
                                                   self._encoder_decoder_model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradient_update, self._encoder_decoder_model.trainable_variables))

            # Atualiza métrica
            self._total_loss_tracker.update_state(update_gradient_loss)

            return {"loss": self._total_loss_tracker.result()}

        def get_samples(self, number_samples_per_class: Dict[str, Any]) -> Dict[int, numpy.ndarray]:
            """
            Gera amostras de dados sintéticos para cada classe especificada.

            Args:
                number_samples_per_class (dict): Dicionário especificando amostras por classe.

            Returns:
                dict: Dicionário com amostras geradas por classe.
            """
            generated_data = {}

            for label_class, number_instances in number_samples_per_class["classes"].items():
                # Cria labels one-hot encoded
                label_samples_generated = to_categorical(
                    [label_class] * number_instances,
                    num_classes=number_samples_per_class["number_classes"]
                )

                # Gera vetores de ruído latente
                latent_noise = numpy.random.normal(
                    self._latent_mean_distribution,
                    self._latent_stander_deviation,
                    (number_instances, self._latent_dimension)
                )

                # Gera amostras
                generated_samples = self._decoder.predict([latent_noise, label_samples_generated], verbose=0)
                generated_samples = numpy.rint(generated_samples)

                generated_data[label_class] = generated_samples

            return generated_data

        def save_model(self, directory: str, file_name: str) -> None:
            """Salva os modelos encoder e decoder."""
            if not os.path.exists(directory):
                os.makedirs(directory)

            encoder_file_name = os.path.join(directory, f"fold_{file_name}_encoder")
            decoder_file_name = os.path.join(directory, f"fold_{file_name}_decoder")

            # Salva encoder
            encoder_json = self._encoder.to_json()
            with open(f"{encoder_file_name}.json", "w") as json_file:
                json.dump(encoder_json, json_file)
            self._encoder.save_weights(f"{encoder_file_name}.weights.h5")

            # Salva decoder
            decoder_json = self._decoder.to_json()
            with open(f"{decoder_file_name}.json", "w") as json_file:
                json.dump(decoder_json, json_file)
            self._decoder.save_weights(f"{decoder_file_name}.weights.h5")

        def load_models(self, directory: str, file_name: str) -> None:
            """Carrega os modelos encoder e decoder."""
            from tensorflow.keras.models import model_from_json

            encoder_file_name = os.path.join(directory, f"{file_name}_encoder")
            decoder_file_name = os.path.join(directory, f"{file_name}_decoder")

            # Carrega encoder
            with open(f"{encoder_file_name}.json", 'r') as json_file:
                encoder_json = json.load(json_file)
            self._encoder = model_from_json(encoder_json)
            self._encoder.load_weights(f"{encoder_file_name}.weights.h5")

            # Carrega decoder
            with open(f"{decoder_file_name}.json", 'r') as json_file:
                decoder_json = json.load(json_file)
            self._decoder = model_from_json(decoder_json)
            self._decoder.load_weights(f"{decoder_file_name}.weights.h5")

        @property
        def decoder(self):
            """Retorna o decoder."""
            return self._decoder

        @property
        def encoder(self):
            """Retorna o encoder."""
            return self._encoder

        @decoder.setter
        def decoder(self, decoder):
            """Define o decoder."""
            self._decoder = decoder

        @encoder.setter
        def encoder(self, encoder):
            """Define o encoder."""
            self._encoder = encoder


elif ML_FRAMEWORK == 'pytorch':
    class AutoencoderAlgorithm(nn.Module):
        """
        AutoencoderAlgorithm (PyTorch)

        Implementação de AutoEncoder usando PyTorch, herdando de nn.Module.
        Fornece métodos para treinamento, geração de dados sintéticos, salvamento e carregamento.

        Args:
            encoder_model: A parte encoder do AutoEncoder.
            decoder_model: A parte decoder do AutoEncoder.
            loss_function: A função de perda para treinamento.
            file_name_encoder (str): O nome do arquivo para salvar o encoder.
            file_name_decoder (str): O nome do arquivo para salvar o decoder.
            models_saved_path (str): O caminho para salvar os modelos.
            latent_mean_distribution (float): Média da distribuição do espaço latente.
            latent_stander_deviation (float): Desvio padrão da distribuição do espaço latente.
            latent_dimension (int): A dimensionalidade do espaço latente.

        Example:
            >>> import os
            >>> os.environ['ML_FRAMEWORK'] = 'pytorch'
            >>> autoencoder = AutoencoderAlgorithm(
            ...     encoder_model=encoder_model,
            ...     decoder_model=decoder_model,
            ...     loss_function=loss_fn,
            ...     file_name_encoder="encoder_model",
            ...     file_name_decoder="decoder_model",
            ...     models_saved_path="./autoencoder_models/",
            ...     latent_mean_distribution=0.0,
            ...     latent_stander_deviation=1.0,
            ...     latent_dimension=64
            ... )
            >>> autoencoder.compile(optimizer=optimizer)
            >>> autoencoder.fit(train_dataset, epochs=50)
        """

        def __init__(self,
                     encoder_model,
                     decoder_model,
                     loss_function,
                     file_name_encoder,
                     file_name_decoder,
                     models_saved_path,
                     latent_mean_distribution,
                     latent_stander_deviation,
                     latent_dimension):

            super().__init__()

            # Validações
            if not isinstance(encoder_model, nn.Module):
                raise TypeError("encoder_model must be a nn.Module instance.")

            if not isinstance(decoder_model, nn.Module):
                raise TypeError("decoder_model must be a nn.Module instance.")

            if not isinstance(file_name_encoder, str) or not file_name_encoder:
                raise ValueError("file_name_encoder must be a non-empty string.")

            if not isinstance(file_name_decoder, str) or not file_name_decoder:
                raise ValueError("file_name_decoder must be a non-empty string.")

            if not isinstance(models_saved_path, str) or not models_saved_path:
                raise ValueError("models_saved_path must be a non-empty string.")

            if not isinstance(latent_mean_distribution, (int, float)):
                raise TypeError("latent_mean_distribution must be a number.")

            if not isinstance(latent_stander_deviation, (int, float)):
                raise TypeError("latent_stander_deviation must be a number.")

            if latent_stander_deviation <= 0:
                raise ValueError("latent_stander_deviation must be greater than 0.")

            if not isinstance(latent_dimension, int) or latent_dimension <= 0:
                raise ValueError("latent_dimension must be a positive integer.")

            # Inicializa modelos
            self._encoder = encoder_model
            self._decoder = decoder_model
            self._loss_function = loss_function
            self._latent_mean_distribution = latent_mean_distribution
            self._latent_stander_deviation = latent_stander_deviation
            self._latent_dimension = latent_dimension
            self._file_name_encoder = file_name_encoder
            self._file_name_decoder = file_name_decoder
            self._models_saved_path = models_saved_path
            self.optimizer = None
            self.loss_accumulator = []

        @property
        def framework(self) -> str:
            """Retorna o framework sendo utilizado."""
            return 'pytorch'

        def compile(self, optimizer=None, loss=None, *args, **kwargs):
            """Compila o modelo definindo otimizador e loss."""
            if optimizer is not None:
                self.optimizer = optimizer
            if loss is not None:
                self._loss_function = loss

        def forward(self, data_input, label_input):
            """
            Forward pass do autoencoder.

            Args:
                data_input: Tensor de dados de entrada
                label_input: Tensor de labels (one-hot encoded)

            Returns:
                Tensor reconstruído
            """
            # O encoder retorna (latent, labels)
            latent, _ = self._encoder(data_input, label_input)
            # O decoder recebe (latent, labels)
            reconstructed = self._decoder(latent, label_input)
            return reconstructed

        def train_step(self, batch):
            """
            Executa um passo de treinamento.

            Args:
                batch: Tupla (batch_x, batch_y) onde:
                    - batch_x: Tupla (features, labels)
                    - batch_y: Features target (para reconstrução)
            """
            # Desempacota o batch
            batch_x, batch_y = batch

            # batch_x é uma tupla (features, labels)
            if isinstance(batch_x, (tuple, list)):
                features, labels = batch_x
            else:
                raise ValueError("batch_x must be a tuple of (features, labels)")

            self.train()
            self.optimizer.zero_grad()

            # Forward pass com features e labels
            reconstructed_data = self.forward(features, labels)

            # Calcula perda
            loss = F.mse_loss(reconstructed_data, batch_y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Acumula perda
            self.loss_accumulator.append(loss.item())
            avg_loss = sum(self.loss_accumulator[-100:]) / len(self.loss_accumulator[-100:])

            return {"loss": avg_loss}

        def fit(self, x=None, y=None, batch_size=32, epochs=1, verbose=1,
                validation_data=None, shuffle=True, callbacks=None, *args, **kwargs):
            """
            Treina o modelo autoencoder (PyTorch).

            Args:
                x: Pode ser:
                   - Tupla (features, labels) onde ambos são arrays
                   - Array de features (requer y)
                y: Target data (features para reconstrução)
                batch_size: Tamanho do batch
                epochs: Número de épocas
                verbose: Nível de verbosidade
                validation_data: Dados de validação (não implementado)
                shuffle: Se deve embaralhar os dados
                callbacks: Lista de callbacks (não implementado completamente)
            """
            if self.optimizer is None:
                raise ValueError("Model must be compiled before training. Call .compile(optimizer=...) first.")

            # Processa entrada
            if isinstance(x, (tuple, list)) and len(x) == 2:
                features, labels = x
                target = y if y is not None else features
            elif y is not None:
                features = x
                labels = None  # Labels serão derivadas se necessário
                target = y
            else:
                raise ValueError("You must provide either x as (features, labels) tuple or both x and y separately.")

            # Converte para numpy
            if not isinstance(features, numpy.ndarray):
                features = numpy.array(features)
            if labels is not None and not isinstance(labels, numpy.ndarray):
                labels = numpy.array(labels)
            if not isinstance(target, numpy.ndarray):
                target = numpy.array(target)

            num_samples = len(features)
            history = {'loss': []}

            # Loop de treinamento
            for epoch in range(epochs):
                epoch_losses = []

                if shuffle:
                    indices = numpy.random.permutation(num_samples)
                    features = features[indices]
                    if labels is not None:
                        labels = labels[indices]
                    target = target[indices]

                num_batches = int(numpy.ceil(num_samples / batch_size))

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, num_samples)

                    batch_features = features[start_idx:end_idx]
                    batch_target = target[start_idx:end_idx]

                    # Converte para tensores
                    batch_features_tensor = torch.from_numpy(batch_features).float()
                    batch_target_tensor = torch.from_numpy(batch_target).float()

                    if labels is not None:
                        batch_labels = labels[start_idx:end_idx]
                        batch_labels_tensor = torch.from_numpy(batch_labels).float()
                    else:
                        # Se labels não fornecidas, cria dummy labels
                        batch_labels_tensor = torch.zeros(len(batch_features), 2).float()

                    # batch_x deve ser uma tupla (features, labels)
                    batch_x = (batch_features_tensor, batch_labels_tensor)
                    batch_y = batch_target_tensor

                    loss_dict = self.train_step((batch_x, batch_y))
                    epoch_losses.append(float(loss_dict['loss']))

                avg_loss = numpy.mean(epoch_losses)
                history['loss'].append(avg_loss)

                if verbose > 0:
                    if verbose == 1:
                        progress = (epoch + 1) / epochs
                        bar_length = 30
                        filled = int(bar_length * progress)
                        bar = '=' * filled + '>' + '.' * (bar_length - filled - 1)
                        print(f'\rEpoch {epoch + 1}/{epochs} [{bar}] - loss: {avg_loss:.4f}', end='', flush=True)
                        if epoch == epochs - 1:
                            print()
                    elif verbose == 2:
                        print(f'Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}')

            class History:
                def __init__(self, history_dict):
                    self.history = history_dict

            return History(history)

        def get_samples(self, number_samples_per_class: Dict[str, Any]) -> Dict[int, numpy.ndarray]:
            """Gera amostras sintéticas."""
            generated_data = {}
            self.eval()

            for label_class, number_instances in number_samples_per_class["classes"].items():
                # Cria labels one-hot
                label_samples_generated = numpy.zeros((number_instances, number_samples_per_class["number_classes"]))
                label_samples_generated[:, label_class] = 1

                # Gera ruído latente
                latent_noise = numpy.random.normal(
                    self._latent_mean_distribution,
                    self._latent_stander_deviation,
                    (number_instances, self._latent_dimension)
                )

                # Gera amostras
                with torch.no_grad():
                    latent_tensor = torch.from_numpy(latent_noise).float()
                    labels_tensor = torch.from_numpy(label_samples_generated).float()
                    generated_samples = self._decoder(latent_tensor, labels_tensor)
                    generated_samples = generated_samples.cpu().numpy()

                generated_samples = numpy.rint(generated_samples)
                generated_data[label_class] = generated_samples

            return generated_data

        def save_model(self, directory: str, file_name: str) -> None:
            """Salva os modelos."""
            if not os.path.exists(directory):
                os.makedirs(directory)

            encoder_file_name = os.path.join(directory, f"fold_{file_name}_encoder.pth")
            decoder_file_name = os.path.join(directory, f"fold_{file_name}_decoder.pth")

            torch.save({'model_state_dict': self._encoder.state_dict()}, encoder_file_name)
            torch.save({'model_state_dict': self._decoder.state_dict()}, decoder_file_name)

        def load_models(self, directory: str, file_name: str) -> None:
            """Carrega os modelos."""
            encoder_file_name = os.path.join(directory, f"{file_name}_encoder.pth")
            decoder_file_name = os.path.join(directory, f"{file_name}_decoder.pth")

            encoder_checkpoint = torch.load(encoder_file_name)
            self._encoder.load_state_dict(encoder_checkpoint['model_state_dict'])

            decoder_checkpoint = torch.load(decoder_file_name)
            self._decoder.load_state_dict(decoder_checkpoint['model_state_dict'])

        @property
        def decoder(self):
            return self._decoder

        @property
        def encoder(self):
            return self._encoder

        @decoder.setter
        def decoder(self, decoder):
            self._decoder = decoder

        @encoder.setter
        def encoder(self, encoder):
            self._encoder = encoder