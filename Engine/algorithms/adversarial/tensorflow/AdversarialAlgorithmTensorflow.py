#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__last_update__ = '2025/12/07'

# MIT License - Copyright (c) 2025 Synthetic Ocean AI

try:
    import os
    import sys
    import numpy

    import logging
    import tensorflow

    from pathlib import Path

    from tensorflow.keras.models import Model
    from tensorflow.keras.metrics import Mean
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import model_from_json

    from tensorflow.keras.losses import BinaryCrossentropy

except ImportError as error:
    logging.error(error)
    sys.exit(-1)


class AdversarialAlgorithmTensorflow(Model):
    """
    Implements an adversarial training algorithm, typically used in Generative adversarial Networks (GANs).
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
                 latent_standard_deviation,
                 smoothing_rate,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        if latent_dimension <= 0:
            raise ValueError("Latent dimension must be greater than 0.")
        if not isinstance(file_name_discriminator, str) or not file_name_discriminator:
            raise ValueError("Discriminator file name must be a non-empty string.")
        if not isinstance(file_name_generator, str) or not file_name_generator:
            raise ValueError("Generator file name must be a non-empty string.")
        if not isinstance(models_saved_path, str) or not models_saved_path:
            raise ValueError("architectures saved path must be a non-empty string.")
        if not isinstance(latent_mean_distribution, (int, float)):
            raise TypeError("Latent mean distribution must be a number.")
        if not isinstance(latent_standard_deviation, (int, float)):
            raise TypeError("Latent standard deviation must be a number.")
        if latent_standard_deviation <= 0:
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
        self._latent_standard_deviation = latent_standard_deviation
        self._file_name_discriminator = file_name_discriminator
        self._file_name_generator = file_name_generator
        self._models_saved_path = models_saved_path

        # ** FIX: Initialize loss trackers **
        self._loss_d_tracker = Mean(name="loss_d")
        self._loss_g_tracker = Mean(name="loss_g")
        self._total_loss_tracker = Mean(name="loss")

    def compile(self, optimizer_generator, optimizer_discriminator, loss_generator, loss_discriminator, *args,
                **kwargs):
        super().compile(*args, **kwargs)
        self._optimizer_generator = optimizer_generator
        self._optimizer_discriminator = optimizer_discriminator
        self._loss_generator = loss_generator
        self._loss_discriminator = loss_discriminator

    def call(self, inputs, training=False):
        """
        Define the forward pass of the model (required by Keras Model API).
        For a GAN, this generates synthetic samples.

        Args:
            inputs: Can be either:
                - A tuple (latent_vector, labels) for conditional generation
                - Just latent_vector for unconditional generation
            training: Boolean indicating if in training mode

        Returns:
            Generated synthetic samples
        """
        # Se inputs for uma tupla, desempacote
        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 2:
                latent_vector, labels = inputs
            else:
                latent_vector = inputs[0]
                labels = None
        else:
            latent_vector = inputs
            labels = None

        # Gerar amostras sintéticas usando o gerador
        if labels is not None:
            # Geração condicional (com labels)
            synthetic_samples = self._generator([latent_vector, labels], training=training)
        else:
            # Geração incondicional (sem labels)
            synthetic_samples = self._generator(latent_vector, training=training)

        return synthetic_samples

    @tensorflow.function
    def train_step(self, batch):
        """
        Performs a single training step for both generator and discriminator.

        Args:
            batch: Tuple of (real_features, real_labels)

        Returns:
            Dictionary with loss metrics
        """
        # Unpack the batch into real features and real labels
        real_feature, real_samples_label = batch

        # Get the current batch size
        batch_size = tensorflow.shape(real_feature)[0]

        # FIX 1: Expandir labels corretamente (verificar se precisa)
        if len(real_samples_label.shape) == 1:
            real_samples_label = tensorflow.expand_dims(real_samples_label, axis=-1)

        # ==================================================
        # FASE 1: TREINAR O DISCRIMINADOR
        # ==================================================

        # Gerar ruído latente para o gerador
        latent_space = tensorflow.random.normal(
            shape=(batch_size, self._latent_dimension),
            mean=self._latent_mean_distribution,
            stddev=self._latent_standard_deviation
        )

        # FIX 2: Gerar amostras sintéticas com training=True
        synthetic_feature = self._generator([latent_space, real_samples_label], training=True)

        # Treinar discriminador
        with tensorflow.GradientTape() as discriminator_gradient:
            # Obter predições do discriminador para amostras reais e sintéticas
            label_predicted_real = self._discriminator([real_feature, real_samples_label], training=True)
            label_predicted_synthetic = self._discriminator([synthetic_feature, real_samples_label], training=True)

            # Concatenar todas as predições
            label_predicted_all_samples = tensorflow.concat(
                [label_predicted_real, label_predicted_synthetic],
                axis=0
            )

            # FIX 3: Label smoothing correto e simplificado
            # Para amostras reais: labels próximos de 0 (0.0 a 0.15)
            # Para amostras sintéticas: labels próximos de 1 (0.85 a 1.0)
            smooth_real_labels = tensorflow.random.uniform(
                shape=tensorflow.shape(label_predicted_real),
                minval=0.0,
                maxval=0.15
            )

            smooth_synthetic_labels = tensorflow.random.uniform(
                shape=tensorflow.shape(label_predicted_synthetic),
                minval=0.85,
                maxval=1.0
            )

            # Concatenar labels suavizados
            tensor_labels_predicted = tensorflow.concat(
                [smooth_real_labels, smooth_synthetic_labels],
                axis=0
            )

            # Calcular perda do discriminador
            loss_d = self._loss_discriminator(tensor_labels_predicted, label_predicted_all_samples)

        # FIX 4: Aplicar gradientes apenas se houver variáveis treináveis
        trainable_vars_discriminator = self._discriminator.trainable_variables
        if trainable_vars_discriminator:
            gradient_discriminator = discriminator_gradient.gradient(loss_d, trainable_vars_discriminator)
            self._optimizer_discriminator.apply_gradients(zip(gradient_discriminator, trainable_vars_discriminator))

        # ==================================================
        # FASE 2: TREINAR O GERADOR
        # ==================================================

        with tensorflow.GradientTape() as generator_gradient:
            # FIX 5: Gerar NOVO ruído latente (não reutilizar o anterior)
            latent_space_generator = tensorflow.random.normal(
                shape=(batch_size, self._latent_dimension),
                mean=self._latent_mean_distribution,
                stddev=self._latent_standard_deviation
            )

            # Gerar amostras sintéticas
            synthetic_feature_generator = self._generator(
                [latent_space_generator, real_samples_label],
                training=True
            )

            # FIX 6: Discriminador em modo de inferência durante treinamento do gerador
            predicted_labels = self._discriminator(
                [synthetic_feature_generator, real_samples_label],
                training=False
            )

            # Perda do gerador - queremos que o discriminador classifique como real (0)
            loss_g = self._loss_generator(tensorflow.zeros_like(predicted_labels), predicted_labels)

        # Aplicar gradientes ao gerador
        trainable_vars_generator = self._generator.trainable_variables
        if trainable_vars_generator:
            gradient_generator = generator_gradient.gradient(loss_g, trainable_vars_generator)
            self._optimizer_generator.apply_gradients(zip(gradient_generator, trainable_vars_generator))

        # ==================================================
        # ATUALIZAR MÉTRICAS E RETORNAR
        # ==================================================

        # Atualizar trackers de perda
        self._loss_d_tracker.update_state(loss_d)
        self._loss_g_tracker.update_state(loss_g)

        # Perda combinada
        combined_loss = (loss_d + loss_g) / 2.0
        self._total_loss_tracker.update_state(combined_loss)

        # Retornar todas as métricas como dicionário
        return {
            "loss": combined_loss,  # Perda total (esperada pelo fit)
            "loss_d": loss_d,  # Perda do discriminador
            "loss_g": loss_g  # Perda do gerador
        }

    @property
    def metrics(self):
        """Return metrics for tracking."""
        return [self._loss_d_tracker, self._loss_g_tracker, self._total_loss_tracker]

    def fit(self, x=None, y=None, batch_size=32, epochs=1, verbose=1,
            callbacks=None, validation_data=None, shuffle=True,
            initial_epoch=0, steps_per_epoch=None, validation_steps=None,
            validation_freq=1, optimizer=None, learning_rate=0.001, **kwargs):
        """
        Train the model with a simplified progress bar.
        """
        # Set optimizer if provided
        if optimizer is not None:
            self.optimizer = optimizer
        elif not hasattr(self, 'optimizer') or self.optimizer is None:
            self.optimizer = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate)

        # Prepare the dataset
        if isinstance(x, tensorflow.data.Dataset):
            train_dataset = x
        else:
            if y is None:
                y = x
            train_dataset = tensorflow.data.Dataset.from_tensor_slices((x, y))
            if shuffle:
                train_dataset = train_dataset.shuffle(buffer_size=len(x))
            train_dataset = train_dataset.batch(batch_size)

        # Calculate steps per epoch if not provided
        if steps_per_epoch is None:
            steps_per_epoch = len(train_dataset)

        # History to store metrics
        history = {'loss': [], 'loss_d': [], 'loss_g': []}

        # Training loop
        for epoch in range(initial_epoch, epochs):
            # ** FIX: Reset all trackers at the start of each epoch **
            self._total_loss_tracker.reset_state()
            self._loss_d_tracker.reset_state()
            self._loss_g_tracker.reset_state()

            if verbose == 1:
                print(f'\nEpoch {epoch + 1}/{epochs}')

            # Progress tracking
            step = 0
            for batch_data in train_dataset:
                step += 1

                # Perform training step
                metrics = self.train_step(batch_data)

                # ** FIX: Now metrics['loss'] exists **
                current_loss = float(metrics['loss'])
                current_loss_d = float(metrics['loss_d'])
                current_loss_g = float(metrics['loss_g'])

                # Simple progress bar
                if verbose == 1:
                    progress = int(50 * step / steps_per_epoch)
                    bar = '=' * progress + '>' + '.' * (50 - progress - 1)
                    print(
                        f'\r[{bar}] {step}/{steps_per_epoch} - loss: {current_loss:.4f} - loss_d: {current_loss_d:.4f} - loss_g: {current_loss_g:.4f}',
                        end='', flush=True)

                if step >= steps_per_epoch:
                    break

            # Store epoch losses
            epoch_loss = float(self._total_loss_tracker.result())
            epoch_loss_d = float(self._loss_d_tracker.result())
            epoch_loss_g = float(self._loss_g_tracker.result())

            history['loss'].append(epoch_loss)
            history['loss_d'].append(epoch_loss_d)
            history['loss_g'].append(epoch_loss_g)

            if verbose == 1:
                print(f' - loss: {epoch_loss:.4f} - loss_d: {epoch_loss_d:.4f} - loss_g: {epoch_loss_g:.4f}')
            elif verbose == 2:
                print(
                    f'Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.4f} - loss_d: {epoch_loss_d:.4f} - loss_g: {epoch_loss_g:.4f}')

            # Validation
            if validation_data is not None and (epoch + 1) % validation_freq == 0:
                val_loss = self._evaluate_validation(validation_data, validation_steps)
                if 'val_loss' not in history:
                    history['val_loss'] = []
                history['val_loss'].append(val_loss)

                if verbose >= 1:
                    print(f' - val_loss: {val_loss:.4f}')

            # callbacks
            if callbacks is not None:
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, {
                            'loss': epoch_loss,
                            'loss_d': epoch_loss_d,
                            'loss_g': epoch_loss_g
                        })

        # Return history object
        class History:
            def __init__(self, history_dict):
                self.history = history_dict

        return History(history)

    def _evaluate_validation(self, validation_data, validation_steps=None):
        """
        Evaluate the model on validation data WITHOUT updating weights.
        """
        # FIX: Resetar trackers antes da validação
        val_loss_tracker = Mean(name="val_loss")
        val_loss_d_tracker = Mean(name="val_loss_d")
        val_loss_g_tracker = Mean(name="val_loss_g")

        step = 0
        for batch_data in validation_data:
            real_feature, real_samples_label = batch_data
            batch_size = tensorflow.shape(real_feature)[0]
            real_samples_label = tensorflow.expand_dims(real_samples_label, axis=-1)

            # Gerar dados sintéticos
            latent_space = tensorflow.random.normal(shape=(batch_size, self._latent_dimension))
            synthetic_feature = self._generator([latent_space, real_samples_label], training=False)

            # Avaliar discriminador (SEM treinar)
            label_predicted_real = self._discriminator([real_feature, real_samples_label], training=False)
            label_predicted_synthetic = self._discriminator([synthetic_feature, real_samples_label], training=False)

            label_predicted_all = tensorflow.concat([label_predicted_real, label_predicted_synthetic], axis=0)
            tensor_labels = tensorflow.concat([
                tensorflow.zeros_like(label_predicted_real),
                tensorflow.ones_like(label_predicted_synthetic)
            ], axis=0)

            # Calcular perdas (sem aplicar gradientes)
            loss_d = self._loss_discriminator(tensor_labels, label_predicted_all)

            predicted_labels = self._discriminator([synthetic_feature, real_samples_label], training=False)
            loss_g = self._loss_generator(tensorflow.zeros_like(predicted_labels), predicted_labels)

            combined_loss = (loss_d + loss_g) / 2.0

            # Atualizar trackers
            val_loss_tracker.update_state(combined_loss)
            val_loss_d_tracker.update_state(loss_d)
            val_loss_g_tracker.update_state(loss_g)

            step += 1
            if validation_steps is not None and step >= validation_steps:
                break

        return float(val_loss_tracker.result())

    def get_samples(self, number_samples_per_class):
        """
        Generates synthetic data samples for each specified class using the trained generator.
        """
        generated_data = {}

        for label_class, number_instances in number_samples_per_class["classes"].items():
            # Create one-hot encoded labels
            label_samples_generated = to_categorical(
                [label_class] * number_instances,
                num_classes=number_samples_per_class["number_classes"]
            )

            # Generate random noise vectors
            latent_noise = numpy.random.normal(
                self._latent_mean_distribution,
                self._latent_standard_deviation,
                (number_instances, self._latent_dimension)
            )

            # Generate synthetic samples
            generated_samples = self._generator.predict([latent_noise, label_samples_generated], verbose=0)

            # Store in dictionary
            generated_data[label_class] = generated_samples

        return generated_data

    def save_model(self, path_output, k_fold):
        try:
            logging.info("Starting to save adversarial Model for fold {}...".format(k_fold))

            path_directory = os.path.join(path_output, self._models_saved_path)
            Path(path_directory).mkdir(parents=True, exist_ok=True)

            # FIX: Usar k_fold consistentemente (sem +1)
            discriminator_file_name = self._file_name_discriminator + "_" + str(k_fold)
            generator_file_name = self._file_name_generator + "_" + str(k_fold)

            # FIX: Não criar subpasta fold_X ou ser consistente com load_models
            path_model = path_directory  # Salvar direto no path_directory

            discriminator_file_name = os.path.join(path_model, discriminator_file_name)
            generator_file_name = os.path.join(path_model, generator_file_name)

            # Salvar discriminador
            discriminator_model_json = self._discriminator.to_json()
            with open(discriminator_file_name + ".json", "w") as json_file:
                json_file.write(discriminator_model_json)
            self._discriminator.save_weights(discriminator_file_name + ".h5")

            # Salvar gerador
            generator_model_json = self._generator.to_json()
            with open(generator_file_name + ".json", "w") as json_file:
                json_file.write(generator_model_json)
            self._generator.save_weights(generator_file_name + ".h5")

            logging.info("models saved successfully for fold {}".format(k_fold))

        except Exception as e:
            logging.error("An error occurred while saving the models: {}".format(e))
            raise

    def load_models(self, path_output, k_fold):
        try:
            logging.info("Loading adversarial Model for fold {}...".format(k_fold))

            path_directory = os.path.join(path_output, self._models_saved_path)

            # FIX: Usar k_fold consistentemente (igual ao save_model)
            discriminator_file_name = self._file_name_discriminator + "_" + str(k_fold)
            generator_file_name = self._file_name_generator + "_" + str(k_fold)

            discriminator_file_name = os.path.join(path_directory, discriminator_file_name)
            generator_file_name = os.path.join(path_directory, generator_file_name)

            # Carregar discriminador
            with open(discriminator_file_name + ".json", 'r') as json_file:
                discriminator_model_json = json_file.read()
            self._discriminator = model_from_json(discriminator_model_json)
            self._discriminator.load_weights(discriminator_file_name + ".h5")

            # Carregar gerador
            with open(generator_file_name + ".json", 'r') as json_file:
                generator_model_json = json_file.read()
            self._generator = model_from_json(generator_model_json)
            self._generator.load_weights(generator_file_name + ".h5")

            logging.info("models loaded successfully for fold {}".format(k_fold))

        except Exception as e:
            logging.error("An error occurred while loading the models: {}".format(e))
            raise

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