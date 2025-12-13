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
    Implements an adversarial training algorithm, typically used in Generative Adversarial Networks (GANs).
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

        super().__init__(*args, **kwargs)

        if latent_dimension <= 0:
            raise ValueError("Latent dimension must be greater than 0.")
        if not isinstance(file_name_discriminator, str) or not file_name_discriminator:
            raise ValueError("Discriminator file name must be a non-empty string.")
        if not isinstance(file_name_generator, str) or not file_name_generator:
            raise ValueError("Generator file name must be a non-empty string.")
        if not isinstance(models_saved_path, str) or not models_saved_path:
            raise ValueError("Architectures saved path must be a non-empty string.")
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

    @tensorflow.function
    def train_step(self, batch):
        """
        Performs a single training step for both generator and discriminator.
        """
        # Unpack the batch into real features and real labels
        real_feature, real_samples_label = batch

        # Get the current batch size
        batch_size = tensorflow.shape(real_feature)[0]

        # Expand the label tensor to match the expected shape
        real_samples_label = tensorflow.expand_dims(real_samples_label, axis=-1)

        # Sample random noise vectors for the generator input
        latent_space = tensorflow.random.normal(shape=(batch_size, self._latent_dimension))

        # Generate synthetic features
        synthetic_feature = self._generator([latent_space, real_samples_label], training=False)

        # Train discriminator
        with tensorflow.GradientTape() as discriminator_gradient:
            # Get discriminator predictions on real and synthetic samples
            label_predicted_real = self._discriminator([real_feature, real_samples_label], training=True)
            label_predicted_synthetic = self._discriminator([synthetic_feature, real_samples_label], training=True)

            # Concatenate predictions
            label_predicted_all_samples = tensorflow.concat([label_predicted_real, label_predicted_synthetic], axis=0)

            # Create ground-truth labels (real=0, fake=1)
            list_all_labels_predicted = [
                tensorflow.zeros_like(label_predicted_real),
                tensorflow.ones_like(label_predicted_synthetic)
            ]
            tensor_labels_predicted = tensorflow.concat(list_all_labels_predicted, axis=0)

            # Label smoothing
            smooth_tensor_real_data = 0.15 * tensorflow.random.uniform(tensorflow.shape(label_predicted_real))
            smooth_tensor_synthetic_data = -0.15 * tensorflow.random.uniform(
                tensorflow.shape(label_predicted_synthetic))
            tensor_labels_predicted += tensorflow.concat([smooth_tensor_real_data, smooth_tensor_synthetic_data],
                                                         axis=0)

            # Compute discriminator loss
            loss_d = self._loss_discriminator(tensor_labels_predicted, label_predicted_all_samples)

        # Update discriminator
        gradient_tape_loss = discriminator_gradient.gradient(loss_d, self._discriminator.trainable_variables)
        self._optimizer_discriminator.apply_gradients(zip(gradient_tape_loss, self._discriminator.trainable_variables))

        # Train generator
        with tensorflow.GradientTape() as generator_gradient:
            # Generate synthetic samples
            latent_space = tensorflow.random.normal(shape=(batch_size, self._latent_dimension))
            synthetic_feature = self._generator([latent_space, real_samples_label], training=True)

            # Get discriminator predictions for synthetic samples
            predicted_labels = self._discriminator([synthetic_feature, real_samples_label], training=False)

            # Compute generator loss
            loss_g = self._loss_generator(tensorflow.zeros_like(predicted_labels), predicted_labels)

        # Update generator
        gradient_tape_loss = generator_gradient.gradient(loss_g, self._generator.trainable_variables)
        self._optimizer_generator.apply_gradients(zip(gradient_tape_loss, self._generator.trainable_variables))

        # ** FIX: Update loss trackers **
        self._loss_d_tracker.update_state(loss_d)
        self._loss_g_tracker.update_state(loss_g)
        # Combined loss for overall tracking
        combined_loss = (loss_d + loss_g) / 2.0
        self._total_loss_tracker.update_state(combined_loss)

        # ** FIX: Return all three metrics **
        return {
            "loss_d": loss_d,
            "loss_g": loss_g,
            "loss": combined_loss  # Add this key that fit() expects
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

            # Callbacks
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
        Evaluate the model on validation data.
        """
        val_losses = []
        step = 0

        for batch_data in validation_data:
            batch_x, batch_y = batch_data
            # Perform validation step (without updating weights)
            metrics = self.train_step(batch_data)
            val_losses.append(float(metrics['loss']))

            step += 1
            if validation_steps is not None and step >= validation_steps:
                break

        return numpy.mean(val_losses) if val_losses else 0.0

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
                self._latent_stander_deviation,
                (number_instances, self._latent_dimension)
            )

            # Generate synthetic samples
            generated_samples = self._generator.predict([latent_noise, label_samples_generated], verbose=0)

            # Round to nearest integer
            generated_samples = numpy.rint(generated_samples)

            # Store in dictionary
            generated_data[label_class] = generated_samples

        return generated_data

    def save_model(self, path_output, k_fold):
        try:
            logging.info("Starting to save Adversarial Model for fold {}...".format(k_fold))

            # Create directory for saving models
            path_directory = os.path.join(path_output, self._models_saved_path)
            Path(path_directory).mkdir(parents=True, exist_ok=True)
            logging.info("Created/verified directory at: {}".format(path_directory))

            # Filenames
            discriminator_file_name = self._file_name_discriminator + "_" + str(k_fold)
            generator_file_name = self._file_name_generator + "_" + str(k_fold)

            # Directory for current fold
            path_model = os.path.join(path_directory, "fold_" + str(k_fold + 1))
            Path(path_model).mkdir(parents=True, exist_ok=True)
            logging.info("Created/verified fold directory at: {}".format(path_model))

            # Full paths
            discriminator_file_name = os.path.join(path_model, discriminator_file_name)
            generator_file_name = os.path.join(path_model, generator_file_name)

            # Save discriminator
            logging.info("Saving discriminator model...")
            discriminator_model_json = self._discriminator.to_json()
            with open(discriminator_file_name + ".json", "w") as json_file:
                json_file.write(discriminator_model_json)
            self._discriminator.save_weights(discriminator_file_name + ".h5")
            logging.info("Discriminator model saved at: {}.json and {}.h5".format(discriminator_file_name,
                                                                                  discriminator_file_name))

            # Save generator
            logging.info("Saving generator model...")
            generator_model_json = self._generator.to_json()
            with open(generator_file_name + ".json", "w") as json_file:
                json_file.write(generator_model_json)
            self._generator.save_weights(generator_file_name + ".h5")
            logging.info("Generator model saved at: {}.json and {}.h5".format(generator_file_name,
                                                                              generator_file_name))

        except FileExistsError:
            logging.error("Model file already exists. Aborting.")
            exit(-1)
        except Exception as e:
            logging.error("An error occurred while saving the models: {}".format(e))
            exit(-1)

    def load_models(self, path_output, k_fold):
        try:
            logging.info("Loading Adversarial Model for fold {}...".format(k_fold + 1))

            # Directory containing saved models
            path_directory = os.path.join(path_output, self._models_saved_path)

            # Filenames
            discriminator_file_name = self._file_name_discriminator + "_" + str(k_fold + 1)
            generator_file_name = self._file_name_generator + "_" + str(k_fold + 1)

            # Full paths
            discriminator_file_name = os.path.join(path_directory, discriminator_file_name)
            generator_file_name = os.path.join(path_directory, generator_file_name)

            # Load discriminator
            logging.info("Loading discriminator model from: {}.json".format(discriminator_file_name))
            with open(discriminator_file_name + ".json", 'r') as json_file:
                discriminator_model_json = json_file.read()
            self._discriminator = model_from_json(discriminator_model_json)
            self._discriminator.load_weights(discriminator_file_name + ".h5")
            logging.info("Loaded discriminator weights from: {}.h5".format(discriminator_file_name))

            # Load generator
            logging.info("Loading generator model from: {}.json".format(generator_file_name))
            with open(generator_file_name + ".json", 'r') as json_file:
                generator_model_json = json_file.read()
            self._generator = model_from_json(generator_model_json)
            self._generator.load_weights(generator_file_name + ".h5")
            logging.info("Loaded generator weights from: {}.h5".format(generator_file_name))

        except FileNotFoundError:
            logging.error("Model file not found. Please provide an existing and valid model.")
            exit(-1)
        except Exception as e:
            logging.error("An error occurred while loading the models: {}".format(e))
            exit(-1)

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