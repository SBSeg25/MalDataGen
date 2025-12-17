#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com'
__version__ = '{1}.{0}.{2}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/16'
__credits__ = ['Kayuã Oleques']

# MIT License
#
# Copyright (c) 2025 Synthetic Ocean AI

try:
    import os
    import sys
    import json
    import numpy

    import tensorflow

    from abc import ABC
    from typing import Any
    from typing import Callable

    from tensorflow.keras import Model
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.losses import BinaryCrossentropy

except ImportError as error:
    print(error)
    sys.exit(-1)


class WassersteinGPAlgorithmTensorflow(Model):
    """
    A Wasserstein GAN with Gradient Penalty (WGAN-GP) model with adaptive input handling.

    This class represents a WGAN-GP consisting of a generator and discriminator model.
    It implements the Wasserstein loss with gradient penalty to improve training stability.

    This implementation automatically adapts to any input shape: (x), (x, y), (x, y, z), etc.

    Reference:
        Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein GAN.
        In Proceedings of the 34th International Conference on Machine Learning (ICML 2017) (Vol. 70, pp. 214-223).
        http://proceedings.mlr.press/v70/arjovsky17a.html

    Example:
        >>> # Example with 1D data
        >>> wgan_gp_1d = WassersteinGPAlgorithmTensorflow(
        ...     generator_model=generator_1d,
        ...     discriminator_model=discriminator_1d,
        ...     latent_dimension=64,
        ...     input_shape=(100,)
        ... )
        >>> wgan_gp_1d.fit(data_1d, epochs=50)

        >>> # Example with 2D data (images)
        >>> wgan_gp_2d = WassersteinGPAlgorithmTensorflow(
        ...     generator_model=generator_2d,
        ...     discriminator_model=discriminator_2d,
        ...     latent_dimension=128,
        ...     input_shape=(28, 28)
        ... )
        >>> wgan_gp_2d.fit(data_2d, labels_2d, epochs=100)
    """

    def __init__(self,
                 generator_model,
                 discriminator_model,
                 latent_dimension,
                 generator_loss_fn=None,
                 discriminator_loss_fn=None,
                 file_name_discriminator=None,
                 file_name_generator=None,
                 models_saved_path=None,
                 latent_mean_distribution=0.0,
                 latent_standard_deviation=1.0,
                 smoothing_rate=0.0,
                 gradient_penalty_weight=10.0,
                 discriminator_steps=5,
                 input_shape=None,
                 auto_adapt_shape=True,
                 *args,
                 **kwargs):
        """
        Initialize the WGAN-GP model.

        Args:
            generator_model: The generator model
            discriminator_model: The discriminator model
            latent_dimension: The dimension of the latent space
            generator_loss_fn: Loss function for generator
            discriminator_loss_fn: Loss function for discriminator
            file_name_discriminator: Filename for saving discriminator
            file_name_generator: Filename for saving generator
            models_saved_path: Path where models are saved
            latent_mean_distribution: Mean of latent distribution
            latent_standard_deviation: Std dev of latent distribution
            smoothing_rate: Rate for label smoothing
            gradient_penalty_weight: Weight for gradient penalty
            discriminator_steps: Number of discriminator updates per generator update
            input_shape: Expected input shape (without batch dimension)
            auto_adapt_shape: Whether to automatically adapt to input data shape
        """
        super().__init__(*args, **kwargs)

        # Initialize instance variables
        self._generator_optimizer = None
        self._discriminator_optimizer = None
        self._generator = generator_model
        self._discriminator = discriminator_model
        self._latent_dimension = latent_dimension
        self._discriminator_loss_fn = discriminator_loss_fn
        self._generator_loss_fn = generator_loss_fn
        self._gradient_penalty_weight = gradient_penalty_weight
        self._smooth_rate = smoothing_rate
        self._latent_mean_distribution = latent_mean_distribution
        self._latent_standard_deviation = latent_standard_deviation
        self._file_name_discriminator = file_name_discriminator
        self._file_name_generator = file_name_generator
        self._models_saved_path = models_saved_path
        self._discriminator_steps = discriminator_steps

        # Shape adaptation
        self._input_shape = input_shape
        self._auto_adapt_shape = auto_adapt_shape
        self._inferred_shape = None

        # Initialize loss tracker
        self._total_loss_tracker = tensorflow.keras.metrics.Mean(name="total_loss")

    @staticmethod
    def _infer_data_shape(data):
        """
        Infer the shape of input data, excluding the batch dimension.

        Args:
            data: Input data (tensor, array, or tuple/list).

        Returns:
            tuple: Shape of the data excluding batch dimension.
        """
        if isinstance(data, (tuple, list)):
            # If data is a tuple/list, infer from first element
            data = data[0]

        if tensorflow.is_tensor(data):
            shape = tuple(data.shape[1:])
        elif isinstance(data, numpy.ndarray):
            shape = data.shape[1:] if len(data.shape) > 1 else data.shape
        else:
            # Try to convert to tensor and get shape
            try:
                tensor_data = tensorflow.constant(data)
                shape = tuple(tensor_data.shape[1:])
            except:
                raise ValueError(f"Cannot infer shape from data of type {type(data)}")

        return shape

    def _validate_and_adapt_shape(self, data):
        """
        Validate input data shape and adapt if necessary.

        Args:
            data: Input data.

        Returns:
            bool: True if shape is valid or successfully adapted.
        """
        current_shape = self._infer_data_shape(data)

        if self._inferred_shape is None:
            self._inferred_shape = current_shape
            if self._input_shape is not None and self._input_shape != current_shape:
                print(f"Warning: Specified input_shape {self._input_shape} differs from inferred shape {current_shape}")
                if self._auto_adapt_shape:
                    print(f"Auto-adapting to shape: {current_shape}")
                    self._input_shape = current_shape
            elif self._input_shape is None:
                self._input_shape = current_shape
                print(f"Inferred input shape: {current_shape}")
        else:
            if current_shape != self._inferred_shape:
                if self._auto_adapt_shape:
                    print(f"Warning: Input shape changed from {self._inferred_shape} to {current_shape}")
                    self._inferred_shape = current_shape
                else:
                    raise ValueError(
                        f"Input shape mismatch: expected {self._inferred_shape}, got {current_shape}. "
                        f"Set auto_adapt_shape=True to allow dynamic shape changes."
                    )

        return True

    @staticmethod
    def _prepare_batch(batch):
        """
        Prepare batch data, handling different input formats.

        Args:
            batch: Input batch (can be single tensor, tuple, or list).

        Returns:
            tuple: (batch_x, batch_labels) where batch_x is the feature data.
        """
        if isinstance(batch, (tuple, list)):
            if len(batch) == 1:
                # Single input, no labels
                batch_x = batch[0]
                batch_labels = None
            elif len(batch) == 2:
                # Input and labels provided
                batch_x, batch_labels = batch
            else:
                # Multiple inputs, use first as input, second as labels
                batch_x = batch[0]
                batch_labels = batch[1]
        else:
            # Single tensor, no labels
            batch_x = batch
            batch_labels = None

        return batch_x, batch_labels

    def compile(self, optimizer_generator=None, optimizer_discriminator=None,
                loss_generator=None, loss_discriminator=None, *args, **kwargs):
        """
        Compile the WGAN-GP with custom optimizers and loss functions.

        Args:
            optimizer_generator: The optimizer for the generator.
            optimizer_discriminator: The optimizer for the discriminator.
            loss_generator: The loss function for the generator.
            loss_discriminator: The loss function for the discriminator.
        """
        super().compile()

        if optimizer_discriminator is not None:
            self._discriminator_optimizer = optimizer_discriminator
        if optimizer_generator is not None:
            self._generator_optimizer = optimizer_generator
        if loss_discriminator is not None:
            self._discriminator_loss_fn = loss_discriminator
        if loss_generator is not None:
            self._generator_loss_fn = loss_generator

    def gradient_penalty(self, batch_size, real_feature, real_label, synthetic_feature):
        """
        Compute the gradient penalty for the Wasserstein GAN.

        The gradient penalty is used to enforce the Lipschitz constraint on the discriminator's output.

        Parameters:
            batch_size (int): The batch size of the input data.
            real_feature (tensorflow.Tensor): Real data features.
            real_label (tensorflow.Tensor): Real data labels.
            synthetic_feature (tensorflow.Tensor): Synthetic (generated) data features.
        """
        # Generate random epsilon for interpolation
        epsilon = tensorflow.random.uniform(
            shape=[batch_size, 1],
            minval=0.0,
            maxval=1.0
        )

        # Interpolate between real and synthetic features
        interpolated_feature = epsilon * real_feature + (1.0 - epsilon) * synthetic_feature

        with tensorflow.GradientTape() as tape:
            # Watch the interpolated features for gradient computation
            tape.watch(interpolated_feature)

            # Get discriminator's output for the interpolated features
            labels_predicted = self.discriminator([interpolated_feature, real_label], training=True)

        # Calculate the gradient of the discriminator's output with respect to the interpolated features
        gradients = tape.gradient(labels_predicted, interpolated_feature)

        # Compute the L2 norm of gradients
        gradients_sqr = tensorflow.square(gradients)
        gradients_sqr_sum = tensorflow.reduce_sum(gradients_sqr, axis=1)
        gradient_l2_norm = tensorflow.sqrt(gradients_sqr_sum + 1e-12)

        # Calculate the gradient penalty as the mean squared difference from 1.0
        gradient_penalty_final = tensorflow.reduce_mean(tensorflow.square(gradient_l2_norm - 1.0))

        return gradient_penalty_final

    @staticmethod
    def calculate_samples_per_class(y_labels):
        """
        Calculate the distribution of samples per class from labels.

        Args:
            y_labels (array-like): Labels array

        Returns:
            dict: Dictionary with 'classes' and 'number_classes' keys
        """
        # Convert to numpy if needed
        if tensorflow.is_tensor(y_labels):
            y_labels = y_labels.numpy()

        # Handle one-hot encoded labels
        if len(y_labels.shape) == 2 and y_labels.shape[1] > 1:
            y_labels = numpy.argmax(y_labels, axis=1)

        # Count samples per class
        unique, counts = numpy.unique(y_labels, return_counts=True)

        return {
            "classes": dict(zip(unique.tolist(), counts.tolist())),
            "number_classes": len(unique)
        }

    def fit(self, x=None, y=None, batch_size=32, epochs=1, verbose=1,
            callbacks=None, validation_data=None, shuffle=True,
            initial_epoch=0, steps_per_epoch=None, validation_steps=None,
            validation_freq=1, optimizer=None, learning_rate=0.001, **kwargs):
        """
        Train the model with automatic shape adaptation.

        Args:
            x: Input data (any shape) or tf.data.Dataset.
            y: Target labels (if None, generates without labels).
            batch_size: Number of samples per gradient update.
            epochs: Number of epochs to train.
            verbose: 0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks: List of callbacks to apply during training.
            validation_data: Data for validation.
            shuffle: Whether to shuffle data before each epoch.
            initial_epoch: Epoch at which to start training.
            steps_per_epoch: Number of steps per epoch.
            validation_steps: Number of validation steps.
            validation_freq: Validation frequency.
            optimizer: Optimizer (dict with 'generator' and 'discriminator' keys).
            learning_rate: Learning rate for optimizers (if optimizer is None).

        Returns:
            History object containing training loss history.
        """

        # Set optimizers if provided
        if optimizer is not None:
            if isinstance(optimizer, dict):
                self._generator_optimizer = optimizer.get('generator')
                self._discriminator_optimizer = optimizer.get('discriminator')
            else:
                self._generator_optimizer = optimizer
                self._discriminator_optimizer = optimizer
        elif self._generator_optimizer is None or self._discriminator_optimizer is None:
            # Create default optimizers if none exist
            self._generator_optimizer = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate)
            self._discriminator_optimizer = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate)

        # Prepare the dataset
        if isinstance(x, tensorflow.data.Dataset):
            train_dataset = x
            # Try to infer shape from dataset
            for batch in train_dataset:
                self._validate_and_adapt_shape(batch)
                break
        else:
            # Validate and adapt shape
            self._validate_and_adapt_shape(x)

            # Handle different input formats
            if isinstance(x, tuple):
                # x is tuple: (data, labels)
                if len(x) == 2:
                    x_data, labels = x
                else:
                    x_data = x[0]
                    labels = None
            else:
                x_data = x
                labels = y

            # Convert to tensors
            if isinstance(x_data, numpy.ndarray):
                x_data = tensorflow.constant(x_data, dtype=tensorflow.float32)
            if labels is not None and isinstance(labels, numpy.ndarray):
                labels = tensorflow.constant(labels, dtype=tensorflow.float32)

            # Create dataset
            if labels is not None:
                train_dataset = tensorflow.data.Dataset.from_tensor_slices((x_data, labels))
            else:
                # Create dummy labels if none provided
                num_samples = x_data.shape[0]
                labels = tensorflow.zeros((num_samples,), dtype=tensorflow.int32)
                train_dataset = tensorflow.data.Dataset.from_tensor_slices((x_data, labels))

            if shuffle:
                train_dataset = train_dataset.shuffle(buffer_size=len(x_data) if hasattr(x_data, '__len__') else 10000)
            train_dataset = train_dataset.batch(batch_size)

        # Calculate steps per epoch if not provided
        if steps_per_epoch is None:
            try:
                steps_per_epoch = len(train_dataset)
            except:
                steps_per_epoch = 100  # Default fallback

        # History to store metrics
        history = {'loss': [], 'd_loss': [], 'g_loss': []}

        # Training loop
        for epoch in range(initial_epoch, epochs):
            self._total_loss_tracker.reset_state()

            # Trackers for epoch metrics
            epoch_d_losses = []
            epoch_g_losses = []

            if verbose == 1:
                print(f'\nEpoch {epoch + 1}/{epochs}')

            # Progress tracking
            step = 0
            for batch_data in train_dataset:
                step += 1

                # Perform training step
                losses = self.train_step(batch_data)
                current_d_loss = float(losses['d_loss'])
                current_g_loss = float(losses['g_loss'])

                # Track losses for this epoch
                epoch_d_losses.append(current_d_loss)
                epoch_g_losses.append(current_g_loss)

                # Update total loss tracker
                self._total_loss_tracker.update_state(current_d_loss + current_g_loss)

                # Simple progress bar
                if verbose == 1:
                    progress = int(50 * step / steps_per_epoch)
                    bar = '=' * progress + '>' + '.' * (50 - progress - 1)
                    print(
                        f'\r[{bar}] {step}/{steps_per_epoch} - d_loss: {current_d_loss:.4f} - g_loss: {current_g_loss:.4f}',
                        end='', flush=True)

                if step >= steps_per_epoch:
                    break

            # Store epoch losses
            epoch_loss = float(self._total_loss_tracker.result())
            epoch_d_loss = numpy.mean(epoch_d_losses) if epoch_d_losses else 0.0
            epoch_g_loss = numpy.mean(epoch_g_losses) if epoch_g_losses else 0.0

            history['loss'].append(epoch_loss)
            history['d_loss'].append(epoch_d_loss)
            history['g_loss'].append(epoch_g_loss)

            if verbose == 1:
                print(f' - d_loss: {epoch_d_loss:.4f} - g_loss: {epoch_g_loss:.4f} - total: {epoch_loss:.4f}')
            elif verbose == 2:
                print(f'Epoch {epoch + 1}/{epochs} - d_loss: {epoch_d_loss:.4f} - g_loss: {epoch_g_loss:.4f}')

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
                            'd_loss': epoch_d_loss,
                            'g_loss': epoch_g_loss
                        })

        # Return history object
        class History:
            def __init__(self, history_dict):
                self.history = history_dict

        return History(history)

    def _evaluate_validation(self, validation_data, validation_steps=None):
        """
        Evaluate the model on validation data with automatic shape handling.

        Args:
            validation_data: Validation dataset (tf.data.Dataset or tuple).
            validation_steps: Number of validation steps.

        Returns:
            Average validation loss.
        """
        val_d_losses = []
        val_g_losses = []
        step = 0

        # Prepare validation dataset
        if isinstance(validation_data, tensorflow.data.Dataset):
            val_dataset = validation_data
        elif isinstance(validation_data, tuple) and len(validation_data) == 2:
            val_x, val_y = validation_data
            if isinstance(val_x, numpy.ndarray):
                val_x = tensorflow.constant(val_x, dtype=tensorflow.float32)
            if isinstance(val_y, numpy.ndarray):
                val_y = tensorflow.constant(val_y, dtype=tensorflow.float32)
            val_dataset = tensorflow.data.Dataset.from_tensor_slices((val_x, val_y))
            val_dataset = val_dataset.batch(32)
        else:
            raise ValueError("validation_data must be a tf.data.Dataset or tuple of (x_val, y_val)")

        for batch_data in val_dataset:
            real_feature, real_samples_label = self._prepare_batch(batch_data)
            batch_size = tensorflow.shape(real_feature)[0]

            # Handle labels
            if real_samples_label is not None:
                if len(real_samples_label.shape) == 1:
                    real_samples_label = tensorflow.expand_dims(real_samples_label, axis=-1)
            else:
                real_samples_label = tensorflow.zeros((batch_size, 1))

            # Generate synthetic samples
            latent_space = tensorflow.random.normal(
                (batch_size, self._latent_dimension),
                mean=self._latent_mean_distribution,
                stddev=self._latent_standard_deviation
            )

            synthetic_feature = self._generator([latent_space, real_samples_label], training=False)

            # Get discriminator predictions
            label_predicted_real = self._discriminator([real_feature, real_samples_label], training=False)
            label_predicted_synthetic = self._discriminator([synthetic_feature, real_samples_label], training=False)

            # Calculate losses
            if self._discriminator_loss_fn is not None:
                d_loss = self._discriminator_loss_fn(
                    real_img=label_predicted_real,
                    fake_img=label_predicted_synthetic
                )
            else:
                # Default Wasserstein loss
                d_loss = tensorflow.reduce_mean(label_predicted_synthetic) - tensorflow.reduce_mean(
                    label_predicted_real)

            if self._generator_loss_fn is not None:
                g_loss = self._generator_loss_fn(label_predicted_synthetic)
            else:
                # Default Wasserstein loss
                g_loss = -tensorflow.reduce_mean(label_predicted_synthetic)

            val_d_losses.append(float(d_loss))
            val_g_losses.append(float(g_loss))

            step += 1
            if validation_steps is not None and step >= validation_steps:
                break

        # Return average of discriminator and generator losses
        avg_val_loss = (numpy.mean(val_d_losses) + numpy.mean(val_g_losses)) / 2
        return avg_val_loss if val_d_losses else 0.0

    @tensorflow.function
    def train_step(self, batch):
        """
        Executes one training step for the GAN model with automatic batch handling.

        Args:
            batch: Input batch (can be single tensor, tuple, or list).

        Returns:
            dict: Dictionary containing the discriminator and generator loss.
        """
        # Prepare batch data
        real_feature, real_samples_label = self._prepare_batch(batch)
        batch_size = tensorflow.shape(real_feature)[0]

        # Handle labels
        if real_samples_label is not None:
            if len(real_samples_label.shape) == 1:
                real_samples_label = tensorflow.expand_dims(real_samples_label, axis=-1)
        else:
            # Create dummy labels
            real_samples_label = tensorflow.zeros((batch_size, 1))

        # === Discriminator Training Loop ===
        for _ in range(self._discriminator_steps):
            # Generate random noise vectors for the latent space
            latent_space = tensorflow.random.normal((batch_size, self._latent_dimension),
                                                    mean=self._latent_mean_distribution,
                                                    stddev=self._latent_standard_deviation)

            with tensorflow.GradientTape() as discriminator_gradient:
                # Generate synthetic samples
                synthetic_feature = self._generator([latent_space, real_samples_label], training=False)

                # Predict labels
                label_predicted_real = self._discriminator([real_feature, real_samples_label], training=True)
                label_predicted_synthetic = self._discriminator([synthetic_feature, real_samples_label], training=True)

                # Compute discriminator loss
                if self._discriminator_loss_fn is not None:
                    discriminator_loss_result = self._discriminator_loss_fn(
                        real_img=label_predicted_real, fake_img=label_predicted_synthetic)
                else:
                    # Default Wasserstein loss
                    discriminator_loss_result = tensorflow.reduce_mean(
                        label_predicted_synthetic) - tensorflow.reduce_mean(label_predicted_real)

                # Compute gradient penalty
                gradient_penalty = self.gradient_penalty(batch_size,
                                                         real_feature,
                                                         real_samples_label,
                                                         synthetic_feature)

                # Combine loss with gradient penalty
                all_discriminator_loss = discriminator_loss_result + gradient_penalty * self._gradient_penalty_weight

            # Update discriminator
            gradient_computed = discriminator_gradient.gradient(all_discriminator_loss,
                                                                self.discriminator.trainable_variables)
            self._discriminator_optimizer.apply_gradients(zip(gradient_computed,
                                                              self.discriminator.trainable_variables))

        # === Generator Training Step ===
        latent_space = tensorflow.random.normal((batch_size, self._latent_dimension),
                                                mean=self._latent_mean_distribution,
                                                stddev=self._latent_standard_deviation)

        with tensorflow.GradientTape() as generator_gradient:
            # Generate synthetic samples
            synthetic_feature = self._generator([latent_space, real_samples_label], training=True)

            # Predict labels
            predicted_labels = self._discriminator([synthetic_feature, real_samples_label], training=False)

            # Compute generator loss
            if self._generator_loss_fn is not None:
                all_generator_loss = self._generator_loss_fn(predicted_labels)
            else:
                # Default Wasserstein loss
                all_generator_loss = -tensorflow.reduce_mean(predicted_labels)

        # Update generator
        gradient_computed = generator_gradient.gradient(all_generator_loss, self._generator.trainable_variables)
        self._generator_optimizer.apply_gradients(zip(gradient_computed, self._generator.trainable_variables))

        return {"d_loss": all_discriminator_loss, "g_loss": all_generator_loss}

    def get_input_shape(self):
        """
        Get the current input shape.

        Returns:
            tuple: Current input shape (excluding batch dimension).
        """
        return self._inferred_shape if self._inferred_shape is not None else self._input_shape

    @staticmethod
    def reshape_data(data, target_shape):
        """
        Reshape data to target shape if needed.

        Args:
            data: Input data.
            target_shape: Desired shape (excluding batch dimension).

        Returns:
            Reshaped data.
        """
        if isinstance(data, numpy.ndarray):
            if data.shape[1:] != target_shape:
                batch_size = data.shape[0]
                return data.reshape((batch_size,) + target_shape)
        elif tensorflow.is_tensor(data):
            if tuple(data.shape[1:]) != target_shape:
                batch_size = data.shape[0]
                return tensorflow.reshape(data, (batch_size,) + target_shape)
        return data

    def get_samples(self, number_samples_per_class):
        """
        Generates synthetic samples for each specified class using the trained generator.

        Args:
            number_samples_per_class (dict): A dictionary containing:
                - "classes" (dict): Mapping of class labels to number of samples.
                - "number_classes" (int): Total number of classes.

        Returns:
            dict: Dictionary with class labels as keys and generated samples as values.
        """
        if number_samples_per_class is None:
            raise ValueError("number_samples_per_class is required for generating samples")

        # Dictionary to store generated samples for each class
        generated_data = {}

        # Loop through each class and the desired number of samples for that class
        for label_class, number_instances in number_samples_per_class["classes"].items():
            # Create one-hot encoded labels for all samples of the current class
            label_samples_generated = to_categorical([label_class] * number_instances,
                                                     num_classes=number_samples_per_class["number_classes"])

            # Sample random noise vectors from a normal distribution
            latent_noise = numpy.random.normal(loc=self._latent_mean_distribution,
                                               scale=self._latent_standard_deviation,
                                               size=(number_instances, self._latent_dimension))

            # Generate synthetic samples using the generator
            generated_samples = self._generator.predict([latent_noise, label_samples_generated], verbose=0)

            # Store generated samples for the current class
            generated_data[label_class] = generated_samples

        return generated_data

    def save_model(self, directory, file_name):
        """
        Save the generator and discriminator models with shape information.

        Args:
            directory (str): Directory where models will be saved.
            file_name (str): Base file name for saving models.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Construct file names for generator and discriminator models
        generator_file_name = os.path.join(directory, f"fold_{file_name}_generator")
        discriminator_file_name = os.path.join(directory, f"fold_{file_name}_discriminator")

        # Save models
        self._save_model_to_json(self._generator, f"{generator_file_name}.json")
        self._generator.save_weights(f"{generator_file_name}.weights.h5")

        self._save_model_to_json(self._discriminator, f"{discriminator_file_name}.json")
        self._discriminator.save_weights(f"{discriminator_file_name}.weights.h5")

        # Save shape information
        shape_info = {
            'input_shape': self._input_shape,
            'inferred_shape': self._inferred_shape,
            'latent_dimension': self._latent_dimension,
            'gradient_penalty_weight': self._gradient_penalty_weight
        }
        shape_file = os.path.join(directory, f"fold_{file_name}_shape_info.json")
        with open(shape_file, 'w') as f:
            json.dump(shape_info, f)

    @staticmethod
    def _save_model_to_json(model, file_path):
        """
        Save model architecture to a JSON file.

        Args:
            model (tf.keras.Model): Model to save.
            file_path (str): Path to the JSON file.
        """
        with open(file_path, "w") as json_file:
            json.dump(model.to_json(), json_file)

    def load_models(self, directory, file_name):
        """
        Load the generator and discriminator models from a directory with shape information.

        Args:
            directory (str): Directory where models are stored.
            file_name (str): Base file name for loading models.
        """
        # Construct file names for generator and discriminator models
        generator_file_name = os.path.join(directory, f"fold_{file_name}_generator")
        discriminator_file_name = os.path.join(directory, f"fold_{file_name}_discriminator")

        # Load weights
        self._generator.load_weights(f"{generator_file_name}.weights.h5")
        self._discriminator.load_weights(f"{discriminator_file_name}.weights.h5")

        # Load shape information if available
        shape_file = os.path.join(directory, f"fold_{file_name}_shape_info.json")
        if os.path.exists(shape_file):
            with open(shape_file, 'r') as f:
                shape_info = json.load(f)
                self._input_shape = tuple(shape_info.get('input_shape', ()))
                self._inferred_shape = tuple(shape_info.get('inferred_shape', ()))
                self._latent_dimension = shape_info.get('latent_dimension', self._latent_dimension)
                self._gradient_penalty_weight = shape_info.get('gradient_penalty_weight', self._gradient_penalty_weight)

    # ========== PROPERTIES ==========

    @property
    def generator(self) -> Any:
        """Get the generator model instance."""
        return self._generator

    @generator.setter
    def generator(self, value: Any) -> None:
        """Set the generator model instance."""
        self._generator = value

    @property
    def discriminator(self) -> Any:
        """Get the discriminator model instance."""
        return self._discriminator

    @discriminator.setter
    def discriminator(self, value: Any) -> None:
        """Set the discriminator model instance."""
        self._discriminator = value

    @property
    def latent_dimension(self) -> int:
        """Get the dimension of the latent space."""
        return self._latent_dimension

    @latent_dimension.setter
    def latent_dimension(self, value: int) -> None:
        """Set the dimension of the latent space."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Latent dimension must be a positive integer")
        self._latent_dimension = value

    @property
    def discriminator_loss_fn(self) -> Callable:
        """Get the discriminator loss function."""
        return self._discriminator_loss_fn

    @discriminator_loss_fn.setter
    def discriminator_loss_fn(self, value: Callable) -> None:
        """Set the discriminator loss function."""
        self._discriminator_loss_fn = value

    @property
    def generator_loss_fn(self) -> Callable:
        """Get the generator loss function."""
        return self._generator_loss_fn

    @generator_loss_fn.setter
    def generator_loss_fn(self, value: Callable) -> None:
        """Set the generator loss function."""
        self._generator_loss_fn = value

    @property
    def gradient_penalty_weight(self) -> float:
        """Get the weight for gradient penalty in WGAN-GP."""
        return self._gradient_penalty_weight

    @gradient_penalty_weight.setter
    def gradient_penalty_weight(self, value: float) -> None:
        """Set the weight for gradient penalty in WGAN-GP."""
        if value < 0:
            raise ValueError("Gradient penalty weight cannot be negative")
        self._gradient_penalty_weight = value

    @property
    def smooth_rate(self) -> float:
        """Get the label smoothing rate."""
        return self._smooth_rate

    @smooth_rate.setter
    def smooth_rate(self, value: float) -> None:
        """Set the label smoothing rate."""
        if not 0 <= value <= 1:
            raise ValueError("Smoothing rate must be between 0 and 1")
        self._smooth_rate = value

    @property
    def latent_mean_distribution(self) -> float:
        """Get the mean of the latent space distribution."""
        return self._latent_mean_distribution

    @latent_mean_distribution.setter
    def latent_mean_distribution(self, value: float) -> None:
        """Set the mean of the latent space distribution."""
        self._latent_mean_distribution = value

    @property
    def latent_standard_deviation(self) -> float:
        """Get the standard deviation of the latent space distribution."""
        return self._latent_standard_deviation

    @latent_standard_deviation.setter
    def latent_standard_deviation(self, value: float) -> None:
        """Set the standard deviation of the latent space distribution."""
        if value <= 0:
            raise ValueError("Standard deviation must be positive")
        self._latent_standard_deviation = value

    @property
    def file_name_discriminator(self) -> str:
        """Get the discriminator model save filename."""
        return self._file_name_discriminator

    @file_name_discriminator.setter
    def file_name_discriminator(self, value: str) -> None:
        """Set the discriminator model save filename."""
        self._file_name_discriminator = value

    @property
    def file_name_generator(self) -> str:
        """Get the generator model save filename."""
        return self._file_name_generator

    @file_name_generator.setter
    def file_name_generator(self, value: str) -> None:
        """Set the generator model save filename."""
        self._file_name_generator = value

    @property
    def models_saved_path(self) -> str:
        """Get the path for saving models."""
        return self._models_saved_path

    @models_saved_path.setter
    def models_saved_path(self, value: str) -> None:
        """Set the path for saving models."""
        self._models_saved_path = value

    @property
    def discriminator_steps(self) -> int:
        """Get the number of discriminator steps per iteration."""
        return self._discriminator_steps

    @discriminator_steps.setter
    def discriminator_steps(self, value: int) -> None:
        """Set the number of discriminator steps per iteration."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Discriminator steps must be a positive integer")
        self._discriminator_steps = value

    @property
    def input_shape(self):
        """Get the current input shape."""
        return self.get_input_shape()

    @property
    def metrics(self):
        """Returns dictionary of metrics tracked during training."""
        return {
            "loss": float(self._total_loss_tracker.result()),
        }