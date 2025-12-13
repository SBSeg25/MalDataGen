#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
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
    A wassersteingp Generative adversarial Network (wassersteingp GAN) model.

    This class represents a wassersteingp GAN consisting of a generator and discriminator model.
    It implements the wassersteingp loss with gradient penalty to improve the training of the discriminator and generator.

    Reference:
        Arjovsky, M., Chintala, S., & Bottou, L. (2017). wassersteingp GAN.
        In Proceedings of the 34th International Conference on Machine Learning (ICML 2017) (Vol. 70, pp. 214-223).
        http://proceedings.mlr.press/v70/arjovsky17a.html

    Attributes:
        @_generator (Model):
            The generator model responsible for generating synthetic data.
        @_discriminator (Model):
            The discriminator model used to evaluate the authenticity of generated data.
        @_latent_dimension (int):
            The dimension of the latent space from which the generator takes input.
        @_generator_optimizer (Optimizer):
            Optimizer used for training the generator.
        @_discriminator_optimizer (Optimizer):
            Optimizer used for training the discriminator.
        @_generator_loss_fn (function):
            Loss function used for training the generator.
        @_discriminator_loss_fn (function):
            Loss function used for training the discriminator.
        @_latent_mean_distribution (float):
            Mean of the latent space distribution.
        @_latent_stander_deviation (float):
            Standard deviation of the latent space distribution.
        @_smoothing_rate (float):
            Rate for label smoothing applied to the discriminator's true labels.
        @_gradient_penalty_weight (float):
            Weight for the gradient penalty term in the wassersteingp loss.
        @_discriminator_steps (int):
            Number of discriminator updates per generator update.
        @_file_name_discriminator (str):
            File name for saving/loading the discriminator model.
        @_file_name_generator (str):
            File name for saving/loading the generator model.
        @_models_saved_path (str):
            Path where the models are saved.

    Raises:
        ValueError:
            Raised if:
            - The latent dimension is non-positive.
            - The gradient penalty weight is non-positive.
            - The smoothing rate is outside the valid range (0, 1).
            - The number of discriminator steps is non-positive.

    Example:
        >>> generator = build_generator_model()
        >>> discriminator = build_discriminator_model()
        >>> wgan = WassersteinGPAlgorithm(
        ...     generator_model=generator,
        ...     discriminator_model=discriminator,
        ...     latent_dimension=100,
        ...     generator_loss_fn=generator_loss_fn,
        ...     discriminator_loss_fn=discriminator_loss_fn,
        ...     file_name_discriminator='discriminator_model.h5',
        ...     file_name_generator='generator_model.h5',
        ...     models_saved_path='./models/',
        ...     latent_mean_distribution=0.0,
        ...     latent_stander_deviation=1.0,
        ...     smoothing_rate=0.1,
        ...     gradient_penalty_weight=10.0,
        ...     discriminator_steps=5
        ... )
        >>> # number_samples_per_class is calculated automatically during fit()
        >>> wgan.train_step(real_data, batch_size=64)
    """

    def __init__(self,
                 generator_model,
                 discriminator_model,
                 latent_dimension,
                 generator_loss_fn,
                 discriminator_loss_fn,
                 file_name_discriminator,
                 file_name_generator,
                 models_saved_path,
                 latent_mean_distribution,
                 latent_stander_deviation,
                 smoothing_rate,
                 gradient_penalty_weight,
                 discriminator_steps,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        # Initialize instance variables with provided or default values
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
        self._latent_stander_deviation = latent_stander_deviation
        self._file_name_discriminator = file_name_discriminator
        self._file_name_generator = file_name_generator
        self._models_saved_path = models_saved_path
        self._discriminator_steps = discriminator_steps

    def compile(self, optimizer_generator, optimizer_discriminator,
                loss_generator, loss_discriminator, *args, **kwargs):
        """
        Compile the wassersteingp Generative adversarial Network (WGAN) with custom optimizers and loss functions.

        Args:
            optimizer_generator (str):
                The optimizer for the generator.
            optimizer_discriminator (str):
                The optimizer for the discriminator.
            loss_generator (str):
                The loss function for the generator.
            loss_discriminator (str):
                The loss function for the discriminator.
            *args:
                Additional positional arguments.
            **kwargs:
                Additional keyword arguments.

        This method compiles the GAN with custom optimizers and loss functions specified as arguments.
        It sets the optimizer and loss for both the generator and discriminator.
        """
        super().compile()
        self._discriminator_optimizer = optimizer_discriminator
        self._generator_optimizer = optimizer_generator
        self._discriminator_loss_fn = loss_discriminator
        self._generator_loss_fn = loss_generator

    def gradient_penalty(self, batch_size, real_feature, real_label, synthetic_feature):
        """
        Compute the gradient penalty for the wassersteingp GAN.

        The gradient penalty is used to enforce the Lipschitz constraint on the discriminator's output.

        Parameters:
            batch_size (int):
                The batch size of the input data.
            real_feature (Tensorflow.Tensor):
                Real data features.
            synthetic_feature (Tensorflow.Tensor):
                Synthetic (generated) data features.

        """
        # Generate random noise for smoothing.
        random_smooth = tensorflow.random.normal([batch_size, 1], 0.0, 0.1)

        # Calculate the linear distance between real and synthetic features.
        linear_distance = synthetic_feature - real_feature

        # Interpolate between real and synthetic features using the random noise.
        interpolated_feature = real_feature + random_smooth * linear_distance

        with tensorflow.GradientTape() as gradient_penalty:
            # Watch the interpolated features for gradient computation.
            gradient_penalty.watch(interpolated_feature)

            # Get discriminator's output for the interpolated features.
            labels_predicted = self.discriminator([interpolated_feature, real_label], training=True)

        # Calculate the gradient of the discriminator's output with respect to the interpolated features.
        gradient_computed = gradient_penalty.gradient(labels_predicted, [interpolated_feature])[0]

        # Compute the gradient magnitude and normalize it.
        gradient_normalized = tensorflow.sqrt(tensorflow.reduce_sum(tensorflow.square(gradient_computed), axis=[1]))

        # Calculate the final gradient penalty as the mean squared difference from 1.0 and return.
        gradient_penalty_final = tensorflow.reduce_mean((gradient_normalized - 1.0) ** 2)

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
        Train the model for a fixed number of epochs with a simplified progress bar.

        This method mimics Keras's fit() API for compatibility.

        Args:
            x: Input data (array-like or tf.data.Dataset).
            y: Target data (labels). Not needed if x is a Dataset.
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
            **kwargs: Additional arguments for compatibility.

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
        elif not hasattr(self, '_generator_optimizer') or self._generator_optimizer is None:
            # Create default optimizers if none exist
            self._generator_optimizer = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate)
            self._discriminator_optimizer = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate)

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
            try:
                steps_per_epoch = len(train_dataset)
            except:
                steps_per_epoch = int(numpy.ceil(len(x) / batch_size))

        # History to store metrics
        history = {'d_loss': [], 'g_loss': []}

        # Training loop
        for epoch in range(initial_epoch, epochs):
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
            epoch_d_loss = numpy.mean(epoch_d_losses) if epoch_d_losses else 0.0
            epoch_g_loss = numpy.mean(epoch_g_losses) if epoch_g_losses else 0.0

            history['d_loss'].append(epoch_d_loss)
            history['g_loss'].append(epoch_g_loss)

            if verbose == 1:
                print(f' - d_loss: {epoch_d_loss:.4f} - g_loss: {epoch_g_loss:.4f}')
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
        Evaluate the model on validation data.

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
        else:
            val_x, val_y = validation_data
            val_dataset = tensorflow.data.Dataset.from_tensor_slices((val_x, val_y))
            val_dataset = val_dataset.batch(32)

        for batch_data in val_dataset:
            real_feature, real_samples_label = batch_data
            batch_size = tensorflow.shape(real_feature)[0]

            # Expand label dimensions
            real_samples_label = tensorflow.expand_dims(real_samples_label, axis=-1)

            # Generate synthetic samples
            latent_space = tensorflow.random.normal(
                (batch_size, self._latent_dimension),
                mean=self._latent_mean_distribution,
                stddev=self._latent_stander_deviation
            )

            synthetic_feature = self._generator([latent_space, real_samples_label], training=False)

            # Get discriminator predictions
            label_predicted_real = self._discriminator([real_feature, real_samples_label], training=False)
            label_predicted_synthetic = self._discriminator([synthetic_feature, real_samples_label], training=False)

            # Calculate losses
            d_loss = self._discriminator_loss_fn(
                real_img=label_predicted_real,
                fake_img=label_predicted_synthetic
            )
            g_loss = self._generator_loss_fn(label_predicted_synthetic)

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
        Executes one training step for the GAN model.

        This step updates both the discriminator and the generator.
        The discriminator is updated multiple times (controlled by self._discriminator_steps),
        while the generator is updated once.

        Args:
            batch (tuple): A tuple containing:
                - real_feature: A batch of real data samples (features).
                - real_samples_label: Corresponding class labels for each sample.

        Returns:
            dict: Dictionary containing the discriminator and generator loss for the current training step.
        """

        # Unpack batch into features and labels.
        real_feature, real_samples_label = batch
        batch_size = tensorflow.shape(real_feature)[0]

        # Expand label dimensions to match input expectations (e.g., (batch_size, 1)).
        real_samples_label = tensorflow.expand_dims(real_samples_label, axis=-1)

        # === Discriminator Training Loop ===
        for _ in range(self._discriminator_steps):
            # Generate random noise vectors for the latent space.
            latent_space = tensorflow.random.normal((batch_size, self._latent_dimension),
                                                    mean=self._latent_mean_distribution,
                                                    stddev=self._latent_stander_deviation)

            with tensorflow.GradientTape() as discriminator_gradient:
                # Generate synthetic samples from the generator using noise and labels.
                synthetic_feature = self._generator([latent_space, real_samples_label], training=False)

                # Predict "real/fake" labels using the discriminator for real and synthetic samples.
                label_predicted_real = self._discriminator([real_feature, real_samples_label], training=True)
                label_predicted_synthetic = self._discriminator([synthetic_feature, real_samples_label], training=True)

                # Compute discriminator loss (real vs fake).
                discriminator_loss_result = self._discriminator_loss_fn(
                    real_img=label_predicted_real, fake_img=label_predicted_synthetic)

                # Compute gradient penalty for improved stability (WGAN-GP, etc.).
                gradient_penalty = self.gradient_penalty(batch_size,
                                                         real_feature,
                                                         real_samples_label,
                                                         synthetic_feature)

                # Combine loss with gradient penalty.
                all_discriminator_loss = discriminator_loss_result + gradient_penalty * self._gradient_penalty_weight

            # Compute and apply gradients to update the discriminator's weights.
            gradient_computed = discriminator_gradient.gradient(all_discriminator_loss,
                                                                self.discriminator.trainable_variables)
            self._discriminator_optimizer.apply_gradients(zip(gradient_computed,
                                                              self.discriminator.trainable_variables))

        # === Generator Training Step ===
        # Generate fresh random noise vectors for the latent space.
        latent_space = tensorflow.random.normal((batch_size, self._latent_dimension),
                                                mean=self._latent_mean_distribution,
                                                stddev=self._latent_stander_deviation)

        with tensorflow.GradientTape() as generator_gradient:
            # Generate synthetic samples from the generator.
            synthetic_feature = self._generator([latent_space, real_samples_label], training=True)

            # Predict "real/fake" labels for synthetic samples using the discriminator.
            predicted_labels = self._discriminator([synthetic_feature, real_samples_label], training=False)

            # Compute generator loss (how well generator fools the discriminator).
            all_generator_loss = self._generator_loss_fn(predicted_labels)

        # Compute and apply gradients to update the generator's weights.
        gradient_computed = generator_gradient.gradient(all_generator_loss, self._generator.trainable_variables)
        self._generator_optimizer.apply_gradients(zip(gradient_computed, self._generator.trainable_variables))

        # Return the loss values for monitoring/tracking.
        return {"d_loss": all_discriminator_loss, "g_loss": all_generator_loss}

    def get_samples(self, number_samples_per_class):
        """
        Generates synthetic samples for each specified class using the trained generator.

        This method creates samples conditioned on class labels, using random noise vectors
        and the generator to produce the samples.

        Args:
            number_samples_per_class (dict): A dictionary containing:
                - "classes" (dict): Mapping of class labels to the number of samples to generate for each class.
                - "number_classes" (int): Total number of classes (used for one-hot encoding).

        Returns:
            dict: A dictionary where each key is a class label and the value is an array of generated samples.

        Raises:
            ValueError: If number_samples_per_class is not provided.
        """
        if number_samples_per_class is None:
            raise ValueError("number_samples_per_class is required for generating samples")

        # Dictionary to store generated samples for each class.
        generated_data = {}

        # Loop through each class and the desired number of samples for that class.
        for label_class, number_instances in number_samples_per_class["classes"].items():
            # Create one-hot encoded labels for all samples of the current class.
            label_samples_generated = to_categorical([label_class] * number_instances,
                                                     num_classes=number_samples_per_class["number_classes"])

            # Sample random noise vectors from a normal distribution.
            latent_noise = numpy.random.normal(loc=self._latent_mean_distribution,
                                               scale=self._latent_stander_deviation,
                                               size=(number_instances, self._latent_dimension))

            # Generate synthetic samples using the generator.
            generated_samples = self._generator.predict([latent_noise, label_samples_generated], verbose=0)

            # Round the generated samples to integer values
            # (if samples are intended to be binary, e.g., images with pixel values 0 or 1).
            generated_samples = numpy.rint(generated_samples)

            # Store generated samples for the current class.
            generated_data[label_class] = generated_samples

        # Return the dictionary containing generated samples for all requested classes.
        return generated_data

    def save_model(self, directory, file_name):
        """
        Save the encoder and decoder models in both JSON and H5 formats.

        Args:
            directory (str): Directory where models will be saved.
            file_name (str): Base file name for saving models.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Construct file names for encoder and decoder models
        generator_file_name = os.path.join(directory, f"fold_{file_name}_generator")
        discriminator_file_name = os.path.join(directory, f"fold_{file_name}_discriminator")

        # Save encoder model
        self._save_model_to_json(self._generator, f"{generator_file_name}.json")
        self._generator.save_weights(f"{generator_file_name}.weights.h5")

        # Save decoder model
        self._save_model_to_json(self._discriminator, f"{discriminator_file_name}.json")
        self._discriminator.save_weights(f"{discriminator_file_name}.weights.h5")

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
        Load the generator and discriminator models from a directory.

        Args:
            directory (str): Directory where models are stored.
            file_name (str): Base file name for loading models.
        """

        # Construct file names for generator and discriminator models
        generator_file_name = "{}_generator".format(file_name)
        discriminator_file_name = "{}_discriminator".format(file_name)

        # Load the generator and discriminator models from the specified directory
        self._generator = self._save_neural_network_model(generator_file_name, directory)
        self._discriminator = self._save_neural_network_model(discriminator_file_name, directory)

    @property
    def discriminator(self) -> Any:
        """Get the discriminator model instance.

        Returns:
            The discriminator model used in GAN training.
        """
        return self._discriminator

    @discriminator.setter
    def discriminator(self, value: Any) -> None:
        """Set the discriminator model instance.

        Args:
            value: The discriminator model to set.
        """
        self._discriminator = value

    @property
    def latent_dimension(self) -> int:
        """Get the dimension of the latent space.

        Returns:
            The size of the latent space dimension (positive integer).
        """
        return self._latent_dimension

    @latent_dimension.setter
    def latent_dimension(self, value: int) -> None:
        """Set the dimension of the latent space.

        Args:
            value: The latent dimension size (must be positive).

        Raises:
            ValueError: If value is not a positive integer.
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Latent dimension must be a positive integer")
        self._latent_dimension = value

    @property
    def discriminator_loss_fn(self) -> Callable:
        """Get the discriminator loss function.

        Returns:
            The loss function used for discriminator training.
        """
        return self._discriminator_loss_fn

    @discriminator_loss_fn.setter
    def discriminator_loss_fn(self, value: Callable) -> None:
        """Set the discriminator loss function.

        Args:
            value: The loss function to use for discriminator training.
        """
        self._discriminator_loss_fn = value

    @property
    def generator_loss_fn(self) -> Callable:
        """Get the generator loss function.

        Returns:
            The loss function used for generator training.
        """
        return self._generator_loss_fn

    @generator_loss_fn.setter
    def generator_loss_fn(self, value: Callable) -> None:
        """Set the generator loss function.

        Args:
            value: The loss function to use for generator training.
        """
        self._generator_loss_fn = value

    @property
    def gradient_penalty_weight(self) -> float:
        """Get the weight for gradient penalty in WGAN-GP.

        Returns:
            The weight factor for gradient penalty term.
        """
        return self._gradient_penalty_weight

    @gradient_penalty_weight.setter
    def gradient_penalty_weight(self, value: float) -> None:
        """Set the weight for gradient penalty in WGAN-GP.

        Args:
            value: The penalty weight (must be non-negative).

        Raises:
            ValueError: If value is negative.
        """
        if value < 0:
            raise ValueError("Gradient penalty weight cannot be negative")
        self._gradient_penalty_weight = value

    @property
    def smooth_rate(self) -> float:
        """Get the label smoothing rate.

        Returns:
            The rate used for one-sided label smoothing.
        """
        return self._smooth_rate

    @smooth_rate.setter
    def smooth_rate(self, value: float) -> None:
        """Set the label smoothing rate.

        Args:
            value: The smoothing rate (typically between 0 and 0.3).

        Raises:
            ValueError: If value is not between 0 and 1.
        """
        if not 0 <= value <= 1:
            raise ValueError("Smoothing rate must be between 0 and 1")
        self._smooth_rate = value

    @property
    def latent_mean_distribution(self) -> float:
        """Get the mean of the latent space distribution.

        Returns:
            The mean value used for latent space sampling.
        """
        return self._latent_mean_distribution

    @latent_mean_distribution.setter
    def latent_mean_distribution(self, value: float) -> None:
        """Set the mean of the latent space distribution.

        Args:
            value: The mean value for latent distribution.
        """
        self._latent_mean_distribution = value

    @property
    def latent_stander_deviation(self) -> float:
        """Get the standard deviation of the latent space distribution.

        Returns:
            The standard deviation used for latent space sampling.
        """
        return self._latent_stander_deviation

    @latent_stander_deviation.setter
    def latent_stander_deviation(self, value: float) -> None:
        """Set the standard deviation of the latent space distribution.

        Args:
            value: The standard deviation (must be positive).

        Raises:
            ValueError: If value is not positive.
        """
        if value <= 0:
            raise ValueError("Standard deviation must be positive")
        self._latent_stander_deviation = value

    @property
    def file_name_discriminator(self) -> str:
        """Get the discriminator model save filename.

        Returns:
            The filename pattern for saving discriminator models.
        """
        return self._file_name_discriminator

    @file_name_discriminator.setter
    def file_name_discriminator(self, value: str) -> None:
        """Set the discriminator model save filename.

        Args:
            value: The filename pattern to use.
        """
        self._file_name_discriminator = value

    @property
    def file_name_generator(self) -> str:
        """Get the generator model save filename.

        Returns:
            The filename pattern for saving generator models.
        """
        return self._file_name_generator

    @file_name_generator.setter
    def file_name_generator(self, value: str) -> None:
        """Set the generator model save filename.

        Args:
            value: The filename pattern to use.
        """
        self._file_name_generator = value

    @property
    def models_saved_path(self) -> str:
        """Get the path for saving models.

        Returns:
            The directory path where models are saved.
        """
        return self._models_saved_path

    @models_saved_path.setter
    def models_saved_path(self, value: str) -> None:
        """Set the path for saving models.

        Args:
            value: The directory path to use for saving models.
        """
        self._models_saved_path = value

    @property
    def discriminator_steps(self) -> int:
        """Get the number of discriminator steps per iteration.

        Returns:
            The number of discriminator training steps per GAN iteration.
        """
        return self._discriminator_steps

    @discriminator_steps.setter
    def discriminator_steps(self, value: int) -> None:
        """Set the number of discriminator steps per iteration.

        Args:
            value: The number of steps (must be positive integer).

        Raises:
            ValueError: If value is not a positive integer.
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Discriminator steps must be a positive integer")
        self._discriminator_steps = value