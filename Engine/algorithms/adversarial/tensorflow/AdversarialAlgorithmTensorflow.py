#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adversarial Training Algorithm Implementation for Generative Adversarial Networks (GANs)

This module implements a comprehensive GAN training framework using TensorFlow/Keras,
supporting both conditional and unconditional generation with advanced training techniques.

Mathematical Overview:
----------------------
A GAN consists of two neural networks:
1. Generator G(z): Maps latent noise z ~ p_z to data space x' = G(z)
2. Discriminator D(x): Classifies real vs generated data: D(x) ∈ [0,1]

The minimax objective function:
min_G max_D V(D,G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]

In practice, we use non-saturating loss for generator training:
L_D = -[log(D(x)) + log(1 - D(G(z)))]
L_G = -log(D(G(z)))  # Non-saturating alternative

For conditional GANs (cGAN), both networks receive class labels y:
G(z|y) → x'
D(x|y) → [0,1]

"""

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__last_update__ = '2025/12/07'

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
    Implements a complete adversarial training framework for Generative Adversarial Networks.

    This class provides:
    - Conditional and unconditional GAN training
    - Label smoothing for discriminator stabilization
    - Custom training loop with TensorFlow graph execution
    - Model persistence and loading capabilities
    - Synthetic data generation utilities

    Mathematical Components:
    -----------------------
    1. Latent Space: z ~ N(μ, σ²) where z ∈ ℝ^{latent_dim}
    2. Generator: G(z|y) → x̂ ∈ ℝ^{feature_dim}
    3. Discriminator: D(x|y) → [0,1]
    4. Loss Functions:
        - Standard: L_D = BCE(y_true, y_pred)
        - Label Smoothing: y_smooth ~ U(α, β) where α,β ∈ [0,1]
    5. Optimization: Alternate gradient updates for G and D

    Attributes:
    -----------
    _generator : tf.keras.Model
        Generator network that creates synthetic samples
    _discriminator : tf.keras.Model
        Discriminator network that classifies real vs fake
    _latent_dimension : int
        Dimensionality of the latent space (z-vector)
    _optimizer_generator : tf.keras.optimizers.Optimizer
        Optimizer for generator network
    _optimizer_discriminator : tf.keras.optimizers.Optimizer
        Optimizer for discriminator network
    _loss_generator : callable
        Loss function for generator training
    _loss_discriminator : callable
        Loss function for discriminator training
    _smoothing_rate : float
        Degree of label smoothing (0.0 to 1.0)
    _latent_mean_distribution : float
        Mean of the latent noise distribution (μ)
    _latent_standard_deviation : float
        Standard deviation of latent noise (σ)
    _loss_d_tracker : tf.keras.metrics.Mean
        Tracks discriminator loss over training
    _loss_g_tracker : tf.keras.metrics.Mean
        Tracks generator loss over training
    _total_loss_tracker : tf.keras.metrics.Mean
        Tracks combined loss (average of D and G losses)
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
        """
        Initialize the adversarial training algorithm.

        Parameters:
        -----------
        generator_model : tf.keras.Model
            Generator neural network architecture
        discriminator_model : tf.keras.Model
            Discriminator neural network architecture
        latent_dimension : int
            Size of latent space vector, must be > 0
        loss_generator : callable
            Loss function for generator optimization
        loss_discriminator : callable
            Loss function for discriminator optimization
        file_name_discriminator : str
            Base filename for saving discriminator model
        file_name_generator : str
            Base filename for saving generator model
        models_saved_path : str
            Directory path for model persistence
        latent_mean_distribution : float
            Mean (μ) for latent noise sampling: z ~ N(μ, σ²)
        latent_standard_deviation : float
            Standard deviation (σ) for latent noise sampling, must be > 0
        smoothing_rate : float
            Label smoothing factor ∈ [0, 1]
            0 = no smoothing, 1 = maximum smoothing
        """
        super().__init__(*args, **kwargs)

        # Mathematical validation of parameters
        if latent_dimension <= 0:
            raise ValueError("Latent dimension must be greater than 0.")
        if not isinstance(file_name_discriminator, str) or not file_name_discriminator:
            raise ValueError("Discriminator file name must be a non-empty string.")
        if not isinstance(file_name_generator, str) or not file_name_generator:
            raise ValueError("Generator file name must be a non-empty string.")
        if not isinstance(models_saved_path, str) or not models_saved_path:
            raise ValueError("models saved path must be a non-empty string.")
        if not isinstance(latent_mean_distribution, (int, float)):
            raise TypeError("Latent mean distribution must be a number.")
        if not isinstance(latent_standard_deviation, (int, float)):
            raise TypeError("Latent standard deviation must be a number.")
        if latent_standard_deviation <= 0:
            raise ValueError("Latent standard deviation must be greater than 0.")
        if not (0.0 <= smoothing_rate <= 1.0):
            raise ValueError("Smoothing rate must be between 0 and 1.")

        # Core GAN components
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

        # Loss tracking metrics - initialized to track training statistics
        self._loss_d_tracker = Mean(name="loss_d")
        self._loss_g_tracker = Mean(name="loss_g")
        self._total_loss_tracker = Mean(name="loss")

    def compile(self, optimizer_generator, optimizer_discriminator, loss_generator, loss_discriminator, *args,
                **kwargs):
        """
        Configure the training algorithm with optimizers and loss functions.

        This method follows the Keras Model.compile() pattern but extends it for GANs.

        Parameters:
        -----------
        optimizer_generator : tf.keras.optimizers.Optimizer
            Optimizer for generator network (e.g., Adam, RMSprop)
        optimizer_discriminator : tf.keras.optimizers.Optimizer
            Optimizer for discriminator network
        loss_generator : callable
            Loss function for generator: L_G = f(y_true, D(G(z)))
        loss_discriminator : callable
            Loss function for discriminator: L_D = f(y_true, D(x))
        """
        super().compile(*args, **kwargs)
        self._optimizer_generator = optimizer_generator
        self._optimizer_discriminator = optimizer_discriminator
        self._loss_generator = loss_generator
        self._loss_discriminator = loss_discriminator

    def call(self, inputs, training=False):
        """
        Define the forward pass of the model (required by Keras Model API).

        For a GAN, this generates synthetic samples from latent vectors.
        Supports both conditional and unconditional generation.

        Mathematical Operation:
        ----------------------
        For unconditional: x̂ = G(z)
        For conditional: x̂ = G(z|y)

        Parameters:
        -----------
        inputs : tuple or tf.Tensor
            If tuple: (latent_vector, labels) for conditional generation
            If tf.Tensor: latent_vector for unconditional generation
        training : bool
            Whether the model is in training mode (affects batch normalization, dropout)

        Returns:
        --------
        synthetic_samples : tf.Tensor
            Generated synthetic data samples
        """
        # Unpack inputs based on structure
        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 2:
                latent_vector, labels = inputs
            else:
                latent_vector = inputs[0]
                labels = None
        else:
            latent_vector = inputs
            labels = None

        # Generate synthetic samples using the generator
        if labels is not None:
            # Conditional generation: G(z|y)
            synthetic_samples = self._generator([latent_vector, labels], training=training)
        else:
            # Unconditional generation: G(z)
            synthetic_samples = self._generator(latent_vector, training=training)

        return synthetic_samples

    @tensorflow.function
    def train_step(self, batch):
        """
        Perform a single training step for both generator and discriminator.

        This implements the core GAN training algorithm with:
        1. Discriminator update: max_D V(D,G)
        2. Generator update: min_G V(D,G)

        Mathematical Steps:
        ------------------
        1. Sample batch: (x, y) ~ p_data
        2. Sample noise: z ~ N(μ, σ²)
        3. Generate: x̂ = G(z|y)
        4. Discriminator loss: L_D = BCE(y_smooth, D(x|y)) + BCE(y_smooth, D(x̂|y))
        5. Generator loss: L_G = BCE(0, D(G(z|y)|y))  # Non-saturating loss

        Parameters:
        -----------
        batch : tuple
            Contains (real_features, real_labels) from training dataset

        Returns:
        --------
        metrics : dict
            Dictionary containing loss metrics:
            - 'loss': Combined loss (average of D and G losses)
            - 'loss_d': Discriminator loss
            - 'loss_g': Generator loss
        """
        # Unpack the batch into real features and real labels
        real_feature, real_samples_label = batch

        # Get the current batch size for tensor shape operations
        batch_size = tensorflow.shape(real_feature)[0]

        # Ensure labels have correct shape for conditional generation
        if len(real_samples_label.shape) == 1:
            real_samples_label = tensorflow.expand_dims(real_samples_label, axis=-1)

        # ==================================================
        # PHASE 1: TRAIN THE DISCRIMINATOR
        # Objective: max_D V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]
        # ==================================================

        # Generate latent noise from normal distribution
        latent_space = tensorflow.random.normal(
            shape=(batch_size, self._latent_dimension),
            mean=self._latent_mean_distribution,
            stddev=self._latent_standard_deviation
        )

        # Generate synthetic features with training=True for proper batch norm behavior
        synthetic_feature = self._generator([latent_space, real_samples_label], training=True)

        # Train discriminator with gradient tape for automatic differentiation
        with tensorflow.GradientTape() as discriminator_gradient:
            # Get discriminator predictions for real and synthetic samples
            label_predicted_real = self._discriminator([real_feature, real_samples_label], training=True)
            label_predicted_synthetic = self._discriminator([synthetic_feature, real_samples_label], training=True)

            # Concatenate all predictions for batch loss computation
            label_predicted_all_samples = tensorflow.concat(
                [label_predicted_real, label_predicted_synthetic],
                axis=0
            )

            # Label smoothing technique to prevent discriminator from becoming overconfident
            # Real samples: labels near 0 (0.0 to 0.15)
            # Synthetic samples: labels near 1 (0.85 to 1.0)
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

            # Concatenate smoothed labels
            tensor_labels_predicted = tensorflow.concat(
                [smooth_real_labels, smooth_synthetic_labels],
                axis=0
            )

            # Calculate discriminator loss using binary cross-entropy
            loss_d = self._loss_discriminator(tensor_labels_predicted, label_predicted_all_samples)

        # Apply gradients to discriminator if it has trainable variables
        trainable_vars_discriminator = self._discriminator.trainable_variables
        if trainable_vars_discriminator:
            gradient_discriminator = discriminator_gradient.gradient(loss_d, trainable_vars_discriminator)
            self._optimizer_discriminator.apply_gradients(zip(gradient_discriminator, trainable_vars_discriminator))

        # ==================================================
        # PHASE 2: TRAIN THE GENERATOR
        # Objective: min_G V(D,G) = E[log(1 - D(G(z)))] (non-saturating version)
        # ==================================================

        with tensorflow.GradientTape() as generator_gradient:
            # Generate new latent noise (don't reuse from discriminator phase)
            latent_space_generator = tensorflow.random.normal(
                shape=(batch_size, self._latent_dimension),
                mean=self._latent_mean_distribution,
                stddev=self._latent_standard_deviation
            )

            # Generate synthetic features
            synthetic_feature_generator = self._generator(
                [latent_space_generator, real_samples_label],
                training=True
            )

            # Get discriminator predictions on synthetic samples
            # Note: discriminator in inference mode (training=False) during generator update
            predicted_labels = self._discriminator(
                [synthetic_feature_generator, real_samples_label],
                training=False
            )

            # Generator loss: we want discriminator to classify synthetic as real (label 0)
            # Using non-saturating loss: L_G = -log(D(G(z)))
            loss_g = self._loss_generator(tensorflow.zeros_like(predicted_labels), predicted_labels)

        # Apply gradients to generator
        trainable_vars_generator = self._generator.trainable_variables
        if trainable_vars_generator:
            gradient_generator = generator_gradient.gradient(loss_g, trainable_vars_generator)
            self._optimizer_generator.apply_gradients(zip(gradient_generator, trainable_vars_generator))

        # ==================================================
        # UPDATE METRICS AND RETURN
        # ==================================================

        # Update loss trackers with current batch losses
        self._loss_d_tracker.update_state(loss_d)
        self._loss_g_tracker.update_state(loss_g)

        # Combined loss (average of discriminator and generator losses)
        combined_loss = (loss_d + loss_g) / 2.0
        self._total_loss_tracker.update_state(combined_loss)

        # Return all metrics as dictionary
        return {
            "loss": combined_loss,  # Total loss (expected by Keras fit)
            "loss_d": loss_d,  # Discriminator loss
            "loss_g": loss_g  # Generator loss
        }

    @property
    def metrics(self):
        """
        Return metrics for tracking during training.

        Returns:
        --------
        list of tf.keras.metrics.Mean
            List containing loss trackers for monitoring
        """
        return [self._loss_d_tracker, self._loss_g_tracker, self._total_loss_tracker]

    def fit(self, x=None, y=None, batch_size=32, epochs=1, verbose=1,
            callbacks=None, validation_data=None, shuffle=True,
            initial_epoch=0, steps_per_epoch=None, validation_steps=None,
            validation_freq=1, optimizer=None, learning_rate=0.001, **kwargs):
        """
        Train the model with a simplified progress bar and custom training loop.

        This method implements the complete training pipeline with:
        - Epoch-based training
        - Progress visualization
        - Validation monitoring
        - Callback support

        Parameters:
        -----------
        x : tf.data.Dataset or array-like
            Training data features
        y : array-like, optional
            Training labels for conditional GANs
        batch_size : int
            Number of samples per gradient update
        epochs : int
            Number of epochs to train
        verbose : int
            Verbosity mode: 0 = silent, 1 = progress bar, 2 = one line per epoch
        callbacks : list of tf.keras.callbacks.Callback
            List of callbacks to apply during training
        validation_data : tuple or tf.data.Dataset
            Data on which to evaluate loss after each epoch
        shuffle : bool
            Whether to shuffle the training data
        initial_epoch : int
            Epoch at which to start training
        steps_per_epoch : int
            Number of steps (batches) per epoch
        validation_steps : int
            Number of validation steps
        validation_freq : int
            Frequency (in epochs) of validation
        optimizer : tf.keras.optimizers.Optimizer
            Optimizer to use (defaults to Adam)
        learning_rate : float
            Learning rate if using default Adam optimizer

        Returns:
        --------
        history : object
            Training history with loss metrics
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

        # Training loop over epochs
        for epoch in range(initial_epoch, epochs):
            # Reset all trackers at the start of each epoch
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

                # Extract current losses
                current_loss = float(metrics['loss'])
                current_loss_d = float(metrics['loss_d'])
                current_loss_g = float(metrics['loss_g'])

                # Simple progress bar visualization
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

            # Print epoch summary
            if verbose == 1:
                print(f' - loss: {epoch_loss:.4f} - loss_d: {epoch_loss_d:.4f} - loss_g: {epoch_loss_g:.4f}')
            elif verbose == 2:
                print(
                    f'Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.4f} - loss_d: {epoch_loss_d:.4f} - loss_g: {epoch_loss_g:.4f}')

            # Validation phase
            if validation_data is not None and (epoch + 1) % validation_freq == 0:
                val_loss = self._evaluate_validation(validation_data, validation_steps)
                if 'val_loss' not in history:
                    history['val_loss'] = []
                history['val_loss'].append(val_loss)

                if verbose >= 1:
                    print(f' - val_loss: {val_loss:.4f}')

            # Execute callbacks
            if callbacks is not None:
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, {
                            'loss': epoch_loss,
                            'loss_d': epoch_loss_d,
                            'loss_g': epoch_loss_g
                        })

        # Return history object compatible with Keras
        class History:
            def __init__(self, history_dict):
                self.history = history_dict

        return History(history)

    def _evaluate_validation(self, validation_data, validation_steps=None):
        """
        Evaluate the model on validation data WITHOUT updating weights.

        This method computes validation loss to monitor overfitting.

        Mathematical Operation:
        ----------------------
        1. For each validation batch:
            x_val, y_val ~ p_val
            z ~ N(0,1)
            x̂ = G(z|y_val)
        2. L_D_val = BCE(0, D(x_val|y_val)) + BCE(1, D(x̂|y_val))
        3. L_G_val = BCE(0, D(x̂|y_val))
        4. L_val = (L_D_val + L_G_val) / 2

        Parameters:
        -----------
        validation_data : tf.data.Dataset or tuple
            Validation dataset
        validation_steps : int, optional
            Number of validation batches to process

        Returns:
        --------
        float
            Average validation loss
        """
        # Initialize validation trackers
        val_loss_tracker = Mean(name="val_loss")
        val_loss_d_tracker = Mean(name="val_loss_d")
        val_loss_g_tracker = Mean(name="val_loss_g")

        step = 0
        for batch_data in validation_data:
            real_feature, real_samples_label = batch_data
            batch_size = tensorflow.shape(real_feature)[0]
            real_samples_label = tensorflow.expand_dims(real_samples_label, axis=-1)

            # Generate synthetic data from latent noise
            latent_space = tensorflow.random.normal(shape=(batch_size, self._latent_dimension))
            synthetic_feature = self._generator([latent_space, real_samples_label], training=False)

            # Evaluate discriminator on real and synthetic data
            label_predicted_real = self._discriminator([real_feature, real_samples_label], training=False)
            label_predicted_synthetic = self._discriminator([synthetic_feature, real_samples_label], training=False)

            # Concatenate predictions
            label_predicted_all = tensorflow.concat([label_predicted_real, label_predicted_synthetic], axis=0)

            # Create target labels: 0 for real, 1 for synthetic
            tensor_labels = tensorflow.concat([
                tensorflow.zeros_like(label_predicted_real),
                tensorflow.ones_like(label_predicted_synthetic)
            ], axis=0)

            # Calculate discriminator loss (no gradient applied)
            loss_d = self._loss_discriminator(tensor_labels, label_predicted_all)

            # Calculate generator loss
            predicted_labels = self._discriminator([synthetic_feature, real_samples_label], training=False)
            loss_g = self._loss_generator(tensorflow.zeros_like(predicted_labels), predicted_labels)

            # Combined validation loss
            combined_loss = (loss_d + loss_g) / 2.0

            # Update validation trackers
            val_loss_tracker.update_state(combined_loss)
            val_loss_d_tracker.update_state(loss_d)
            val_loss_g_tracker.update_state(loss_g)

            step += 1
            if validation_steps is not None and step >= validation_steps:
                break

        return float(val_loss_tracker.result())

    def get_samples(self, number_samples_per_class):
        """
        Generate synthetic data samples for each specified class using the trained generator.

        Mathematical Operation:
        ----------------------
        For each class c:
            1. Create labels: y = one_hot(c)
            2. Sample noise: z ~ N(μ, σ²)
            3. Generate: x̂ = G(z|y)

        Parameters:
        -----------
        number_samples_per_class : dict
            Dictionary containing:
            - "classes": dict of {class_label: number_of_samples}
            - "number_classes": total number of classes

        Returns:
        --------
        generated_data : dict
            Dictionary mapping class labels to generated samples
        """
        generated_data = {}

        for label_class, number_instances in number_samples_per_class["classes"].items():
            # Create one-hot encoded labels for conditional generation
            label_samples_generated = to_categorical(
                [label_class] * number_instances,
                num_classes=number_samples_per_class["number_classes"]
            )

            # Generate random noise vectors from normal distribution
            latent_noise = numpy.random.normal(
                self._latent_mean_distribution,
                self._latent_standard_deviation,
                (number_instances, self._latent_dimension)
            )

            # Generate synthetic samples using the trained generator
            generated_samples = self._generator.predict([latent_noise, label_samples_generated], verbose=0)

            # Store in dictionary by class
            generated_data[label_class] = generated_samples

        return generated_data

    def save_model(self, path_output, k_fold):
        """
        Save generator and discriminator models to disk.

        Saves both architecture (JSON) and weights (HDF5) for persistence.

        Parameters:
        -----------
        path_output : str
            Base output directory path
        k_fold : int
            Current fold number for cross-validation naming
        """
        try:
            logging.info("Starting to save adversarial Model for fold {}...".format(k_fold))

            # Create directory if it doesn't exist
            path_directory = os.path.join(path_output, self._models_saved_path)
            Path(path_directory).mkdir(parents=True, exist_ok=True)

            # Create filenames with fold identifier
            discriminator_file_name = self._file_name_discriminator + "_" + str(k_fold)
            generator_file_name = self._file_name_generator + "_" + str(k_fold)

            # Full paths for model files
            path_model = path_directory  # Save directly in the directory
            discriminator_file_name = os.path.join(path_model, discriminator_file_name)
            generator_file_name = os.path.join(path_model, generator_file_name)

            # Save discriminator: architecture as JSON, weights as HDF5
            discriminator_model_json = self._discriminator.to_json()
            with open(discriminator_file_name + ".json", "w") as json_file:
                json_file.write(discriminator_model_json)
            self._discriminator.save_weights(discriminator_file_name + ".h5")

            # Save generator: architecture as JSON, weights as HDF5
            generator_model_json = self._generator.to_json()
            with open(generator_file_name + ".json", "w") as json_file:
                json_file.write(generator_model_json)
            self._generator.save_weights(generator_file_name + ".h5")

            logging.info("models saved successfully for fold {}".format(k_fold))

        except Exception as e:
            logging.error("An error occurred while saving the models: {}".format(e))
            raise

    def load_models(self, path_output, k_fold):
        """
        Load generator and discriminator models from disk.

        Parameters:
        -----------
        path_output : str
            Base directory path containing saved models
        k_fold : int
            Fold number for model file naming

        Raises:
        -------
        Exception
            If model files are not found or cannot be loaded
        """
        try:
            logging.info("Loading adversarial Model for fold {}...".format(k_fold))

            # Construct directory path
            path_directory = os.path.join(path_output, self._models_saved_path)

            # Create filenames with fold identifier
            discriminator_file_name = self._file_name_discriminator + "_" + str(k_fold)
            generator_file_name = self._file_name_generator + "_" + str(k_fold)

            # Full paths for model files
            discriminator_file_name = os.path.join(path_directory, discriminator_file_name)
            generator_file_name = os.path.join(path_directory, generator_file_name)

            # Load discriminator: architecture from JSON, weights from HDF5
            with open(discriminator_file_name + ".json", 'r') as json_file:
                discriminator_model_json = json_file.read()
            self._discriminator = model_from_json(discriminator_model_json)
            self._discriminator.load_weights(discriminator_file_name + ".h5")

            # Load generator: architecture from JSON, weights from HDF5
            with open(generator_file_name + ".json", 'r') as json_file:
                generator_model_json = json_file.read()
            self._generator = model_from_json(generator_model_json)
            self._generator.load_weights(generator_file_name + ".h5")

            logging.info("models loaded successfully for fold {}".format(k_fold))

        except Exception as e:
            logging.error("An error occurred while loading the models: {}".format(e))
            raise

    # ==================================================
    # SETTER METHODS FOR DYNAMIC CONFIGURATION
    # ==================================================

    def set_generator(self, generator):
        """Set or replace the generator model."""
        self._generator = generator

    def set_discriminator(self, discriminator):
        """Set or replace the discriminator model."""
        self._discriminator = discriminator

    def set_latent_dimension(self, latent_dimension):
        """Update the latent space dimension."""
        self._latent_dimension = latent_dimension

    def set_optimizer_generator(self, optimizer_generator):
        """Update the generator optimizer."""
        self._optimizer_generator = optimizer_generator

    def set_optimizer_discriminator(self, optimizer_discriminator):
        """Update the discriminator optimizer."""
        self._optimizer_discriminator = optimizer_discriminator

    def set_loss_generator(self, loss_generator):
        """Update the generator loss function."""
        self._loss_generator = loss_generator

    def set_loss_discriminator(self, loss_discriminator):
        """Update the discriminator loss function."""
        self._loss_discriminator = loss_discriminator