#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com'
__version__ = '{1}.{1}.{0}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
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

    from tensorflow.keras import Model

    from tensorflow.keras.metrics import Mean

    from tensorflow.keras.utils import to_categorical

    from tensorflow.keras.losses import BinaryCrossentropy

except ImportError as error:
    print(error)
    sys.exit(-1)


class VariationalAutoencoderAlgorithmTensorflow(Model):
    """
    Implements an adaptive Variational AutoEncoder (VAE) model for generating synthetic data.

    The model includes an encoder and a decoder for encoding input data and reconstructing
    it from a learned latent space. During training, it computes both the reconstruction loss
    and the KL divergence loss. The trained decoder can be used to generate synthetic data.

    This class supports customizable latent space parameters and loss functions, making it
    adaptable for different generative tasks. It automatically adapts to any input shape:
    (x), (x, y), (x, y, z), etc.

    Example:
        >>> # Example with 1D data
        >>> vae_1d = VariationalAutoencoderAlgorithmTensorflow(
        ...     encoder_model=encoder_1d,
        ...     decoder_model=decoder_1d,
        ...     loss_function='mse',
        ...     latent_dimension=64,
        ...     input_shape=(100,)
        ... )
        >>> vae_1d.fit(data_1d, epochs=50)

        >>> # Example with 2D data (images)
        >>> vae_2d = VariationalAutoencoderAlgorithmTensorflow(
        ...     encoder_model=encoder_2d,
        ...     decoder_model=decoder_2d,
        ...     loss_function='bce',
        ...     latent_dimension=128,
        ...     input_shape=(28, 28)
        ... )
        >>> vae_2d.fit(data_2d, epochs=100)
    """

    def __init__(self,
                 encoder_model,
                 decoder_model,
                 loss_function,
                 latent_dimension,
                 decoder_latent_dimension=None,
                 latent_mean_distribution=0.0,
                 latent_standard_deviation=1.0,
                 file_name_encoder=None,
                 file_name_decoder=None,
                 models_saved_path=None,
                 input_shape=None,
                 auto_adapt_shape=True,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)
        """
        Initializes the VariationalAutoencoderAlgorithmTensorflow model with provided encoder and decoder models,
        loss function, and latent space parameters.

        Args:
            @encoder_model (Model): The encoder model.
            @decoder_model (Model): The decoder model.
            @loss_function (callable or str): The loss function (auto-compiled if string).
            @latent_dimension (int): The dimensionality of the latent space.
            @decoder_latent_dimension (int): The dimensionality for decoder (defaults to latent_dimension).
            @latent_mean_distribution (float): The mean of the latent distribution.
            @latent_standard_deviation (float): The standard deviation of the latent distribution.
            @file_name_encoder (str): The filename for saving the encoder model.
            @file_name_decoder (str): The filename for saving the decoder model.
            @models_saved_path (str): The directory where models will be saved.
            @input_shape (tuple): Expected input shape (without batch dimension).
            @auto_adapt_shape (bool): Whether to automatically adapt to input data shape.
        """
        # Initialize the encoder and decoder models
        self._encoder = encoder_model
        self._decoder = decoder_model

        # Loss function - will be auto-compiled if string
        self._loss_function_string = loss_function if isinstance(loss_function, str) else None
        self._loss_function = self._convert_loss_to_object(loss_function)

        # Metrics for tracking losses
        self._total_loss_tracker = Mean(name="loss")
        self._reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self._kl_loss_tracker = Mean(name="kl_loss")

        self._latent_mean_distribution = latent_mean_distribution
        self._latent_standard_deviation = latent_standard_deviation
        self._latent_dimension = latent_dimension
        self._decoder_latent_dimension = decoder_latent_dimension if decoder_latent_dimension is not None else latent_dimension

        # File names for saving models
        self._file_name_encoder = file_name_encoder
        self._file_name_decoder = file_name_decoder

        # Path for saving models
        self._models_saved_path = models_saved_path

        # Shape adaptation
        self._input_shape = input_shape
        self._auto_adapt_shape = auto_adapt_shape
        self._inferred_shape = None

        # Compiled flag
        self._is_compiled = True  # Auto-compiled

        self.configure_optimizer()

    def _convert_loss_to_object(self, loss):
        """
        Convert string loss names to TensorFlow loss objects.

        Args:
            loss: Loss function (string, callable, or loss object)

        Returns:
            Loss object or callable
        """
        if loss is None:
            return tensorflow.keras.losses.BinaryCrossentropy()

        if isinstance(loss, str):
            loss_map = {
                'mse': tensorflow.keras.losses.MeanSquaredError(),
                'mean_squared_error': tensorflow.keras.losses.MeanSquaredError(),
                'mae': tensorflow.keras.losses.MeanAbsoluteError(),
                'mean_absolute_error': tensorflow.keras.losses.MeanAbsoluteError(),
                'bce': tensorflow.keras.losses.BinaryCrossentropy(),
                'binary_crossentropy': tensorflow.keras.losses.BinaryCrossentropy(),
                'crossentropy': tensorflow.keras.losses.CategoricalCrossentropy(),
                'categorical_crossentropy': tensorflow.keras.losses.CategoricalCrossentropy(),
                'sparse_categorical_crossentropy': tensorflow.keras.losses.SparseCategoricalCrossentropy(),
                'huber': tensorflow.keras.losses.Huber(),
                'kld': tensorflow.keras.losses.KLDivergence(),
                'kullback_leibler_divergence': tensorflow.keras.losses.KLDivergence(),
            }
            loss_lower = loss.lower()
            if loss_lower in loss_map:
                return loss_map[loss_lower]
            else:
                raise ValueError(f"Unknown loss function: {loss}. Available: {list(loss_map.keys())}")

        return loss

    def compile(self, loss=None, optimizer=None, metrics=None, **kwargs):
        """
        Compile the VAE model (TensorFlow-compatible version).

        Args:
            loss: Loss function (can be string name, function, or loss object)
            optimizer: Optimizer (can be passed to fit() method instead)
            metrics: Metrics to track (stored but not implemented yet)
            **kwargs: Additional arguments (ignored)

        Returns:
            self: Returns self for method chaining
        """
        if loss is not None:
            self._loss_function = self._convert_loss_to_object(loss)
            self._loss_function_string = loss if isinstance(loss, str) else None

        self._is_compiled = True
        return self

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

        if isinstance(data, tensorflow.Tensor):
            shape = tuple(data.shape.as_list()[1:])
        elif isinstance(data, numpy.ndarray):
            shape = data.shape[1:] if len(data.shape) > 1 else data.shape
        else:
            # Try to convert to tensor and get shape
            try:
                tensor_data = tensorflow.convert_to_tensor(data)
                shape = tuple(tensor_data.shape.as_list()[1:])
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
            tuple: (batch_x, batch_y) where batch_y is the reconstruction target.
        """
        if isinstance(batch, (tuple, list)):
            if len(batch) == 1:
                # Single input, use as both input and target
                batch_x = batch[0]
                batch_y = batch[0]
            elif len(batch) == 2:
                # Input and target provided
                batch_x, batch_y = batch
            else:
                # Multiple inputs, use first as input and last as target
                batch_x = batch[0]
                batch_y = batch[-1]
        else:
            # Single tensor, use as both input and target
            batch_x = batch
            batch_y = batch

        return batch_x, batch_y

    @tensorflow.function
    def train_step(self, batch):
        """
        Perform a training step for the Variational AutoEncoder (VAE).
        Automatically adapts to different batch formats.

        Args:
            batch: Input data batch.

        Returns:
            dict: Dictionary containing the loss values (total loss, reconstruction loss, KL divergence loss).
        """
        # Prepare batch data
        batch_x, batch_y = self._prepare_batch(batch)

        with tensorflow.GradientTape() as tape:
            # Forward pass: Encode input data and sample from the latent space
            latent_mean, latent_log_variation, latent, label = self._encoder(batch_x)

            # Decode the sampled latent space and generate reconstructed data
            reconstruction_data = self._decoder([latent, label])

            # Calculate reconstruction loss
            if isinstance(self._loss_function, tensorflow.keras.losses.BinaryCrossentropy):
                binary_cross_entropy_loss = tensorflow.keras.losses.binary_crossentropy(batch_y,
                                                                                        reconstruction_data)
                reconstruction_loss = tensorflow.reduce_mean(binary_cross_entropy_loss)
            else:
                reconstruction_loss = self._loss_function(batch_y, reconstruction_data)

            # KL divergence: -0.5 * sum(1 + log(var) - mean^2 - var)
            kl_loss = -0.5 * (1 + latent_log_variation - tensorflow.square(latent_mean) - tensorflow.exp(
                latent_log_variation))
            kl_divergence_loss = tensorflow.reduce_mean(tensorflow.reduce_sum(kl_loss, axis=1))

            # Total loss is the sum of reconstruction loss and KL divergence loss
            loss_model_in_reconstruction = reconstruction_loss + kl_divergence_loss

        # Compute gradients and update model weights
        gradient_update = tape.gradient(loss_model_in_reconstruction, self.trainable_weights)

        # Update loss metrics
        self.optimizer.apply_gradients(zip(gradient_update, self.trainable_weights))
        self._total_loss_tracker.update_state(loss_model_in_reconstruction)
        self._reconstruction_loss_tracker.update_state(reconstruction_loss)
        self._kl_loss_tracker.update_state(kl_divergence_loss)

        # Return a dictionary containing the current loss values
        return {"loss": self._total_loss_tracker.result(),
                "reconstruction_loss": self._reconstruction_loss_tracker.result(),
                "kl_loss": self._kl_loss_tracker.result()}

    def fit(self, x=None, y=None, batch_size=32, epochs=1, verbose=1,
            callbacks=None, validation_data=None, shuffle=True,
            initial_epoch=0, steps_per_epoch=None, validation_steps=None,
            validation_freq=1, optimizer=None, learning_rate=0.001, **kwargs):
        """
        Train the model with automatic shape adaptation.

        Args:
            x: Input data (any shape).
            y: Target data (if None, x is used as target).
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
            optimizer: TensorFlow optimizer (if None, uses already compiled optimizer).
            learning_rate: Learning rate for optimizer (only used if optimizer is None).

        Returns:
            A History object with training metrics.
        """

        # Set optimizer if provided
        if optimizer is not None:
            self.optimizer = optimizer
        elif not hasattr(self, 'optimizer') or self.optimizer is None:
            # Create default optimizer if none exists
            self.optimizer = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate)

        # Prepare the dataset
        if isinstance(x, tensorflow.data.Dataset):
            train_dataset = x
            # Try to infer shape from dataset
            for batch in train_dataset.take(1):
                self._validate_and_adapt_shape(batch)
        else:
            # Validate and adapt shape
            self._validate_and_adapt_shape(x)

            if y is None:
                y = x

            train_dataset = tensorflow.data.Dataset.from_tensor_slices((x, y))
            if shuffle:
                buffer_size = len(x) if hasattr(x, '__len__') else 1000
                train_dataset = train_dataset.shuffle(buffer_size=buffer_size)
            train_dataset = train_dataset.batch(batch_size)

        # Calculate steps per epoch if not provided
        if steps_per_epoch is None:
            try:
                steps_per_epoch = len(train_dataset)
            except:
                steps_per_epoch = 100

        # History to store metrics
        history = {'loss': [], 'reconstruction_loss': [], 'kl_loss': []}

        # Training loop
        for epoch in range(initial_epoch, epochs):
            self._total_loss_tracker.reset_state()
            self._reconstruction_loss_tracker.reset_state()
            self._kl_loss_tracker.reset_state()

            # Trackers for epoch metrics
            epoch_losses = []
            epoch_recon_losses = []
            epoch_kl_losses = []

            if verbose == 1:
                print(f'\nEpoch {epoch + 1}/{epochs}')

            # Progress tracking
            step = 0
            for batch_data in train_dataset:
                step += 1

                # Perform training step
                metrics = self.train_step(batch_data)
                current_loss = float(metrics['loss'])
                current_recon_loss = float(metrics['reconstruction_loss'])
                current_kl_loss = float(metrics['kl_loss'])

                # Track losses for this epoch
                epoch_losses.append(current_loss)
                epoch_recon_losses.append(current_recon_loss)
                epoch_kl_losses.append(current_kl_loss)

                # Simple progress bar
                if verbose == 1:
                    progress = int(50 * step / steps_per_epoch)
                    bar = '=' * progress + '>' + '.' * (50 - progress - 1)
                    print(
                        f'\r[{bar}] {step}/{steps_per_epoch} - loss: {current_loss:.4f} - recon_loss: {current_recon_loss:.4f} - kl_loss: {current_kl_loss:.4f}',
                        end='', flush=True)

                if step >= steps_per_epoch:
                    break

            # Store epoch losses
            epoch_loss = float(self._total_loss_tracker.result())
            epoch_recon_loss = numpy.mean(epoch_recon_losses) if epoch_recon_losses else 0.0
            epoch_kl_loss = numpy.mean(epoch_kl_losses) if epoch_kl_losses else 0.0

            history['loss'].append(epoch_loss)
            history['reconstruction_loss'].append(epoch_recon_loss)
            history['kl_loss'].append(epoch_kl_loss)

            if verbose == 1:
                print(f' - loss: {epoch_loss:.4f} - recon_loss: {epoch_recon_loss:.4f} - kl_loss: {epoch_kl_loss:.4f}')
            elif verbose == 2:
                print(
                    f'Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.4f} - recon_loss: {epoch_recon_loss:.4f} - kl_loss: {epoch_kl_loss:.4f}')

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
                            'reconstruction_loss': epoch_recon_loss,
                            'kl_loss': epoch_kl_loss
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
        val_losses = []
        val_recon_losses = []
        val_kl_losses = []
        step = 0

        # Prepare validation dataset
        if isinstance(validation_data, tensorflow.data.Dataset):
            val_dataset = validation_data
        else:
            val_x, val_y = validation_data
            val_dataset = tensorflow.data.Dataset.from_tensor_slices((val_x, val_y))
            val_dataset = val_dataset.batch(32)

        for batch_data in val_dataset:
            batch_x, batch_y = self._prepare_batch(batch_data)

            # Forward pass through encoder and decoder
            latent_mean, latent_log_variation, latent, label = self._encoder(batch_x, training=False)
            reconstruction_data = self._decoder([latent, label], training=False)

            # Calculate reconstruction loss
            if isinstance(self._loss_function, tensorflow.keras.losses.BinaryCrossentropy):
                binary_cross_entropy_loss = tensorflow.keras.losses.binary_crossentropy(batch_y,
                                                                                        reconstruction_data)
                reconstruction_loss = tensorflow.reduce_mean(binary_cross_entropy_loss)
            else:
                reconstruction_loss = self._loss_function(batch_y, reconstruction_data)

            # KL divergence
            kl_loss = -0.5 * (1 + latent_log_variation - tensorflow.square(latent_mean) - tensorflow.exp(
                latent_log_variation))
            kl_divergence_loss = tensorflow.reduce_mean(tensorflow.reduce_sum(kl_loss, axis=1))

            # Total loss
            total_loss = reconstruction_loss + kl_divergence_loss

            val_losses.append(float(total_loss))
            val_recon_losses.append(float(reconstruction_loss))
            val_kl_losses.append(float(kl_divergence_loss))

            step += 1
            if validation_steps is not None and step >= validation_steps:
                break

        return numpy.mean(val_losses) if val_losses else 0.0

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

    def configure_optimizer(self,
                            learning_rate=0.001,
                            beta_1=0.9,
                            beta_2=0.999,
                            epsilon=1e-7,
                            amsgrad=False,
                            weight_decay=None):
        """
        Configure the Adam optimizer with custom parameters.
        """
        # Nota: weight_decay não é mais suportado como 'decay' no Keras moderno
        optimizer_kwargs = {
            'learning_rate': learning_rate,
            'beta_1': beta_1,
            'beta_2': beta_2,
            'epsilon': epsilon,
            'amsgrad': amsgrad
        }

        self.optimizer = tensorflow.keras.optimizers.Adam(**optimizer_kwargs)

        # Compile the model after setting up the optimizer
        super().compile(optimizer=self.optimizer)

    def get_decoder_trained(self):
        return self._decoder

    def get_encoder_trained(self):
        return self._encoder

    def create_embedding(self, data, labels=None):
        """
        Generates latent space embeddings using the trained encoder.

        Args:
            data (ndarray): Input data to encode.
            labels (ndarray, optional): Optional labels for conditional encoding.

        Returns:
            ndarray: Latent space representations (mean vectors).
        """
        if labels is not None:
            encoder_output = self._encoder.predict([data, labels], batch_size=32)
        else:
            encoder_output = self._encoder.predict(data, batch_size=32)

        # Return only the mean (first output)
        return encoder_output[0]

    def get_samples(self, number_samples_per_class):
        """
        Generate synthetic samples for each specified class using the trained decoder.

        Args:
            number_samples_per_class (dict):
                Dictionary specifying the number of samples to generate for each class.
                Expected structure:
                {
                    "classes": {class_label: number_of_samples, ...},
                    "number_classes": total_number_of_classes
                }

        Returns:
            dict:
                A dictionary where each key is a class label and the value is an array of generated samples.
        """

        # Initialize a dictionary to store the generated samples for each class
        generated_data = {}

        # Iterate over each class and the corresponding number of samples to generate
        for label_class, number_instances in number_samples_per_class["classes"].items():
            # Create a one-hot encoded label array for all samples in the current class
            label_samples_generated = to_categorical([label_class] * number_instances,
                                                     num_classes=number_samples_per_class["number_classes"])

            # Sample random latent vectors from a standard normal distribution
            latent_noise = numpy.random.normal(size=(number_instances, self._decoder_latent_dimension))

            # Use the decoder to generate samples conditioned on the latent vectors and class labels
            generated_samples = self._decoder.predict([latent_noise, label_samples_generated], verbose=0)

            # Store the generated samples in the dictionary under the corresponding class label
            generated_data[label_class] = generated_samples

        # Return the dictionary with all generated samples, organized by class
        return generated_data

    def generate_synthetic_data(self, number_samples_generate, label_class, num_classes, latent_dimension=None):
        """
        Generate synthetic data using the Variational AutoEncoder (VAE).

        Args:
            number_samples_generate (int): Number of synthetic samples to generate.
            label_class (int): Class label to generate.
            num_classes (int): Total number of classes.
            latent_dimension (int, optional): Dimension of the latent space.

        Returns:
            numpy.ndarray: Synthetic data generated by the decoder.
        """
        if latent_dimension is None:
            latent_dimension = self._latent_dimension

        # Generate random noise samples in the latent space
        random_noise_generate = tensorflow.random.normal(
            shape=(number_samples_generate, latent_dimension),
            mean=self._latent_mean_distribution,
            stddev=self._latent_standard_deviation,
            dtype=tensorflow.float32
        )

        # Create one-hot encoded labels for compatibility with decoder
        label_list = tensorflow.one_hot(
            tensorflow.fill((number_samples_generate,), label_class),
            depth=num_classes
        )
        label_list = tensorflow.cast(label_list, dtype=tensorflow.float32)

        # Generate synthetic data by passing random noise and labels through the decoder
        synthetic_data = self._decoder.predict([random_noise_generate, label_list], verbose=0)

        # Return the generated synthetic data
        return synthetic_data

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
        return data

    @property
    def metrics(self):
        """
        Returns:
            list: List of metrics to track during training.
        """
        return [self._total_loss_tracker, self._reconstruction_loss_tracker, self._kl_loss_tracker]

    @property
    def input_shape(self):
        return self.get_input_shape()

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
        encoder_file_name = os.path.join(directory, f"fold_{file_name}_encoder")
        decoder_file_name = os.path.join(directory, f"fold_{file_name}_decoder")

        # Save encoder model
        self._save_model_to_json(self._encoder, f"{encoder_file_name}.json")
        self._encoder.save_weights(f"{encoder_file_name}.weights.h5")

        # Save decoder model
        self._save_model_to_json(self._decoder, f"{decoder_file_name}.json")
        self._decoder.save_weights(f"{decoder_file_name}.weights.h5")

        # Save shape information
        shape_info = {
            'input_shape': self._input_shape,
            'inferred_shape': self._inferred_shape,
            'latent_dimension': self._latent_dimension,
            'decoder_latent_dimension': self._decoder_latent_dimension
        }
        shape_file = os.path.join(directory, f"fold_{file_name}_shape_info.json")
        with open(shape_file, 'w') as f:
            json.dump(shape_info, f)

    @staticmethod
    def _save_model_to_json(model, file_path):
        """
        Save model architecture to a JSON file.

        Args:
            model (Model): Model to save.
            file_path (str): Path to the JSON file.
        """
        with open(file_path, "w") as json_file:
            json.dump(model.to_json(), json_file)

    def load_models(self, directory, file_name):
        """
        Load the encoder and decoder models from a directory.

        Args:
            directory (str): Directory where models are stored.
            file_name (str): Base file name for loading models.
        """

        # Construct file names for encoder and decoder models
        encoder_file_name = "{}_encoder".format(file_name)
        decoder_file_name = "{}_decoder".format(file_name)

        # Load the encoder and decoder models from the specified directory
        self._encoder = self._save_neural_network_model(encoder_file_name, directory)
        self._decoder = self._save_neural_network_model(decoder_file_name, directory)

        # Load shape information if available
        shape_file = os.path.join(directory, f"{file_name}_shape_info.json")
        if os.path.exists(shape_file):
            with open(shape_file, 'r') as f:
                shape_info = json.load(f)
                self._input_shape = tuple(shape_info.get('input_shape', ()))
                self._inferred_shape = tuple(shape_info.get('inferred_shape', ()))
                self._latent_dimension = shape_info.get('latent_dimension', self._latent_dimension)
                self._decoder_latent_dimension = shape_info.get('decoder_latent_dimension',
                                                                self._decoder_latent_dimension)

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