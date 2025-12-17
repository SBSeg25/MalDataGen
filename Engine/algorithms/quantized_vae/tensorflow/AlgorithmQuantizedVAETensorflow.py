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

    from tensorflow.keras import Model
    from tensorflow.keras.metrics import Mean
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.losses import BinaryCrossentropy

except ImportError as error:
    print(error)
    sys.exit(-1)


class QuantizedVAEAlgorithmTensorflow(Model):
    """
    Implements a Vector Quantized Variational Autoencoder (VQ-VAE) with adaptive input handling.

    This model combines an encoder, decoder, and vector quantization layer to learn compressed
    representations of input data. It automatically adapts to any input shape: (x), (x, y), (x, y, z), etc.

    The algorithm supports training with reconstruction loss and commitment loss for
    the quantization layer, enabling stable training of discrete latent variables.

    References:
        - van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). "Neural Discrete
        Representation Learning." Advances in Neural Information Processing Systems (NeurIPS).
        Available at: https://arxiv.org/abs/1711.00937

    Example:
        >>> # Example with 1D data
        >>> vq_vae_1d = QuantizedVAEAlgorithmTensorflow(
        ...     encoder_model=encoder_1d,
        ...     decoder_model=decoder_1d,
        ...     quantized_vae_model=vq_vae_1d,
        ...     train_variance=0.1,
        ...     latent_dimension=64,
        ...     number_embeddings=512,
        ...     input_shape=(100,)
        ... )
        >>> vq_vae_1d.fit(data_1d, epochs=50)

        >>> # Example with 2D data (images)
        >>> vq_vae_2d = QuantizedVAEAlgorithmTensorflow(
        ...     encoder_model=encoder_2d,
        ...     decoder_model=decoder_2d,
        ...     quantized_vae_model=vq_vae_2d,
        ...     train_variance=0.1,
        ...     latent_dimension=128,
        ...     number_embeddings=512,
        ...     input_shape=(28, 28)
        ... )
        >>> vq_vae_2d.fit(data_2d, labels_2d, epochs=100)
    """

    def __init__(self,
                 encoder_model,
                 decoder_model,
                 quantized_vae_model,
                 train_variance,
                 latent_dimension,
                 number_embeddings,
                 file_name_encoder=None,
                 file_name_decoder=None,
                 models_saved_path=None,
                 input_shape=None,
                 auto_adapt_shape=True,
                 **kwargs):
        """
        Initializes the QuantizedVAEAlgorithm with encoder, decoder, and VQ-VAE components.

        Args:
            encoder_model: Encoder network that compresses input data to latent space
            decoder_model: Decoder network that reconstructs data from quantized latent codes
            quantized_vae_model: Complete VQ-VAE model including quantization layer
            train_variance: Data variance used to scale reconstruction loss
            latent_dimension: Dimensionality of latent space before quantization
            number_embeddings: Size of quantization codebook
            file_name_encoder: Filename for saving encoder weights
            file_name_decoder: Filename for saving decoder weights
            models_saved_path: Directory path for model weight storage
            input_shape: Expected input shape (without batch dimension)
            auto_adapt_shape: Whether to automatically adapt to input data shape
        """
        super().__init__(**kwargs)

        self._train_variance = train_variance
        self._latent_dimension = latent_dimension
        self._number_embeddings = number_embeddings

        self._encoder = encoder_model
        self._decoder = decoder_model
        self._quantized_vae_model = quantized_vae_model

        self._file_name_encoder = file_name_encoder
        self._file_name_decoder = file_name_decoder
        self._models_saved_path = models_saved_path

        # Shape adaptation
        self._input_shape = input_shape
        self._auto_adapt_shape = auto_adapt_shape
        self._inferred_shape = None

        # Metrics
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.vq_loss_tracker = Mean(name="vq_loss")

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
            tuple: (batch_x, batch_y) formatted for VQ-VAE training.
        """
        if isinstance(batch, (tuple, list)):
            if len(batch) == 1:
                # Single input
                batch_x = batch[0]
                batch_y = batch[0]  # Autoencoder: reconstruct input
            elif len(batch) == 2:
                # Check if it's (input, target) or (input, labels)
                x, y = batch
                # VQ-VAE typically uses input as both x and y for reconstruction
                # If y has different shape than x, treat it as labels
                if tensorflow.is_tensor(x) and tensorflow.is_tensor(y):
                    if x.shape == y.shape:
                        batch_x = (x, y)
                        batch_y = y
                    else:
                        # y is labels, x is both input and target
                        batch_x = (x, y)
                        batch_y = x
                else:
                    batch_x = (x, y)
                    batch_y = x
            else:
                # Multiple inputs
                batch_x = batch
                batch_y = batch[0]
        else:
            # Single tensor
            batch_x = (batch, batch)
            batch_y = batch

        return batch_x, batch_y

    @property
    def metrics(self):
        """
        Returns the list of metrics tracked during training.

        Returns:
            list: List of metric trackers.
        """
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.vq_loss_tracker]

    @tensorflow.function
    def train_step(self, data):
        """
        Performs a single training step on a batch of data with automatic batch handling.

        Args:
            data: Input data batch.

        Returns:
            dict: Dictionary with loss metrics for the current step.
        """
        # Prepare batch
        x, y = self._prepare_batch(data)

        # Extract output tensor (for VQ-VAE, it's typically the first element)
        if isinstance(x, (tuple, list)):
            output_tensor = x[0]
        else:
            output_tensor = x

        with tensorflow.GradientTape() as tape:
            # Forward pass through VQ-VAE
            reconstructions = self._quantized_vae_model(x)

            # Compute reconstruction loss (MSE normalized by data variance)
            reconstruction_loss = (
                    tensorflow.reduce_mean((output_tensor - reconstructions) ** 2) / self._train_variance
            )

            # Total loss is reconstruction loss plus VQ losses (commitment + codebook)
            vae_model_loss = reconstruction_loss + sum(self._quantized_vae_model.losses)

        # Compute and apply gradients
        gradient_flow = tape.gradient(vae_model_loss, self._quantized_vae_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient_flow, self._quantized_vae_model.trainable_variables))

        # Update metric trackers
        self.total_loss_tracker.update_state(vae_model_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self._quantized_vae_model.losses))

        # Return results
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }

    def fit(self, x=None, y=None, batch_size=32, epochs=1, verbose=1,
            callbacks=None, validation_data=None, shuffle=True,
            initial_epoch=0, steps_per_epoch=None, validation_steps=None,
            validation_freq=1, optimizer=None, learning_rate=0.001, **kwargs):
        """
        Train the model with automatic shape adaptation.

        Args:
            x: Input data (any shape) or tf.data.Dataset.
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
            for batch in train_dataset:
                self._validate_and_adapt_shape(batch)
                break
        else:
            # Validate and adapt shape
            self._validate_and_adapt_shape(x)

            # Handle different input formats
            if isinstance(x, tuple):
                # x is tuple: (data, labels) or (data, target)
                if len(x) == 2:
                    x_data, target_or_labels = x
                else:
                    x_data = x[0]
                    target_or_labels = None
            else:
                x_data = x
                target_or_labels = y

            # Convert to tensors
            if isinstance(x_data, numpy.ndarray):
                x_data = tensorflow.constant(x_data, dtype=tensorflow.float32)

            if target_or_labels is not None:
                if isinstance(target_or_labels, numpy.ndarray):
                    target_or_labels = tensorflow.constant(target_or_labels, dtype=tensorflow.float32)
                # Create dataset with both
                train_dataset = tensorflow.data.Dataset.from_tensor_slices((x_data, target_or_labels))
            else:
                # Use x_data as both input and target (autoencoder behavior)
                train_dataset = tensorflow.data.Dataset.from_tensor_slices((x_data, x_data))

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
        history = {'loss': [], 'reconstruction_loss': [], 'vqvae_loss': []}

        # Training loop
        for epoch in range(initial_epoch, epochs):
            self.total_loss_tracker.reset_state()
            self.reconstruction_loss_tracker.reset_state()
            self.vq_loss_tracker.reset_state()

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
                current_vq_loss = float(metrics['vqvae_loss'])

                # Simple progress bar
                if verbose == 1:
                    progress = int(50 * step / steps_per_epoch)
                    bar = '=' * progress + '>' + '.' * (50 - progress - 1)
                    print(
                        f'\r[{bar}] {step}/{steps_per_epoch} - loss: {current_loss:.4f} - recon: {current_recon_loss:.4f} - vq: {current_vq_loss:.4f}',
                        end='', flush=True)

                if step >= steps_per_epoch:
                    break

            # Store epoch losses
            epoch_loss = float(self.total_loss_tracker.result())
            epoch_reconstruction_loss = float(self.reconstruction_loss_tracker.result())
            epoch_vq_loss = float(self.vq_loss_tracker.result())

            history['loss'].append(epoch_loss)
            history['reconstruction_loss'].append(epoch_reconstruction_loss)
            history['vqvae_loss'].append(epoch_vq_loss)

            if verbose == 1:
                print(f' - loss: {epoch_loss:.4f} - recon: {epoch_reconstruction_loss:.4f} - vq: {epoch_vq_loss:.4f}')
            elif verbose == 2:
                print(f'Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.4f}')

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
                            'reconstruction_loss': epoch_reconstruction_loss,
                            'vqvae_loss': epoch_vq_loss
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
            validation_data: Validation dataset.
            validation_steps: Number of validation steps.

        Returns:
            Average validation loss.
        """
        val_losses = []
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
            batch_x, batch_y = self._prepare_batch(batch_data)

            # Get output tensor
            if isinstance(batch_x, (tuple, list)):
                output_tensor = batch_x[0]
            else:
                output_tensor = batch_x

            reconstructed = self._quantized_vae_model(batch_x, training=False)
            loss = tensorflow.reduce_mean(tensorflow.square(output_tensor - reconstructed))
            val_losses.append(float(loss))

            step += 1
            if validation_steps is not None and step >= validation_steps:
                break

        return numpy.mean(val_losses) if val_losses else 0.0

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

    def get_samples(self, number_samples_per_class):
        """
        Generates samples from the latent space using the decoder.

        Samples are generated by:
        1. Randomly selecting indices from the codebook
        2. Gathering corresponding latent vectors
        3. Decoding these vectors conditioned on class labels

        Args:
            number_samples_per_class (dict): Dictionary with 'classes' and 'number_classes' keys

        Returns:
            dict: Generated samples keyed by class label.
        """
        generated_data = {}

        # Get codebook embeddings from the quantization layer
        codebook = self._quantized_vae_model.get_layer("vector_quantizer").embeddings

        number_embeddings = self._number_embeddings

        for label_class, number_instances in number_samples_per_class["classes"].items():
            # Create one-hot encoded labels for the samples
            label_samples_generated = to_categorical([label_class] * number_instances,
                                                     num_classes=number_samples_per_class["number_classes"])

            # Randomly sample from codebook
            sampled_indices = numpy.random.choice(number_embeddings, size=number_instances)

            # Get corresponding latent vectors
            quantized_vectors = tensorflow.gather(codebook, sampled_indices)
            quantized_vectors = tensorflow.convert_to_tensor(quantized_vectors)

            # Decode the latent vectors
            generated_samples = self._decoder.predict([quantized_vectors, label_samples_generated], verbose=0)

            generated_data[label_class] = generated_samples

        return generated_data

    def save_model(self, directory, file_name):
        """
        Save the encoder and decoder models with shape information.

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
            'number_embeddings': self._number_embeddings,
            'train_variance': self._train_variance
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
        Load the encoder and decoder models with shape information.

        Args:
            directory (str): Directory where models are stored.
            file_name (str): Base file name for loading models.
        """
        # Construct file names for encoder and decoder models
        encoder_file_name = os.path.join(directory, f"fold_{file_name}_encoder")
        discriminator_file_name = os.path.join(directory, f"fold_{file_name}_decoder")

        # Load weights
        self._encoder.load_weights(f"{encoder_file_name}.weights.h5")
        self._decoder.load_weights(f"{discriminator_file_name}.weights.h5")

        # Load shape information if available
        shape_file = os.path.join(directory, f"fold_{file_name}_shape_info.json")
        if os.path.exists(shape_file):
            with open(shape_file, 'r') as f:
                shape_info = json.load(f)
                self._input_shape = tuple(shape_info.get('input_shape', ()))
                self._inferred_shape = tuple(shape_info.get('inferred_shape', ()))
                self._latent_dimension = shape_info.get('latent_dimension', self._latent_dimension)
                self._number_embeddings = shape_info.get('number_embeddings', self._number_embeddings)
                self._train_variance = shape_info.get('train_variance', self._train_variance)

    # ========== PROPERTIES ==========

    @property
    def encoder(self):
        """Get the encoder model."""
        return self._encoder

    @encoder.setter
    def encoder(self, value):
        """Set the encoder model."""
        self._encoder = value

    @property
    def decoder(self):
        """Get the decoder model."""
        return self._decoder

    @decoder.setter
    def decoder(self, value):
        """Set the decoder model."""
        self._decoder = value

    @property
    def quantized_vae_model(self):
        """Get the complete VQ-VAE model."""
        return self._quantized_vae_model

    @quantized_vae_model.setter
    def quantized_vae_model(self, value):
        """Set the complete VQ-VAE model."""
        self._quantized_vae_model = value

    @property
    def train_variance(self):
        """Get the training variance."""
        return self._train_variance

    @train_variance.setter
    def train_variance(self, value):
        """Set the training variance."""
        if value <= 0:
            raise ValueError("Train variance must be positive")
        self._train_variance = value

    @property
    def latent_dimension(self):
        """Get the latent dimension."""
        return self._latent_dimension

    @latent_dimension.setter
    def latent_dimension(self, value):
        """Set the latent dimension."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Latent dimension must be a positive integer")
        self._latent_dimension = value

    @property
    def number_embeddings(self):
        """Get the number of embeddings."""
        return self._number_embeddings

    @number_embeddings.setter
    def number_embeddings(self, value):
        """Set the number of embeddings."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Number of embeddings must be a positive integer")
        self._number_embeddings = value

    @property
    def input_shape(self):
        """Get the current input shape."""
        return self.get_input_shape()

    @property
    def file_name_encoder(self):
        """Get the encoder filename."""
        return self._file_name_encoder

    @file_name_encoder.setter
    def file_name_encoder(self, value):
        """Set the encoder filename."""
        self._file_name_encoder = value

    @property
    def file_name_decoder(self):
        """Get the decoder filename."""
        return self._file_name_decoder

    @file_name_decoder.setter
    def file_name_decoder(self, value):
        """Set the decoder filename."""
        self._file_name_decoder = value

    @property
    def models_saved_path(self):
        """Get the models save path."""
        return self._models_saved_path

    @models_saved_path.setter
    def models_saved_path(self, value):
        """Set the models save path."""
        self._models_saved_path = value