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

    from tensorflow.keras.metrics import Mean
    from tensorflow.keras.models import Model

    from tensorflow.keras.utils import to_categorical

except ImportError as error:
    print(error)
    sys.exit(-1)

DEFAULT_LATENT_MEAN_DISTRIBUTION = 0.0
DEFAULT_latent_standard_deviation = 1.0
DEFAULT_LATENT_DIMENSION = 64
DEFAULT_NUMBER_CLASSES = 2


class AutoencoderAlgorithmTensorflow(Model):
    """
    An adaptive AutoEncoder class that handles any input shape dynamically.

    This class provides a foundation for AutoEncoder models with methods for training,
    generating synthetic data, saving and loading models. It automatically adapts to
    different input shapes: (x), (x, y), (x, y, z), etc.

    Args:
        @encoder_model (Model, optional):
            The encoder part of the AutoEncoder.
        @decoder_model (Model, optional):
            The decoder part of the AutoEncoder.
        @loss_function (loss, optional):
            The loss function for training.
        @file_name_encoder (str, optional):
            The file name for saving the encoder model.
        @file_name_decoder (str, optional):
            The file name for saving the decoder model.
        @models_saved_path (str, optional):
            The path to save the models.
        @latent_mean_distribution (float, optional):
            Mean of the latent space distribution.
        @latent_standard_deviation (float, optional):
            Standard deviation of the latent space distribution.
        @latent_dimension (int, optional):
            The dimensionality of the latent space.
        @input_shape (tuple, optional):
            Expected input shape. If None, will be inferred from data.
        @auto_adapt_shape (bool, optional):
            If True, automatically adapts to input data shape. Default: True

    Example:
        >>> # Example with 1D data
        >>> encoder_1d = build_encoder(input_shape=(100,), latent_dimension=64)
        >>> decoder_1d = build_decoder(latent_dimension=64, output_shape=(100,))
        >>> autoencoder_1d = AutoencoderAlgorithmTensorflow(
        ...     encoder_model=encoder_1d,
        ...     decoder_model=decoder_1d,
        ...     input_shape=(100,)
        ... )
        >>> autoencoder_1d.compile(loss='mse')  # Compile before training!
        >>> autoencoder_1d.fit(data_1d, epochs=10)

        >>> # Example with 2D data
        >>> encoder_2d = build_encoder(input_shape=(28, 28), latent_dimension=64)
        >>> decoder_2d = build_decoder(latent_dimension=64, output_shape=(28, 28))
        >>> autoencoder_2d = AutoencoderAlgorithmTensorflow(
        ...     encoder_model=encoder_2d,
        ...     decoder_model=decoder_2d,
        ...     input_shape=(28, 28)
        ... )
        >>> autoencoder_2d.compile(loss='mae')
        >>> autoencoder_2d.fit(data_2d, epochs=10)

        >>> # Example with 3D data (images) and custom loss
        >>> encoder_3d = build_encoder(input_shape=(128, 128, 3), latent_dimension=64)
        >>> decoder_3d = build_decoder(latent_dimension=64, output_shape=(128, 128, 3))
        >>> autoencoder_3d = AutoencoderAlgorithmTensorflow(
        ...     encoder_model=encoder_3d,
        ...     decoder_model=decoder_3d,
        ...     loss_function=tensorflow.keras.losses.MeanSquaredError(),
        ...     input_shape=(128, 128, 3)
        ... )
        >>> autoencoder_3d.fit(data_3d, epochs=10)  # Loss already set in __init__
    """

    def __init__(self,
                 encoder_model,
                 decoder_model,
                 loss_function=None,
                 file_name_encoder=None,
                 file_name_decoder=None,
                 models_saved_path=None,
                 latent_mean_distribution=DEFAULT_LATENT_MEAN_DISTRIBUTION,
                 latent_standard_deviation=DEFAULT_latent_standard_deviation,
                 latent_dimension=DEFAULT_LATENT_DIMENSION,
                 input_shape=None,
                 auto_adapt_shape=True):

        super().__init__()
        """
        Initializes an adaptive AutoEncoder model.

        Args:
            @encoder_model (Model):
                The encoder part of the AutoEncoder.
            @decoder_model (Model):
                The decoder part of the AutoEncoder.
            @loss_function (loss):
                The loss function used for training.
            @file_name_encoder (str):
                The filename for saving the trained encoder model.
            @file_name_decoder (str):
                The filename for saving the trained decoder model.
            @models_saved_path (str):
                The directory path where models should be saved.
            @latent_mean_distribution (float):
                The mean of the latent noise distribution.
            @latent_standard_deviation (float):
                The standard deviation of the latent noise distribution.
            @latent_dimension (int):
                The number of dimensions in the latent space.
            @input_shape (tuple):
                Expected input shape (without batch dimension).
            @auto_adapt_shape (bool):
                Whether to automatically adapt to input data shape.
        """
        if not isinstance(encoder_model, tensorflow.keras.Model):
            raise TypeError("encoder_model must be a tf.keras.Model instance.")

        if not isinstance(decoder_model, tensorflow.keras.Model):
            raise TypeError("decoder_model must be a tf.keras.Model instance.")

        if file_name_encoder is not None and (not isinstance(file_name_encoder, str) or not file_name_encoder):
            raise ValueError("file_name_encoder must be a non-empty string or None.")

        if file_name_decoder is not None and (not isinstance(file_name_decoder, str) or not file_name_decoder):
            raise ValueError("file_name_decoder must be a non-empty string or None.")

        if models_saved_path is not None and (not isinstance(models_saved_path, str) or not models_saved_path):
            raise ValueError("models_saved_path must be a non-empty string or None.")

        if not isinstance(latent_mean_distribution, (int, float)):
            raise TypeError("latent_mean_distribution must be a number.")

        if not isinstance(latent_standard_deviation, (int, float)):
            raise TypeError("latent_standard_deviation must be a number.")

        if latent_standard_deviation <= 0:
            raise ValueError("latent_standard_deviation must be greater than 0.")

        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")

        # Initialize the encoder and decoder models
        self._encoder = encoder_model
        self._decoder = decoder_model

        # loss function and metric for tracking total loss
        self._loss_function = loss_function
        self._total_loss_tracker = Mean(name="loss")
        self._latent_mean_distribution = latent_mean_distribution
        self._latent_standard_deviation = latent_standard_deviation
        self._latent_dimension = latent_dimension

        # File names for saving models
        self._file_name_encoder = file_name_encoder
        self._file_name_decoder = file_name_decoder

        # Path for saving models
        self._models_saved_path = models_saved_path

        # Shape adaptation
        self._input_shape = input_shape
        self._auto_adapt_shape = auto_adapt_shape
        self._inferred_shape = None

        # Combined encoder-decoder model
        self._encoder_decoder_model = Model(self._encoder.input, self._decoder(self._encoder.output))

        # Compiled flag
        self._is_compiled = False

    def compile(self, loss=None, optimizer=None, metrics=None, **kwargs):
        """
        Compile the autoencoder model (TensorFlow-compatible version).

        This method allows setting the loss function using string names or function objects,
        similar to Keras' compile() method.

        Args:
            loss: Loss function (can be string name, function, or loss object)
            optimizer: Optimizer (can be passed to fit() method instead)
            metrics: Metrics to track (stored but not implemented yet)
            **kwargs: Additional arguments (ignored)

        Returns:
            self: Returns self for method chaining
        """
        if loss is not None:
            if isinstance(loss, str):
                # Convert string loss names to TensorFlow loss functions
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
                    self._loss_function = loss_map[loss_lower]
                else:
                    raise ValueError(f"Unknown loss function: {loss}. Available options: {list(loss_map.keys())}")
            elif callable(loss):
                self._loss_function = loss
            else:
                raise TypeError("loss must be a string, callable function, or TensorFlow loss object")

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
    def train_step(self, batch, optimizer=None):
        """
        Perform a training step for the AutoEncoder.
        Automatically adapts to different batch formats.

        Args:
            batch: Input data batch (can be single tensor, tuple, or list).
            optimizer: Optional optimizer (if None, uses self.optimizer).

        Returns:
            dict: Dictionary containing the loss value.
        """
        # Prepare batch data
        batch_x, batch_y = self._prepare_batch(batch)

        with tensorflow.GradientTape() as gradient_ae:
            # Forward pass: Generate reconstructed data using the encoder-decoder model
            reconstructed_data = self._encoder_decoder_model(batch_x, training=True)

            # Calculate the loss between target and reconstructed data
            if self._loss_function is not None:
                # Check if loss_function is a string (not compiled)
                if isinstance(self._loss_function, str):
                    raise RuntimeError(
                        "Model not compiled. Please call model.compile(loss='mse') or similar before training."
                    )
                # Use the configured loss function
                update_gradient_loss = self._loss_function(batch_y, reconstructed_data)
            else:
                # Default to mean squared error
                update_gradient_loss = tensorflow.reduce_mean(tensorflow.square(batch_y - reconstructed_data))

        # Calculate gradients of the loss with respect to trainable variables
        gradient_update = gradient_ae.gradient(update_gradient_loss, self._encoder_decoder_model.trainable_variables)

        # Apply gradients using the optimizer
        self.optimizer.apply_gradients(zip(gradient_update, self._encoder_decoder_model.trainable_variables))

        # Update the total loss metric
        self._total_loss_tracker.update_state(update_gradient_loss)

        # Return a dictionary containing the current loss value
        return {"loss": self._total_loss_tracker.result()}

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

        # Check if loss_function is a string and warn user
        if isinstance(self._loss_function, str):
            raise RuntimeError(
                f"Model not compiled. Loss function is set to string '{self._loss_function}'. "
                f"Please call model.compile(loss='{self._loss_function}') before training."
            )

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
                steps_per_epoch = 100  # Default value if length cannot be determined

        # History to store metrics
        history = {'loss': []}

        # Training loop
        for epoch in range(initial_epoch, epochs):
            self._total_loss_tracker.reset_state()

            if verbose == 1:
                print(f'\nEpoch {epoch + 1}/{epochs}')

            # Progress tracking
            step = 0
            for batch_data in train_dataset:
                step += 1

                # Perform training step
                metrics = self.train_step(batch_data)
                current_loss = float(metrics['loss'])

                # Simple progress bar
                if verbose == 1:
                    progress = int(50 * step / steps_per_epoch)
                    bar = '=' * progress + '>' + '.' * (50 - progress - 1)
                    print(f'\r[{bar}] {step}/{steps_per_epoch} - loss: {current_loss:.4f}',
                          end='', flush=True)

                if step >= steps_per_epoch:
                    break

            # Store epoch loss
            epoch_loss = float(self._total_loss_tracker.result())
            history['loss'].append(epoch_loss)

            if verbose == 1:
                print(f' - loss: {epoch_loss:.4f}')
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

            # callbacks
            if callbacks is not None:
                for callback in callbacks:
                    callback.on_epoch_end(epoch, {'loss': epoch_loss})

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

        for batch_data in validation_data:
            batch_x, batch_y = self._prepare_batch(batch_data)
            reconstructed = self._encoder_decoder_model(batch_x, training=False)

            if self._loss_function is not None:
                # Check if loss_function is a string (not compiled)
                if isinstance(self._loss_function, str):
                    raise RuntimeError(
                        "Model not compiled. Please call model.compile(loss='mse') or similar before evaluation."
                    )
                loss = self._loss_function(batch_y, reconstructed)
            else:
                loss = tensorflow.reduce_mean(tensorflow.square(batch_y - reconstructed))

            val_losses.append(float(loss))

            step += 1
            if validation_steps is not None and step >= validation_steps:
                break

        return numpy.mean(val_losses) if val_losses else 0.0

    def get_samples(self, number_samples_per_class):
        """
        Generates synthetic data samples for each specified class using the trained decoder.
        This function creates synthetic samples conditioned on class labels, typically used
        when working with conditional generative models (like conditional VAEs or conditional GANs).

        Args:
            number_samples_per_class (dict):
                A dictionary specifying how many synthetic samples should be generated per class.
                Expected structure:
                {
                    "classes": {class_label: number_of_samples, ...},
                    "number_classes": total_number_of_classes
                }

        Returns:
            dict:
                A dictionary where each key is a class label and the value is an array of generated samples.
                Each array contains the synthetic samples generated for the corresponding class.
        """

        # Initialize an empty dictionary to store generated samples grouped by class label
        generated_data = {}

        # Loop through each class label and the corresponding number of samples to generate
        for label_class, number_instances in number_samples_per_class["classes"].items():
            # Create a batch of one-hot encoded class labels, all set to the current class
            label_samples_generated = to_categorical(
                [label_class] * number_instances,
                num_classes=number_samples_per_class["number_classes"]
            )

            # Generate random noise vectors (latent space vectors) for each sample
            latent_noise = numpy.random.normal(
                self._latent_mean_distribution,
                self._latent_standard_deviation,
                (number_instances, self._latent_dimension)
            )

            # Use the decoder to generate synthetic samples from the latent space and class labels
            generated_samples = self._decoder.predict([latent_noise, label_samples_generated], verbose=0)

            # Store the generated samples in the dictionary under the corresponding class label
            generated_data[label_class] = generated_samples

        # Return the dictionary containing all generated samples, organized by class
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
            'latent_dimension': self._latent_dimension
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
    def decoder(self):
        return self._decoder

    @property
    def encoder(self):
        return self._encoder

    @property
    def input_shape(self):
        return self.get_input_shape()

    @decoder.setter
    def decoder(self, decoder):
        self._decoder = decoder
        # Update combined model
        self._encoder_decoder_model = Model(self._encoder.input, self._decoder(self._encoder.output))

    @encoder.setter
    def encoder(self, encoder):
        self._encoder = encoder
        # Update combined model
        self._encoder_decoder_model = Model(self._encoder.input, self._decoder(self._encoder.output))