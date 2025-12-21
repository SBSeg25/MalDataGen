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
    from typing import Any

    from tensorflow.keras.utils import to_categorical

except ImportError as error:
    print(error)
    sys.exit(-1)


class AlgorithmDenoisingDiffusionTensorflow(tensorflow.keras.Model):
    """
    Implements a diffusion process using UNet architectures with adaptive input handling.

    This model integrates an autoencoder and a diffusion network, enabling both data
    reconstruction and controlled generative modeling through Gaussian diffusion.
    It automatically adapts to any input shape: (x), (x, y), (x, y, z), etc.

    This class supports exponential moving average (EMA) updates for stable training,
    multiple training stages, and customizable hyperparameters to adapt to different tasks.

    References:
        - Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models."
        Advances in Neural Information Processing Systems (NeurIPS).
        Available at: https://arxiv.org/abs/2006.11239

    Example:
        >>> # Example with 1D time series data
        >>> diffusion_1d = AlgorithmDenoisingDiffusionTensorflow(
        ...     output_shape=100,
        ...     first_unet_model=primary_unet,
        ...     second_unet_model=ema_unet,
        ...     gdf_util=gaussian_diffusion,
        ...     optimizer_autoencoder=tf.keras.optimizers.Adam(1e-4),
        ...     optimizer_diffusion=tf.keras.optimizers.Adam(2e-4),
        ...     time_steps=1000,
        ...     ema=0.999,
        ...     margin=0.1,
        ...     input_shape=(100,)
        ... )
        >>> diffusion_1d.fit(data_1d, labels_1d, epochs=50)

        >>> # Example with 2D image data
        >>> diffusion_2d = AlgorithmDenoisingDiffusionTensorflow(
        ...     output_shape=784,
        ...     first_unet_model=unet_2d,
        ...     second_unet_model=ema_unet_2d,
        ...     gdf_util=gaussian_diffusion,
        ...     optimizer_autoencoder=tf.keras.optimizers.Adam(1e-4),
        ...     optimizer_diffusion=tf.keras.optimizers.Adam(2e-4),
        ...     time_steps=1000,
        ...     ema=0.999,
        ...     margin=0.1,
        ...     input_shape=(28, 28)
        ... )
        >>> diffusion_2d.fit(images, labels, epochs=100)
    """

    def __init__(self,
                 output_shape,
                 first_unet_model,
                 second_unet_model,
                 gdf_util,
                 optimizer_autoencoder,
                 optimizer_diffusion,
                 time_steps,
                 ema,
                 margin,
                 train_stage='all',
                 input_shape=None,
                 auto_adapt_shape=True):
        """
        Initializes the DiffusionModel with provided sub-models, optimizers, and hyperparameters.

        Args:
            output_shape: Expected output dimension for generated data
            first_unet_model: Primary UNet model for diffusion-based generation
            second_unet_model: Secondary UNet model for maintaining EMA-based weight updates
            gdf_util: Utility object responsible for Gaussian diffusion operations
            optimizer_autoencoder: Optimizer handling the training of the encoder-decoder network
            optimizer_diffusion: Optimizer applied to the diffusion process
            time_steps: Number of discrete time steps for the diffusion process
            ema: Exponential moving average decay factor
            margin: Margin value used in loss calculations or regularization
            train_stage: Current training stage ('all', 'diffusion', etc.)
            input_shape: Expected input shape (without batch dimension)
            auto_adapt_shape: Whether to automatically adapt to input data shape
        """
        super().__init__()

        self._ema = ema
        self._margin = margin
        self._gdf_util = gdf_util
        self._time_steps = time_steps
        self._train_stage = train_stage
        self._network = first_unet_model
        self._output_shape = output_shape
        self._original_shape = output_shape
        self._second_unet_model = second_unet_model
        self._optimizer_diffusion = optimizer_diffusion
        self._optimizer_autoencoder = optimizer_autoencoder
        self._total_loss_tracker = tensorflow.keras.metrics.Mean(name='total_loss')

        # Shape adaptation
        self._input_shape = input_shape
        self._auto_adapt_shape = auto_adapt_shape
        self._inferred_shape = None

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
            tuple: (batch_x, batch_y) formatted for diffusion training.
        """
        if isinstance(batch, (tuple, list)):
            if len(batch) == 1:
                # Single input - need labels for diffusion
                raise ValueError("Diffusion model requires both data and labels")
            elif len(batch) == 2:
                # (data, labels)
                batch_x = batch[0]
                batch_y = batch[1]
            else:
                # Multiple inputs
                batch_x = batch[0]
                batch_y = batch[1]
        else:
            # Single tensor - need labels
            raise ValueError("Diffusion model requires both data and labels")

        return batch_x, batch_y

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

    def set_stage_training(self, training_stage):
        """
        Sets the current training stage.

        Args:
            training_stage (str): New training stage ('all', 'diffusion', etc.).
        """
        self._train_stage = training_stage

    @tensorflow.function
    def train_step(self, data):
        """
        Performs a single training step with automatic batch handling.

        Args:
            data: Input data batch.

        Returns:
            dict: A dictionary with the computed loss for diffusion.
        """
        raw_data, label = self._prepare_batch(data)

        loss_diffusion = self.train_diffusion_model(raw_data, label)
        return {"Diffusion_loss": loss_diffusion if loss_diffusion is not None else 0}

    def fit(self, x=None, y=None, batch_size=32, epochs=1, verbose=1,
            callbacks=None, validation_data=None, shuffle=True,
            initial_epoch=0, steps_per_epoch=None, validation_steps=None,
            validation_freq=1, optimizer=None, learning_rate=0.001, **kwargs):
        """
        Train the model with automatic shape adaptation.

        Args:
            x: Input data (any shape) or tf.data.Dataset.
            y: Target data (labels).
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
            self._optimizer_diffusion = optimizer
        elif not hasattr(self, '_optimizer_diffusion') or self._optimizer_diffusion is None:
            # Create default optimizer if none exists
            self._optimizer_diffusion = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate)

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
                    x_data, target_or_labels = x
                else:
                    x_data = x[0]
                    target_or_labels = None
            else:
                x_data = x
                target_or_labels = y

            if target_or_labels is None:
                raise ValueError("Labels (y) must be provided for diffusion training")

            # Convert to tensors
            if isinstance(x_data, numpy.ndarray):
                x_data = tensorflow.constant(x_data, dtype=tensorflow.float32)

            if isinstance(target_or_labels, numpy.ndarray):
                target_or_labels = tensorflow.constant(target_or_labels, dtype=tensorflow.float32)

            # Create dataset
            train_dataset = tensorflow.data.Dataset.from_tensor_slices((x_data, target_or_labels))

            if shuffle:
                buffer_size = len(x_data) if hasattr(x_data, '__len__') else 10000
                train_dataset = train_dataset.shuffle(buffer_size=buffer_size)

            train_dataset = train_dataset.batch(batch_size)

        # Calculate steps per epoch if not provided
        if steps_per_epoch is None:
            try:
                steps_per_epoch = len(train_dataset)
            except:
                steps_per_epoch = 100  # Default fallback

        # History to store metrics
        history = {'Diffusion_loss': []}

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
                current_loss = float(metrics['Diffusion_loss'])

                # Update total loss tracker
                self._total_loss_tracker.update_state(current_loss)

                # Simple progress bar
                if verbose == 1:
                    progress = int(50 * step / steps_per_epoch)
                    bar = '=' * progress + '>' + '.' * (50 - progress - 1)
                    print(f'\r[{bar}] {step}/{steps_per_epoch} - Diffusion_loss: {current_loss:.4f}',
                          end='', flush=True)

                if step >= steps_per_epoch:
                    break

            # Store epoch loss
            epoch_loss = float(self._total_loss_tracker.result())
            history['Diffusion_loss'].append(epoch_loss)

            if verbose == 1:
                print(f' - Diffusion_loss: {epoch_loss:.4f}')
            elif verbose == 2:
                print(f'Epoch {epoch + 1}/{epochs} - Diffusion_loss: {epoch_loss:.4f}')

            # Validation
            if validation_data is not None and (epoch + 1) % validation_freq == 0:
                val_loss = self._evaluate_validation(validation_data, validation_steps, batch_size)
                if 'val_Diffusion_loss' not in history:
                    history['val_Diffusion_loss'] = []
                history['val_Diffusion_loss'].append(val_loss)

                if verbose >= 1:
                    print(f' - val_Diffusion_loss: {val_loss:.4f}')

            # Callbacks
            if callbacks is not None:
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, {'Diffusion_loss': epoch_loss})

        # Return history object
        class History:
            def __init__(self, history_dict):
                self.history = history_dict

        return History(history)

    def _evaluate_validation(self, validation_data, validation_steps=None, batch_size=32):
        """
        Evaluate the model on validation data with automatic shape handling.

        Args:
            validation_data: Validation dataset.
            validation_steps: Number of validation steps.
            batch_size: Batch size for validation.

        Returns:
            Average validation loss.
        """
        val_losses = []
        step = 0

        # Prepare validation dataset
        if isinstance(validation_data, tensorflow.data.Dataset):
            val_dataset = validation_data
        elif isinstance(validation_data, tuple) and len(validation_data) == 2:
            x_val, y_val = validation_data
            if isinstance(x_val, numpy.ndarray):
                x_val = tensorflow.constant(x_val, dtype=tensorflow.float32)
            if isinstance(y_val, numpy.ndarray):
                y_val = tensorflow.constant(y_val, dtype=tensorflow.float32)
            val_dataset = tensorflow.data.Dataset.from_tensor_slices((x_val, y_val))
            val_dataset = val_dataset.batch(batch_size)
        else:
            raise ValueError("validation_data must be a tf.data.Dataset or tuple of (x_val, y_val)")

        for batch_data in val_dataset:
            raw_data, labels = self._prepare_batch(batch_data)

            # Prepare data
            embedding_data_expanded = self._padding_input_tensor(raw_data)
            embedding_data_expanded = tensorflow.cast(embedding_data_expanded, tensorflow.float32)
            batch_size_val = tensorflow.shape(raw_data)[0]

            # Sample random time steps
            random_time_steps = tensorflow.random.uniform(
                minval=0,
                maxval=self._time_steps,
                shape=(batch_size_val,),
                dtype=tensorflow.int32
            )

            # Sample random noise
            random_noise = tensorflow.random.normal(
                shape=tensorflow.shape(embedding_data_expanded),
                dtype=embedding_data_expanded.dtype
            )

            # Apply forward diffusion process
            embedding_with_noise = self._gdf_util.q_sample(
                embedding_data_expanded,
                random_time_steps,
                random_noise
            )

            # Predict noise using the network (without training)
            predicted_noise = self._network(
                [embedding_with_noise, random_time_steps, labels],
                training=False
            )

            # Compute validation loss
            loss = self.loss(random_noise, tensorflow.squeeze(predicted_noise, axis=-1))
            val_losses.append(float(loss))

            step += 1
            if validation_steps is not None and step >= validation_steps:
                break

        return numpy.mean(val_losses) if val_losses else 0.0

    def train_diffusion_model(self, data, ground_truth):
        """
        Performs a single training step for the diffusion model.

        This method applies the forward diffusion process (adding noise to the data),
        predicts the noise using the model, computes the loss, and updates the model weights.

        Args:
            data (tf.Tensor): Input data embeddings (e.g., image or text embeddings).
            ground_truth (tf.Tensor): Corresponding class labels or conditioning embeddings.

        Returns:
            tf.Tensor: The computed loss for this training step.
        """

        # Labels (conditioning information) and input data embeddings
        embedding_label = ground_truth
        embedding_data_expanded = data

        # Batch size of the current data batch
        batch_size = tensorflow.shape(data)[0]

        embedding_data_expanded = self._padding_input_tensor(embedding_data_expanded)
        embedding_data_expanded = tensorflow.cast(embedding_data_expanded, tensorflow.float32)

        static_shape = embedding_data_expanded.shape

        if static_shape[-2] is not None:
            self._output_shape = static_shape[-2]
        else:
            self._output_shape = tensorflow.shape(embedding_data_expanded)[-2]

        # Sample random time steps for each sample in the batch
        random_time_steps = tensorflow.random.uniform(minval=0,
                                                      maxval=self._time_steps,
                                                      shape=(batch_size,),
                                                      dtype=tensorflow.int32)
        loss_diffusion = 0

        # Track gradients for the diffusion model's weights
        with tensorflow.GradientTape() as tape:

            # Sample random noise to add to the data
            random_noise = tensorflow.random.normal(shape=tensorflow.shape(embedding_data_expanded),
                                                    dtype=embedding_data_expanded.dtype)

            # Apply forward diffusion process
            embedding_with_noise = self._gdf_util.q_sample(embedding_data_expanded,
                                                           random_time_steps,
                                                           random_noise)

            # Predict noise using the diffusion model
            predicted_noise = self._network([embedding_with_noise, random_time_steps, embedding_label], training=True)

            # Compute the loss
            loss_diffusion = self.loss(random_noise, tensorflow.squeeze(predicted_noise, axis=-1))

        # Compute gradients
        gradients = tape.gradient(loss_diffusion, self._network.trainable_weights)

        # Apply gradients
        self._optimizer_diffusion.apply_gradients(zip(gradients, self._network.trainable_weights))

        return loss_diffusion

    def update_ema_weights(self):
        for w, ew in zip(self._network.trainable_weights,
                         self._second_unet_model.trainable_weights):
            ew.assign_add((1.0 - self._ema) * (w - ew))

    def generate_data(self, labels, batch_size):
        """
        Generates synthetic data by reversing the diffusion process.

        Args:
            labels (tf.Tensor): Class labels used to condition the generated data.
            batch_size (int): Number of data samples to generate in a single batch.

        Returns:
            numpy.ndarray: Generated synthetic data samples.
        """

        # Start with random noise
        synthetic_data = tensorflow.random.normal(
            shape=(labels.shape[0], self._output_shape, 1),
            dtype=tensorflow.float32
        )

        # Reshape labels
        labels_vector = tensorflow.expand_dims(labels, axis=-1)

        # Reverse diffusion process
        for time_step in reversed(range(0, self._time_steps)):
            array_time = tensorflow.cast(tensorflow.fill([labels_vector.shape[0]], time_step),
                                         dtype=tensorflow.int32)

            predicted_noise = self._network.predict([synthetic_data, array_time, labels_vector],
                                                    verbose=0, batch_size=32)

            synthetic_data = self._gdf_util.p_sample(predicted_noise[0], synthetic_data, array_time,
                                                     clip_denoised=True)

        # Crop to original size
        generated_data = self._crop_tensor_to_original_size(synthetic_data, self._original_shape)

        return generated_data

    @staticmethod
    def _crop_tensor_to_original_size(tensor: numpy.ndarray, original_size: int) -> numpy.ndarray:
        """
        Crops the input tensor along the second dimension to match the original size.

        Args:
            tensor (np.ndarray): A 3D NumPy array of shape (X, Y, Z).
            original_size (int): The desired size for the second dimension (Y).

        Returns:
            np.ndarray: A cropped 3D tensor with shape (X, original_size, Z).
        """

        if tensor.ndim != 3:
            raise ValueError(f"Expected 3D tensor (X, Y, Z), got shape: {tensor.shape}")

        current_size = tensor.shape[1]

        if current_size <= original_size:
            return tensor

        return tensor[:, :original_size, :]

    def _padding_input_tensor(self, input_tensor):
        """
        Pads the input tensor along the feature dimension to match the expected input shape.

        Args:
            input_tensor (tensorflow.Tensor): Tensor of shape (batch_size, seq_len, channels).

        Returns:
            tensorflow.Tensor: Padded tensor.
        """
        input_tensor = tensorflow.cast(input_tensor, tensorflow.float32)

        input_shape_dynamic = tensorflow.shape(input_tensor)
        input_rank = tensorflow.rank(input_tensor)

        target_dimension = self._network.input_shape[0][-2]

        static_channels = input_tensor.shape[-1]

        current_dimension = input_shape_dynamic[-2]

        padding_needed = tensorflow.maximum(0, target_dimension - current_dimension)

        tensor_paddings = tensorflow.concat([
            tensorflow.zeros([input_rank - 2, 2], dtype=tensorflow.int32),
            [[0, padding_needed]],
            tensorflow.zeros([1, 2], dtype=tensorflow.int32)
        ], axis=0)

        padded_tensor = tensorflow.cond(
            tensorflow.equal(padding_needed, 0),
            lambda: input_tensor,
            lambda: tensorflow.pad(input_tensor, paddings=tensor_paddings, mode="CONSTANT", constant_values=0)
        )

        padded_tensor = tensorflow.ensure_shape(padded_tensor, [None, target_dimension, static_channels])

        return padded_tensor

    def get_samples(self, number_samples_per_class):
        """
        Generates synthetic data samples for each class.

        Args:
            number_samples_per_class (dict): Dictionary with 'classes' and 'number_classes' keys.

        Returns:
            dict: Generated samples keyed by class label.
        """
        generated_data = {}

        for label_class, number_instances in number_samples_per_class["classes"].items():
            label_samples_generated = to_categorical([label_class] * number_instances,
                                                     num_classes=number_samples_per_class["number_classes"])

            generated_samples = self.generate_data(numpy.array(label_samples_generated, dtype=numpy.float32),
                                                   batch_size=64)

            generated_samples = numpy.squeeze(generated_samples, axis=-1)

            generated_data[label_class] = generated_samples

        return generated_data

    def save_model(self, directory, file_name):
        """
        Save the models with shape information.

        Args:
            directory (str): Directory where models will be saved.
            file_name (str): Base file name for saving models.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        first_unet_file_name = os.path.join(directory, f"{file_name}_first_unet")
        second_unet_file_name = os.path.join(directory, f"{file_name}_second_unet")

        # Save UNet models
        self._save_model_to_json(self._network, f"{first_unet_file_name}.json")
        self._network.save_weights(f"{first_unet_file_name}.weights.h5")

        self._save_model_to_json(self._second_unet_model, f"{second_unet_file_name}.json")
        self._second_unet_model.save_weights(f"{second_unet_file_name}.weights.h5")

        # Save shape information
        shape_info = {
            'input_shape': self._input_shape,
            'inferred_shape': self._inferred_shape,
            'output_shape': self._output_shape,
            'original_shape': self._original_shape,
            'time_steps': self._time_steps,
            'ema': self._ema,
            'margin': self._margin
        }
        shape_file = os.path.join(directory, f"{file_name}_shape_info.json")
        with open(shape_file, 'w') as f:
            json.dump(shape_info, f)

    def load_models(self, directory, file_name):
        """
        Load the models with shape information.

        Args:
            directory (str): Directory where models are stored.
            file_name (str): Base file name for loading models.
        """
        first_unet_file_name = os.path.join(directory, f"{file_name}_first_unet")
        second_unet_file_name = os.path.join(directory, f"{file_name}_second_unet")

        # Load weights
        self._network.load_weights(f"{first_unet_file_name}.weights.h5")
        self._second_unet_model.load_weights(f"{second_unet_file_name}.weights.h5")

        # Load shape information if available
        shape_file = os.path.join(directory, f"{file_name}_shape_info.json")
        if os.path.exists(shape_file):
            with open(shape_file, 'r') as f:
                shape_info = json.load(f)
                self._input_shape = tuple(shape_info.get('input_shape', ()))
                self._inferred_shape = tuple(shape_info.get('inferred_shape', ()))
                self._output_shape = shape_info.get('output_shape', self._output_shape)
                self._original_shape = shape_info.get('original_shape', self._original_shape)
                self._time_steps = shape_info.get('time_steps', self._time_steps)
                self._ema = shape_info.get('ema', self._ema)
                self._margin = shape_info.get('margin', self._margin)

    @staticmethod
    def _save_model_to_json(model, file_path):
        """
        Save model architecture to a JSON file.

        Args:
            model (tf.keras.Model): Model to save.
            file_path (str): Path to the JSON file.
        """
        try:
            with open(file_path, "w") as json_file:
                json.dump(model.to_json(), json_file)
            print(f"Model successfully saved to {file_path}.")
        except Exception as e:
            error_message = f"Error occurred while saving model: {str(e)}"
            with open(file_path, "w") as error_file:
                error_file.write(error_message)
            print(f"An error occurred and was saved to {file_path}: {error_message}")

    # ========== PROPERTIES ==========

    @property
    def ema(self) -> Any:
        """Get the Exponential Moving Average (EMA) model."""
        return self._ema

    @ema.setter
    def ema(self, value: Any) -> None:
        """Set the Exponential Moving Average (EMA) model."""
        self._ema = value

    @property
    def margin(self) -> float:
        """Get the margin value used in contrastive loss."""
        return self._margin

    @margin.setter
    def margin(self, value: float) -> None:
        """Set the margin value for contrastive loss."""
        if value <= 0:
            raise ValueError("Margin must be positive")
        self._margin = value

    @property
    def gdf_util(self) -> Any:
        """Get the Gradient Descent Filter utility."""
        return self._gdf_util

    @gdf_util.setter
    def gdf_util(self, value: Any) -> None:
        """Set the Gradient Descent Filter utility."""
        self._gdf_util = value

    @property
    def time_steps(self) -> int:
        """Get the number of diffusion time steps."""
        return self._time_steps

    @time_steps.setter
    def time_steps(self, value: int) -> None:
        """Set the number of diffusion time steps."""
        if value <= 0:
            raise ValueError("Time steps must be positive")
        self._time_steps = value

    @property
    def train_stage(self) -> str:
        """Get the current training stage."""
        return self._train_stage

    @train_stage.setter
    def train_stage(self, value: str) -> None:
        """Set the current training stage."""
        self._train_stage = value

    @property
    def network(self) -> Any:
        """Get the primary U-Net model."""
        return self._network

    @network.setter
    def network(self, value: Any) -> None:
        """Set the primary U-Net model."""
        self._network = value

    @property
    def second_unet_model(self) -> Any:
        """Get the secondary U-Net model."""
        return self._second_unet_model

    @second_unet_model.setter
    def second_unet_model(self, value: Any) -> None:
        """Set the secondary U-Net model."""
        self._second_unet_model = value

    @property
    def optimizer_diffusion(self) -> Any:
        """Get the diffusion model optimizer."""
        return self._optimizer_diffusion

    @optimizer_diffusion.setter
    def optimizer_diffusion(self, value: Any) -> None:
        """Set the diffusion model optimizer."""
        self._optimizer_diffusion = value

    @property
    def optimizer_autoencoder(self) -> Any:
        """Get the autoencoder optimizer."""
        return self._optimizer_autoencoder

    @optimizer_autoencoder.setter
    def optimizer_autoencoder(self, value: Any) -> None:
        """Set the autoencoder optimizer."""
        self._optimizer_autoencoder = value

    @property
    def input_shape(self):
        """Get the current input shape."""
        return self.get_input_shape()

    @property
    def output_shape(self):
        """Get the output shape."""
        return self._output_shape

    @output_shape.setter
    def output_shape(self, value):
        """Set the output shape."""
        self._output_shape = value
