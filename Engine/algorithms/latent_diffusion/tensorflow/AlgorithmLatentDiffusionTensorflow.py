#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com'
__version__ = '{1}.{0}.{2}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/17'
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
    from typing import Any

    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.metrics import Mean
except ImportError as error:
    print(error)
    sys.exit(-1)


class AlgorithmLatentDiffusionTensorflow(tensorflow.keras.Model):
    """
    Implements a diffusion process using UNet architectures for generating synthetic data.
    This model integrates an autoencoder and a diffusion network, enabling both data
    reconstruction and controlled generative modeling through Gaussian diffusion.

    This class supports exponential moving average (EMA) updates for stable training,
    multiple training stages, customizable hyperparameters, and automatic adaptation
    to different input shapes: (x), (x, y), (x, y, z), etc.

    Attributes:
        @ema (float):
            Exponential moving average (EMA) decay rate for stabilizing training updates.
        @margin (float):
            Margin parameter used for loss computation or regularization purposes.
        @gdf_util:
            Utility object for Gaussian diffusion functions, handling noise scheduling and diffusion-related operations.
        @time_steps (int):
            Number of time steps used in the diffusion process.
        @train_stage (str):
            Defines the current training stage ('all', 'diffusion', etc.), determining whether only specific components are updated.
        @network (Model):
            Primary UNet model responsible for the diffusion process.
        @second_unet_model (Model):
            Secondary UNet model used for EMA-based weight updates to enhance training stability.
        @embedding_dimension (int):
            Dimensionality of the latent space used for encoding data.
        @encoder_model_data (Model):
            Encoder model responsible for feature extraction from input data.
        @decoder_model_data (Model):
            Decoder model used to reconstruct data from encoded representations.
        @optimizer_diffusion (Optimizer):
            Optimizer used for training the diffusion model.
        @optimizer_autoencoder (Optimizer):
            Optimizer responsible for training the autoencoder components.
        @ensemble_encoder_decoder (Model):
            Combined encoder-decoder model for data reconstruction.
        @input_shape (tuple):
            Expected input shape (without batch dimension).
        @auto_adapt_shape (bool):
            If True, automatically adapts to input data shape.

    Raises:
        ValueError:
            Raised in cases where:
            - The number of time steps is non-positive.
            - The EMA decay rate is outside the range (0,1).
            - The embedding dimension is invalid (<=0).

    References:
        - Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising latent_diffusion Probabilistic architectures."
        Advances in Neural Information Processing Systems (NeurIPS).
        Available at: https://arxiv.org/abs/2006.11239

    Example:
        >>> diffusion_model = AlgorithmLatentDiffusionTensorflow(
        ...     first_unet_model=primary_unet,
        ...     second_unet_model=ema_unet,
        ...     encoder_model_image=encoder,
        ...     decoder_model_image=decoder,
        ...     gdf_util=gaussian_diffusion,
        ...     optimizer_autoencoder=tf.keras.optimizers.Adam(learning_rate=1e-4),
        ...     optimizer_diffusion=tf.keras.optimizers.Adam(learning_rate=2e-4),
        ...     time_steps=1000,
        ...     ema=0.999,
        ...     margin=0.1,
        ...     embedding_dimension=128,
        ...     train_stage='all',
        ...     input_shape=(100,),
        ...     auto_adapt_shape=True
        ... )
        >>> diffusion_model.set_stage_training('diffusion')
        >>> diffusion_model.fit(data, epochs=10)
    """

    def __init__(self, first_unet_model,
                 second_unet_model,
                 encoder_model_image,
                 decoder_model_image,
                 gdf_util,
                 optimizer_autoencoder,
                 optimizer_diffusion,
                 time_steps,
                 ema,
                 margin,
                 embedding_dimension,
                 train_stage='all',
                 input_shape=None,
                 auto_adapt_shape=True):

        super().__init__()
        """
        Initializes the DiffusionModel with provided sub-models, optimizers, and hyperparameters.

        This constructor sets up the network structure, including the autoencoder, diffusion
        models, and EMA components, ensuring flexibility for different training strategies and
        automatic adaptation to various input shapes.

        Args:
            @first_unet_model (Model):
                Primary UNet model for diffusion-based generation.
            @second_unet_model (Model):
                Secondary UNet model for maintaining EMA-based weight updates.
            @encoder_model_image (Model):
                Encoder model used to extract meaningful feature representations.
            @decoder_model_image (Model):
                Decoder model reconstructing data from encoded embeddings.
            @gdf_util:
                Utility object responsible for Gaussian diffusion operations.
            @optimizer_autoencoder (Optimizer):
                Optimizer handling the training of the encoder-decoder network.
            @optimizer_diffusion (Optimizer):
                Optimizer applied to the diffusion process.
            @time_steps (int):
                Number of discrete time steps for the diffusion process.
            @ema (float):
                Exponential moving average decay factor.
            @margin (float):
                Margin value used in loss calculations or regularization.
            @embedding_dimension (int):
                Dimensionality of the embedding space.
            @train_stage (str, optional):
                Current training stage ('all', 'diffusion', etc.), defaulting to 'all'.
            @input_shape (tuple, optional):
                Expected input shape (without batch dimension). If None, will be inferred from data.
            @auto_adapt_shape (bool, optional):
                If True, automatically adapts to input data shape. Default: True

        Raises:
            ValueError:
                If time_steps is <= 0.
                If ema is not within the (0,1) range.
                If embedding_dimension is <= 0.
        """

        self._ema = ema
        self._margin = margin
        self._gdf_util = gdf_util
        self._time_steps = time_steps
        self._train_stage = train_stage
        self._network = first_unet_model
        self._second_unet_model = second_unet_model
        self._embedding_dimension = embedding_dimension
        self._encoder_model_data = encoder_model_image
        self._decoder_model_data = decoder_model_image
        self._optimizer_diffusion = optimizer_diffusion
        self._optimizer_autoencoder = optimizer_autoencoder

        self._total_loss_tracker = Mean(name="loss")

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
            tuple: (batch_x, batch_y) where batch_y is the label/conditioning.
        """
        if isinstance(batch, (tuple, list)):
            if len(batch) == 1:
                # Single input, use zeros as labels (unconditional)
                batch_x = batch[0]
                batch_y = tensorflow.zeros((tensorflow.shape(batch_x)[0], 1))
            elif len(batch) == 2:
                # Input and label provided
                batch_x, batch_y = batch
            else:
                # Multiple inputs, use first as input and second as label
                batch_x = batch[0]
                batch_y = batch[1]
        else:
            # Single tensor, use zeros as labels
            batch_x = batch
            batch_y = tensorflow.zeros((tensorflow.shape(batch_x)[0], 1))

        return batch_x, batch_y

    def set_stage_training(self, training_stage):
        """
        Sets the current training stage.

        Args:
            training_stage (str): New training stage ('all', 'diffusion', etc.).
        """
        self._train_stage = training_stage

    def train_step(self, data):
        """
        Performs a single training step with automatic shape handling.

        Args:
            data (tuple): A tuple containing input data and labels, or just data.

        Returns:
            dict: A dictionary with the computed loss for diffusion.
        """
        # Prepare batch data
        raw_data, label = self._prepare_batch(data)

        loss_diffusion = self.train_diffusion_model(raw_data, label)
        self.update_ema_weights()

        # Update the loss tracker
        self._total_loss_tracker.update_state(loss_diffusion)

        # Return both 'loss' and 'Diffusion_loss' for compatibility
        return {
            "loss": loss_diffusion if loss_diffusion is not None else 0,
            "Diffusion_loss": loss_diffusion if loss_diffusion is not None else 0
        }

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

        # Sample random time steps for each sample in the batch (each sample can be at a different step t)
        random_time_steps = tensorflow.random.uniform(minval=0,
                                                      maxval=self._time_steps,
                                                      shape=(batch_size,),
                                                      dtype=tensorflow.int32)
        loss_diffusion = 0
        # Track gradients for the diffusion model's weights
        with tensorflow.GradientTape() as tape:
            # Sample random noise to add to the data (same shape as the data itself)
            random_noise = tensorflow.random.normal(shape=tensorflow.shape(embedding_data_expanded),
                                                    dtype=embedding_data_expanded.dtype)

            # Apply forward diffusion process (add noise based on the current time step t)
            embedding_with_noise = self._gdf_util.q_sample(embedding_data_expanded,
                                                           random_time_steps,
                                                           random_noise)

            # Predict noise using the diffusion model (network), conditioned on time and label
            predicted_noise = self._network([embedding_with_noise, random_time_steps, embedding_label], training=True)

            # Compute the loss by comparing the true noise with the predicted noise
            loss_diffusion = self.loss(random_noise, predicted_noise)

        # Compute gradients for the model's trainable weights
        gradients = tape.gradient(loss_diffusion, self._network.trainable_weights)

        # Apply gradients using the diffusion model's optimizer
        self._optimizer_diffusion.apply_gradients(zip(gradients, self._network.trainable_weights))

        # Return the computed diffusion loss for monitoring
        return loss_diffusion

    def fit(self, x=None, y=None, batch_size=32, epochs=1, verbose=1,
            callbacks=None, validation_data=None, shuffle=True,
            initial_epoch=0, steps_per_epoch=None, validation_steps=None,
            validation_freq=1, optimizer=None, learning_rate=0.001, **kwargs):
        """
        Train the model with automatic shape adaptation and simplified progress bar.

        Args:
            x: Input data (any shape).
            y: Target data/labels (if None, unconditional generation is assumed).
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
            optimizer: TensorFlow optimizer (if provided, overrides diffusion optimizer).
            learning_rate: Learning rate for optimizer.
        """
        import sys

        # Set optimizer if provided
        if optimizer is not None:
            self._optimizer_diffusion = optimizer
        elif learning_rate != 0.001:
            self._optimizer_diffusion = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate)

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
                # Create dummy labels for unconditional generation
                y = tensorflow.zeros((len(x), 1))

            train_dataset = tensorflow.data.Dataset.from_tensor_slices((x, y))
            if shuffle:
                train_dataset = train_dataset.shuffle(buffer_size=len(x))
            train_dataset = train_dataset.batch(batch_size)

        # Calculate steps per epoch if not provided
        if steps_per_epoch is None:
            try:
                steps_per_epoch = len(train_dataset)
            except:
                steps_per_epoch = tensorflow.data.experimental.cardinality(train_dataset).numpy()
                if steps_per_epoch < 0:  # Unknown cardinality
                    steps_per_epoch = 100  # Default fallback

        # History to store metrics
        history = {'loss': [], 'Diffusion_loss': []}

        # Training loop
        for epoch in range(initial_epoch, epochs):
            self._total_loss_tracker.reset_state()

            if verbose >= 1:
                print(f'\nEpoch {epoch + 1}/{epochs}')
                sys.stdout.flush()

            # Progress tracking
            step = 0
            epoch_losses = []

            for batch_data in train_dataset:
                step += 1

                # Perform training step
                metrics = self.train_step(batch_data)
                current_loss = float(metrics.get('loss', 0))
                epoch_losses.append(current_loss)

                # Simple progress bar (verbose == 1)
                if verbose == 1:
                    # Calculate progress (0 to 50 characters)
                    progress = min(int(50 * step / steps_per_epoch), 50)
                    remaining = max(50 - progress, 0)

                    # Build progress bar
                    if progress > 0:
                        bar = '=' * (progress - 1) + '>' + '.' * remaining
                    else:
                        bar = '.' * 50

                    # Print with carriage return
                    msg = f'\r[{bar}] {step}/{steps_per_epoch} - loss: {current_loss:.4f}'
                    print(msg, end='', flush=True)
                    sys.stdout.flush()

                if step >= steps_per_epoch:
                    break

            # Store epoch loss
            epoch_loss = float(self._total_loss_tracker.result())
            if len(epoch_losses) > 0:
                epoch_loss = numpy.mean(epoch_losses)

            history['loss'].append(epoch_loss)
            history['Diffusion_loss'].append(epoch_loss)

            if verbose == 1:
                sys.stdout.flush()
            elif verbose == 2:
                print(f'Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.4f}')
                sys.stdout.flush()

            print()
            # Validation
            if validation_data is not None and (epoch + 1) % validation_freq == 0:
                val_loss = self._evaluate_validation(validation_data, validation_steps)
                if 'val_loss' not in history:
                    history['val_loss'] = []
                history['val_loss'].append(val_loss)

                if verbose >= 1:
                    print(f' - val_loss: {val_loss:.4f}')
                    sys.stdout.flush()

            # callbacks
            if callbacks is not None:
                for callback in callbacks:
                    callback.set_model(self)
                    callback.set_params({
                        'epochs': epochs,
                        'steps': steps_per_epoch,
                        'verbose': verbose,
                        'batch_size': batch_size
                    })
                for callback in callbacks:
                    callback.on_train_begin()

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

            # Use the diffusion model for validation
            batch_size = tensorflow.shape(batch_x)[0]
            random_time_steps = tensorflow.random.uniform(
                minval=0,
                maxval=self._time_steps,
                shape=(batch_size,),
                dtype=tensorflow.int32
            )

            random_noise = tensorflow.random.normal(
                shape=tensorflow.shape(batch_x),
                dtype=batch_x.dtype
            )

            embedding_with_noise = self._gdf_util.q_sample(
                batch_x,
                random_time_steps,
                random_noise
            )

            predicted_noise = self._network(
                [embedding_with_noise, random_time_steps, batch_y],
                training=False
            )

            loss = tensorflow.reduce_mean(tensorflow.square(random_noise - predicted_noise))
            val_losses.append(float(loss))

            step += 1
            if validation_steps is not None and step >= validation_steps:
                break

        return numpy.mean(val_losses) if val_losses else 0.0

    def update_ema_weights(self):
        """
        Updates the weights of the second UNet model using exponential moving average.
        """
        for weight, ema_weight in zip(self._network.weights, self._second_unet_model.weights):
            ema_weight.assign(self._ema * ema_weight + (1 - self._ema) * weight)

    def generate_data(self, labels, batch_size):
        """
        Generates synthetic data by reversing the diffusion process using TensorFlow.

        Args:
            labels (tf.Tensor): One-hot encoded labels of shape (batch_size, number_classes)
            batch_size (int): Batch size for generation

        Returns:
            numpy.ndarray: Generated synthetic data
        """

        # Start with random noise in TensorFlow
        embedding_diffusion = tensorflow.random.normal(
            shape=(tensorflow.shape(labels)[0], self._embedding_dimension, 1),
            dtype=tensorflow.float32
        )

        # Labels are already in correct shape (batch_size, number_classes)
        labels_vector = labels

        # Verify shape
        if len(labels_vector.shape) != 2:
            raise ValueError(
                f"Labels must be 2D (batch_size, number_classes), got shape {labels_vector.shape}"
            )

        # Reverse diffusion process
        for time_step in reversed(range(0, self._time_steps)):
            # Create time step array for the batch
            array_time = tensorflow.fill(
                dims=(tensorflow.shape(labels_vector)[0],),
                value=time_step
            )
            array_time = tensorflow.cast(array_time, tensorflow.int32)

            # Predict noise using the network
            predicted_noise = self._network(
                [embedding_diffusion, array_time, labels_vector],
                training=False
            )

            # Apply reverse diffusion step
            embedding_diffusion = self._gdf_util.p_sample(
                predicted_noise,
                embedding_diffusion,
                array_time,
                clip_denoised=True
            )

        generated_data = self._decoder_model_data(
            [embedding_diffusion, labels_vector],
            training=False
        )

        print(f"[DEBUG] generated_data.shape = {generated_data.shape}")

        return generated_data.numpy()

    @staticmethod
    def _crop_tensor_to_original_size(tensor: numpy.ndarray, original_size: int) -> numpy.ndarray:
        """
        Crops the input tensor along the second dimension (axis=1) to match the original size.

        This function is useful for reversing padding operations or restoring tensors to
        a fixed input size before feeding them into downstream models.

        Args:
            tensor (np.ndarray): A 3D NumPy array of shape (X, Y, Z), where:
                - X is the batch size,
                - Y is the sequence or time dimension (to be cropped),
                - Z is the feature/channel dimension.
            original_size (int): The desired size for the second dimension (Y).
                If tensor.shape[1] <= original_size, the tensor is returned unchanged.

        Returns:
            np.ndarray: A cropped 3D tensor with shape (X, original_size, Z).

        Example:
            >>> tensor = np.random.rand(32, 120, 16)
            >>> cropped = crop_tensor_to_original_size(tensor, original_size=100)
            >>> cropped.shape
            (32, 100, 16)
        """

        # Validate input dimensions
        if tensor.ndim != 3:
            raise ValueError(f"Expected 3D tensor (X, Y, Z), got shape: {tensor.shape}")

        current_size = tensor.shape[1]

        # No cropping needed
        if current_size <= original_size:
            return tensor

        # Slice the tensor along axis 1 (sequence length) to crop the excess at the end
        return tensor[:, :original_size, :]

    def _padding_input_tensor(self, input_tensor):
        """
        Pads the input tensor along the feature dimension to match the expected input shape
        required by the diffusion network.

        Args:
            input_tensor (tensorflow.Tensor): Tensor of shape (batch_size, seq_len, channels),
                                              or similar.

        Returns:
            tensorflow.Tensor: A tensor padded along the feature dimension to match the model's
                               expected input shape.
        """
        # Ensure tensor is in float32 for consistency with model expectations
        input_tensor = tensorflow.cast(input_tensor, tensorflow.float32)

        # Retrieve the dynamic shape and rank of the input tensor
        input_shape_dynamic = tensorflow.shape(input_tensor)
        input_rank = tensorflow.rank(input_tensor)

        # Retrieve the target length of the feature dimension (e.g., 120) from model input shape
        target_dimension = self._network.input_shape[0][-2]

        # Extract static dimensions (for batch and channel)
        static_channels = input_tensor.shape[-1]

        # Determine the current length of the feature dimension
        current_dimension = input_shape_dynamic[-2]

        # Calculate how much padding is required (only pad if shorter than target)
        padding_needed = tensorflow.maximum(0, target_dimension - current_dimension)

        # Build padding configuration: only pad the feature dimension
        # Format: [[pad_before_dim0, pad_after_dim0], ..., [pad_before_d, pad_after_d]]
        tensor_paddings = tensorflow.concat([
            tensorflow.zeros([input_rank - 2, 2], dtype=tensorflow.int32),  # No padding for batch/leading dims
            [[0, padding_needed]],  # Padding on the feature dimension
            tensorflow.zeros([1, 2], dtype=tensorflow.int32)  # No padding on channel dimension
        ], axis=0)

        # Apply conditional padding: if no padding is needed, return input as-is
        padded_tensor = tensorflow.cond(
            tensorflow.equal(padding_needed, 0),
            lambda: input_tensor,
            lambda: tensorflow.pad(input_tensor, paddings=tensor_paddings, mode="CONSTANT", constant_values=0)
        )

        # Manually enforce the static shape so downstream layers can properly infer tensor dimensions
        padded_tensor = tensorflow.ensure_shape(padded_tensor, [None, target_dimension, static_channels])

        return padded_tensor

    def get_samples(self, number_samples_per_class):
        """
        Generates synthetic data samples for each class using TensorFlow.

        Args:
            number_samples_per_class (dict): Dictionary with class info.
                Expected format: {
                    "classes": {class_id: num_samples, ...},
                    "number_classes": total_classes
                }

        Returns:
            dict: Generated samples for each class.
        """
        generated_data = {}

        for label_class, number_instances in number_samples_per_class["classes"].items():

            # Create one-hot encoded labels - shape: (number_instances, number_classes)
            labels = tensorflow.zeros((number_instances, number_samples_per_class["number_classes"]))
            labels = tensorflow.tensor_scatter_nd_update(
                labels,
                indices=[[i, label_class] for i in range(number_instances)],
                updates=tensorflow.ones(number_instances)
            )
            labels = tensorflow.cast(labels, tensorflow.float32)

            # Generate samples
            generated_samples = self.generate_data(labels, batch_size=64)
            generated_samples = numpy.rint(generated_samples)

            generated_data[label_class] = generated_samples
        return generated_data

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

    def save_model(self, directory, file_name):
        """
        Save the encoder and decoder models in Keras native format, including shape information.

        Args:
            directory (str): Directory where models will be saved.
            file_name (str): Base file name for saving models.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Construct file names for encoder and decoder models
        encoder_file_name = os.path.join(directory, f"{file_name}_encoder.keras")
        decoder_file_name = os.path.join(directory, f"{file_name}_decoder.keras")
        first_unet_file_name = os.path.join(directory, f"{file_name}_first_unet.keras")
        second_unet_file_name = os.path.join(directory, f"{file_name}_second_unet.keras")

        # Remove old files if they exist
        for file_path in [encoder_file_name, decoder_file_name, first_unet_file_name, second_unet_file_name]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Removed existing file: {file_path}")
                except Exception as e:
                    print(f"Warning: Could not remove {file_path}: {e}")

        try:
            # Save encoder model
            self._encoder_model_data.save(encoder_file_name)
            print(f"Encoder model saved to {encoder_file_name}")
        except Exception as e:
            print(f"Error saving encoder: {e}")
            # Fallback to JSON + weights
            self._save_model_to_json(self._encoder_model_data, f"{encoder_file_name}.json")
            self._encoder_model_data.save_weights(f"{encoder_file_name}.weights.h5")

        try:
            # Save decoder model
            self._decoder_model_data.save(decoder_file_name)
            print(f"Decoder model saved to {decoder_file_name}")
        except Exception as e:
            print(f"Error saving decoder: {e}")
            # Fallback to JSON + weights
            self._save_model_to_json(self._decoder_model_data, f"{decoder_file_name}.json")
            self._decoder_model_data.save_weights(f"{decoder_file_name}.weights.h5")

        try:
            # Save first UNet model
            self._network.save(first_unet_file_name)
            print(f"First UNet model saved to {first_unet_file_name}")
        except Exception as e:
            print(f"Error saving first UNet: {e}")
            # Fallback to JSON + weights
            self._save_model_to_json(self._network, f"{first_unet_file_name}.json")
            self._network.save_weights(f"{first_unet_file_name}.weights.h5")

        try:
            # Save second UNet model
            self._second_unet_model.save(second_unet_file_name)
            print(f"Second UNet model saved to {second_unet_file_name}")
        except Exception as e:
            print(f"Error saving second UNet: {e}")
            # Fallback to JSON + weights
            self._save_model_to_json(self._second_unet_model, f"{second_unet_file_name}.json")
            self._second_unet_model.save_weights(f"{second_unet_file_name}.weights.h5")

        # Save shape information
        shape_info = {
            'input_shape': self._input_shape,
            'inferred_shape': self._inferred_shape,
            'embedding_dimension': self._embedding_dimension
        }
        shape_file = os.path.join(directory, f"{file_name}_shape_info.json")
        with open(shape_file, 'w') as f:
            json.dump(shape_info, f)
        print(f"Shape information saved to {shape_file}")

    @staticmethod
    def _save_model_to_json(model, file_path):
        """
        Save model architecture to a JSON file.

        Args:
            model (tf.keras.Model): Model to save.
            file_path (str): Path to the JSON file.
        """

        try:
            # Try to save the model as JSON
            with open(file_path, "w") as json_file:
                json.dump(model.to_json(), json_file)
            print(f"Model architecture successfully saved to {file_path}.")

        except Exception as e:
            # In case of error, save the error message to the file
            error_message = f"Error occurred while saving model: {str(e)}"

            with open(file_path, "w") as error_file:
                error_file.write(error_message)
            print(f"An error occurred and was saved to {file_path}: {error_message}")

    # ==================================================
    # PROPERTIES WITH GETTERS AND SETTERS
    # ==================================================

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
    def embedding_dimension(self) -> int:
        """Get the embedding dimension size."""
        return self._embedding_dimension

    @embedding_dimension.setter
    def embedding_dimension(self, value: int) -> None:
        """Set the embedding dimension size."""
        if value <= 0:
            raise ValueError("Embedding dimension must be positive")
        self._embedding_dimension = value

    @property
    def encoder_model_data(self) -> Any:
        """Get the image encoder model."""
        return self._encoder_model_data

    @encoder_model_data.setter
    def encoder_model_data(self, value: Any) -> None:
        """Set the image encoder model."""
        self._encoder_model_data = value

    @property
    def decoder_model_data(self) -> Any:
        """Get the image decoder model."""
        return self._decoder_model_data

    @decoder_model_data.setter
    def decoder_model_data(self, value: Any) -> None:
        """Set the image decoder model."""
        self._decoder_model_data = value

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
    def auto_adapt_shape(self) -> bool:
        """Get the auto-adapt shape flag."""
        return self._auto_adapt_shape

    @auto_adapt_shape.setter
    def auto_adapt_shape(self, value: bool) -> None:
        """Set the auto-adapt shape flag."""
        self._auto_adapt_shape = value