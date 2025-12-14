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

    from tensorflow.keras import Model

    from tensorflow.keras.metrics import Mean

    from tensorflow.keras.utils import to_categorical

    from tensorflow.keras.losses import BinaryCrossentropy

except ImportError as error:
    print(error)
    sys.exit(-1)


class VAELatentDiffusionAlgorithmTensorflow(Model):
    """
    Implements a Variational AutoEncoder (VAE) model for generating synthetic data.

    The model includes an encoder and a decoder for encoding input data and reconstructing
    it from a learned latent space. During training, it computes both the reconstruction loss
    and the KL divergence loss. The trained decoder can be used to generate synthetic data.

    This class supports customizable latent space parameters and loss functions, making it
    adaptable for different generative tasks.

    Attributes:
        @_encoder (Model):
            Encoder model that encodes input data into the latent space.
        @_decoder (Model):
            Decoder model that reconstructs data from the latent representation.
        @_loss_function (callable):
            Function used to compute the total loss during training.
        @_total_loss_tracker (Mean):
            Tracks the overall loss during training.
        @_reconstruction_loss_tracker (Mean):
            Tracks the reconstruction loss during training.
        @_kl_loss_tracker (Mean):
            Tracks the KL divergence loss during training.
        @_latent_mean_distribution (float):
            Mean of the latent distribution.
        @_latent_standard_deviation (float):
            Standard deviation of the latent distribution.
        @_latent_dimension (int):
            Dimensionality of the latent space.
        @_decoder_latent_dimension (int):
            Dimensionality of the latent space used by the decoder.
        @_file_name_encoder (str):
            File name for saving the encoder model.
        @_file_name_decoder (str):
            File name for saving the decoder model.
        @_models_saved_path (str):
            Directory path where the encoder and decoder models are saved.

    Raises:
        ValueError:
            Raised in cases where:
            - The latent dimension is non-positive.
            - The standard deviation of the latent space is non-positive.
            - The file paths are invalid.

    Example:
        >>> vae_model = VAELatentDiffusionAlgorithmTensorflow(
        ...     encoder_model=encoder,
        ...     decoder_model=decoder,
        ...     loss_function=custom_loss_function,
        ...     latent_dimension=128,
        ...     latent_mean_distribution=0.0,
        ...     latent_standard_deviation=1.0,
        ...     file_name_encoder="encoder_model.h5",
        ...     file_name_decoder="decoder_model.h5",
        ...     models_saved_path="models/"
        ... )
        >>> vae_model.train_step(data)
    """

    def __init__(self,
                 encoder_model,
                 decoder_model,
                 loss_function,
                 latent_dimension,
                 decoder_latent_dimension,
                 latent_mean_distribution,
                 latent_standard_deviation,
                 file_name_encoder,
                 file_name_decoder,
                 models_saved_path,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)
        """
        Initializes the VariationalAlgorithm model with provided encoder and decoder models, 
        loss function, and latent space parameters.

        This constructor sets up the architecture, metrics, and paths for saving the models.

        Args:
            @encoder_model (Model):
                The encoder model responsible for encoding input data into latent variables.
            @decoder_model (Model):
                The decoder model responsible for reconstructing data from the latent space.
            @loss_function (callable):
                The loss function used to compute the training loss.
            @latent_dimension (int):
                The dimensionality of the latent space.
            @latent_mean_distribution (float):
                The mean of the latent distribution (usually 0).
            @latent_standard_deviation (float):
                The standard deviation of the latent distribution (usually 1).
            @file_name_encoder (str):
                The filename for saving the encoder model.
            @file_name_decoder (str):
                The filename for saving the decoder model.
            @models_saved_path (str):
                The directory where the models will be saved.
            @*args:
                Additional arguments for the parent class.
            @**kwargs:
                Additional keyword arguments for the parent class.

        Raises:
            ValueError:
                If latent_dimension <= 0.
                If latent_standard_deviation <= 0.
                If file paths are invalid.
        """
        # Initialize the encoder and decoder models
        self._encoder = encoder_model
        self._decoder = decoder_model

        # Loss function and metrics for tracking losses
        self._loss_function = loss_function
        self._total_loss_tracker = Mean(name="loss")
        self._reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self._kl_loss_tracker = Mean(name="kl_loss")
        self._latent_mean_distribution = latent_mean_distribution
        self._latent_standard_deviation = latent_standard_deviation
        self._latent_dimension = latent_dimension
        self._decoder_latent_dimension = decoder_latent_dimension
        # File names for saving models
        self._file_name_encoder = file_name_encoder
        self._file_name_decoder = file_name_decoder

        # Path for saving models
        self._models_saved_path = models_saved_path

    @tensorflow.function
    def train_step(self, batch):
        """
        Perform a training step for the Variational AutoEncoder (VAE).

        Args:
            batch: Input data batch.

        Returns:
            dict: Dictionary containing the loss values (total loss, reconstruction loss, KL divergence loss).
        """
        # Use tf.function decorator for improved TensorFlow performance
        batch_x, batch_y = batch

        with tensorflow.GradientTape() as tape:
            # Forward pass: Encode input data and sample from the latent space
            latent_mean, latent_log_variation, latent, label = self._encoder(batch_x)

            # Decode the sampled latent space and generate reconstructed data
            reconstruction_data = self._decoder([latent, label])

            # Calculate binary cross-entropy loss for reconstruction
            binary_cross_entropy_loss = tensorflow.keras.losses.binary_crossentropy(batch_y, reconstruction_data)
            sum_reduced = binary_cross_entropy_loss
            reconstruction_loss = tensorflow.reduce_mean(sum_reduced)

            # Calculate KL divergence loss
            encoder_output = (1 + latent_log_variation - tensorflow.square(latent_mean))
            kl_divergence_loss = -0.5 * (encoder_output - tensorflow.exp(latent_log_variation))
            kl_divergence_loss = tensorflow.reduce_mean(tensorflow.reduce_sum(kl_divergence_loss, axis=1))

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
        Train the model with a simplified progress bar.

        Args:
            x: Input data.
            y: Target data.
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
        history = {'loss': [], 'Diffusion_loss': []}

        # Initialize callbacks properly
        if callbacks is not None:
            for callback in callbacks:
                callback.set_model(self)
                # Set params that callbacks might need
                callback.set_params({
                    'epochs': epochs,
                    'steps': steps_per_epoch,
                    'verbose': verbose,
                    'batch_size': batch_size
                })
            # Call on_train_begin for all callbacks
            for callback in callbacks:
                callback.on_train_begin()

        # Training loop
        for epoch in range(initial_epoch, epochs):
            self._total_loss_tracker.reset_state()

            # Call on_epoch_begin for all callbacks
            if callbacks is not None:
                for callback in callbacks:
                    callback.on_epoch_begin(epoch)

            if verbose == 1:
                print(f'\nEpoch {epoch + 1}/{epochs}')

            # Progress tracking
            step = 0
            epoch_losses = []

            for batch_data in train_dataset:
                step += 1

                # Call on_batch_begin for all callbacks
                if callbacks is not None:
                    for callback in callbacks:
                        callback.on_batch_begin(step - 1)

                # Perform training step
                metrics = self.train_step(batch_data)
                current_loss = float(metrics.get('loss', 0))
                epoch_losses.append(current_loss)

                # Call on_batch_end for all callbacks
                if callbacks is not None:
                    logs = {'loss': current_loss, 'Diffusion_loss': current_loss}
                    for callback in callbacks:
                        callback.on_batch_end(step - 1, logs)

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
            if len(epoch_losses) > 0:
                epoch_loss = numpy.mean(epoch_losses)

            history['loss'].append(epoch_loss)
            history['Diffusion_loss'].append(epoch_loss)

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

            # Call on_epoch_end for all callbacks with proper logs
            if callbacks is not None:
                logs = {'loss': epoch_loss, 'Diffusion_loss': epoch_loss}
                if 'val_loss' in history and len(history['val_loss']) > 0:
                    logs['val_loss'] = history['val_loss'][-1]
                for callback in callbacks:
                    callback.on_epoch_end(epoch, logs)

        # Call on_train_end for all callbacks
        if callbacks is not None:
            for callback in callbacks:
                callback.on_train_end()

        # Return history object
        class History:
            def __init__(self, history_dict):
                self.history = history_dict

        return History(history)

    def _evaluate_validation(self, validation_data, validation_steps=None):
        """
        Evaluate the model on validation data.

        Args:
            validation_data: Validation dataset.
            validation_steps: Number of validation steps.

        Returns:
            Average validation loss.
        """
        val_losses = []
        step = 0

        for batch_data in validation_data:
            batch_x, batch_y = batch_data
            reconstructed = self._encoder_decoder_model(batch_x, training=False)
            loss = tensorflow.reduce_mean(tensorflow.square(batch_y - reconstructed))
            val_losses.append(float(loss))

            step += 1
            if validation_steps is not None and step >= validation_steps:
                break

        return numpy.mean(val_losses) if val_losses else 0.0


    def get_decoder_trained(self):

        return self._decoder

    def get_encoder_trained(self):

        return self._encoder


    def create_embedding(self, data):
        """
        Generates latent space embeddings using the trained encoder.

        Args:
            data (ndarray): Input data to encode.

        Returns:
            ndarray: Latent space representations.
        """
        return self._encoder.predict(data, batch_size=32)[0]


    def get_samples(self, number_samples_per_class):
        """
        Generate synthetic samples for each specified class using the trained decoder.

        This function generates samples by sampling from a normal distribution in the latent space
        and conditioning the generation process on class labels.

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
                Each array contains the synthetic samples generated for the corresponding class.
        """

        # Initialize a dictionary to store the generated samples for each class
        generated_data = {}

        # Iterate over each class and the corresponding number of samples to generate
        for label_class, number_instances in number_samples_per_class["classes"].items():
            # Create a one-hot encoded label array for all samples in the current class
            # Example: if label_class = 1 and number_instances = 3, this generates:
            # [[0, 1], [0, 1], [0, 1]]
            label_samples_generated = to_categorical([label_class] * number_instances,
                                                     num_classes=number_samples_per_class["number_classes"])

            # Sample random latent vectors from a standard normal distribution
            # Shape: (number_instances, decoder_latent_dimension)
            latent_noise = numpy.random.normal(size=(number_instances, self._decoder_latent_dimension))

            # Use the decoder to generate samples conditioned on the latent vectors and class labels
            # Inputs: (latent vectors, class labels)
            # 'verbose=0' suppresses any print output from the prediction process
            generated_samples = self._decoder.predict([latent_noise, label_samples_generated], verbose=0)

            # Round the generated samples to the nearest integer
            # This is useful for discrete data, like binary features (0/1) or integer values
            generated_samples = numpy.rint(generated_samples)

            # Store the generated samples in the dictionary under the corresponding class label
            generated_data[label_class] = generated_samples

        # Return the dictionary with all generated samples, organized by class
        return generated_data


    def generate_synthetic_data(self, number_samples_generate, labels, latent_dimension):
        """
        Generate synthetic data using the Variational AutoEncoder (VAE).

        Args:
            number_samples_generate (int): Number of synthetic samples to generate.
            labels: Labels for the generated data.
            latent_dimension (int): Dimension of the latent space.

        Returns:
            tf.Tensor: Synthetic data generated by the decoder.
        """

        # Generate random noise samples in the latent space
        random_noise_generate = tensorflow.random.normal(shape=(number_samples_generate, latent_dimension),
                                                 mean=self.latent_mean_distribution, stddev=self.latent_deviation,
                                                 dtype=tensorflow.float32)

        # Create label vectors for the generated data
        label_list = tensorflow.cast(tensorflow.fill((number_samples_generate, 1), labels), dtype=tensorflow.float32)

        # Generate synthetic data by passing random noise and labels through the decoder
        synthetic_data = self._decoder.predict(numpy.array([random_noise_generate, label_list]))

        # Return the generated synthetic data as a TensorFlow tensor
        return synthetic_data

    @property
    def metrics(self):
        """
        Returns:
            list: List of metrics to track during training.
        """
        return [self._total_loss_tracker, self._reconstruction_loss_tracker, self._kl_loss_tracker]

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
