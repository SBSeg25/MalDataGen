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

    from tensorflow.keras.metrics import Mean
    from tensorflow.keras.models import Model

    from tensorflow.keras.utils import to_categorical

except ImportError as error:
    print(error)
    sys.exit(-1)

DEFAULT_LATENT_MEAN_DISTRIBUTION=0.0
DEFAULT_LATENT_STANDER_DEVIATION=1.0
DEFAULT_LATENT_DIMENSION=64
DEFAULT_NUMBER_CLASSES=2


class AutoencoderAlgorithmTensorflow(Model):
    """
    An abstract class for AutoEncoder models.

    This class provides a foundation for AutoEncoder models with methods for training,
    generating synthetic data, saving and loading models.

    Args:
        @encoder_model (Model, optional):
            The encoder part of the AutoEncoder.
        @decoder_model (Model, optional):
            The decoder part of the AutoEncoder.
        @loss_function (Loss, optional):
            The loss function for training.
        @file_name_encoder (str, optional):
            The file name for saving the encoder model.
        @file_name_decoder (str, optional):
            The file name for saving the decoder model.
        @models_saved_path (str, optional):
            The path to save the models.
        @latent_mean_distribution (float, optional):
            Mean of the latent space distribution.
        @latent_stander_deviation (float, optional):
            Standard deviation of the latent space distribution.
        @latent_dimension (int, optional):
            The dimensionality of the latent space.

    Attributes:
        @_encoder (Model):
            The encoder part of the AutoEncoder.
        @_decoder (Model):
            The decoder part of the AutoEncoder.
        @_loss_function (Loss):
            Loss function for training.
        @_total_loss_tracker (Mean):
            Metric for tracking total loss.
        @_file_name_encoder (str):
            File name for saving the encoder model.
        @_file_name_decoder (str):
            File name for saving the decoder model.
        @_models_saved_path (str):
            Path to save the models.
        @_encoder_decoder_model (Model):
            Combined encoder-decoder model.

    Example:
        >>> encoder_model = build_encoder(input_shape=(128, 128, 3), latent_dimension=64)
        >>> decoder_model = build_decoder(latent_dimension=64, output_shape=(128, 128, 3))
        ...     autoencoder = AutoencoderAlgorithm(
        ...     encoder_model=encoder_model,
        ...     decoder_model=decoder_model,
        ...     loss_function=_tensorflow.keras.losses.MeanSquaredError(),
        ...     file_name_encoder="encoder_model.h5",
        ...     file_name_decoder="decoder_model.h5",
        ...     models_saved_path="./autoencoder_models/",
        ...     latent_mean_distribution=0.0,
        ...     latent_stander_deviation=1.0,
        ...     latent_dimension=64
        ...     )
        ...     autoencoder.compile(optimizer=_tensorflow.keras.optimizers.Adam(learning_rate=0.001))
        >>> autoencoder.fit_model(train_dataset, epochs=50)
    """

    def __init__(self,
                 encoder_model,
                 decoder_model,
                 loss_function=None,
                 file_name_encoder=None,
                 file_name_decoder=None,
                 models_saved_path=None,
                 latent_mean_distribution=DEFAULT_LATENT_MEAN_DISTRIBUTION,
                 latent_stander_deviation=DEFAULT_LATENT_STANDER_DEVIATION,
                 latent_dimension=DEFAULT_LATENT_DIMENSION):

        super().__init__()
        """
        Initializes an AutoEncoder model with an encoder, decoder, and necessary configurations.

        Args:
            @encoder_model (Model):
                The encoder part of the AutoEncoder.
            @decoder_model (Model):
                The decoder part of the AutoEncoder.
            @loss_function (Loss):
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

        Attributes:
            @_encoder (Model):
                The encoder model.
            @_decoder (Model):
                The decoder model.
            @_loss_function (Loss):
                The loss function used for optimization.
            @_total_loss_tracker (Mean):
                Metric for tracking total loss during training.
            @_latent_mean_distribution (float):
                The mean of the latent space distribution.
            @_latent_standard_deviation (float):
                The standard deviation of the latent space distribution.
            @_latent_dimension (int):
                The dimensionality of the latent space.
            @_file_name_encoder (str):
                Name of the file where the encoder model is saved.
            @_file_name_decoder (str): 
                Name of the file where the decoder model is saved.
            @_models_saved_path (str):
                Path where models are saved.
            @_encoder_decoder_model (Model):
                A combined model that links the encoder and decoder.

        """
        if not isinstance(encoder_model, tensorflow.keras.Model):
            raise TypeError("encoder_model must be a tf.keras.Model instance.")

        if not isinstance(decoder_model, tensorflow.keras.Model):
            raise TypeError("decoder_model must be a tf.keras.Model instance.")

        if not isinstance(file_name_encoder, str) or not file_name_encoder:
            raise ValueError("file_name_encoder must be a non-empty string.")

        if not isinstance(file_name_decoder, str) or not file_name_decoder:
            raise ValueError("file_name_decoder must be a non-empty string.")

        if not isinstance(models_saved_path, str) or not models_saved_path:
            raise ValueError("models_saved_path must be a non-empty string.")

        if not isinstance(latent_mean_distribution, (int, float)):
            raise TypeError("latent_mean_distribution must be a number.")

        if not isinstance(latent_stander_deviation, (int, float)):
            raise TypeError("latent_stander_deviation must be a number.")

        if latent_stander_deviation <= 0:
            raise ValueError("latent_stander_deviation must be greater than 0.")

        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")

        # Initialize the encoder and decoder models
        self._encoder = encoder_model
        self._decoder = decoder_model

        # Loss function and metric for tracking total loss
        self._loss_function = loss_function
        self._total_loss_tracker = Mean(name="loss")
        self._latent_mean_distribution = latent_mean_distribution
        self._latent_stander_deviation = latent_stander_deviation
        self._latent_dimension = latent_dimension

        # File names for saving models
        self._file_name_encoder = file_name_encoder
        self._file_name_decoder = file_name_decoder

        # Path for saving models
        self._models_saved_path = models_saved_path

        # Combined encoder-decoder model
        self._encoder_decoder_model = Model(self._encoder.input, self._decoder(self._encoder.output))


    @tensorflow.function
    def train_step(self, batch, optimizer = None):
        """
        Perform a training step for the AutoEncoder.

        Args:
            batch: Input data batch.

        Returns:
            dict: Dictionary containing the loss value.
        """
        batch_x, batch_y = batch

        # Use tf.function decorator for improved TensorFlow performance

        with tensorflow.GradientTape() as gradient_ae:
            # Forward pass: Generate reconstructed data using the encoder-decoder model
            reconstructed_data = self._encoder_decoder_model(batch_x, training=True)

            # Calculate the mean squared error loss between input batch and reconstructed data
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

            # Callbacks
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
            # Example: if label_class = 1 and number_instances = 3, this creates:
            # [[0, 1], [0, 1], [0, 1]]
            label_samples_generated = to_categorical(
                [label_class] * number_instances,
                num_classes=number_samples_per_class["number_classes"]
            )

            # Generate random noise vectors (latent space vectors) for each sample
            # Shape: (number_instances, latent_dimension)
            latent_noise = numpy.random.normal(
                self._latent_mean_distribution,  # Mean of the latent distribution
                self._latent_stander_deviation,  # Standard deviation of the latent distribution
                (number_instances, self._latent_dimension)
            )

            # Use the decoder to generate synthetic samples from the latent space and class labels
            # Inputs: (latent vectors, class labels)
            # 'verbose=0' suppresses any output from the decoder's predict method
            generated_samples = self._decoder.predict([latent_noise, label_samples_generated], verbose=0)

            # Round the output values to the nearest integer
            # This is useful if the output is binary (like 0/1) or for discrete data types
            generated_samples = numpy.rint(generated_samples)

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
