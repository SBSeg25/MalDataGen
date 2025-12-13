#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

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

    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import model_from_json

    from tensorflow.keras.losses import BinaryCrossentropy


except ImportError as error:
    logging.error(error)
    sys.exit(-1)


class AdversarialAlgorithmTensorflow(Model):
    """
    Implements an adversarial training algorithm, typically used in Generative Adversarial Networks (GANs).

    This class performs adversarial training by utilizing a generator and a discriminator,
    optimizing the generator to produce realistic data while training the discriminator to differentiate
    between real and fake data.

    The concept of Generative Adversarial Networks was introduced by Ian Goodfellow and his collaborators in
    the following paper:


    Attributes:
        @generator_model (Model):
            The generator model.
        @discriminator_model (Model):
            The discriminator model.
        @latent_dimension (int):
            Dimensionality of the latent space.
        @loss_generator (function):
            Loss function for the generator.
        @loss_discriminator (function):
            Loss function for the discriminator.
        @file_name_discriminator (str):
            Filename for saving the discriminator model.
        @file_name_generator (str):
            Filename for saving the generator model.
        @models_saved_path (str):
            Path where models will be saved.
        @latent_mean_distribution (float):
            Mean of the latent noise distribution.
        @latent_stander_deviation (float):
            Standard deviation of the latent noise distribution.
        @smoothing_rate (float):
            Smoothing rate applied to discriminator labels.

    Example:
        >>> generator_model = build_generator(latent_dimension=100)
        ...     discriminator_model = build_discriminator()
        ...     adversarial_algorithm = AdversarialAlgorithm(
        ...     generator_model=generator_model,
        ...     discriminator_model=discriminator_model,
        ...     latent_dimension=100,
        ...     loss_generator=tf.keras.losses.BinaryCrossEntropy(),
        ...     loss_discriminator=tf.keras.losses.BinaryCrossEntropy(),
        ...     file_name_discriminator="discriminator.h5",
        ...     file_name_generator="generator.h5",
        ...     models_saved_path="./models/",
        ...     latent_mean_distribution=0.0,
        ...     latent_stander_deviation=1.0,
        ...     smoothing_rate=0.1
        ...     )
        ...     adversarial_algorithm.compile(
        ...     optimizer_generator=tf.keras.optimizers.Adam(learning_rate=0.0002),
        ...     optimizer_discriminator=tf.keras.optimizers.Adam(learning_rate=0.0002)
        ...     )
        # Train the model
        >>> adversarial_algorithm.fit_model(train_dataset)

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
                 latent_stander_deviation,
                 smoothing_rate,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)
        """
        Initializes the adversarial algorithm with the specified generator, discriminator, and other configurations.

        Args:
            @generator_model (Model):
                The generator model.
            @discriminator_model (Model):
                The discriminator model.
            @latent_dimension (int):
                Latent space dimension.
            @loss_generator (function):
                Generator's loss function.
            @loss_discriminator (function):
                Discriminator's loss function.
            @file_name_discriminator (str):
                Filename for discriminator model.
            @file_name_generator (str):
                Filename for generator model.
            @models_saved_path (str):
                Path for saving models.
            @latent_mean_distribution (float):
                Mean of the latent noise distribution.
            @latent_stander_deviation (float):
                Standard deviation of the latent noise.
            @smoothing_rate (float):
                Label smoothing rate.
            @*args, **kwargs:
                Additional arguments.
        """

        if latent_dimension <= 0:
            raise ValueError("Latent dimension must be greater than 0.")

        if not isinstance(file_name_discriminator, str) or not file_name_discriminator:
            raise ValueError("Discriminator file name must be a non-empty string.")

        if not isinstance(file_name_generator, str) or not file_name_generator:
            raise ValueError("Generator file name must be a non-empty string.")

        if not isinstance(models_saved_path, str) or not models_saved_path:
            raise ValueError("Architectures saved path must be a non-empty string.")

        if not isinstance(latent_mean_distribution, (int, float)):
            raise TypeError("Latent mean distribution must be a number.")

        if not isinstance(latent_stander_deviation, (int, float)):
            raise TypeError("Latent standard deviation must be a number.")

        if latent_stander_deviation <= 0:
            raise ValueError("Latent standard deviation must be greater than 0.")

        if not (0.0 <= smoothing_rate <= 1.0):
            raise ValueError("Smoothing rate must be between 0 and 1.")


        self._generator = generator_model
        self._discriminator = discriminator_model
        self._latent_dimension = latent_dimension
        self._optimizer_generator = None
        self._optimizer_discriminator = None
        self._loss_generator = loss_generator
        self._loss_discriminator = loss_discriminator
        self._smoothing_rate = smoothing_rate
        self._latent_mean_distribution = latent_mean_distribution
        self._latent_stander_deviation = latent_stander_deviation
        self._file_name_discriminator = file_name_discriminator
        self._file_name_generator = file_name_generator
        self._models_saved_path = models_saved_path


    def compile(self, optimizer_generator, optimizer_discriminator, loss_generator, loss_discriminator, *args,
                **kwargs):
        super().compile(*args, **kwargs)
        """
        Compiles the adversarial algorithm by setting optimizers and loss functions for both generator and discriminator.

        Args:
            optimizer_generator (Optimizer): Optimizer for the generator.
            optimizer_discriminator (Optimizer): Optimizer for the discriminator.
            loss_generator (function): Generator's loss function.
            loss_discriminator (function): Discriminator's loss function.
            *args, **kwargs: Additional arguments.
        """

        self._optimizer_generator = optimizer_generator
        self._optimizer_discriminator = optimizer_discriminator
        self._loss_generator = loss_generator
        self._loss_discriminator = loss_discriminator

    @tensorflow.function
    def train_step(self, batch):
        """
        Performs a single training step for both generator and discriminator.
        This function is decorated with @Tensorflow.function to optimize graph execution.

        Args:
            batch (tuple): A tuple containing real features (input data) and their corresponding labels.

        Returns:
            dict: A dictionary containing the loss values for both generator (loss_g) and discriminator (loss_d).
        """

        # Unpack the batch into real features and real labels
        real_feature, real_samples_label = batch

        # Get the current batch size (number of samples in this batch)
        batch_size = tensorflow.shape(real_feature)[0]

        # Expand the label tensor to match the expected shape (add a new axis at the end)
        real_samples_label = tensorflow.expand_dims(real_samples_label, axis=-1)

        # Sample random noise vectors (latent space) for the generator input
        latent_space = tensorflow.random.normal(shape=(batch_size, self._latent_dimension))

        # Use the generator to create synthetic (fake) features using noise and real labels (conditioning)
        synthetic_feature = self._generator([latent_space, real_samples_label], training=False)

        # Start discriminator training within a gradient tape context to track gradients
        with tensorflow.GradientTape() as discriminator_gradient:
            # Get discriminator prediction on real samples (real features + real labels)
            label_predicted_real = self._discriminator([real_feature, real_samples_label], training=True)

            # Get discriminator prediction on synthetic (fake) samples (synthetic features + real labels)
            label_predicted_synthetic = self._discriminator([synthetic_feature, real_samples_label], training=True)

            # Concatenate predictions for real and synthetic samples into a single tensor
            label_predicted_all_samples = tensorflow.concat([label_predicted_real, label_predicted_synthetic], axis=0)

            # Create the ground-truth labels for the discriminator
            # Real samples should be labeled as 0, synthetic samples as 1
            list_all_labels_predicted = [
                tensorflow.zeros_like(label_predicted_real),  # Real labels = 0
                tensorflow.ones_like(label_predicted_synthetic)  # Fake labels = 1
            ]
            tensor_labels_predicted = tensorflow.concat(list_all_labels_predicted, axis=0)

            # Label smoothing for regularization
            # Add random noise to real labels (smooth them slightly downwards)
            smooth_tensor_real_data = 0.15 * tensorflow.random.uniform(tensorflow.shape(label_predicted_real))

            # Add random noise to fake labels (smooth them slightly upwards)
            smooth_tensor_synthetic_data = -0.15 * tensorflow.random.uniform(
                tensorflow.shape(label_predicted_synthetic))

            # Combine smoothed noise into the label tensor
            tensor_labels_predicted += tensorflow.concat(
                [smooth_tensor_real_data, smooth_tensor_synthetic_data], axis=0
            )

            # Compute the discriminator loss comparing predicted labels to (smoothed) ground-truth labels
            loss_value = self._loss_discriminator(tensor_labels_predicted, label_predicted_all_samples)

        # Compute gradients of the discriminator loss with respect to discriminator trainable weights
        gradient_tape_loss = discriminator_gradient.gradient(loss_value, self._discriminator.trainable_variables)

        # Apply the computed gradients to update discriminator weights
        self._optimizer_discriminator.apply_gradients(zip(gradient_tape_loss, self._discriminator.trainable_variables))

        # Start generator training within a gradient tape context to track gradients
        with tensorflow.GradientTape() as generator_gradient:
            # Generate synthetic samples using the generator (in training mode)
            latent_space = tensorflow.random.normal(shape=(batch_size, self._latent_dimension))
            synthetic_feature = self._generator([latent_space, real_samples_label], training=True)

            # Get discriminator predictions for the synthetic samples (real labels used for conditioning)
            predicted_labels = self._discriminator([synthetic_feature, real_samples_label], training=False)

            # Compute the generator loss
            # The generator wants the discriminator to classify synthetic data as real (label = 0)
            total_loss_g = self._loss_generator(tensorflow.zeros_like(predicted_labels), predicted_labels)

        # Compute gradients of the generator loss with respect to generator trainable weights
        gradient_tape_loss = generator_gradient.gradient(total_loss_g, self._generator.trainable_variables)

        # Apply the computed gradients to update generator weights
        self._optimizer_generator.apply_gradients(zip(gradient_tape_loss, self._generator.trainable_variables))

        # Return a dictionary containing both losses for tracking
        return {"loss_d": loss_value, "loss_g": total_loss_g}

    def get_samples(self, number_samples_per_class):
        """
        Generates synthetic data samples for each specified class using the trained generator.

        Args:
            number_samples_per_class (dict):
                A dictionary specifying the number of synthetic samples to generate per class.
                Expected structure:
                {
                    "classes": {class_label: number_of_samples, ...},
                    "number_classes": total_number_of_classes
                }

        Returns:
            dict:
                A dictionary where each key is a class label and the value is an array of generated samples for that class.
        """

        # Initialize an empty dictionary to store generated samples grouped by class
        generated_data = {}

        # Iterate over each class and the corresponding number of samples to generate
        for label_class, number_instances in number_samples_per_class["classes"].items():
            # Create one-hot encoded labels for all generated samples in this class
            # Example: if label_class = 2 and number_instances = 5, this will generate:
            # [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
            label_samples_generated = to_categorical(
                [label_class] * number_instances,
                num_classes=number_samples_per_class["number_classes"]
            )

            # Generate random noise vectors (latent vectors) for each sample
            # Shape: (number_instances, latent_dimension)
            latent_noise = numpy.random.normal(
                self._latent_mean_distribution,  # Mean of the latent space distribution
                self._latent_stander_deviation,  # Standard deviation of the latent space distribution
                (number_instances, self._latent_dimension)
            )

            # Use the generator to produce synthetic samples conditioned on the class labels
            # Inputs: latent noise vectors and one-hot class labels
            # 'verbose=0' suppresses console output from the predict call
            generated_samples = self._generator.predict([latent_noise, label_samples_generated], verbose=0)

            # Round generated sample values to nearest integer (useful if generating binary data, like images with pixel values 0/1)
            generated_samples = numpy.rint(generated_samples)

            # Store the generated samples in the dictionary under the corresponding class label
            generated_data[label_class] = generated_samples

        # Return the dictionary containing all generated samples, grouped by class
        return generated_data

    def save_model(self, path_output, k_fold):

        try:
            logging.info("Starting to save Adversarial Model for fold {}...".format(k_fold))

            # Create directory for saving models
            path_directory = os.path.join(path_output, self._models_saved_path)
            Path(path_directory).mkdir(parents=True, exist_ok=True)
            logging.info("Created/verified directory at: {}".format(path_directory))

            # Filenames for the discriminator and generator models
            discriminator_file_name = self._file_name_discriminator + "_" + str(k_fold)
            generator_file_name = self._file_name_generator + "_" + str(k_fold)

            # Directory for the current fold
            path_model = os.path.join(path_directory, "fold_" + str(k_fold + 1))
            Path(path_model).mkdir(parents=True, exist_ok=True)
            logging.info("Created/verified fold directory at: {}".format(path_model))

            # Full paths for the model files
            discriminator_file_name = os.path.join(path_model, discriminator_file_name)
            generator_file_name = os.path.join(path_model, generator_file_name)

            # Saving discriminator model to JSON and weights
            logging.info("Saving discriminator model...")
            discriminator_model_json = self._discriminator.to_json()

            with open(discriminator_file_name + ".json", "w") as json_file:
                json_file.write(discriminator_model_json)
            self._discriminator.save_weights(discriminator_file_name + ".h5")
            logging.info("Discriminator model saved at: {}.json and {}.h5".format(discriminator_file_name,
                                                                                  discriminator_file_name))

            # Saving generator model to JSON and weights
            logging.info("Saving generator model...")
            generator_model_json = self._generator.to_json()

            with open(generator_file_name + ".json", "w") as json_file:
                json_file.write(generator_model_json)
            self._generator.save_weights(generator_file_name + ".h5")
            logging.info("Generator model saved at: {}.json and {}.h5".format(generator_file_name,
                                                                              generator_file_name))

        except FileExistsError:
            logging.error("Model file already exists. Aborting.")
            exit(-1)

        except Exception as e:
            logging.error("An error occurred while saving the models: {}".format(e))
            exit(-1)

    def load_models(self, path_output, k_fold):

        try:
            logging.info("Loading Adversarial Model for fold {}...".format(k_fold + 1))

            # Directory containing saved models
            path_directory = os.path.join(path_output, self._models_saved_path)

            # Filenames for the discriminator and generator models
            discriminator_file_name = self._file_name_discriminator + "_" + str(k_fold + 1)
            generator_file_name = self._file_name_generator + "_" + str(k_fold + 1)

            # Full paths to the model files
            discriminator_file_name = os.path.join(path_directory, discriminator_file_name)
            generator_file_name = os.path.join(path_directory, generator_file_name)

            # Load discriminator model
            logging.info("Loading discriminator model from: {}.json".format(discriminator_file_name))
            with open(discriminator_file_name + ".json", 'r') as json_file:
                discriminator_model_json = json_file.read()

            self._discriminator = model_from_json(discriminator_model_json)
            self._discriminator.load_weights(discriminator_file_name + ".h5")
            logging.info("Loaded discriminator weights from: {}.h5".format(discriminator_file_name))

            # Load generator model
            logging.info("Loading generator model from: {}.json".format(generator_file_name))
            with open(generator_file_name + ".json", 'r') as json_file:
                generator_model_json = json_file.read()

            self._generator = model_from_json(generator_model_json)
            self._generator.load_weights(generator_file_name + ".h5")
            logging.info("Loaded generator weights from: {}.h5".format(generator_file_name))

        except FileNotFoundError:
            logging.error("Model file not found. Please provide an existing and valid model.")
            exit(-1)
        except Exception as e:
            logging.error("An error occurred while loading the models: {}".format(e))
            exit(-1)

    def set_generator(self, generator):
        self._generator = generator

    def set_discriminator(self, discriminator):
        self._discriminator = discriminator

    def set_latent_dimension(self, latent_dimension):
        self._latent_dimension = latent_dimension

    def set_optimizer_generator(self, optimizer_generator):
        self._optimizer_generator = optimizer_generator

    def set_optimizer_discriminator(self, optimizer_discriminator):
        self._optimizer_discriminator = optimizer_discriminator

    def set_loss_generator(self, loss_generator):
        self._loss_generator = loss_generator

    def set_loss_discriminator(self, loss_discriminator):
        self._loss_discriminator = loss_discriminator

