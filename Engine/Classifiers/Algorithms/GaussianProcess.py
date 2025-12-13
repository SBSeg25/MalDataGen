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
    import sys
    import numpy
    import logging

    from sklearn.exceptions import NotFittedError

    from sklearn.gaussian_process import GaussianProcessClassifier

except ImportError as error:
    print(error)
    sys.exit(-1)

class GaussianProcess:
    """
    A Gaussian Process classifier wrapper that encapsulates the configuration and training process.

    Attributes:
        _gaussian_process_kernel (str): Kernel function for the Gaussian Process.
        _gaussian_process_max_iterations (int): Maximum number of iterations for predictions.
        _gaussian_process_optimizer (str): Optimizer used to maximize the log-marginal likelihood.

    Methods:
        get_model(x_samples_training, y_samples_training, dataset_type, input_dataset_shape):
            Trains a Gaussian Process model using the provided training samples and labels.
    """

    def __init__(self, arguments):
        """
        Initializes the GaussianProcess class with hyperparameters.

        Args:
            arguments: An object containing hyperparameters for the Gaussian Process model.
        """
        self._gaussian_process_kernel = arguments.gaussian_process_kernel
        self._gaussian_process_max_iterations = arguments.gaussian_process_max_iterations
        self._gaussian_process_optimizer = arguments.gaussian_process_optimizer

        logging.debug(f"GaussianProcess initialized with kernel={self._gaussian_process_kernel}, "
                      f"max_iterations={self._gaussian_process_max_iterations}, "
                      f"optimizer={self._gaussian_process_optimizer}")

    def get_model(self, x_samples_training, y_samples_training, dataset_type, input_dataset_shape):
        """
        Trains a Gaussian Process classifier using the provided training samples and labels.

        Args:
            x_samples_training (array-like): The training feature samples.
            y_samples_training (array-like): The training labels corresponding to the samples.
            dataset_type: The data type for the training samples (e.g., float32).
            input_dataset_shape: The shape of the input dataset (used for logging purposes).

        Returns:
            GaussianProcessClassifier: A trained Gaussian Process classifier instance.

        Raises:
            ValueError: If training samples or labels are empty or do not match in shape.
            Exception: For any other issues encountered during model fitting.
        """
        logging.info("Starting training classifier: GAUSSIAN PROCESS")

        try:
            # Convert input samples to the specified data type
            logging.debug(f"Converting training samples to numpy arrays with type {dataset_type}.")
            x_samples_training = numpy.array(x_samples_training, dtype=dataset_type)
            y_samples_training = numpy.array(y_samples_training, dtype=dataset_type)

            logging.debug(f"Training data shape: {x_samples_training.shape}, Labels shape: {y_samples_training.shape}")

            # Validate training data
            if x_samples_training.size == 0 or y_samples_training.size == 0:
                raise ValueError("Training samples or labels are empty.")
            if x_samples_training.shape[0] != y_samples_training.shape[0]:
                raise ValueError("The number of samples in training data and labels do not match.")

            # Create and train the Gaussian Process classifier
            instance_model_classifier = GaussianProcessClassifier(
                kernel=self._gaussian_process_kernel,
                optimizer=self._gaussian_process_optimizer,
                max_iter_predict=self._gaussian_process_max_iterations
            )

            logging.info("Fitting the Gaussian Process model to the training data.")
            instance_model_classifier.fit(x_samples_training, y_samples_training)
            logging.info("Finished training GAUSSIAN PROCESS model.")
            return instance_model_classifier

        except ValueError as ve:
            logging.error(f"ValueError occurred: {ve}")
            raise  # Re-raise to propagate the exception
        except Exception as e:
            logging.error(f"An error occurred during model training: {e}")
            raise  # Re-raise to propagate the exception

    def set_gaussian_process_kernel(self, gaussian_process_kernel):
        """
        Sets a new kernel function for the Gaussian Process.

        Args:
            gaussian_process_kernel (str): New kernel function.
        """
        logging.debug(f"Setting new Gaussian Process kernel: kernel={gaussian_process_kernel}")
        self._gaussian_process_kernel = gaussian_process_kernel

    def set_gaussian_process_max_iterations(self, gaussian_process_max_iterations):
        """
        Sets the maximum number of iterations for the Gaussian Process predictions.

        Args:
            gaussian_process_max_iterations (int): New maximum number of iterations.
        """
        logging.debug(f"Setting new Gaussian Process max iterations: max_iterations={gaussian_process_max_iterations}")
        self._gaussian_process_max_iterations = gaussian_process_max_iterations

    def set_gaussian_process_optimizer(self, gaussian_process_optimizer):
        """
        Sets a new optimizer for the Gaussian Process model.

        Args:
            gaussian_process_optimizer (str): New optimizer for the Gaussian Process.
        """
        logging.debug(f"Setting new Gaussian Process optimizer: optimizer={gaussian_process_optimizer}")
        self._gaussian_process_optimizer = gaussian_process_optimizer

    def predict(self, model, x_samples):
        """
        Makes predictions using the trained Gaussian Process model.

        Args:
            model (GaussianProcessClassifier): The trained Gaussian Process classifier.
            x_samples (array-like): The samples for which to make predictions.

        Returns:
            array: Predictions for the provided samples.

        Raises:
            NotFittedError: If the model has not been trained yet.
            ValueError: If the input samples are empty or do not match the model's expected input shape.
        """
        try:
            if not isinstance(model, GaussianProcessClassifier):
                raise NotFittedError("The model is not fitted. Please train the model first.")

            x_samples = numpy.array(x_samples, dtype=numpy.float32)

            if x_samples.size == 0:
                raise ValueError("Input samples for prediction cannot be empty.")

            predictions = model.predict(x_samples)
            logging.debug(f"Predictions made for samples: {predictions}")
            return predictions

        except ValueError as ve:
            logging.error(f"ValueError during prediction: {ve}")
            raise  # Re-raise to propagate the exception
        except NotFittedError as nfe:
            logging.error(f"Model not fitted error: {nfe}")
            raise  # Re-raise to propagate the exception
        except Exception as e:
            logging.error(f"An error occurred during prediction: {e}")
            raise  # Re-raise to propagate the exception