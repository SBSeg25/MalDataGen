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
    from sklearn.svm import SVC

except ImportError as error:
    print(error)
    sys.exit(-1)

class SupportVectorMachine:
    """
    A Support Vector Machine (SVM) classifier wrapper that encapsulates the configuration and training process.

    Attributes:
        _support_vector_machine_normalization (float): Regularization parameter (C).
        _support_vector_machine_kernel (str): Kernel type (linear, poly, rbf, etc.).
        _support_vector_machine_kernel_degree (int): Degree of the polynomial kernel function (if poly is used).
        _support_vector_machine_gamma (str or float): Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.

    Methods:
        get_model(x_samples_training, y_samples_training, dataset_type, input_dataset_shape):
            Trains an SVM model using the provided training samples and labels.
    """

    def __init__(self, arguments):
        """
        Initializes the SupportVectorMachine class with hyperparameters.

        Args:
            arguments: An object containing hyperparameters for the SVM model.
        """
        self._support_vector_machine_normalization = arguments.support_vector_machine_regularization
        self._support_vector_machine_kernel = arguments.support_vector_machine_kernel
        self._support_vector_machine_kernel_degree = arguments.support_vector_machine_kernel_degree
        self._support_vector_machine_gamma = arguments.support_vector_machine_gamma

        logging.debug(f"SupportVectorMachine initialized with C={self._support_vector_machine_normalization}, "
                      f"kernel={self._support_vector_machine_kernel}, degree={self._support_vector_machine_kernel_degree}, "
                      f"gamma={self._support_vector_machine_gamma}")

    def get_model(self, x_samples_training, y_samples_training, dataset_type, input_dataset_shape):
        """
        Trains a Support Vector Machine (SVM) classifier using the provided training samples and labels.

        Args:
            x_samples_training (array-like): The training feature samples.
            y_samples_training (array-like): The training labels corresponding to the samples.
            dataset_type: The data type for the training samples (e.g., float32).
            input_dataset_shape: The shape of the input dataset (used for logging purposes).

        Returns:
            SVC: A trained Support Vector Machine classifier instance.

        Raises:
            ValueError: If training samples or labels are empty or do not match in shape.
            Exception: For any other issues encountered during model fitting.
        """
        logging.info("Starting training classifier: SUPPORT VECTOR MACHINE")

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

            # Create and train the Support Vector Machine classifier
            instance_model_classifier = SVC(
                C=self._support_vector_machine_normalization,
                kernel=self._support_vector_machine_kernel,
                degree=self._support_vector_machine_kernel_degree,
                gamma=self._support_vector_machine_gamma
            )

            logging.info("Fitting the Support Vector Machine model to the training data.")
            instance_model_classifier.fit(x_samples_training, y_samples_training)
            logging.info("Finished training SUPPORT VECTOR MACHINE model.")
            return instance_model_classifier

        except ValueError as ve:
            logging.error(f"ValueError occurred: {ve}")
            raise  # Re-raise to propagate the exception
        except Exception as e:
            logging.error(f"An error occurred during model training: {e}")
            raise  # Re-raise to propagate the exception

    def set_support_vector_machine_normalization(self, support_vector_machine_normalization):
        """
        Sets a new regularization parameter for the Support Vector Machine.

        Args:
            support_vector_machine_normalization (float): New regularization parameter (C).
        """
        logging.debug(f"Setting new SVM regularization parameter: C={support_vector_machine_normalization}")
        self._support_vector_machine_normalization = support_vector_machine_normalization

    def set_support_vector_machine_kernel(self, support_vector_machine_kernel):
        """
        Sets a new kernel type for the Support Vector Machine.

        Args:
            support_vector_machine_kernel (str): New kernel type.
        """
        logging.debug(f"Setting new SVM kernel: kernel={support_vector_machine_kernel}")
        self._support_vector_machine_kernel = support_vector_machine_kernel

    def set_support_vector_machine_kernel_degree(self, support_vector_machine_kernel_degree):
        """
        Sets a new degree for the polynomial kernel of the Support Vector Machine.

        Args:
            support_vector_machine_kernel_degree (int): New kernel degree.
        """
        logging.debug(f"Setting new SVM kernel degree: degree={support_vector_machine_kernel_degree}")
        self._support_vector_machine_kernel_degree = support_vector_machine_kernel_degree

    def set_support_vector_machine_gamma(self, support_vector_machine_gamma):
        """
        Sets a new gamma value for the Support Vector Machine.

        Args:
            support_vector_machine_gamma (str or float): New gamma value.
        """
        logging.debug(f"Setting new SVM gamma: gamma={support_vector_machine_gamma}")
        self._support_vector_machine_gamma = support_vector_machine_gamma

    def predict(self, model, x_samples):
        """
        Makes predictions using the trained Support Vector Machine model.

        Args:
            model (SVC): The trained Support Vector Machine classifier.
            x_samples (array-like): The samples for which to make predictions.

        Returns:
            array: Predictions for the provided samples.

        Raises:
            NotFittedError: If the model has not been trained yet.
            ValueError: If the input samples are empty or do not match the model's expected input shape.
        """
        try:
            if not isinstance(model, SVC):
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
