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

    from sklearn.linear_model import LinearRegression

except ImportError as error:
    print(error)
    sys.exit(-1)


class LinearRegressionModel:
    """
    Class that encapsulates the Linear Regression model and its parameters.

    Attributes:
        _linear_regression_fit_intercept (bool): Whether to calculate the intercept for the model.
        _linear_regression_normalize (bool): This parameter is deprecated and has no effect in the latest versions.
        _linear_regression_copy_X (bool): Whether to copy X (input data) before fitting the model.
        _linear_regression_n_jobs (int): Number of jobs to use for the computation. -1 means using all processors.
    """

    def __init__(self, arguments):
        """
        Initializes the Linear Regression model with the provided hyperparameters.

        Args:
            arguments: Object containing the hyperparameters for the Linear Regression model.
        """
        self._linear_regression_fit_intercept = arguments.linear_regression_fit_intercept
        self._linear_regression_normalize = arguments.linear_regression_normalize  # Deprecated in newer versions
        self._linear_regression_copy_X = arguments.linear_regression_copy_X
        self._linear_regression_n_jobs = arguments.linear_regression_number_jobs

        logging.debug(f"Linear Regression initialized with fit_intercept={self._linear_regression_fit_intercept}, "
                      f"copy_X={self._linear_regression_copy_X}, n_jobs={self._linear_regression_n_jobs}")

    def get_model(self, x_samples_training, y_samples_training, dataset_type, input_dataset_shape):
        """
        Trains the Linear Regression model using the provided training data.

        Args:
            x_samples_training (array-like): Feature samples for training.
            y_samples_training (array-like): Target values corresponding to the training samples.
            dataset_type: Data type to convert the samples into.
            input_dataset_shape: Expected shape of the input dataset (for logging purposes).

        Returns:
            LinearRegression: The trained Linear Regression model.

        Raises:
            ValueError: If the training samples are empty or incompatible.
        """
        logging.info("Starting training model: LINEAR REGRESSION")

        try:
            # Convert the training samples and labels to numpy arrays
            logging.debug(f"Converting training samples and labels to numpy arrays with type {dataset_type}.")
            x_samples_training = numpy.array(x_samples_training, dtype=dataset_type)
            y_samples_training = numpy.array(y_samples_training, dtype=dataset_type)

            logging.debug(f"Training data shape: {x_samples_training.shape}, Labels shape: {y_samples_training.shape}")

            # Validate the training data
            if x_samples_training.size == 0 or y_samples_training.size == 0:
                raise ValueError("Training samples or labels are empty.")

            # Create and train the Linear Regression model
            instance_model_regressor = LinearRegression(
                fit_intercept=self._linear_regression_fit_intercept,
                copy_X=self._linear_regression_copy_X,
                n_jobs=self._linear_regression_n_jobs
            )

            logging.info("Fitting the Linear Regression model to the training data.")
            instance_model_regressor.fit(x_samples_training, y_samples_training)
            logging.info("Finished training Linear Regression model.")

            return instance_model_regressor

        except ValueError as ve:
            logging.error(f"ValueError: {ve}")
            raise  # Re-raise the exception
        except Exception as e:
            logging.error(f"An error occurred during training: {e}")
            raise  # Re-raise the exception

    def set_lr_fit_intercept(self, lr_fit_intercept):
        """
        Sets the fit_intercept parameter for the Linear Regression model.

        Args:
            lr_fit_intercept (bool): Whether to calculate the intercept for the model.
        """
        logging.debug(f"Setting new Linear Regression fit_intercept: {lr_fit_intercept}")
        self._linear_regression_fit_intercept = lr_fit_intercept

    def set_lr_normalize(self, lr_normalize):
        """
        Sets the normalize parameter for the Linear Regression model.
        (Note: The normalize parameter is deprecated in the latest versions of sklearn.)

        Args:
            lr_normalize (bool): Whether to normalize the regressors (deprecated).
        """
        logging.debug(f"Setting new Linear Regression normalize (deprecated): {lr_normalize}")
        self._linear_regression_normalize = lr_normalize

    def set_lr_copy_X(self, lr_copy_X):
        """
        Sets the copy_X parameter for the Linear Regression model.

        Args:
            lr_copy_X (bool): Whether to copy X (input data) before fitting the model.
        """
        logging.debug(f"Setting new Linear Regression copy_X: {lr_copy_X}")
        self._linear_regression_copy_X = lr_copy_X

    def set_lr_n_jobs(self, lr_n_jobs):
        """
        Sets the n_jobs parameter for the Linear Regression model.

        Args:
            lr_n_jobs (int): The number of jobs to use for the computation. -1 means using all processors.
        """
        logging.debug(f"Setting new Linear Regression n_jobs: {lr_n_jobs}")
        self._linear_regression_n_jobs = lr_n_jobs