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

    from sklearn.ensemble import GradientBoostingClassifier

except ImportError as error:
    print(error)
    sys.exit(-1)

class GradientBoosting:
    """
    Class that encapsulates the process of training a Gradient Boosting classifier.

    Attributes:
        _gradient_boosting_loss (str): The loss function to be used in boosting (default is 'deviance').
        _gradient_boosting_learning_rate (float): Learning rate.
        _gradient_boosting_number_estimators (int): Number of estimators.
        _gradient_boosting_subsample (float): The fraction of samples to be used to fit each base estimator.
        _gradient_boosting_criterion (str): Criterion to measure the quality of a split.
    """

    def __init__(self, arguments):
        """
        Initializes the Gradient Boosting model with the provided hyperparameters.

        Args:
            arguments: Object containing the hyperparameters for the Gradient Boosting model.
        """
        self._gradient_boosting_loss = arguments.gradient_boosting_loss
        self._gradient_boosting_learning_rate = arguments.gradient_boosting_learning_rate
        self._gradient_boosting_number_estimators = arguments.gradient_boosting_number_estimators
        self._gradient_boosting_subsample = arguments.gradient_boosting_subsample
        self._gradient_boosting_criterion = arguments.gradient_boosting_criterion

        logging.debug(f"GradientBoosting initialized with loss={self._gradient_boosting_loss}, "
                      f"learning_rate={self._gradient_boosting_learning_rate}, "
                      f"n_estimators={self._gradient_boosting_number_estimators}, "
                      f"subsample={self._gradient_boosting_subsample}, criterion={self._gradient_boosting_criterion}")

    def get_model(self, x_samples_training, y_samples_training, dataset_type, input_dataset_shape):
        """
        Trains the Gradient Boosting classifier using the provided training data.

        Args:
            x_samples_training (array-like): Feature samples for training.
            y_samples_training (array-like): Labels corresponding to the training samples.
            dataset_type: Data type to convert the samples.
            input_dataset_shape: Expected shape of the input dataset (for logging purposes).

        Returns:
            GradientBoostingClassifier: Trained model.

        Raises:
            ValueError: If the training samples or labels are empty or have incompatible shapes.
        """
        logging.info("Starting training classifier: GRADIENT BOOSTING")

        try:
            # Convert the input data to the specified type
            logging.debug(f"Converting training samples to numpy arrays with type {dataset_type}.")
            x_samples_training = numpy.array(x_samples_training, dtype=dataset_type)
            y_samples_training = numpy.array(y_samples_training, dtype=dataset_type)

            logging.debug(
                f"Training data shape: {x_samples_training.shape}, Labels shape: {y_samples_training.shape}")

            # Validate the training data
            if x_samples_training.size == 0 or y_samples_training.size == 0:
                raise ValueError("The training samples or labels are empty.")
            if x_samples_training.shape[0] != y_samples_training.shape[0]:
                raise ValueError("The number of samples and labels do not match.")

            # Create and train the Gradient Boosting classifier
            instance_model_classifier = GradientBoostingClassifier(
                loss=self._gradient_boosting_loss,
                learning_rate=self._gradient_boosting_learning_rate,
                n_estimators=self._gradient_boosting_number_estimators,
                subsample=self._gradient_boosting_subsample,
                criterion=self._gradient_boosting_criterion
            )

            logging.info("Fitting the Gradient Boosting model to the training data.")
            instance_model_classifier.fit(x_samples_training, y_samples_training)
            logging.info("Gradient Boosting model training completed.")
            return instance_model_classifier

        except ValueError as ve:
            logging.error(f"Value error: {ve}")
            raise  # Re-raise to propagate the exception
        except Exception as e:
            logging.error(f"An error occurred during training: {e}")
            raise  # Re-raise to propagate the exception

    def set_gradient_boosting_loss(self, gb_loss):
        """
        Sets the loss function for Gradient Boosting.

        Args:
            gb_loss (str): Loss function ('deviance' or 'exponential').
        """
        logging.debug(f"Setting new loss function: {gb_loss}")
        self._gradient_boosting_loss = gb_loss

    def set_gradient_boosting_learning_rate(self, gb_learning_rate):
        """
        Sets the learning rate for Gradient Boosting.

        Args:
            gb_learning_rate (float): The new learning rate.
        """
        logging.debug(f"Setting new learning rate: {gb_learning_rate}")
        self._gradient_boosting_learning_rate = gb_learning_rate

    def set_gradient_boosting_number_estimators(self, gb_n_estimators):
        """
        Sets the number of estimators for Gradient Boosting.

        Args:
            gb_n_estimators (int): The number of boosting stages to perform.
        """
        logging.debug(f"Setting new number of estimators: {gb_n_estimators}")
        self._gradient_boosting_number_estimators = gb_n_estimators

    def set_gradient_boosting_subsample(self, gb_subsample):
        """
        Sets the fraction of samples to be used to fit each base estimator.

        Args:
            gb_subsample (float): The new fraction of samples.
        """
        logging.debug(f"Setting new subsample fraction: {gb_subsample}")
        self._gradient_boosting_subsample = gb_subsample

    def set_gradient_boosting_criterion(self, gb_criterion):
        """
        Sets the criterion to measure the quality of a split.

        Args:
            gb_criterion (str): The new criterion ('friedman_mse', 'mse', 'mae').
        """
        logging.debug(f"Setting new criterion: {gb_criterion}")
        self._gradient_boosting_criterion = gb_criterion