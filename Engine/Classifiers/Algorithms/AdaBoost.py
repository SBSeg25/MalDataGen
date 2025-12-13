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
    from sklearn.ensemble import AdaBoostClassifier

except ImportError as error:
    print(error)
    sys.exit(-1)

class AdaBoost:
    """
    An AdaBoost classifier wrapper that encapsulates the configuration and training process.

    Attributes:
        _ada_boost_estimator: Base estimator from which the boosted ensemble is built.
        _ada_boost_number_estimators (int): The maximum number of estimators at which boosting is terminated.
        _ada_boost_learning_rate (float): Learning rate shrinks the contribution of each classifier.
        _ada_boost_algorithm (str): Algorithm used for boosting ('SAMME' or 'SAMME.R').

    Methods:
        get_model(x_samples_training, y_samples_training, dataset_type, input_dataset_shape):
            Trains an AdaBoost model using the provided training samples and labels.
    """

    def __init__(self, arguments):
        """
        Initializes the AdaBoost class with hyperparameters.

        Args:
            arguments: An object containing hyperparameters for the AdaBoost model.
        """
        self._ada_boost_estimator = arguments.ada_boost_estimator
        self._ada_boost_number_estimators = arguments.ada_boost_number_estimators
        self._ada_boost_learning_rate = arguments.ada_boost_learning_rate
        self._ada_boost_algorithm = arguments.ada_boost_algorithm

        logging.debug(f"AdaBoost initialized with estimator={self._ada_boost_estimator}, "
                      f"n_estimators={self._ada_boost_number_estimators}, learning_rate={self._ada_boost_learning_rate}, "
                      f"algorithm={self._ada_boost_algorithm}")

    def get_model(self, x_samples_training, y_samples_training, dataset_type, input_dataset_shape):
        """
        Trains an AdaBoost classifier using the provided training samples and labels.

        Args:
            x_samples_training (array-like): The training feature samples.
            y_samples_training (array-like): The training labels corresponding to the samples.
            dataset_type: The data type for the training samples (e.g., float32).
            input_dataset_shape: The shape of the input dataset (used for logging purposes).

        Returns:
            AdaBoostClassifier: A trained AdaBoost classifier instance.

        Raises:
            ValueError: If training samples or labels are empty or do not match in shape.
            Exception: For any other issues encountered during model fitting.
        """
        logging.info("Starting training classifier: ADA BOOST")

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

            # Create and train the AdaBoost classifier
            instance_model_classifier = AdaBoostClassifier(
                estimator=self._ada_boost_estimator,
                algorithm=self._ada_boost_algorithm,
                n_estimators=self._ada_boost_number_estimators,
                learning_rate=self._ada_boost_learning_rate
            )

            logging.info("Fitting the AdaBoost model to the training data.")
            instance_model_classifier.fit(x_samples_training, y_samples_training)
            logging.info("Finished training ADA BOOST model.")
            return instance_model_classifier

        except ValueError as ve:
            logging.error(f"ValueError occurred: {ve}")
            raise  # Re-raise to propagate the exception
        except Exception as e:
            logging.error(f"An error occurred during model training: {e}")
            raise  # Re-raise to propagate the exception

    def set_ada_boost_estimator(self, ada_boost_estimator):
        """
        Sets the base estimator for the AdaBoost algorithm.

        Args:
            ada_boost_estimator: The base estimator (weak learner) to use.
        """
        logging.debug(f"Setting new AdaBoost estimator: {ada_boost_estimator}")
        self._ada_boost_estimator = ada_boost_estimator

    def set_ada_boost_number_estimators(self, ada_boost_number_estimators):
        """
        Sets the number of estimators (iterations) for AdaBoost.

        Args:
            ada_boost_number_estimators (int): The number of boosting stages.
        """
        logging.debug(f"Setting new number of AdaBoost estimators: {ada_boost_number_estimators}")
        self._ada_boost_number_estimators = ada_boost_number_estimators

    def set_ada_boost_learning_rate(self, ada_boost_learning_rate):
        """
        Sets the learning rate for the AdaBoost algorithm.

        Args:
            ada_boost_learning_rate (float): The learning rate for shrinking the contribution of each classifier.
        """
        logging.debug(f"Setting new AdaBoost learning rate: {ada_boost_learning_rate}")
        self._ada_boost_learning_rate = ada_boost_learning_rate

    def set_ada_boost_algorithm(self, ada_boost_algorithm):
        """
        Sets the boosting algorithm ('SAMME' or 'SAMME.R').

        Args:
            ada_boost_algorithm (str): The boosting algorithm to use.
        """
        logging.debug(f"Setting new AdaBoost algorithm: {ada_boost_algorithm}")
        self._ada_boost_algorithm = ada_boost_algorithm

    def predict(self, model, x_samples):
        """
        Makes predictions using the trained AdaBoost model.

        Args:
            model (AdaBoostClassifier): The trained AdaBoost classifier.
            x_samples (array-like): The samples for which to make predictions.

        Returns:
            array: Predictions for the provided samples.

        Raises:
            NotFittedError: If the model has not been trained yet.
            ValueError: If the input samples are empty or do not match the model's expected input shape.
        """
        try:
            if not isinstance(model, AdaBoostClassifier):
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
