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

    from sklearn.naive_bayes import GaussianNB

except ImportError as error:
    print(error)
    sys.exit(-1)

class NaiveBayes:
    """
    Class that encapsulates the Naive Bayes classifier and its parameters.

    Attributes:
        _naive_bayes_priors (array-like, optional): Prior probabilities of the classes. If specified, it replaces the
                                                    uniform prior.
        _naive_bayes_variation_smoothing (float): Portion of the largest variance of all features added to variances
                                                  for numerical stability.
    """

    def __init__(self, arguments):
        """
        Initializes the Naive Bayes classifier with the provided hyperparameters.

        Args:
            arguments: Object containing the hyperparameters for the Naive Bayes model.
        """
        self._naive_bayes_priors = arguments.naive_bayes_priors
        self._naive_bayes_variation_smoothing = arguments.naive_bayes_variation_smoothing

        logging.debug(f"Naive Bayes initialized with priors={self._naive_bayes_priors}, "
                      f"var_smoothing={self._naive_bayes_variation_smoothing}")

    def get_model(self, x_samples_training, y_samples_training, dataset_type, input_dataset_shape):
        """
        Trains the Naive Bayes classifier using the provided training data.

        Args:
            x_samples_training (array-like): Feature samples for training.
            y_samples_training (array-like): Target values corresponding to the training samples.
            dataset_type: Data type to convert the samples into.
            input_dataset_shape: Expected shape of the input dataset (for logging purposes).

        Returns:
            GaussianNB: The trained Naive Bayes classifier.

        Raises:
            ValueError: If the training samples are empty or incompatible.
        """
        logging.info("Starting training classifier: NAIVE BAYES")

        try:
            # Convert the training samples and labels to numpy arrays
            logging.debug(f"Converting training samples and labels to numpy arrays with type {dataset_type}.")
            x_samples_training = numpy.array(x_samples_training, dtype=dataset_type)
            y_samples_training = numpy.array(y_samples_training, dtype=dataset_type)

            logging.debug(f"Training data shape: {x_samples_training.shape}, Labels shape: {y_samples_training.shape}")

            # Validate the training data
            if x_samples_training.size == 0 or y_samples_training.size == 0:
                raise ValueError("Training samples or labels are empty.")

            # Create and train the Naive Bayes classifier
            instance_model_classifier = GaussianNB(priors=self._naive_bayes_priors,
                                                   var_smoothing=self._naive_bayes_variation_smoothing)

            logging.info("Fitting the Naive Bayes classifier to the training data.")
            instance_model_classifier.fit(x_samples_training, y_samples_training)
            logging.info("Finished training Naive Bayes classifier.")

            return instance_model_classifier

        except ValueError as ve:
            logging.error(f"ValueError: {ve}")
            raise  # Re-raise the exception
        except Exception as e:
            logging.error(f"An error occurred during training: {e}")
            raise  # Re-raise the exception

    def set_naive_bayes_priors(self, naive_bayes_priors):
        """
        Sets the priors parameter for the Naive Bayes classifier.

        Args:
            naive_bayes_priors (array-like): Prior probabilities of the classes.
        """
        logging.debug(f"Setting new Naive Bayes priors: {naive_bayes_priors}")
        self._naive_bayes_priors = naive_bayes_priors

    def set_naive_bayes_variation_smoothing(self, naive_bayes_variation_smoothing):
        """
        Sets the variation smoothing parameter for the Naive Bayes classifier.

        Args:
            naive_bayes_variation_smoothing (float): Portion of the largest variance of all features added to variances
                                                     for numerical stability.
        """
        logging.debug(f"Setting new Naive Bayes var_smoothing: {naive_bayes_variation_smoothing}")
        self._naive_bayes_variation_smoothing = naive_bayes_variation_smoothing
