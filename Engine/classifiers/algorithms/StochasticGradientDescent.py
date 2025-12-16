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

    from sklearn.linear_model import SGDClassifier

except ImportError as error:
    print(error)
    sys.exit(-1)


class StochasticGradientDescent:
    """
    Class that encapsulates the training process of a Stochastic Gradient Descent (SGD) classifier.

    Attributes:
        _stochastic_gradient_descent_loss (str): The loss function to be used in SGD ('hinge', 'log', etc.).
        _stochastic_gradient_descent_penalty (str): The penalty (regularization term) to be used ('l2', 'l1', or 'elasticnet').
        _stochastic_gradient_descent_alpha (float): The regularization strength (constant that multiplies the regularization term).
        _stochastic_gradient_descent_max_iterations (int): Maximum number of iterations over the training data.
        _stochastic_gradient_descent_tol (float): Tolerance for stopping criteria.
    """

    def __init__(self, arguments):
        """
        Initializes the Stochastic Gradient Descent classifier with the provided hyperparameters.

        Args:
            arguments: Object containing the hyperparameters for the SGD model.
        """
        self._stochastic_gradient_descent_loss = arguments.stochastic_gradient_descent_loss
        self._stochastic_gradient_descent_penalty = arguments.stochastic_gradient_descent_penalty
        self._stochastic_gradient_descent_alpha = arguments.stochastic_gradient_descent_alpha
        self._stochastic_gradient_descent_max_iterations = arguments.stochastic_gradient_descent_max_iterations
        self._stochastic_gradient_descent_tol = arguments.stochastic_gradient_descent_tolerance

        logging.debug(f"SGD initialized with loss={self._stochastic_gradient_descent_loss}, "
                      f"penalty={self._stochastic_gradient_descent_penalty},"
                      f" alpha={self._stochastic_gradient_descent_alpha}, "
                      f"max_iter={self._stochastic_gradient_descent_max_iterations},"
                      f" tol={self._stochastic_gradient_descent_tol}")

    def get_model(self, x_samples_training, y_samples_training, dataset_type, input_dataset_shape):
        """
        Trains the SGD classifier using the provided training data.

        Args:
            x_samples_training (array-like): Feature samples for training.
            y_samples_training (array-like): Labels corresponding to the training samples.
            dataset_type: Data type to convert the samples into.
            input_dataset_shape: Expected shape of the input dataset (for logging purposes).

        Returns:
            SGDClassifier: The trained SGD classifier.

        Raises:
            ValueError: If the training samples or labels are empty or incompatible.
        """
        logging.info("Starting training classifier: STOCHASTIC GRADIENT DESCENT")

        try:
            # Convert the training samples and labels to numpy arrays
            logging.debug(f"Converting training samples to numpy arrays with type {dataset_type}.")
            x_samples_training = numpy.array(x_samples_training, dtype=dataset_type)
            y_samples_training = numpy.array(y_samples_training, dtype=dataset_type)

            logging.debug(f"Training data shape: {x_samples_training.shape}, Labels shape: {y_samples_training.shape}")

            # Validate the training data
            if x_samples_training.size == 0 or y_samples_training.size == 0:
                raise ValueError("Training samples or labels are empty.")
            if x_samples_training.shape[0] != y_samples_training.shape[0]:
                raise ValueError("Mismatch between the number of samples and labels.")

            # Create and train the SGD classifier
            instance_model_classifier = SGDClassifier(
                loss=self._stochastic_gradient_descent_loss,
                penalty=self._stochastic_gradient_descent_penalty,
                alpha=self._stochastic_gradient_descent_alpha,
                max_iter=self._stochastic_gradient_descent_max_iterations,
                tol=self._stochastic_gradient_descent_tol
            )

            logging.info("Fitting the SGD model to the training data.")
            instance_model_classifier.fit(x_samples_training, y_samples_training)
            logging.info("Finished training the SGD model.")

            return instance_model_classifier

        except ValueError as ve:
            logging.error(f"ValueError: {ve}")
            raise  # Re-raise the exception
        except Exception as e:
            logging.error(f"An error occurred during training: {e}")
            raise  # Re-raise the exception

    def set_sgd_loss(self, sgd_loss):
        """
        Sets the loss function for the SGD classifier.

        Args:
            sgd_loss (str): The new loss function ('hinge', 'log', etc.).
        """
        logging.debug(f"Setting new SGD loss function: {sgd_loss}")
        self._stochastic_gradient_descent_loss = sgd_loss

    def set_sgd_penalty(self, sgd_penalty):
        """
        Sets the penalty (regularization term) for the SGD classifier.

        Args:
            sgd_penalty (str): The new penalty ('l2', 'l1', or 'elasticnet').
        """
        logging.debug(f"Setting new SGD penalty: {sgd_penalty}")
        self._stochastic_gradient_descent_penalty = sgd_penalty

    def set_sgd_alpha(self, sgd_alpha):
        """
        Sets the regularization strength for the SGD classifier.

        Args:
            sgd_alpha (float): The new regularization strength.
        """
        logging.debug(f"Setting new SGD alpha (regularization strength): {sgd_alpha}")
        self._stochastic_gradient_descent_alpha = sgd_alpha

    def set_sgd_max_iter(self, sgd_max_iter):
        """
        Sets the maximum number of iterations for the SGD classifier.

        Args:
            sgd_max_iter (int): The new maximum number of iterations.
        """
        logging.debug(f"Setting new SGD max iterations: {sgd_max_iter}")
        self._stochastic_gradient_descent_max_iterations = sgd_max_iter

    def set_sgd_tol(self, sgd_tol):
        """
        Sets the tolerance for stopping criteria for the SGD classifier.

        Args:
            sgd_tol (float): The new tolerance for stopping criteria.
        """
        logging.debug(f"Setting new SGD tolerance: {sgd_tol}")
        self._stochastic_gradient_descent_tol = sgd_tol