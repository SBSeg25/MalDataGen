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

    from Engine.Classifiers.Algorithms.PerceptronModel import PerceptronMultilayer

except ImportError as error:
    print(error)
    sys.exit(-1)

class Perceptron:
    """
    Class that encapsulates the Multilayer Perceptron (MLP) classifier and its hyperparameters.

    Attributes:
        Various hyperparameters related to the MLP architecture, activation functions, training algorithms, etc.
    """

    def __init__(self, arguments):
        """
        Initializes the MLP classifier with the provided hyperparameters.

        Args:
            arguments: Object containing the hyperparameters for the MLP model.
        """
        self._perceptron_training_algorithm = arguments.perceptron_training_algorithm
        self._perceptron_training_loss = arguments.perceptron_training_loss
        self._perceptron_layer_activation = arguments.perceptron_layer_activation
        self._perceptron_last_layer_activation = arguments.perceptron_last_layer_activation
        self._perceptron_dropout_decay_rate = arguments.perceptron_dropout_decay_rate
        self._perceptron_number_epochs = arguments.perceptron_number_epochs
        self._perceptron_training_metric = arguments.perceptron_training_metric
        self._perceptron_layers_settings = arguments.perceptron_layers_settings

        logging.debug(f"Perceptron initialized with algorithm={self._perceptron_training_algorithm}, "
                      f"loss={self._perceptron_training_loss}, activation={self._perceptron_layer_activation}, "
                      f"epochs={self._perceptron_number_epochs}, metric={self._perceptron_training_metric}, "
                      f"dropout={self._perceptron_dropout_decay_rate}")

    def get_model(self, x_samples_training, y_samples_training, dataset_type, input_dataset_shape):
        """
        Trains the MLP classifier using the provided training data.

        Args:
            x_samples_training (array-like): Feature samples for training.
            y_samples_training (array-like): Target values corresponding to the training samples.
            dataset_type: Data type to convert the samples into.
            input_dataset_shape: Expected shape of the input dataset.

        Returns:
            model_classifier: The trained MLP classifier.

        Raises:
            ValueError: If the training samples or labels are empty or incompatible.
        """
        logging.info("Starting training classifier: MULTILAYER PERCEPTRON")

        try:
            # Convert training samples and labels to numpy arrays
            logging.debug(f"Converting training samples and labels to numpy arrays with type {dataset_type}.")
            x_samples_training = numpy.array(x_samples_training, dtype=dataset_type)
            y_samples_training = numpy.array(y_samples_training, dtype=dataset_type)

            logging.debug(f"Training data shape: {x_samples_training.shape}, Labels shape: {y_samples_training.shape}")

            # Validate training data
            if x_samples_training.size == 0 or y_samples_training.size == 0:
                raise ValueError("Training samples or labels are empty.")

            # Initialize and configure the MLP model
            instance_model_classifier = PerceptronMultilayer(self._perceptron_layers_settings,
                                                         self._perceptron_training_metric,
                                                         self._perceptron_training_loss,
                                                         self._perceptron_training_algorithm,
                                                         dataset_type, self._perceptron_layer_activation,
                                                         self._perceptron_last_layer_activation,
                                                         self._perceptron_dropout_decay_rate)

            logging.info("Building the MLP model.")
            model_classifier = instance_model_classifier.get_model(input_dataset_shape)

            # Train the model
            logging.info("Fitting the MLP model to the training data.")
            model_classifier.fit(x_samples_training, y_samples_training, epochs=self._perceptron_number_epochs, verbose=0)
            logging.info("Finished training Multilayer Perceptron classifier.")

            return model_classifier

        except ValueError as ve:
            logging.error(f"ValueError: {ve}")
            raise  # Re-raise the exception
        except Exception as e:
            logging.error(f"An error occurred during training: {e}")
            raise  # Re-raise the exception

    def set_perceptron_training_algorithm(self, perceptron_training_algorithm):
        """
        Sets the training algorithm for the MLP classifier.

        Args:
            perceptron_training_algorithm (str): The training algorithm to be used.
        """
        logging.debug(f"Setting new Perceptron training algorithm: {perceptron_training_algorithm}")
        self._perceptron_training_algorithm = perceptron_training_algorithm

    def set_perceptron_training_loss(self, perceptron_training_loss):
        """
        Sets the training loss function for the MLP classifier.

        Args:
            perceptron_training_loss (str): The loss function to be used.
        """
        logging.debug(f"Setting new Perceptron training loss: {perceptron_training_loss}")
        self._perceptron_training_loss = perceptron_training_loss

    def set_perceptron_layer_activation(self, perceptron_layer_activation):
        """
        Sets the activation function for the MLP classifier layers.

        Args:
            perceptron_layer_activation (str): The activation function for hidden layers.
        """
        logging.debug(f"Setting new Perceptron layer activation: {perceptron_layer_activation}")
        self._perceptron_layer_activation = perceptron_layer_activation


# class Perceptron:
#
#     def __init__(self, arguments):
#
#         self._perceptron_training_algorithm = arguments.perceptron_training_algorithm
#         self._perceptron_training_loss = arguments.perceptron_training_loss
#         self._perceptron_layer_activation = arguments.perceptron_layer_activation
#         self._perceptron_last_layer_activation = arguments.perceptron_last_layer_activation
#         self._perceptron_dropout_decay_rate = arguments.perceptron_dropout_decay_rate
#         self._perceptron_number_epochs = arguments.perceptron_number_epochs
#         self._perceptron_training_metric = arguments.perceptron_training_metric
#         self._perceptron_layers_settings = arguments.perceptron_layers_settings
#
#     def get_model(self, x_samples_training, y_samples_training, dataset_type, input_dataset_shape):
#         logging.info("    Starting training classifier: MULTILAYER PERCEPTRON")
#
#         x_samples_training = numpy.array(x_samples_training, dtype=dataset_type)
#         y_samples_training = numpy.array(y_samples_training, dtype=dataset_type)
#         instance_model_classifier = PerceptronMultilayer(self._perceptron_layers_settings,
#                                                          self._perceptron_training_metric,
#                                                          self._perceptron_training_loss,
#                                                          self._perceptron_training_algorithm,
#                                                          dataset_type, self._perceptron_layer_activation,
#                                                          self._perceptron_last_layer_activation,
#                                                          self._perceptron_dropout_decay_rate)
#         model_classifier = instance_model_classifier.get_model(input_dataset_shape)
#         model_classifier.fit(x_samples_training, y_samples_training, epochs=self._perceptron_number_epochs, verbose=0)
#         logging.info("\r    Finished training\n")
#
#         return model_classifier
#
#     def set_perceptron_training_algorithm(self, perceptron_training_algorithm):
#         self._perceptron_training_algorithm = perceptron_training_algorithm
#
#     def set_perceptron_training_loss(self, perceptron_training_loss):
#         self._perceptron_training_loss = perceptron_training_loss
#
#     def set_perceptron_layer_activation(self, perceptron_layer_activation):
#         self._perceptron_layer_activation = perceptron_layer_activation
#
