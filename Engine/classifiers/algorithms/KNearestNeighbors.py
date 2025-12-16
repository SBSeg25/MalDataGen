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

    from sklearn.neighbors import KNeighborsClassifier

except ImportError as error:
    print(error)
    sys.exit(-1)


class KNearestNeighbors:
    """
    Class that encapsulates the K-Nearest Neighbors (KNN) classifier model and its parameters.

    Attributes:
        _knn_number_neighbors (int): Number of neighbors to use.
        _knn_weights (str): Weight function used in prediction ('uniform' or 'distance').
        _knn_leaf_size (int): Leaf size for BallTree or KDTree.
        _knn_metric (str): The distance metric to use.
        _knn_algorithm (str): Algorithm used to compute the nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute').
    """

    def __init__(self, arguments):
        """
        Initializes the KNN classifier with the provided hyperparameters.

        Args:
            arguments: Object containing the hyperparameters for the KNN classifier.
        """
        self._knn_number_neighbors = arguments.knn_number_neighbors
        self._knn_weights = arguments.knn_weights
        self._knn_leaf_size = arguments.knn_leaf_size
        self._knn_metric = arguments.knn_metric
        self._knn_algorithm = arguments.knn_algorithm

        logging.debug(f"KNN initialized with n_neighbors={self._knn_number_neighbors}, "
                      f"weights={self._knn_weights}, leaf_size={self._knn_leaf_size}, "
                      f"metric={self._knn_metric}, algorithm={self._knn_algorithm}")

    def get_model(self, x_samples_training, y_samples_training, dataset_type, input_dataset_shape):
        """
        Trains the KNN classifier model using the provided training data.

        Args:
            x_samples_training (array-like): Feature samples for training.
            y_samples_training (array-like): Labels corresponding to the training samples.
            dataset_type: Data type to convert the samples into.
            input_dataset_shape: Expected shape of the input dataset (for logging purposes).

        Returns:
            KNeighborsClassifier: The trained KNN classifier model.

        Raises:
            ValueError: If the training samples are empty or incompatible.
        """
        logging.info("Starting training classifier: K-NEAREST NEIGHBORS")

        try:
            # Convert the training samples and labels to numpy arrays
            logging.debug(f"Converting training samples and labels to numpy arrays with type {dataset_type}.")
            x_samples_training = numpy.array(x_samples_training, dtype=dataset_type)
            y_samples_training = numpy.array(y_samples_training, dtype=dataset_type)

            logging.debug(f"Training data shape: {x_samples_training.shape}, Labels shape: {y_samples_training.shape}")

            # Validate the training data
            if x_samples_training.size == 0 or y_samples_training.size == 0:
                raise ValueError("Training samples or labels are empty.")

            # Create and train the KNN classifier model
            instance_model_classifier = KNeighborsClassifier(
                n_neighbors=self._knn_number_neighbors,
                weights=self._knn_weights,
                algorithm=self._knn_algorithm,
                leaf_size=self._knn_leaf_size,
                metric=self._knn_metric
            )

            logging.info("Fitting the KNN model to the training data.")
            instance_model_classifier.fit(x_samples_training, y_samples_training)
            logging.info("Finished training KNN model.")

            return instance_model_classifier

        except ValueError as ve:
            logging.error(f"ValueError: {ve}")
            raise  # Re-raise the exception
        except Exception as e:
            logging.error(f"An error occurred during training: {e}")
            raise  # Re-raise the exception

    def set_knn_number_neighbors(self, knn_number_neighbors):
        """
        Sets the number of neighbors for the KNN classifier.

        Args:
            knn_number_neighbors (int): The new number of neighbors.
        """
        logging.debug(f"Setting new KNN number of neighbors: {knn_number_neighbors}")
        self._knn_number_neighbors = knn_number_neighbors

    def set_knn_weights(self, knn_weights):
        """
        Sets the weight function used in prediction for the KNN classifier.

        Args:
            knn_weights (str): The new weight function ('uniform' or 'distance').
        """
        logging.debug(f"Setting new KNN weights: {knn_weights}")
        self._knn_weights = knn_weights

    def set_knn_leaf_size(self, knn_leaf_size):
        """
        Sets the leaf size for the BallTree or KDTree in the KNN classifier.

        Args:
            knn_leaf_size (int): The new leaf size.
        """
        logging.debug(f"Setting new KNN leaf size: {knn_leaf_size}")
        self._knn_leaf_size = knn_leaf_size

    def set_knn_metric(self, knn_metric):
        """
        Sets the distance metric used in the KNN classifier.

        Args:
            knn_metric (str): The new distance metric.
        """
        logging.debug(f"Setting new KNN metric: {knn_metric}")
        self._knn_metric = knn_metric

    def set_knn_algorithm(self, knn_algorithm):
        """
        Sets the algorithm used to compute the nearest neighbors in the KNN classifier.

        Args:
            knn_algorithm (str): The new algorithm ('auto', 'ball_tree', 'kd_tree', 'brute').
        """
        logging.debug(f"Setting new KNN algorithm: {knn_algorithm}")
        self._knn_algorithm = knn_algorithm

