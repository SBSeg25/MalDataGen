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
    from sklearn.cluster import SpectralClustering

except ImportError as error:
    print(error)
    sys.exit(-1)

class SpectralClusteringModel:
    """
    Class that encapsulates the spectral clustering model and its parameters.

    Attributes:
        _spectral_number_clusters (int): The number of clusters to form.
        _spectral_eigen_solver (str): The eigenvalue solver to use ('arpack', 'lobpcg', etc.).
        _spectral_affinity (str): The affinity type ('nearest_neighbors', 'precomputed', etc.).
        _spectral_assign_labels (str): The strategy for assigning labels ('kmeans', 'discretize').
        _spectral_random_state (int): Seed used by the random number generator.
    """

    def __init__(self, arguments):
        """
        Initializes the Spectral Clustering model with the provided hyperparameters.

        Args:
            arguments: Object containing the hyperparameters for the spectral clustering model.
        """
        self._spectral_number_clusters = arguments.spectral_number_clusters
        self._spectral_eigen_solver = arguments.spectral_eigen_solver
        self._spectral_affinity = arguments.spectral_affinity
        self._spectral_assign_labels = arguments.spectral_assign_labels
        self._spectral_random_state = arguments.spectral_random_state

        logging.debug(f"Spectral Clustering initialized with number_clusters={self._spectral_number_clusters}, "
                      f"eigen_solver={self._spectral_eigen_solver}, affinity={self._spectral_affinity}, "
                      f"assign_labels={self._spectral_assign_labels}, random_state={self._spectral_random_state}")

    def get_model(self, x_samples_training, y_samples_training, dataset_type, input_dataset_shape):
        """
        Trains the spectral clustering model using the provided training data.

        Args:
            x_samples_training (array-like): Feature samples for training.
            y_samples_training (array-like): Labels corresponding to the training samples (not used in clustering).
            dataset_type: Data type to convert the samples into.
            input_dataset_shape: Expected shape of the input dataset (for logging purposes).

        Returns:
            SpectralClustering: The trained spectral clustering model.

        Raises:
            ValueError: If the training samples are empty or incompatible.
        """
        logging.info("Starting clustering: SPECTRAL CLUSTERING")

        try:
            # Convert the training samples to a numpy array
            logging.debug(f"Converting training samples to numpy arrays with type {dataset_type}.")
            x_samples_training = numpy.array(x_samples_training, dtype=dataset_type)

            logging.debug(f"Training data shape: {x_samples_training.shape}")

            # Validate the training data
            if x_samples_training.size == 0:
                raise ValueError("Training samples are empty.")

            # Create and train the Spectral Clustering model
            instance_model_clustering = SpectralClustering(
                n_clusters=self._spectral_number_clusters,
                eigen_solver=self._spectral_eigen_solver,
                affinity=self._spectral_affinity,
                assign_labels=self._spectral_assign_labels,
                random_state=self._spectral_random_state
            )

            logging.info("Fitting the Spectral Clustering model to the training data.")
            instance_model_clustering.fit(x_samples_training)
            logging.info("Finished clustering.")

            return instance_model_clustering

        except ValueError as ve:
            logging.error(f"ValueError: {ve}")
            raise  # Re-raise the exception
        except Exception as e:
            logging.error(f"An error occurred during clustering: {e}")
            raise  # Re-raise the exception

    def set_spectral_n_clusters(self, spectral_n_clusters):
        """
        Sets the number of clusters for the spectral clustering model.

        Args:
            spectral_n_clusters (int): The new number of clusters.
        """
        logging.debug(f"Setting new spectral number of clusters: {spectral_n_clusters}")
        self._spectral_number_clusters = spectral_n_clusters

    def set_spectral_eigen_solver(self, spectral_eigen_solver):
        """
        Sets the eigenvalue solver for the spectral clustering model.

        Args:
            spectral_eigen_solver (str): The new eigenvalue solver.
        """
        logging.debug(f"Setting new spectral eigen solver: {spectral_eigen_solver}")
        self._spectral_eigen_solver = spectral_eigen_solver

    def set_spectral_affinity(self, spectral_affinity):
        """
        Sets the affinity type for the spectral clustering model.

        Args:
            spectral_affinity (str): The new affinity type.
        """
        logging.debug(f"Setting new spectral affinity: {spectral_affinity}")
        self._spectral_affinity = spectral_affinity

    def set_spectral_assign_labels(self, spectral_assign_labels):
        """
        Sets the label assignment strategy for the spectral clustering model.

        Args:
            spectral_assign_labels (str): The new label assignment strategy.
        """
        logging.debug(f"Setting new spectral assign labels: {spectral_assign_labels}")
        self._spectral_assign_labels = spectral_assign_labels

    def set_spectral_random_state(self, spectral_random_state):
        """
        Sets the random state for the spectral clustering model.

        Args:
            spectral_random_state (int): The new random state.
        """
        logging.debug(f"Setting new spectral random state: {spectral_random_state}")
        self._spectral_random_state = spectral_random_state