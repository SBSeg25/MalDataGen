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

    from sklearn.cluster import KMeans

except ImportError as error:
    print(error)
    sys.exit(-1)

class KMeansClustering:
    """
    Class that encapsulates the KMeans clustering model and its parameters.

    Attributes:
        _k_means_n_clusters (int): The number of clusters to form.
        _k_means_init (str): Method for initialization ('k-means++' or 'random').
        _k_means_max_iterations (int): Maximum number of iterations for the algorithm.
        _k_means_tol (float): Tolerance to declare convergence.
        _k_means_random_state (int): Seed for random number generation.
    """

    def __init__(self, arguments):
        """
        Initializes the KMeans clustering model with the provided hyperparameters.

        Args:
            arguments: Object containing the hyperparameters for the KMeans clustering model.
        """
        self._k_means_n_clusters = arguments.k_means_number_clusters
        self._k_means_init = arguments.k_means_init
        self._k_means_max_iterations = arguments.k_means_max_iterations
        self._k_means_tol = arguments.k_means_tolerance
        self._k_means_random_state = arguments.k_means_random_state

        logging.debug(f"KMeans initialized with n_clusters={self._k_means_n_clusters}, "
                      f"init={self._k_means_init}, max_iter={self._k_means_max_iterations}, "
                      f"tol={self._k_means_tol}, random_state={self._k_means_random_state}")

    def get_model(self, x_samples_training, y_samples_training, dataset_type, input_dataset_shape):
        """
        Trains the KMeans clustering model using the provided training data.

        Args:
            x_samples_training (array-like): Feature samples for training.
            y_samples_training (array-like): Labels corresponding to the training samples (not used in clustering).
            dataset_type: Data type to convert the samples into.
            input_dataset_shape: Expected shape of the input dataset (for logging purposes).

        Returns:
            KMeans: The trained KMeans clustering model.

        Raises:
            ValueError: If the training samples are empty or incompatible.
        """
        logging.info("Starting clustering: K-MEANS")

        try:
            # Convert the training samples to a numpy array
            logging.debug(f"Converting training samples to numpy arrays with type {dataset_type}.")
            x_samples_training = numpy.array(x_samples_training, dtype=numpy.float64)

            logging.debug(f"Training data shape: {x_samples_training.shape}")

            # Validate the training data
            if x_samples_training.size == 0:
                raise ValueError("Training samples are empty.")

            # Create and train the KMeans clustering model
            instance_model_clustering = KMeans(
                n_clusters=self._k_means_n_clusters,
                init=self._k_means_init,
                max_iter=self._k_means_max_iterations,
                tol=self._k_means_tol,
                random_state=self._k_means_random_state
            )

            logging.info("Fitting the KMeans model to the training data.")
            instance_model_clustering.fit(x_samples_training)
            logging.info("Finished clustering.")

            return instance_model_clustering

        except ValueError as ve:
            logging.error(f"ValueError: {ve}")
            raise  # Re-raise the exception
        except Exception as e:
            logging.error(f"An error occurred during clustering: {e}")
            raise  # Re-raise the exception

    def set_kmeans_n_clusters(self, kmeans_n_clusters):
        """
        Sets the number of clusters for the KMeans clustering model.

        Args:
            kmeans_n_clusters (int): The new number of clusters.
        """
        logging.debug(f"Setting new KMeans number of clusters: {kmeans_n_clusters}")
        self._k_means_n_clusters = kmeans_n_clusters

    def set_kmeans_init(self, kmeans_init):
        """
        Sets the initialization method for the KMeans clustering model.

        Args:
            kmeans_init (str): The new initialization method ('k-means++' or 'random').
        """
        logging.debug(f"Setting new KMeans initialization method: {kmeans_init}")
        self._k_means_init = kmeans_init

    def set_kmeans_max_iter(self, kmeans_max_iter):
        """
        Sets the maximum number of iterations for the KMeans clustering model.

        Args:
            kmeans_max_iter (int): The new maximum number of iterations.
        """
        logging.debug(f"Setting new KMeans max iterations: {kmeans_max_iter}")
        self._k_means_max_iterations = kmeans_max_iter

    def set_kmeans_tol(self, kmeans_tol):
        """
        Sets the convergence tolerance for the KMeans clustering model.

        Args:
            kmeans_tol (float): The new tolerance value.
        """
        logging.debug(f"Setting new KMeans tolerance: {kmeans_tol}")
        self._k_means_tol = kmeans_tol

    def set_kmeans_random_state(self, kmeans_random_state):
        """
        Sets the random state for the KMeans clustering model.

        Args:
            kmeans_random_state (int): The new random state.
        """
        logging.debug(f"Setting new KMeans random state: {kmeans_random_state}")
        self._k_means_random_state = kmeans_random_state