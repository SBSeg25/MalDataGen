#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

from Engine.algorithms.smote.AlgorithmSMOTE import SMOTEAlgorithm

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
    import tensorflow

    from tensorflow.keras.optimizers import Adam

    from tensorflow.keras.utils import to_categorical

    from tensorflow.python.keras.losses import MeanSquaredError
    from Engine.callbacks.CallbackEarlyStop import EarlyStopping

    from tensorflow.python.keras.losses import BinaryCrossentropy


except ImportError as error:
    logging.error(error)
    sys.exit(-1)



class Smote:
    """
    A class that implements the Synthetic Minority Over-sampling Technique (smote) for handling
    class imbalance in datasets. smote generates synthetic samples for minority classes by
    interpolating between existing instances, effectively balancing the class distribution.

    Key Components:
    - smote algorithm implementation for synthetic sample generation
    - Configurable neighborhood size for interpolation
    - Flexible sampling strategy for target class distribution
    - Random state control for reproducibility

    Attributes:
        _smote_algorithm: The core smote algorithm instance
        _smote_sampling_strategy: Target sampling strategy for class balancing
        _smote_random_state: Seed for random number generation
        _smote_k_neighbors: Number of nearest neighbors to consider for interpolation
    """
    def __init__(self, arguments):
        """
        Initializes the smote instance with configuration parameters.

        Args:
            arguments (Namespace): Configuration object containing:
                - smote_sampling_strategy: Target class distribution strategy
                - smote_random_state: Random seed for reproducibility
                - smote_k_neighbors: Number of neighbors for synthetic sample generation
        """
        self._smote_algorithm = None

        # smote Configuration Parameters
        self._smote_sampling_strategy = arguments.smote_sampling_strategy
        self._smote_random_state = arguments.smote_random_state
        self._smote_k_neighbors = arguments.smote_k_neighbors


    def _get_smote(self, input_shape):
        """
        Initializes and configures the smote algorithm with the specified parameters.

        This method creates an instance of the SMOTEAlgorithm with the configured:
        - Sampling strategy for target class distribution
        - Random state for reproducible results
        - Number of nearest neighbors for synthetic sample generation

        Args:
            input_shape (tuple): The shape of the input data (unused in smote but kept for interface consistency)

        Initializes:
            self._smote_algorithm (SMOTEAlgorithm): The configured smote algorithm instance
        """
        self._smote_algorithm = SMOTEAlgorithm(sampling_strategy = self._smote_sampling_strategy,
                                               random_state = self._smote_random_state,
                                               k_neighbors = self._smote_k_neighbors)


    def fit_model(self, input_shape, x_real_samples, y_real_samples):
        """
        Executes the smote training process to generate synthetic samples.

        The training process:
        1. Initializes the smote algorithm with configured parameters
        2. Fits the smote model to the input data
        3. Generates synthetic samples for minority classes

        Args:
            input_shape (tuple): Input data dimensions (unused but kept for interface consistency)
            arguments (Namespace): Training configuration (unused in this implementation)
            x_real_samples (ndarray): Original feature vectors
            y_real_samples (ndarray): Corresponding class labels

        Note:
            The method converts labels to categorical format internally to handle multi-class scenarios.
        """
        # Initialize the autoencoder model
        self._get_smote(input_shape)

        # Fit the autoencoder model
        self._smote_algorithm.fit(x_real_samples,
                                  to_categorical(y_real_samples,
                                                 num_classes=self._number_samples_per_class["number_classes"]))

    @property
    def smote_sampling_strategy(self):
        """Get the smote sampling strategy."""
        return self._smote_sampling_strategy

    @smote_sampling_strategy.setter
    def smote_sampling_strategy(self, value):
        """Set the smote sampling strategy."""
        self._smote_sampling_strategy = value

    @property
    def smote_random_state(self):
        """Get the smote random state."""
        return self._smote_random_state

    @smote_random_state.setter
    def smote_random_state(self, value):
        """Set the smote random state."""
        self._smote_random_state = value

    @property
    def smote_k_neighbors(self):
        """Get the smote k-neighbors value."""
        return self._smote_k_neighbors

    @smote_k_neighbors.setter
    def smote_k_neighbors(self, value):
        """Set the smote k-neighbors value."""
        self._smote_k_neighbors = value
