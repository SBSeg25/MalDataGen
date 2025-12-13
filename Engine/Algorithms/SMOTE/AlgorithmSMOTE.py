#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Kayuã Oleques']

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

    from scipy import sparse

    from sklearn.base import BaseEstimator
    from sklearn.utils import check_random_state
    from sklearn.neighbors import NearestNeighbors

except ImportError as error:
    print(error)
    sys.exit(-1)


class SMOTEAlgorithm(BaseEstimator):

    """
    Synthetic Minority Over-sampling Technique (SMOTE) algorithm for class imbalance problems.

    This implementation generates synthetic samples for minority classes by interpolating
    between existing samples in feature space. It follows the scikit-learn estimator API.


    Mathematical Definition:

        Given a minority class sample `x_i` and one of its k-nearest neighbors `x_zi`,
        a new synthetic sample is generated as:

            x_new = x_i + λ * (x_zi - x_i),

        where λ ∈ [0, 1] is a random value sampled from a uniform distribution.

    This implementation follows the scikit-learn estimator API.


    Args:
        @sampling_strategy (str or dict, optional):
            Strategy to determine which classes to oversample. Defaults to "auto".
        @random_state (int or RandomState, optional):
            Controls the randomness of the sample generation. Defaults to None.
        @k_neighbors (int, optional):
            Number of nearest neighbors to use when generating synthetic samples. Defaults to 5.

    Attributes:
        @X_ (array-like or sparse matrix):
            The resampled feature matrix.
        @y_ (array-like):
            The corresponding resampled target values.
        @nn_k_ (NearestNeighbors):
            Nearest neighbors estimator used for synthetic sample generation.

    Reference:
            Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).
            SMOTE: Synthetic Minority Over-sampling Technique.
            Journal of Artificial Intelligence Research, 16, 321–357. https://doi.org/10.1613/jair.953

    Example:
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_classes=2, weights=[0.1, 0.9])
        >>> smote = SMOTEAlgorithm(random_state=42, k_neighbors=5)
        >>> X_resampled, y_resampled = smote.fit(X, y)
        >>> print(f"Original class distribution: {np.bincount(y)}")
        >>> print(f"Resampled class distribution: {np.bincount(y_resampled)}")
    """

    def __init__(self, sampling_strategy="auto", random_state=None, k_neighbors=5):
        self.X_ = None
        self.y_ = None
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.nn_k_ = None

    def compile(self):
        """
        Placeholder method for compatibility with other estimators.

        This method doesn't perform any operation but is included to maintain
        consistency with estimators that require compilation.
        """
        pass

    def fit(self, X, y):
        """
        Initialize the SMOTE algorithm with given parameters.

        Args:
            @sampling_strategy (str or dict):
                Strategy to determine which classes to oversample. "auto" will resample all minority classes
                to have the same number of samples as the majority class. Can also be a dictionary specifying
                exact numbers for each class.
            @random_state (int or RandomState):
                Controls the randomness of the sample generation. Pass an int for reproducible results.
            @k_neighbors (int):
                Number of nearest neighbors to use when generating synthetic samples. Must be positive.

        Raises:
            ValueError: If k_neighbors is not a positive integer.
        """

        # Validate that the estimator (e.g., nearest neighbors) has been properly configured
        self._validate_estimator()

        # Initialize resampled datasets with the original data
        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        # Iterate over each class and the number of samples to generate for it
        for class_sample, number_samples in self._compute_sampling_strategy(y).items():

            # Skip if no new samples need to be generated for this class
            if number_samples == 0:
                continue

            # Find indices of all samples belonging to the target class
            target_class_indices = numpy.flatnonzero(y == class_sample)

            # Extract feature vectors for the target class
            X_class = X[target_class_indices]

            # Ensure there are enough samples to apply k-nearest neighbors
            if len(X_class) <= self.k_neighbors:

                # If not, repeat the class samples enough times to allow k-NN to work
                repeat_factor = int(numpy.ceil((self.k_neighbors + 1) / len(X_class)))
                X_class = numpy.tile(X_class, (repeat_factor, 1))[:self.k_neighbors + 1]

            # Fit the k-NN model to the samples of the target class
            self.nn_k_.fit(X_class)

            # Get the indices of the k nearest neighbors for each sample, excluding the sample itself
            knn_kernels = self.nn_k_.kneighbors(X_class, return_distance=False)[:, 1:]

            # Generate synthetic samples using the selected neighbors
            X_new, y_new = self._make_samples(X_class, y.dtype, class_sample, X_class, knn_kernels, number_samples)

            # Add the newly generated samples to the resampled dataset
            X_resampled.append(X_new)
            y_resampled.append(y_new)

        # If the original input is sparse, stack results using sparse operations
        if sparse.issparse(X):
            X_resampled = sparse.vstack(X_resampled, format=X.format)

        # Otherwise, stack the arrays using NumPy
        else:
            X_resampled = numpy.vstack(X_resampled)

        # Stack the labels into a single array
        y_resampled = numpy.hstack(y_resampled)

        # Store the final resampled data in the instance variables
        self.X_ = X_resampled
        self.y_ = y_resampled


    def _validate_estimator(self):
        """
        Initialize the nearest neighbors estimator.

        This method sets up the NearestNeighbors instance used for finding
        similar samples during synthetic sample generation.
        """
        self.nn_k_ = NearestNeighbors(n_neighbors=self.k_neighbors + 1)

    @staticmethod
    def _compute_sampling_strategy(y):
        """
        Compute the number of synthetic samples to generate for each class.

        Args:
            @y (array-like): Target values.

        Returns:
            dict: A dictionary mapping each class to the number of synthetic samples needed.
        """

        # Get the unique class labels and the number of occurrences for each class in y
        classes, class_counts = numpy.unique(y, return_counts=True)

        # Determine the maximum number of samples across all classes
        max_count = max(class_counts)

        # Create a dictionary mapping each class to the number of samples needed
        # to reach the maximum count (i.e., to balance the class)
        return {cls: max_count - count for cls, count in zip(classes, class_counts)}

    def _make_samples(self, X, y_dtype, y_type, nn_data, nn_number, number_samples, step_size=1.0):
        """
        Generate synthetic samples for a given class.

        Args:
            @X (array-like): Feature matrix for the target class.
            @y_dtype (dtype): Data type of the target values.
            @y_type (int/str): Class label for the generated samples.
            @nn_data (array-like): Feature data for nearest neighbors calculation.
            @nn_number (array-like): Indices of nearest neighbors.
            @number_samples (int): Number of synthetic samples to generate.
            @step_size (float, optional): Control parameter for interpolation. Defaults to 1.0.

        Returns:
            tuple: (X_new, y_new) - Generated samples and their labels.
        """
        # Initialize the random number generator with the provided random state
        random_state = check_random_state(self.random_state)

        # Randomly select indices of samples to generate new synthetic points from
        samples_indices = random_state.randint(0, nn_number.shape[0], size=number_samples)

        # Generate random interpolation steps between 0 and step_size for each new sample
        steps = step_size * random_state.uniform(size=number_samples)[:, numpy.newaxis]

        # Select the corresponding rows (base samples) from the input data
        rows = samples_indices

        # Randomly select a neighbor index (column) for each base sample
        cols = random_state.randint(0, nn_number.shape[1], size=number_samples)

        # Generate new synthetic samples by interpolating between base sample and a selected neighbor
        X_new = X[rows] + steps * (nn_data[nn_number[rows, cols]] - X[rows])

        # Create an array of labels for the synthetic samples, all with the same class (y_type)
        y_new = numpy.full(number_samples, fill_value=y_type, dtype=y_dtype)

        return X_new, y_new

    def get_samples(self, class_sample_map):
        """
        Generate synthetic samples according to the specified class distribution.

        Args:
            @class_sample_map (dict):
                Dictionary specifying how many samples to generate for each class.
                Expected format:
                {
                    "classes": {class_label: number_of_samples, ...},
                    "number_classes": total_number_of_classes (optional)
                }

        Returns:
            dict: A dictionary mapping each class label to its generated samples.

        Note:
            If a class in class_sample_map doesn't exist in the training data,
            no samples will be generated for that class.
        """

        # Extract the dictionary mapping class labels to the number of samples to generate
        class_counts = class_sample_map["classes"]

        # Initialize a dictionary to store the newly generated samples for each class
        generated_data = {}

        # Ensure that self.X_ and self.y_ have the same length
        if len(self.X_) != len(self.y_):
            min_len = min(len(self.X_), len(self.y_))
            self.X_ = self.X_[:min_len]
            self.y_ = self.y_[:min_len]

        # Iterate over each class and the number of samples to generate for that class
        for label, count in class_counts.items():

            # Get indices of all samples in self.y_ that belong to the current class
            target_class_indices = numpy.flatnonzero(self.y_ == label)

            # Ensure that the indices are within bounds of self.X_
            target_class_indices = target_class_indices[target_class_indices < len(self.X_)]

            # If there are no samples available for this class, skip generation
            if len(target_class_indices) == 0:
                continue

            # Extract the feature vectors corresponding to the target class
            X_class = self.X_[target_class_indices]

            # If not enough samples to apply k-NN, repeat them to ensure sufficient quantity
            if len(X_class) <= self.k_neighbors:
                repeat_factor = int(numpy.ceil((self.k_neighbors + 1) / len(X_class)))
                X_class = numpy.tile(X_class, (repeat_factor, 1))[:self.k_neighbors + 1]

            # Fit the nearest neighbors model using the samples from the current class
            self.nn_k_.fit(X_class)

            # Compute the indices of k-nearest neighbors for each sample, excluding itself
            knn_kernels = self.nn_k_.kneighbors(X_class, return_distance=False)[:, 1:]

            # Generate new synthetic samples for the current class
            X_new, _ = self._make_samples(X_class, self.y_.dtype, label, X_class, knn_kernels, count)

            # Store the generated samples in the dictionary under the current class label
            generated_data[label] = X_new

        return generated_data
