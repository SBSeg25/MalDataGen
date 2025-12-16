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
    import numpy as np
    import logging
    from Engine.exception.MetricsException import PermutationTestError
    from sklearn.utils import resample

except ImportError as error:
    print(error)
    sys.exit(-1)

class PermutationTest:
    """
    A class for performing a Permutation Test to compare two binary distributions.

    Attributes:
        n_permutations (int): Number of permutations to perform (default: 1000).
        random_state (int): Seed for reproducibility (default: None).

    Methods:
        get_metric(first_distribution, second_distribution):
            Calculate the p-value using permutation test.

        _calculate_statistic(first_distribution, second_distribution):
            Calculate the test statistic (difference of means by default).

    Exceptions:
        PermutationTestError: Custom exception class for handling permutation test errors.

    Example:
        # Create an instance of the PermutationTest class
        permutation_test = PermutationTest(n_permutations=1000, random_state=42)

        # Define two binary distributions as numpy arrays
        first_distribution = np.array([1, 0, 1, 1, 0, 1])
        second_distribution = np.array([0, 0, 1, 0, 1, 0])

        # Calculate p-value using permutation test
        p_value = permutation_test.get_metric(first_distribution, second_distribution)

        # Print p-value
        print(f"P-value: {p_value}")
    """

    def __init__(self, n_permutations=1000, random_state=None):
        """
        Initialize the PermutationTest class.

        Args:
            n_permutations (int): Number of permutations to perform (default: 1000).
            random_state (int): Seed for reproducibility (default: None).
        """
        self.n_permutations = n_permutations
        self.random_state = random_state

    def get_metric(self, first_distribution, second_distribution):
        """
        Calculate the p-value using permutation test.

        Args:
            first_distribution (numpy.ndarray): First binary distribution as an array of 0s and 1s.
            second_distribution (numpy.ndarray): Second binary distribution as an array of 0s and 1s.

        Returns:
            float: The p-value as a floating-point number.

        Raises:
            PermutationTestError: If inputs are invalid or have different lengths.
        """
        try:
            # Check if inputs are valid
            self._check_input_labels(first_distribution, second_distribution)

            # Calculate observed test statistic
            observed_stat = self._calculate_statistic(first_distribution, second_distribution)

            # Combine the data
            combined = np.concatenate((first_distribution, second_distribution))
            
            # Initialize permutation stats
            permutation_stats = np.zeros(self.n_permutations)
            
            # Perform permutations
            for i in range(self.n_permutations):
                # Shuffle the combined data
                np.random.seed(self.random_state)
                shuffled = resample(combined, replace=False, n_samples=len(combined), 
                                random_state=self.random_state)
                
                # Split into two new samples
                perm_first = shuffled[:len(first_distribution)]
                perm_second = shuffled[len(first_distribution):]
                
                # Calculate permutation statistic
                permutation_stats[i] = self._calculate_statistic(perm_first, perm_second)
            
            # Calculate p-value
            p_value = (np.sum(permutation_stats >= observed_stat) + 1) / (self.n_permutations + 1)
            
            logging.info(f"\t\t\t\t   {self.__class__.__name__}: p-value = {p_value}")
            return float(p_value)

        except PermutationTestError as e:
            return f"Permutation Test Error: {e}"

    def _calculate_statistic(self, first_distribution, second_distribution):
        """
        Calculate the test statistic (difference of means by default).

        Args:
            first_distribution (numpy.ndarray): First binary distribution.
            second_distribution (numpy.ndarray): Second binary distribution.

        Returns:
            float: The test statistic.
        """
        return np.abs(np.mean(first_distribution) - np.mean(second_distribution))

    def _check_input_labels(self, first_distribution, second_distribution):
        """
        Validate input distributions for permutation test.

        Args:
            first_distribution (numpy.ndarray): First distribution to check.
            second_distribution (numpy.ndarray): Second distribution to check.

        Raises:
            PermutationTestError: If inputs are invalid or have different lengths.
        """
        if not isinstance(first_distribution, np.ndarray) or not isinstance(second_distribution, np.ndarray):
            raise PermutationTestError("Inputs must be numpy arrays")

        if len(first_distribution.shape) != 1 or len(second_distribution.shape) != 1:
            raise PermutationTestError("Inputs must be 1-dimensional arrays")

        if first_distribution.shape[0] != second_distribution.shape[0]:
            raise PermutationTestError("Inputs must have the same length")