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
    from Engine.exception.MetricsException import JaccardDistanceError

except ImportError as error:
    print(error)
    sys.exit(-1)

class JaccardDistance:
    """
    A class for calculating Jaccard Distance and Mean Jaccard Distance between two binary distributions.

    Attributes:
        None

    Methods:
        get_metric(first_distribution, second_distribution):
            Calculate the Mean Jaccard Distance between two binary distributions.

        get_jaccard_distance(first_distribution, second_distribution):
            Calculate the Jaccard Distance between two binary distributions.

    Exceptions:
        JaccardDistanceError: Custom exception class for handling Jaccard Distance calculation errors.

    Example:
        # Create an instance of the JaccardDistance class
        jaccard_distance = JaccardDistance()

        # Define two binary distributions as numpy arrays
        first_distribution = np.array([1, 0, 1, 1])
        second_distribution = np.array([0, 0, 1, 0])

        # Calculate Jaccard Distance between the two distributions
        distance = jaccard_distance.get_jaccard_distance(first_distribution, second_distribution)

        # Calculate Mean Jaccard Distance between two distributions
        mean_distance = jaccard_distance.get_metric(first_distribution, second_distribution)

        # Print Jaccard Distance and Mean Jaccard Distance
        print(f"Jaccard Distance: {distance}")
        print(f"Mean Jaccard Distance: {mean_distance}")
    """

    def get_metric(self, first_distribution, second_distribution):
        """
        Calculate the Mean Jaccard Distance between two binary distributions.

        Args:
            first_distribution (numpy.ndarray): First binary distribution as an array of 0s and 1s.
            second_distribution (numpy.ndarray): Second binary distribution as an array of 0s and 1s.

        Returns:
            float: The Mean Jaccard Distance as a floating-point number (normalized to [0, 1]).

        Raises:
            JaccardDistanceError: If inputs are not binary or have different lengths.
        """
        try:
            # Check if inputs are valid
            self._check_input_labels(first_distribution, second_distribution)

            # Calculate total Jaccard distance
            total_distance = 0.0
            for point1, point2 in zip(first_distribution, second_distribution):
                total_distance += self.get_jaccard_distance(point1, point2)

            # Normalize by the number of samples
            mean_distance = total_distance / len(first_distribution) if len(first_distribution) > 0 else 0.0

            logging.info(f"\t\t\t\t   {self.__class__.__name__}: {mean_distance}")
            return float(mean_distance)

        except JaccardDistanceError as e:
            return f"Jaccard Distance Error: {e}"

    def get_jaccard_distance(self, first_distribution, second_distribution):
        """
        Calculate the Jaccard Distance between two binary distributions.

        Args:
            first_distribution (numpy.ndarray): First binary distribution as an array of 0s and 1s.
            second_distribution (numpy.ndarray): Second binary distribution as an array of 0s and 1s.

        Returns:
            float: The Jaccard Distance as a floating-point number (normalized to [0, 1]).

        Raises:
            JaccardDistanceError: If inputs are not binary or have different lengths.
        """
        try:
            self._check_input_labels(first_distribution, second_distribution)
            
            # Calculate intersection and union
            intersection = np.sum(np.logical_and(first_distribution, second_distribution))
            union = np.sum(np.logical_or(first_distribution, second_distribution))
            
            # Avoid division by zero (if both vectors are all zeros)
            if union == 0:
                return 0.0
            
            # Jaccard Similarity = intersection / union
            # Jaccard Distance = 1 - Jaccard Similarity
            jaccard_similarity = intersection / union
            jaccard_distance = 1.0 - jaccard_similarity
            
            return float(jaccard_distance)

        except JaccardDistanceError as e:
            return f"Jaccard Distance Error: {e}"

    def _check_input_labels(self, first_distribution, second_distribution):
        """
        Validate input distributions for Jaccard Distance calculation.

        Args:
            first_distribution (numpy.ndarray): First distribution to check.
            second_distribution (numpy.ndarray): Second distribution to check.

        Raises:
            JaccardDistanceError: If inputs are invalid (non-binary, different lengths, or wrong type).
        """
        if not (isinstance(first_distribution, np.ndarray) or not isinstance(second_distribution, np.ndarray)):
            raise JaccardDistanceError("Inputs must be numpy arrays")

        if first_distribution.shape != second_distribution.shape:
            raise JaccardDistanceError("Inputs must have the same shape")

        # Commented out to allow non-binary inputs (if needed)
        # if not (np.all(np.isin(first_distribution, [0, 1])) or not (np.all(np.isin(second_distribution, [0, 1])): 
        #     raise JaccardDistanceError("Inputs must be binary (0s and 1s only)")