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
    import math
    import numpy
    import logging
    from Engine.exception.MetricsException import EuclideanDistanceError


except ImportError as error:
    print(error)
    sys.exit(-1)

class EuclideanDistance:
    """
    A class for calculating Euclidean distance and Mean Euclidean distance between two distributions.

    Attributes:
        None

    Methods:
        get_mean_euclidean_distance(first_distribution, second_distribution):
            Calculate the Mean Euclidean distance between two distributions.

        get_euclidean_distance(first_distribution, second_distribution):
            Calculate the Euclidean distance between two distributions.

        _check_input_labels(first_distribution, second_distribution):
            Check the validity and type of input distributions.

    Exceptions:
        EuclideanDistanceError: Custom exception class for handling Euclidean distance calculation errors.

    Example:
        # Create an instance of the EuclideanDistance class
        euclidean_distance = EuclideanDistance()

        # Define two distributions as numpy arrays
        first_distribution = np.array([1, 2, 3])
        second_distribution = np.array([4, 5, 6])

        # Calculate Euclidean distance between the two distributions
        distance = euclidean_distance.get_euclidean_distance(first_distribution, second_distribution)

        # Calculate Mean Euclidean distance between two distributions
        mean_distance = euclidean_distance.get_mean_euclidean_distance(first_distribution, second_distribution)

        # Print Euclidean distance and Mean Euclidean distance
        print(f"Euclidean distance: {distance}")
        print(f"Mean Euclidean distance: {mean_distance}")
    """
    

    def get_metric(self, first_distribution, second_distribution):
        """
        Calculate the Mean Euclidean distance between two distributions.

        Args:
            first_distribution (numpy.ndarray): First distribution as an array of numerical values.
            second_distribution (numpy.ndarray): Second distribution as an array of numerical values.

        Returns:
            float: The Mean Euclidean distance as a floating-point number.

        Raises:
            EuclideanDistanceError: Custom exception class for handling Euclidean distance calculation errors.
        """
        # Check if the input distributions are valid and of the correct type

        try:
             
            # Initialize a variable to store the total distance
            total_distance = 0

            # Iterate through corresponding points in both distributions and calculate the distance
            for point1, point2 in zip(first_distribution, second_distribution):
                total_distance += self.get_euclidean_distance(point1, point2)

            # Calculate the average distance by dividing the total distance by the number of points
            average_distance = total_distance / len(first_distribution)
            
            logging.info(f"\t\t\t\t   {self.__class__.__name__}: {average_distance}")
            # Return the computed average distance
            return float(average_distance)

        except EuclideanDistanceError as e:
            # Handle the case where an EuclideanDistanceError is raised and return an error message
            return f"Euclidean distance Error: {e}"

    def get_euclidean_distance(self, first_distribution, second_distribution):
        """
        Calculate the Euclidean distance between two distributions.

        Args:
            first_distribution (numpy.ndarray): First distribution as an array of numerical values.
            second_distribution (numpy.ndarray): Second distribution as an array of numerical values.

        Returns:
            float: The Euclidean distance as a floating-point number.

        Raises:
            EuclideanDistanceError: Custom exception class for handling Euclidean distance calculation errors.
        """
        # Check if the input distributions are valid and of the correct type

        try:
            # Calculate the squared Euclidean distance by summing the squared differences of corresponding elements
            squared_distance = sum((x - y) ** 2 for x, y in zip(first_distribution, second_distribution))

            # Calculate the Euclidean distance by taking the square root of the squared distance
            euclidean_distance = math.sqrt(squared_distance)

            # Return the computed Euclidean distance
            return euclidean_distance

        except EuclideanDistanceError as e:
            # Handle the case where an EuclideanDistanceError is raised and return an error message
            return f"Euclidean distance Error: {e}"
