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

import math


try:
    import sys
    import numpy
    import logging
    from Engine.exception.MetricsException import HellingerDistanceError


except ImportError as error:
    print(error)
    sys.exit(-1)

class HellingerDistance:
    """
    A class for calculating Hellinger distance and Average Hellinger distance between probability distributions.

    Attributes:
        None

    Methods:
        get_hellinger_distance(first_distribution, second_distribution):
            Calculate the Hellinger distance between two probability distributions.

        get_average_hellinger_distance(first_distribution, second_distribution):
            Calculate the Average Hellinger distance between corresponding pairs of probability distributions.

    Exceptions:
        HellingerDistanceError: Custom exception class for handling Hellinger distance calculation errors.

    Example:
        # Create an instance of the HellingerDistance class (not required, as methods are static)
        # hellinger_calculator = HellingerDistance()

        # Define two probability distributions as numpy arrays
        first_distribution = np.array([0.1, 0.4, 0.5])
        second_distribution = np.array([0.2, 0.3, 0.5])

        # Calculate Hellinger distance between the two distributions
        distance = HellingerDistance.get_hellinger_distance(first_distribution, second_distribution)

        # Calculate Average Hellinger distance between two lists of distributions
        avg_distance = HellingerDistance.get_average_hellinger_distance([first_distribution, second_distribution],
                                                                      [second_distribution, first_distribution])

        # Print Hellinger distance and Average Hellinger distance
        print(f"Hellinger distance: {distance}")
        print(f"Average Hellinger distance: {avg_distance}")
    """

    def safe_sqrt(self, data):
        """Calculate sqrt with negative values replaced by 0"""
        # Create mask of valid (non-negative) values
        valid = data >= 0
        # Initialize output array with zeros
        result = numpy.zeros_like(data)
        # Only calculate sqrt where valid
        result[valid] = numpy.sqrt(data[valid])
        return result

    def is_binary_fast(self, array):
        return array.size == numpy.count_nonzero((array == 0) | (array == 1))

    def robust_sqrt(self, data):
        """Calculate sqrt with all invalid values replaced by 0"""
        clean_data = numpy.nan_to_num(data, nan=0, posinf=0, neginf=0)
        clean_data = numpy.clip(clean_data, 0, None)  # Force non-negative
        #return numpy.sqrt(clean_data)
    
        if isinstance(clean_data, (list, numpy.ndarray)):
            if self.is_binary_fast(clean_data):
                return clean_data
            else:
                clean_data = numpy.array(clean_data)
                return numpy.sqrt(clean_data)
        else:  # Single value
            return math.sqrt(clean_data) if clean_data >= 0 else 0

    def get_metric(self, first_distribution, second_distribution):
        """
        Calculate the Hellinger distance between two probability distributions.

        Args:
            first_distribution (numpy.ndarray): First probability distribution as an array of numerical values.
            second_distribution (numpy.ndarray): Second probability distribution as an array of numerical values.

        Returns:
            float: The Hellinger distance as a floating-point number.

        Raises:
            HellingerDistanceError: Custom exception class for handling Hellinger distance calculation errors.
        """
        # Check if the input distributions are valid and of the correct type

        try:
            # Calculate square root of each value in the distributions
            #print(f"first  {first_distribution}")
            #print(f"second {second_distribution}")
           
            sqrt_p =  self.robust_sqrt(first_distribution)
            sqrt_q =  self.robust_sqrt(second_distribution)

            # Calculate Hellinger distance using the formula
            hellinger_distance = numpy.sqrt(0.5 * numpy.sum((sqrt_p - sqrt_q) ** 2))
            
            logging.info(f"\t\t\t\t   {self.__class__.__name__}: {hellinger_distance}")
            return float(hellinger_distance)

        except HellingerDistanceError as e:
            # Handle the case where a HellingerDistanceError is raised and return an error message
            return f"Hellinger distance Error: {e}"

    def get_average_hellinger_distance(self, first_distribution, second_distribution):
        """
        Calculate the Average Hellinger distance between corresponding pairs of probability distributions.

        Args:
            first_distribution (list of numpy.ndarray): List of first probability distributions.
            second_distribution (list of numpy.ndarray): List of second probability distributions.

        Returns:
            float: The Average Hellinger distance as a floating-point number.

        Raises:
            HellingerDistanceError: Custom exception class for handling Hellinger distance calculation errors.
        """
        # Check if the input distributions are valid and of the correct type

        try:
            # Initialize a variable to store the total Hellinger distance
            total_distance = 0

            # Iterate through corresponding pairs of distributions and calculate the Hellinger distance for each pair
            for p, q in zip(first_distribution, second_distribution):

                # Calculate Hellinger distance for each pair of distributions
                distance = HellingerDistance.get_metric(p, q)
                total_distance += distance

            # Calculate the average Hellinger distance by dividing the total distance by the number of distributions
            average_distance = total_distance / len(first_distribution)

            # Return the computed average Hellinger distance
            return average_distance

        except HellingerDistanceError as e:
            # Handle the case where a HellingerDistanceError is raised and return an error message
            return f"Hellinger distance Error: {e}"

