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
    from Engine.exception.MetricsException import ManhattanDistanceError


except ImportError as error:
    print(error)
    sys.exit(-1)

class ManhattanDistance:
    """
    A class for calculating Manhattan Distance, Average Manhattan Distance between two distributions,
    and returning the mean of a single distribution.

    Attributes:
        None

    Methods:
        get_manhattan_distance(first_distribution, second_distribution):
            Calculate the Manhattan Distance between two distributions.

        get_average_manhattan_distance(first_distribution, second_distribution):
            Calculate the Average Manhattan Distance between corresponding pairs of distributions.

        get_mean_distribution(distribution):
            Return the mean of a single distribution.
    """

    def get_metric(self, first_distribution, second_distribution):
        """
        Calculate the Manhattan Distance between two distributions.
        """
        try:
            manhattan_distance = sum(abs(x - y) for x, y in zip(first_distribution, second_distribution))
            logging.info(f"\t\t\t\t   {self.__class__.__name__}: {manhattan_distance}")
            return self.get_mean_distribution(manhattan_distance)
        except Exception as e:
            return f"Manhattan Distance Error: {e}"

    def get_average_manhattan_distance(self, first_distribution, second_distribution):
        """
        Calculate the Average Manhattan Distance between corresponding pairs of distributions.
        """
        try:
            total_distance = 0
            for point1, point2 in zip(first_distribution, second_distribution):
                distance = self.get_metric(point1, point2)
                total_distance += distance
            return total_distance / len(first_distribution)
        except Exception as e:
            return f"Manhattan Distance Error: {e}"

    def get_mean_distribution(self, distribution):
        """
        Return the mean of a single distribution.

        Args:
            distribution (list): A distribution as a list of numerical values.

        Returns:
            float: The mean of the distribution as a floating-point number.
        """
        try:
            # Calculate the mean of the distribution using numpy
            mean_value = float(numpy.mean(distribution))
            return mean_value

        except Exception as e:
            raise ManhattanDistanceError(f"Error calculating the mean of the distribution: {str(e)}")

