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

    from Engine.exception.MetricsException import LogLikelihoodError

except ImportError as error:
    print(error)
    sys.exit(-1)

class LogLikelihood:
    """
    A class for calculating Log-Likelihood between two distributions.

    Attributes:
        None

    Methods:
        get_log_likelihood(first_distribution, second_distribution):
            Calculate the Log-Likelihood between two distributions.

        get_average_log_likelihood(first_distribution, second_distribution):
            Calculate the average Log-Likelihood between a list of pairs of distributions.

    Example:
        # Define two distributions
        distribution1 = np.array([0.2, 0.4, 0.3, 0.1])
        distribution2 = np.array([0.3, 0.3, 0.2, 0.2])

        # Calculate the Log-Likelihood between the two distributions
        log_likelihood_calculator = LogLikelihood()
        log_likelihood = log_likelihood_calculator.get_log_likelihood(distribution1, distribution2)

        # Print the Log-Likelihood value
        print(f"Log-Likelihood: {log_likelihood}")
    """

    def get_metric(self, first_distribution, second_distribution):
        """
        Calculate the Log-Likelihood between two distributions.

        Args:
            first_distribution (numpy.ndarray): First probability distribution.
            second_distribution (numpy.ndarray): Second probability distribution.

        Returns:
            float: The Log-Likelihood value as a floating-point number.

        Raises:
            LogLikelihoodError: Custom exception class for handling Log-Likelihood calculation errors.
        """
        try:

            # Calculate Log-Likelihood
            log_likelihood = numpy.sum(first_distribution * numpy.log(second_distribution))
            
            logging.info(f"\t\t\t\t   {self.__class__.__name__}: {log_likelihood}")
            return float(log_likelihood)

        except LogLikelihoodError as e:
            raise e

    def get_average_log_likelihood(self, first_distribution, second_distribution):
        """
        Calculate the average Log-Likelihood between a list of pairs of distributions.

        Args:
            first_distribution: List of first probability distributions.
            second_distribution: List of second probability distributions.

        Returns:
            float: The average Log-Likelihood value as a floating-point number.

        Raises:
            LogLikelihoodError: Custom exception class for handling Log-Likelihood calculation errors.
        """
        try:

            total_log_likelihood = 0

            # Iterate through pairs of distributions and calculate Log-Likelihood for each pair
            for distribution1, distribution2 in zip(first_distribution, second_distribution):
                log_likelihood = self.get_metric(distribution1, distribution2)
                total_log_likelihood += log_likelihood

            # Check for zero-length to avoid division by zero
            if len(first_distribution) == 0:
                raise LogLikelihoodError("Calculation Error:", "The distribution length must be greater than zero.")

            # Calculate the average Log-Likelihood
            average_log_likelihood = total_log_likelihood / len(first_distribution)

            return average_log_likelihood

        except LogLikelihoodError as e:
            raise e

