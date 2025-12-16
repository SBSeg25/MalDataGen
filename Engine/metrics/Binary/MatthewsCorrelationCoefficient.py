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
    from Engine.exception.MetricsException import MatthewsCorrelationCoefficientError

except ImportError as error:
    print(error)
    sys.exit(-1)

class MatthewsCorrelationCoefficient:
    """
    A class for calculating the Matthews Correlation Coefficient (MCC) between predicted and true labels.

    Attributes:
        None

    Methods:
        get_matthews_correlation_coefficient(predicted_labels, true_labels):
            Calculate the Matthews Correlation Coefficient (MCC) between predicted labels and true labels.

    Example:
        # Define true labels and predicted labels
        true_labels = [1, 0, 1, 0, 1]
        predicted_labels = [1, 1, 0, 0, 1]

        # Calculate the MCC using the MatthewsCorrelationCoefficient class
        mcc_calculator = MatthewsCorrelationCoefficient()
        mcc = mcc_calculator.get_matthews_correlation_coefficient(predicted_labels, true_labels)

        # Print the MCC value
        print(f"Matthews Correlation Coefficient (MCC): {mcc}")
    """

    def get_metric(self, true_labels, predicted_labels):
        """
        Calculate the Matthews Correlation Coefficient (MCC) between predicted labels and true labels.

        Args:
            predicted_labels: List of predicted labels (binary: 0 or 1).
            true_labels: List of true labels (binary: 0 or 1).

        Returns:
            float: The Matthews Correlation Coefficient (MCC) as a floating-point number.

        Raises:
            MatthewsCorrelationCoefficientError: Custom exception class for handling MCC calculation errors.
        """
        # Check if the input distributions are valid and of the correct type
        self._check_input_labels(predicted_labels, true_labels)

        try:

            # Calculate true positives, true negatives, false positives, and false negatives
            true_positives = sum((1 for yt, yp in zip(true_labels, predicted_labels) if yt == 1 and yp == 1))
            true_negatives = sum((1 for yt, yp in zip(true_labels, predicted_labels) if yt == 0 and yp == 0))
            false_positives = sum((1 for yt, yp in zip(true_labels, predicted_labels) if yt == 0 and yp == 1))
            false_negatives = sum((1 for yt, yp in zip(true_labels, predicted_labels) if yt == 1 and yp == 0))

            # Check for non-negative integers in input values
            if true_positives < 0 or true_negatives < 0 or false_positives < 0 or false_negatives < 0:
                raise MatthewsCorrelationCoefficientError("All input values must be non-negative integers.")

            # Calculate MCC components
            matthews_correlation_coefficient_numerator = ((true_positives * true_negatives)
                                                          - (false_positives * false_negatives))
            matthews_correlation_coefficient_denominator = math.sqrt((true_positives + false_positives)
                                                                     * (true_positives + false_negatives)
                                                                     * (true_negatives + false_positives)
                                                                     * (true_negatives + false_negatives))

            # Check if the denominator is zero (avoid division by zero)
            if matthews_correlation_coefficient_denominator == 0:
                return 0.0

            # Calculate the Matthews Correlation Coefficient (MCC)
            mcc = matthews_correlation_coefficient_numerator / matthews_correlation_coefficient_denominator
            logging.info(f"\t\t\t\t   {self.__class__.__name__}: {mcc}")
            return mcc

        except MatthewsCorrelationCoefficientError as e:
            raise e

    @staticmethod
    def _check_input_labels(predicted_labels, true_labels):
        """
        Check the validity and type of input labels.

        Args:
            predicted_labels (numpy.ndarray): Array of predicted labels (0 or 1).
            true_labels (numpy.ndarray): Array of true labels (0 or 1).

        Raises:
            MatthewsCorrelationCoefficientError: Custom exception class for handling accuracy calculation errors.
        """
        # Check if predicted_labels is None
        if predicted_labels is None:
            # Raise an MatthewsCorrelationCoefficientError with an error message
            raise MatthewsCorrelationCoefficientError("Prediction Error:", "Error: The predicted_labels argument "
                                                                           "should be an array but was received a"
                                                                           " None value")
        # Check if predicted_labels is not a numpy array
        elif not isinstance(predicted_labels, numpy.ndarray):
            # Raise anMatthewsCorrelationCoefficientError with an error message
            raise MatthewsCorrelationCoefficientError("Prediction Error:", "Error: The predicted_labels argument"
                                                                           " should be an array but was received an"
                                                                           " invalid type")
        else:
            pass  # No issues with predicted_labels

        # Check if true_labels is None
        if true_labels is None:
            # Raise an MatthewsCorrelationCoefficientError with an error message
            raise MatthewsCorrelationCoefficientError("Prediction Error:", "Error: The true_labels argument"
                                                                           " should be an array but was received"
                                                                           " a None value")
        # Check if true_labels is not a numpy array
        elif not isinstance(true_labels, numpy.ndarray):
            # Raise an MatthewsCorrelationCoefficientError with an error message
            raise MatthewsCorrelationCoefficientError("Prediction Error:", "Error: The true_labels argument"
                                                                           " should be an array but was received an"
                                                                           " invalid type")
        else:
            pass  # No issues with true_labels

        # Check if the dimensions of predicted_labels and true_labels match
        if len(predicted_labels) != len(true_labels):
            # Raise an MatthewsCorrelationCoefficientError with an error message
            raise MatthewsCorrelationCoefficientError("Prediction Error:", "Error: Both predicted_labels and"
                                                                           " true_labels must have the same dimensions"
                                                                           " but are assigned different dimensions")

        # Check if all elements in predicted_labels are 0 or 1
        if not numpy.all(numpy.logical_or(predicted_labels == 0, predicted_labels == 1)):
            # Raise an MatthewsCorrelationCoefficientError with an error message
            raise MatthewsCorrelationCoefficientError("Prediction Error:", "Error: The predicted_labels argument"
                                                                           " must be an array composed of values 0 and"
                                                                           " 1, but given different values")

        # Check if all elements in true_labels are 0 or 1
        if not numpy.all(numpy.logical_or(true_labels == 0, true_labels == 1)):
            # Raise an MatthewsCorrelationCoefficientError with an error message
            raise MatthewsCorrelationCoefficientError("Prediction Error:", "Error: The true_labels argument must"
                                                                           " be an array composed of values 0 and 1,"
                                                                           " but given different values")
