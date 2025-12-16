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
    from Engine.exception.MetricsException import TruePositiveError

except ImportError as error:
    print(error)
    sys.exit(-1)


class TruePositive:
    """
    A class to calculate the amount of True Positive.

    Example:
        Define true binary labels (0 or 1) and predicted binary labels
        true_labels =      [0, 1, 1, 0, 1, 0]
        predicted_labels = [0, 1, 0, 0, 0, 1]

        TN = TrueNegative
        TP = TruePositive
        FN = FalseNegative
        FP = FalsPositive

        True Classification 	 |  0  |  1  |	1  |  0  |
        Predicted Classification |	0  |  1  |	0  |  1  |
        Result 	                 |  TN |  TP |	FN |  FP |
    """

    def get_metric(self, true_labels, predicted_labels):
        """
        Calculate the amount of True Positive.

        Args:
            predicted_labels (list): List of predicted binary labels (0 or 1).
            true_labels (list): List of true binary labels (0 or 1).

        Returns:
            int: Amount of True Positive.

        Raises:
            TruePositiveError: Custom exception class to handle True Positive calculation errors.
        """

        # Check if the input labels are valid and of the correct type
        self._check_input_labels(predicted_labels, true_labels)

        try:
            # print("     true:", true_labels)
            # print("predicted:", predicted_labels)
            true_positives = sum((1 for true, predicted in zip(true_labels, predicted_labels) if true == 1 and predicted == 1))
            # print("true_positives:", true_positives)
            # print("total_samples:", len(predicted_labels))
            logging.info(f"\t\t\t\t   {self.__class__.__name__}: {true_positives}")
            return true_positives

        except TruePositiveError as e:
            raise e

    @staticmethod
    def _check_input_labels(predicted_labels, true_labels):
        """
        Check the validity and type of input labels.

        Args:
            predicted_labels: Array of predicted labels (0 or 1).
            true_labels: Array of true labels (0 or 1).

        Raises:
            TruePositiveError: Custom exception class to handle True Positive calculation errors.
        """

        # Check if predicted_labels is None
        if predicted_labels is None:
            # Raise an TruePositiveError with an error message
            raise TruePositiveError("Prediction Error:", "Error: The predicted_labels argument should be "
                                                             "an array but was received a None value")
        # Check if predicted_labels is not a numpy array
        elif not isinstance(predicted_labels, numpy.ndarray):
            # Raise an TruePositiveError with an error message
            raise TruePositiveError("Prediction Error:", "Error: The predicted_labels argument should be an"
                                                             " array but was received an invalid type")
        else:
            pass  # No issues with predicted_labels

        # Check if true_labels is None
        if true_labels is None:
            # Raise an TruePositiveError with an error message
            raise TruePositiveError("Prediction Error:", "Error: The true_labels argument should be an array"
                                                             " but was received a None value")
        # Check if true_labels is not a numpy array
        elif not isinstance(true_labels, numpy.ndarray):
            # Raise an TruePositiveError with an error message
            raise TruePositiveError("Prediction Error:", "Error: The true_labels argument should be an array"
                                                             " but was received an invalid type")
        else:
            pass  # No issues with true_labels

        # Check if the dimensions of predicted_labels and true_labels match
        if len(predicted_labels) != len(true_labels):
            # Raise an TruePositiveError with an error message
            raise TruePositiveError("Prediction Error:", "Error: Both predicted_labels and true_labels must"
                                                             "have the same dimensions but are assigned different "
                                                             "dimensions")

        # Check if all elements in predicted_labels are 0 or 1
        if not numpy.all(numpy.logical_or(predicted_labels == 0, predicted_labels == 1)):
            # Raise an TruePositiveError with an error message
            raise TruePositiveError("Prediction Error:", "Error: The predicted_labels argument must be an"
                                                             " array composed of values 0 and 1, but given different"
                                                             " values")

        # Check if all elements in true_labels are 0 or 1
        if not numpy.all(numpy.logical_or(true_labels == 0, true_labels == 1)):
            # Raise an TruePositiveError with an error message
            raise TruePositiveError("Prediction Error:", "Error: The true_labels argument must be an array"
                                                             " composed of values 0 and 1, but given different values")
