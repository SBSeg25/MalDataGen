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
    import logging
    import numpy

    from Engine.exception.MetricsException import SpecificityError

except ImportError as error:
    print(error)
    sys.exit(-1)

class Specificity:
    """
    A class for calculating Specificity and validating input labels.

    Attributes:
        None

    Methods:
        get_specificity(predicted_labels, true_labels):
            Calculate Specificity given predicted and true labels.

        _check_input_labels(predicted_labels, true_labels):
            Check the validity and type of input labels.

    Exceptions:
        SpecificityError: Custom exception class for handling Specificity calculation errors.

    Example:
        # Create an instance of the SpecificityCalculator class
        specificity_calculator = SpecificityCalculator()

        # Define predicted and true labels as numpy arrays
        predicted_labels = np.array([1, 0, 1, 0, 1])
        true_labels = np.array([1, 0, 0, 1, 1])

        # Calculate and print the Specificity
        print(f"Specificity: {specificity_calculator.get_specificity(predicted_labels, true_labels)}")
    """

    def get_metric(self, true_labels, predicted_labels):
        """
        Calculate Specificity given predicted and true labels.

        Args:
            predicted_labels (numpy.ndarray): Array of predicted labels (0 or 1).
            true_labels (numpy.ndarray): Array of true labels (0 or 1).

        Returns:
            float: The Specificity as a floating-point number between 0 and 1.

        Raises:
            SpecificityError: Custom exception class for handling Specificity calculation errors.
        """
        try:
            # Check if the input labels are valid and of the correct type
            self._check_input_labels(predicted_labels, true_labels)

            # Count true negatives and false positives
            true_negative = sum(1 for y, true in zip(predicted_labels, true_labels) if y == 0 and true == 0)
            false_positive = sum(1 for y, true in zip(predicted_labels, true_labels) if y == 1 and true == 0)

            if true_negative + false_positive == 0:
                return 0.0

            # Calculate Specificity as the ratio of true negatives to the sum of true negatives and false positives
            specificity = true_negative / (true_negative + false_positive)

            logging.info(f"\t\t\t\t   {self.__class__.__name__}: {specificity}")
            return specificity

        except SpecificityError as e:
            return f"Specificity Error: {e}"

    @staticmethod
    def _check_input_labels(predicted_labels, true_labels):
        """
        Check the validity and type of input labels.

        Args:
            predicted_labels (numpy.ndarray): Array of predicted labels (0 or 1).
            true_labels (numpy.ndarray): Array of true labels (0 or 1).

        Raises:
            SpecificityError: Custom exception class for handling accuracy calculation errors.
        """
        # Check if predicted_labels is None
        if predicted_labels is None:
            # Raise an SpecificityError with an error message
            raise SpecificityError("Prediction Error:", "Error: The predicted_labels argument should be "
                                                        "an array but was received a None value")
        # Check if predicted_labels is not a numpy array
        elif not isinstance(predicted_labels, numpy.ndarray):
            # Raise an SpecificityError with an error message
            raise SpecificityError("Prediction Error:", "Error: The predicted_labels argument should be an"
                                                        " array but was received an invalid type")
        else:
            pass  # No issues with predicted_labels

        # Check if true_labels is None
        if true_labels is None:
            # Raise an SpecificityError with an error message
            raise SpecificityError("Prediction Error:", "Error: The true_labels argument should be an array"
                                                        " but was received a None value")
        # Check if true_labels is not a numpy array
        elif not isinstance(true_labels, numpy.ndarray):
            # Raise an SpecificityError with an error message
            raise SpecificityError("Prediction Error:", "Error: The true_labels argument should be an array"
                                                        " but was received an invalid type")
        else:
            pass  # No issues with true_labels

        # Check if the dimensions of predicted_labels and true_labels match
        if len(predicted_labels) != len(true_labels):
            # Raise an SpecificityError with an error message
            raise SpecificityError("Prediction Error:", "Error: Both predicted_labels and true_labels must"
                                                        "have the same dimensions but are assigned different "
                                                        "dimensions")

        # Check if all elements in predicted_labels are 0 or 1
        if not numpy.all(numpy.logical_or(predicted_labels == 0, predicted_labels == 1)):
            # Raise an SpecificityError with an error message
            raise SpecificityError("Prediction Error:", "Error: The predicted_labels argument must be an"
                                                        " array composed of values 0 and 1, but given different values")

        # Check if all elements in true_labels are 0 or 1
        if not numpy.all(numpy.logical_or(true_labels == 0, true_labels == 1)):
            # Raise an SpecificityError with an error message
            raise SpecificityError("Prediction Error:", "Error: The true_labels argument must be an array"
                                                        " composed of values 0 and 1, but given different values")
