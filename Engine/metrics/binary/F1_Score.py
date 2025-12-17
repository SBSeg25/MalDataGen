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

    from Engine.exception.MetricsException import F1ScoreError

except ImportError as error:
    print(error)
    sys.exit(-1)

class F1Score:
    """
    A class for calculating F1-Score and validating input labels.

    Attributes:
        None

    Methods:
        get_f1_score(predicted_labels, true_labels):
            Calculate F1-Score given predicted and true labels.

        _check_input_labels(predicted_labels, true_labels):
            Check the validity and type of input labels.

    Exceptions:
        F1ScoreError: Custom exception class for handling F1-Score calculation errors.

    Example:
        # Create an instance of the F1Score class
        f1_score_calculator = F1Score()

        # Define predicted and true labels as numpy arrays
        predicted_labels = np.array([1, 0, 1, 1, 0])
        true_labels = np.array([1, 1, 0, 1, 0])

        # Calculate and print the F1-Score
        print(f"F1-Score: {f1_score_calculator.get_f1_score(predicted_labels, true_labels)}")
    """

    def get_metric(self, true_labels, predicted_labels):
        """
        Calculate F1-Score given predicted and true labels.

        Args:
            predicted_labels (numpy.ndarray): Array of predicted labels (0 or 1).
            true_labels (numpy.ndarray): Array of true labels (0 or 1).

        Returns:
            float: The F1-Score as a floating-point number between 0 and 1.

        Raises:
            F1ScoreError: Custom exception class for handling F1-Score calculation errors.
        """

        #logging.info("\t\t\t...")
        # Check if the input labels are valid and of the correct type
        self._check_input_labels(predicted_labels, true_labels)

        try:

            # Count true positives, false positives, and false negatives
            true_positive = sum(1 for y, true in zip(predicted_labels, true_labels) if y == 1 and true == 1)
            false_positive = sum(1 for y, true in zip(predicted_labels, true_labels) if y == 1 and true == 0)
            false_negative = sum(1 for y, true in zip(predicted_labels, true_labels) if y == 0 and true == 1)

            # Calculate precision and recall, handling cases with division by zero
            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
            recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0

            if precision + recall == 0:
                return 0.0

            # Calculate F1-Score using precision and recall
            f1_score = 2 * (precision * recall) / (precision + recall)

             
            logging.info(f"\t\t\t\t   {self.__class__.__name__}: {f1_score}")

            return f1_score

        except F1ScoreError as e:
            return f"F1-Score Error: {e}"

    @staticmethod
    def _check_input_labels(predicted_labels, true_labels):
        """
        Check the validity and type of input labels.

        Args:
            predicted_labels (numpy.ndarray): Array of predicted labels (0 or 1).
            true_labels (numpy.ndarray): Array of true labels (0 or 1).

        Raises:
            F1ScoreError: Custom exception class for handling accuracy calculation errors.
        """
        # Check if predicted_labels is None
        if predicted_labels is None:
            # Raise an F1ScoreError with an error message
            raise F1ScoreError("Prediction Error:", "Error: The predicted_labels argument should be "
                                                    "an array but was received a None value")
        # Check if predicted_labels is not a numpy array
        elif not isinstance(predicted_labels, numpy.ndarray):
            # Raise an F1ScoreError with an error message
            raise F1ScoreError("Prediction Error:", "Error: The predicted_labels argument should be an"
                                                    " array but was received an invalid type")
        else:
            pass  # No issues with predicted_labels

        # Check if true_labels is None
        if true_labels is None:
            # Raise an F1ScoreError with an error message
            raise F1ScoreError("Prediction Error:", "Error: The true_labels argument should be an array"
                                                    " but was received a None value")
        # Check if true_labels is not a numpy array
        elif not isinstance(true_labels, numpy.ndarray):
            # Raise an F1ScoreError with an error message
            raise F1ScoreError("Prediction Error:", "Error: The true_labels argument should be an array"
                                                    " but was received an invalid type")
        else:
            pass  # No issues with true_labels

        # Check if the dimensions of predicted_labels and true_labels match
        if len(predicted_labels) != len(true_labels):
            # Raise an F1ScoreError with an error message
            raise F1ScoreError("Prediction Error:", "Error: Both predicted_labels and true_labels must"
                                                    " have the same dimensions but are assigned different dimensions")

        # Check if all elements in predicted_labels are 0 or 1
        if not numpy.all(numpy.logical_or(predicted_labels == 0, predicted_labels == 1)):
            # Raise an F1ScoreError with an error message
            raise F1ScoreError("Prediction Error:", "Error: The predicted_labels argument must be an"
                                                    " array composed of values 0 and 1, but given different values")

        # Check if all elements in true_labels are 0 or 1
        if not numpy.all(numpy.logical_or(true_labels == 0, true_labels == 1)):
            # Raise an F1ScoreError with an error message
            raise F1ScoreError("Prediction Error:", "Error: The true_labels argument must be an array"
                                                    " composed of values 0 and 1, but given different values")
