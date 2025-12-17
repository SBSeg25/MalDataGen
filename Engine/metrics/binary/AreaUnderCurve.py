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
    from Engine.exception.MetricsException import AreaUnderCurveError

except ImportError as error:
    print(error)
    sys.exit(-1)

class AreaUnderCurve:
    """
    A class for calculating the Area Under the Receiver Operating Characteristic Curve (AUC).

    Attributes:
        None

    Methods:
        get_area_under_curve(true_labels, predicted_probabilities):
            Calculate the AUC given the true labels and predicted probabilities.

    Example:
        # Define true labels (binary, 0 or 1) and predicted probabilities
        true_labels = [0, 1, 1, 0, 1]
        predicted_probabilities = [0.2, 0.8, 0.7, 0.4, 0.9]

        # Calculate the AUC using the AreaUnderCurve class
        auc_calculator = AreaUnderCurve()
        auc = auc_calculator.get_area_under_curve(true_labels, predicted_probabilities)

        # Print the AUC value
        print(f"Area Under the Curve (AUC): {auc}")
    """

    @staticmethod
    def get_metric(true_labels, predicted_probabilities):
        """
        Calculate the Area Under the Receiver Operating Characteristic Curve (AUC).

        Args:
            true_labels (list): List of true binary labels (0 or 1).
            predicted_probabilities (list): List of predicted probabilities corresponding to the true labels.

        Returns:
            float: The AUC (Area Under the ROC Curve) as a floating-point number.

        Raises:
            AreaUnderCurveError: Custom exception class for handling AUC calculation errors.
        """
        try:
            if len(true_labels) != len(predicted_probabilities):
                raise AreaUnderCurveError("The lists of true_labels and predicted_probabilities must have the same "
                                          "length.")

            # Combine true labels and predicted probabilities, then sort by predicted probabilities in descending order
            roc_data = list(zip(true_labels, predicted_probabilities))
            roc_data.sort(key=lambda x: x[1], reverse=True)

            # Calculate the number of positive and negative labels
            num_positive = sum(1 for label in true_labels if label == 1)
            num_negative = sum(1 for label in true_labels if label == 0)

            # Initialize variables for calculating the area under the ROC curve
            area_under_curve = 0.0
            prev_false_positive_rate = 0.0
            prev_true_positive_rate = 0.0

            for label, prob in roc_data:
                if label == 1:
                    # Incrementally calculate the AUC using the trapezoidal rule
                    area_under_curve += (1.0 / num_positive) * (prev_false_positive_rate +
                                                                (0.5 * (1 - prev_false_positive_rate)))
                    prev_true_positive_rate += 1.0 / num_positive
                else:
                    prev_false_positive_rate += 1.0 / num_negative
            
            logging.info(f"\t\t\t\t   {self.__class__.__name__}: {area_under_curve}")
            return area_under_curve

        except AreaUnderCurveError as e:
            # Handle the case where an AreaUnderCurveError is raised and return an error message
            return f"AUC Error: {e}"

    @staticmethod
    def _check_input_labels(predicted_labels, true_labels):
        """
        Check the validity and type of input labels.

        Args:
            predicted_labels (numpy.ndarray): Array of predicted labels (0 or 1).
            true_labels (numpy.ndarray): Array of true labels (0 or 1).

        Raises:
            AccuracyError: Custom exception class for handling accuracy calculation errors.
        """
        # Check if predicted_labels is None
        if predicted_labels is None:
            # Raise an AreaUnderCurveError with an error message
            raise AreaUnderCurveError("Prediction Error:", "Error: The predicted_labels argument should be "
                                                           "an array but was received a None value")
        # Check if predicted_labels is not a numpy array
        elif not isinstance(predicted_labels, numpy.ndarray):
            # Raise an AreaUnderCurveError with an error message
            raise AreaUnderCurveError("Prediction Error:", "Error: The predicted_labels argument should be an"
                                                           " array but was received an invalid type")
        else:
            pass  # No issues with predicted_labels

        # Check if true_labels is None
        if true_labels is None:
            # Raise an AreaUnderCurveError with an error message
            raise AreaUnderCurveError("Prediction Error:", "Error: The true_labels argument should be an array"
                                                           " but was received a None value")
        # Check if true_labels is not a numpy array
        elif not isinstance(true_labels, numpy.ndarray):
            # Raise an AreaUnderCurveError with an error message
            raise AreaUnderCurveError("Prediction Error:", "Error: The true_labels argument should be an array"
                                                           " but was received an invalid type")
        else:
            pass  # No issues with true_labels

        # Check if the dimensions of predicted_labels and true_labels match
        if len(predicted_labels) != len(true_labels):
            # Raise an AreaUnderCurveError with an error message
            raise AreaUnderCurveError("Prediction Error:", "Error: Both predicted_labels and true_labels must"
                                                           " have the same dimensions but are assigned different"
                                                           " dimensions")

        # Check if all elements in predicted_labels are 0 or 1
        if not numpy.all(numpy.logical_or(predicted_labels == 0, predicted_labels == 1)):
            # Raise an AreaUnderCurveError with an error message
            raise AreaUnderCurveError("Prediction Error:", "Error: The predicted_labels argument must be an"
                                                           " array composed of values 0 and 1, but given different"
                                                           " values")

        # Check if all elements in true_labels are 0 or 1
        if not numpy.all(numpy.logical_or(true_labels == 0, true_labels == 1)):
            # Raise an AreaUnderCurveError with an error message
            raise AreaUnderCurveError("Prediction Error:", "Error: The true_labels argument must be an array"
                                                           " composed of values 0 and 1, but given different values")
