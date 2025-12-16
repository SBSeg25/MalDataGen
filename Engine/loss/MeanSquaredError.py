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
    import tensorflow

    from tensorflow.keras.losses import Loss
    from tensorflow.keras import backend as K

except ImportError as error:
    print(error)
    sys.exit(-1)

class MeanSquaredError(Loss):
    """
    Computes the mean of the squared differences between true labels and predicted values.

    The Mean Squared Error (MSE) measures the average squared difference between actual and predicted values.
    Lower values indicate better model performance.

    Mathematical Definition:

    Given `y_true` and `y_predicted`, the MSE is defined as:

        loss = mean((y_true - y_predicted)^2)

    Args:
        reduction (str): Specifies the type of reduction applied to the loss.
            Supported values:
                - "sum": Sums the loss over all elements.
                - "sum_over_batch_size": Sums the loss and divides by the batch size.
                - "mean": Computes the mean loss over all elements.
                - "mean_with_sample_weight": Computes the mean loss weighted by sample importance.
                - "none" or `None`: No reduction applied.
            Defaults to "sum_over_batch_size".
        name (str): Optional name for the loss instance.
        dtype (dtype): Data type for computations.

    Example Usage:

    Example:
    -------
        >>> python3
        ...     import tensorflow
        ...     y_true = tensorflow.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tensorflow.float32)
        ...     y_predicted = tensorflow.constant([[1.5, 2.5], [3.0, 3.5]], dtype=tensorflow.float32)
        ...     loss_fn = MeanSquaredError()
        ...     loss = loss_fn(y_true, y_predicted)
        >>>     print(loss.numpy())  # Output: computed MSE loss

    """

    def __init__(self,
                 reduction=tensorflow.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                 name="mean_squared_error",
                 dtype=None):

        # Calls the parent class constructor and initializes attributes
        super().__init__(reduction=reduction, name=name, dtype=dtype)

    def call(self, y_true, y_predicted):
        """
        Computes the Mean Squared Error loss.

        Args:
            y_true (Tensor): Ground truth labels.
            y_predicted (Tensor): Predicted values.

        Returns:
            Tensor: Computed mean squared error loss.
        """
        y_true = tensorflow.convert_to_tensor(y_true, dtype=self.dtype)  # Convert labels to tensor
        y_predicted = tensorflow.convert_to_tensor(y_predicted, dtype=self.dtype)  # Convert predictions to tensor

        # Compute squared error
        squared_error = tensorflow.square(y_true - y_predicted)

        # Return the mean of squared errors
        return tensorflow.reduce_mean(squared_error)