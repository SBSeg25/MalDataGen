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

class BinaryCrossEntropy(Loss):
    """
    Computes the binary cross-entropy loss between true labels and predicted labels.

    This loss function is commonly used in binary classification tasks where the labels
    are either 0 or 1. It supports both probability-based (`from_logits=False`) and
    logit-based (`from_logits=True`) inputs.

    Mathematical Definition:

    Given the true labels y_true and predicted values y_predicted, the binary
    cross-entropy loss is computed as:

        L(y_true, y_predicted) = -sum(y_true * log(y_predicted) + (1 - y_true) * log(1 - y_predicted))

    If `from_logits=True`, the function applies the sigmoid activation before computing the loss:

        y_sigmoid = 1 / (1 + exp(-y_predicted))

        L(y_true, y_sigmoid) = -sum(y_true * log(y_sigmoid) + (1 - y_true) * log(1 - y_sigmoid))

    Args:
        @from_logits (bool): Whether `y_pred` is a tensor of logit values (unscaled) or probabilities (scaled between 0 and 1).
        @label_smoothing (float): A factor for smoothing the true labels, reducing confidence in absolute predictions.
        @axis (int): The axis along which to compute cross-entropy (default is -1).
        @reduction (str): Specifies the type of reduction applied to the loss. Defaults to "sum_over_batch_size".
        @name (str): Name for the loss instance.
        @dtype (dtype): Data type for computations. Defaults to `None`, which uses Keras backend's float type.

    Example:
    -------
        >>> python3
        ...    import tensorflow
        ...     # Example with logits (from_logits=True)
        ...     y_true = tensorflow.constant([0, 1, 0, 0], dtype=tensorflow.float32)
        ...     y_pred = tensorflow.constant([-18.6, 0.51, 2.94, -12.8], dtype=tensorflow.float32)
        ...     loss_fn = BinaryCrossentropy(from_logits=True)
        ...     loss = loss_fn(y_true, y_pred)
        ...     print(loss.numpy())  # Output: 0.8654
        ...     # Example with probabilities (from_logits=False)
        ...     y_pred_prob = tensorflow.constant([0.6, 0.3, 0.2, 0.8], dtype=tensorflow.float32)
        ...     loss_fn_prob = BinaryCrossentropy(from_logits=False)
        ...     loss_prob = loss_fn_prob(y_true, y_pred_prob)
        >>>     print(loss_prob.numpy())  # Output varies based on probabilities

    """

    def __init__(self,
                 from_logits=False,
                 label_smoothing=0.0,
                 axis=-1,
                 reduction=tensorflow.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                 name="binary_crossentropy",
                 dtype=None):

        # Calls the parent class constructor and initializes attributes
        super().__init__(reduction=reduction, name=name, dtype=dtype)
        self.from_logits = from_logits  # Whether inputs are logits or probabilities
        self.label_smoothing = label_smoothing  # Smoothing factor for true labels
        self.axis = axis  # Axis along which cross-entropy is computed

    def call(self, y_true, y_predicted):
        """
        Computes the binary cross-entropy loss.

        Args:
            y_true (Tensor): Ground truth labels (0 or 1).
            y_predicted (Tensor): Predicted values (either probabilities or logits).

        Returns:
            Tensor: Computed binary cross-entropy loss.
        """
        y_true = tensorflow.convert_to_tensor(y_true, dtype=self.dtype)  # Convert labels to tensor
        y_predicted = tensorflow.convert_to_tensor(y_predicted, dtype=self.dtype)  # Convert predictions to tensor

        # Apply label smoothing if needed
        if self.label_smoothing > 0:
            y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Compute binary cross-entropy loss
        return K.binary_crossentropy(y_true, y_predicted, from_logits=self.from_logits, axis=self.axis)

    @property
    def from_logits(self) -> bool:
        """Whether inputs are logits (unscaled) or probabilities.

        When True, expects raw logits (before softmax).
        When False, expects probability values (after softmax).
        Using logits is numerically more stable but requires different handling.

        Returns:
            bool: True if inputs are logits, False if probabilities
        """
        return self._from_logits

    @from_logits.setter
    def from_logits(self, value: bool) -> None:
        """Sets whether inputs are logits or probabilities.

        Args:
            value: Boolean indicating input type

        Raises:
            ValueError: If value is not a boolean
        """
        if not isinstance(value, bool):
            raise ValueError("from_logits must be a boolean")

        self._from_logits = value

    @property
    def label_smoothing(self) -> float:
        """Gets the label smoothing factor.

        Smooths true labels by mixing with uniform distribution.
        Value of 0.0 means no smoothing (hard labels).
        Value of 1.0 means complete smoothing (ignores true labels).
        Typical values are between 0.0 and 0.2.

        Returns:
            float: Current smoothing factor between [0, 1]
        """
        return self._label_smoothing

    @label_smoothing.setter
    def label_smoothing(self, value: float) -> None:
        """Sets the label smoothing factor.

        Args:
            value: New smoothing value (must be between 0 and 1)

        Raises:
            ValueError: If value is not in range [0, 1]
        """
        if not isinstance(value, (float, int)) or not (0 <= value <= 1):
            raise ValueError("label_smoothing must be a float in range [0, 1]")

        self._label_smoothing = float(value)

    @property
    def axis(self) -> int:
        """Gets the axis along which cross-entropy is computed.

        Typically -1 (the last axis) for class probabilities.
        Must correspond to the axis containing class logits/probabilities.

        Returns:
            int: Axis index (usually -1)
        """
        return self._axis

    @axis.setter
    def axis(self, value: int) -> None:
        """Sets the cross-entropy computation axis.

        Args:
            value: New axis value (must be valid axis index)

        Raises:
            ValueError: If value is not an integer
        """
        if not isinstance(value, int):
            raise ValueError("axis must be an integer")

        self._axis = value