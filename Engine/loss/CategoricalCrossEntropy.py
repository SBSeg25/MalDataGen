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

except ImportError as error:
    print(error)
    sys.exit(-1)


class CategoricalCrossEntropy(Loss):
    """
    Computes the categorical cross entropy loss between true labels and predicted values.

    This loss function is typically used for multi-class classification problems.
    It measures the dissimilarity between true labels (one-hot encoded) and predicted probabilities.

    Args:
        from_logits (bool): Whether `y_predicted` is expected to be logits (before softmax activation).
            If False, `y_predicted` is assumed to be probabilities.
        label_smoothing (float): Float in [0, 1]. Applies label smoothing if > 0.
        reduction (str): Type of reduction to apply to the loss.
        name (str): Optional name for the loss instance.
        dtype (dtype): The dtype of the loss's computations.

    Example:
    -------
        >>> python3
        ...     import tensorflow
        ...     y_true = tensorflow.constant([[0, 1, 0], [0, 0, 1]], dtype=tensorflow.float32)
        ...     y_predicted = tensorflow.constant([[0.05, 0.95, 0], [0.1, 0.8, 0.1]], dtype=tensorflow.float32)
        ...     loss_fn = CategoricalCrossEntropy()
        ...     loss = loss_fn(y_true, y_predicted)
        >>>     print(loss.numpy())  # Output: computed CCE loss

    """

    def __init__(self,
                 from_logits=False,
                 label_smoothing=0.0,
                 reduction=tensorflow.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                 name="categorical_crossentropy",
                 dtype=None):

        super().__init__(reduction=reduction, name=name, dtype=dtype)
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_predicted):
        """
        Computes the categorical crossentropy loss.

        Args:
            y_true (Tensor): One-hot encoded true labels with shape `[batch_size, num_classes]`.
            y_predicted (Tensor): Predicted probabilities or logits with shape `[batch_size, num_classes]`.

        Returns:
            Tensor: Computed loss value.
        """

        # Convert logits to probabilities if from_logits is True
        if self.from_logits:

            # Apply label smoothing
            y_predicted = tensorflow.nn.softmax(y_predicted)

        # Compute categorical crossEntropy loss
        y_true = y_true * (1.0 - self.label_smoothing) + self.label_smoothing / tensorflow.cast(tensorflow.shape(y_true)[-1],
                                                                                        y_true.dtype)

        loss = -tensorflow.reduce_sum(y_true * tensorflow.math.log(y_predicted + tensorflow.keras.backend.epsilon()), axis=-1)

        # Return the mean loss over the batch
        return tensorflow.reduce_mean(loss)

    @property
    def from_logits(self) -> bool:
        """Determines if inputs are raw logits or probability distributions.

        When True, the input values are interpreted as unnormalized logits.
        When False, inputs are assumed to be probability distributions (summing to 1).
        Using logits is numerically more stable but requires proper handling.

        Returns:
            bool: True if inputs are logits, False if probabilities
        """
        return self._from_logits

    @from_logits.setter
    def from_logits(self, value: bool) -> None:
        """Sets whether the inputs are logits or probabilities.

        Args:
            value: Boolean indicating input type

        Raises:
            ValueError: If value is not a boolean
        """
        if not isinstance(value, bool):
            raise ValueError("from_logits must be a boolean (True/False)")

        self._from_logits = value

    @property
    def label_smoothing(self) -> float:
        """Gets the current label smoothing factor.

        Label smoothing replaces hard 0/1 labels with smoothed values,
        which can prevent overconfidence and improve generalization.
        A value of 0.0 means no smoothing (original labels).
        A value of 0.1 is commonly used (replaces 0 with 0.05 and 1 with 0.95).

        Returns:
            float: Current smoothing factor between [0.0, 1.0]
        """
        return self._label_smoothing

    @label_smoothing.setter
    def label_smoothing(self, value: float) -> None:
        """Sets the label smoothing factor.

        Args:
            value: Smoothing factor between 0.0 and 1.0

        Raises:
            ValueError: If value is outside [0.0, 1.0] range
        """
        if not isinstance(value, (float, int)) or value < 0 or value > 1:
            raise ValueError("label_smoothing must be a float between 0.0 and 1.0")

        self._label_smoothing = float(value)