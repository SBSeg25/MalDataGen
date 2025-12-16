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


class CosineSimilarity(Loss):
    """
    Computes the cosine similarity between true labels and predicted values.

    A similarity score of 1 indicates identical vectors, 0 indicates orthogonality,
    and -1 indicates opposite vectors. When used as a loss function, it is typically
    minimized by negating the similarity score.

    Mathematical Definition:

    Given `y_true` and `y_predicted`, the cosine similarity is defined as:

        similarity = sum(y_true * y_predicted) / (||y_true|| * ||y_predicted||)

    As a loss function, we use its negative:

        loss = -sum(y_true * y_predicted) / (||y_true|| * ||y_predicted||)

    If either `y_true` or `y_predicted` is a zero vector, the cosine similarity is set to 0.

    Args:
        @axis (int): The axis along which to compute cosine similarity (default is -1).
        @reduction (str): Specifies the type of reduction applied to the loss.
        @name (str): Name for the loss instance.
        @dtype (dtype): Data type for computations.

    Example:
    -------
        >>> python
        ...     import tensorflow
        ...     y_true = tensorflow.constant([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]], dtype=tensorflow.float32)
        ...     y_predicted = tensorflow.constant([[0.5, 0.5, -0.5], [0.0, 1.0, 1.0]], dtype=tensorflow.float32)
        ...     loss_fn = CosineSimilarity()
        ...     loss = loss_fn(y_true, y_predicted)
        >>>     print(loss.numpy())  # Output: similarity loss value

    """

    def __init__(self,
                 axis=-1,
                 reduction=tensorflow.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                 name="cosine_similarity",
                 dtype=None):

        # Calls the parent class constructor and initializes attributes
        super().__init__(reduction=reduction, name=name, dtype=dtype)
        self.axis = axis  # Axis along which cosine similarity is computed

    def call(self, y_true, y_predicted):
        """
        Computes the cosine similarity loss.

        Args:
            y_true (Tensor): Ground truth labels.
            y_predicted (Tensor): Predicted values.

        Returns:
            Tensor: Computed cosine similarity loss.
        """
        y_true = tensorflow.convert_to_tensor(y_true, dtype=self.dtype)  # Convert labels to tensor
        y_predicted = tensorflow.convert_to_tensor(y_predicted, dtype=self.dtype)  # Convert predictions to tensor

        # Normalize vectors
        y_true = tensorflow.nn.l2_normalize(y_true, axis=self.axis)
        y_predicted = tensorflow.nn.l2_normalize(y_predicted, axis=self.axis)

        # Compute cosine similarity
        similarity = tensorflow.reduce_sum(y_true * y_predicted, axis=self.axis)

        # Return negative similarity as loss
        return -similarity

    @property
    def axis(self) -> int:
        """Gets the axis along which cosine similarity is computed.

        This determines the dimension along which the vectors are normalized
        before computing similarity. Typically -1 (last axis) for most use cases.
        Must be a valid axis index for the input tensor.

        Returns:
            int: Current axis value (usually -1 for last dimension)
        """
        return self._axis

    @axis.setter
    def axis(self, value: int) -> None:
        """Sets the axis for cosine similarity computation.

        Args:
            value: New axis index (must be valid for input tensors)

        Raises:
            ValueError: If value is not an integer
            IndexError: If value is not a valid axis for expected inputs
        """
        if not isinstance(value, int):
            raise ValueError(f"axis must be an integer, got {type(value)}")
        # Note: Actual tensor rank validation should be done during computation
        self._axis = value