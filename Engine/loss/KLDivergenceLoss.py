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

    from tensorflow.python.keras.losses import Loss

except ImportError as error:
    print(error)
    sys.exit(-1)


class KLDivergenceLoss(Loss):
    """
    Kullback-Leibler Divergence loss (KL Divergence).

    This loss function computes the Kullback-Leibler Divergence (KL Divergence)
    between the true distribution `y_true` and the predicted distribution `y_predicted`.
    The KL Divergence measures how one probability distribution diverges from a second,
    expected probability distribution.

    The formula for KL Divergence is as follows:

    Math:

        D_{KL}(P || Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}

    Where:
        - P(i) is the true probability distribution (`y_true`).
        - Q(i) is the predicted probability distribution (`y_pred`).

    This loss function is often used in machine learning models that deal with probabilistic predictions,
    such as in variational autoencoders, generative models, and other tasks involving probability distributions.

    Reference:
    ---------
        - Kullback, S., & Leibler, R. A. (1951). "On information and sufficiency".
         *Annals of Mathematical Statistics*, 22(1), 79-86.

    Attributes:
    ----------
        - name (str): The name of the loss function.
    """

    def __init__(self, name="kl_divergence_loss"):
        """
        Initializes the KLDivergenceLoss class.

        Args:
            name (str): The name of the loss function. Default is "kl_divergence_loss".
        """
        super().__init__(name=name)

    def call(self, y_true, y_predicted):
        """
        Computes the Kullback-Leibler Divergence loss between `y_true` and `y_pred`.

        The KL divergence is calculated using the formula:

        Where:
        - y_true: True probability distribution (P).
        - y_predicted: Predicted probability distribution (Q).

        To avoid undefined logarithms (log(0)), small epsilon values are clipped to the predicted values.

        Args:
            y_true (Tensor): Tensor representing the true probability distribution (P).
            y_predicted (Tensor): Tensor representing the predicted probability distribution (Q).

        Returns:
            Tensor: The computed KL divergence loss.
        """
        # Ensure `y_predicted` does not contain zeros to avoid log(0), which is undefined
        epsilon = tensorflow.keras.backend.epsilon()  # Small value to avoid division by zero
        y_predicted = tensorflow.clip_by_value(y_predicted, epsilon, 1.0)  # Clip values in the range [epsilon, 1.0]

        # Compute the Kullback-Leibler Divergence using the formula
        kl_loss = tensorflow.reduce_sum(y_true * tensorflow.math.log(y_true / y_predicted), axis=-1)

        return kl_loss