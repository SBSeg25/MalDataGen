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

    from tensorflow.keras.utils import io_utils

    from tensorflow.keras.callbacks.callback import Callback

except ImportError as error:
    print(error)
    sys.exit(-1)


class TerminateOnNaN(Callback):
    """
    Callback that terminates training when a NaN (Not a Number) or infinite loss is encountered.

    This callback monitors the loss during training. If the loss becomes NaN or infinite,
    it stops the training process to prevent further computation with invalid loss values.

    Attributes:
    -----------
    model: keras.Model
        The model that the callback is applied to. This is automatically set by Keras during training.

    Methods:
    --------
    on_batch_end(batch, logs=None):
        This method is triggered at the end of each batch during training. It checks if the loss is NaN or infinite,
        and if so, terminates the training process.

    Example:
        >>> python3
        ...     from keras.models import Sequential
        ...     from keras.layers import Dense
        ...     from keras.callbacks import TerminateOnNaN
        ...     import numpy
        ...     # Example of a model with a custom callback
        ...     model = Sequential([
        ...     Dense(32, input_dim=10, activation='relu'),
        ...     Dense(1)])
        ...
        ...     model.compile(optimizer='adam', loss='mse')
        ...
        ...     # Example data
        ...     X = numpy.random.random((100, 10))
        ...     y = numpy.random.random(100)
        ...
        ...     # Instantiate the callback
        ...     terminate_on_nan = TerminateOnNaN()
        ...
        ...     # Fit the model with the callback
        >>>     model.fit(X, y, epochs=10, callbacks=[terminate_on_nan])
    """

    def on_batch_end(self, batch, logs=None):
        """
        This method is called at the end of each batch during training. It checks if the loss is NaN or infinite.

        Parameters:
        -----------
        batch : int
            The index of the current batch.

        logs : dict, optional
            A dictionary of logs containing information about the current batch, including the loss.
            This dictionary is passed automatically by Keras during training.

        If the loss is NaN or infinite, the method will stop the training process.
        """
        logs = logs or {}  # Ensure that logs is not None
        loss = logs.get("loss")  # Extract the loss value from logs

        # Check if loss is NaN or infinite
        if loss is not None:

            if numpy.isnan(loss) or numpy.isinf(loss):

                # Print a message and stop training if the loss is invalid
                io_utils.print_msg(f"Batch {batch}: Invalid loss (NaN or Inf), terminating training.")
                self.model.stop_training = True  # Stop training immediately