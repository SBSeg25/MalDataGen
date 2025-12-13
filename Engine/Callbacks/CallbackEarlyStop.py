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
    import warnings

    import tensorflow as tf

    ops = tf.keras.backend
    #from keras.src.trainers import compile_utils

    from tensorflow.keras.callbacks import Callback
    from tensorflow.python.keras.utils import io_utils

except ImportError as error:
    print(error)
    sys.exit(-1)


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric has stopped improving.

    This callback monitors a specified metric during training. If the metric
    does not improve for a given number of epochs, the training will be stopped early.

    Args:
        monitor (str):
            The quantity to be monitored. Defaults to `"val_loss"`.
        min_delta (float):
            Minimum change in the monitored quantity to qualify as an improvement. Defaults to `0`.
        patience (int):
            Number of epochs with no improvement after which training will be stopped. Defaults to `0`.
        verbose (int):
            Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1 displays messages when the callback takes an action. Defaults to `0`.
        mode (str):
            One of `{"auto", "min", "max"}`. In `"min"` mode, training will stop when the quantity monitored has stopped decreasing;
                in `"max"` mode, it will stop when the quantity monitored has stopped increasing;
                in `"auto"` mode, the direction is automatically inferred from the name of the monitored quantity. Defaults to `"auto"`.
        baseline (float):
            Baseline value for the monitored quantity. If not `None`, training will stop if the model doesn't show
            improvement over the baseline. Defaults to `None`.
        restore_best_weights (bool):
            Whether to restore model weights from the epoch with the best value of the monitored quantity. Defaults to `False`.
        start_from_epoch (int):
            Number of epochs to wait before starting to monitor improvement. This allows for a warm-up period where
             no improvement is expected. Defaults to `0`.

    Example:
        >>> callback = EarlyStopping(monitor='loss', patience=3)
        ...     # This callback will stop training when there is no improvement in the loss for three consecutive epochs.
        ...     model = keras.models.Sequential([keras.layers.Dense(10)])
        ...     model.compile(keras.optimizers.SGD(), loss='mse')
        ...     history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
        ...                         epochs=10, batch_size=1, callbacks=[callback], verbose=0)
        >>> len(history.history['loss'])  # Only 4 epochs are run.

    """

    def __init__(
            self,
            monitor="val_loss",
            min_delta=0,
            patience=0,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=False,
            start_from_epoch=0,
    ):
        super().__init__()

        # Initialize the parameters
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.start_from_epoch = start_from_epoch

        # Validate the mode argument
        if mode not in ["auto", "min", "max"]:
            warnings.warn(
                f"EarlyStopping mode {mode} is unknown, fallback to auto mode.",
                stacklevel=2,
            )
            mode = "auto"
        self.mode = mode
        self.monitor_op = None

    def _set_monitor_op(self):
        """Set the operation based on the mode (min/max/auto)

        This method determines the appropriate operation (`ops.less` or `ops.greater`)
        based on the mode set for the EarlyStopping callback. The operation is used
        to check if the monitored metric is improving or not.

        The 'min' mode is for metrics that should be minimized (e.g., loss),
        while the 'max' mode is for metrics that should be maximized (e.g., accuracy).
        In 'auto' mode, the direction is automatically inferred from the monitored metric's name.
        """

        # Check if mode is 'min' (minimize the monitored quantity)
        if self.mode == "min":

            # Set monitor_op to 'less' since we want the value to decrease
            self.monitor_op = ops.less

        # Check if mode is 'max' (maximize the monitored quantity)
        elif self.mode == "max":

            # Set monitor_op to 'greater' since we want the value to increase
            self.monitor_op = ops.greater

        else:
            # If mode is 'auto', infer the direction from the metric name
            # Remove "val_" prefix if present (as we are looking at the main metric)
            metric_name = self.monitor.removeprefix("val_")

            # If the monitored metric is 'loss', it should be minimized
            if metric_name == "loss":
                self.monitor_op = ops.less

            # Check if the model has any custom metrics defined
            if hasattr(self.model, "metrics"):
                # Initialize a list to store all metrics, including custom ones
                all_metrics = []
                # Iterate through all model's metrics
                for m in self.model.metrics:
                    if isinstance(
                            m,
                            (
                                    compile_utils.CompileMetrics,
                                    compile_utils.MetricsList,
                            ),
                    ):
                        all_metrics.extend(m.metrics)

                # Check each metric's name to match with the monitored one
                for m in all_metrics:

                    if m.name == metric_name:

                        # If the metric has a defined direction, use it to set monitor_op
                        if hasattr(m, "_direction"):

                            if m._direction == "up":
                                self.monitor_op = ops.greater

                            else:
                                self.monitor_op = ops.less

        # Raise an error if monitor_op was not set, which means we couldn't determine the direction
        if self.monitor_op is None:
            raise ValueError(f"EarlyStopping callback received monitor={self.monitor} "
                             "but Keras isn't able to automatically determine whether "
                             "that metric should be maximized or minimized. "
                             "Pass `mode='max'` in order to do early stopping based "
                             "on the highest metric value, or pass `mode='min'` "
                             "in order to use the lowest value.")

        # If the monitor_op is set to 'less', we adjust min_delta to be positive
        # This is because we are looking for a decrease in the monitored value
        if self.monitor_op == ops.less:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        """Initialize tracking variables when training begins"""
        self.wait = 0
        self.stopped_epoch = 0
        self.best = None
        self.best_weights = None
        self.best_epoch = 0
        logging.debug("Training started. Monitoring the metric: %s", self.monitor)

    def on_epoch_end(self, epoch, logs=None):
        """Check for improvement at the end of each epoch"""
        if self.monitor_op is None:
            # Set monitor operation if not already set
            self._set_monitor_op()

        current = self.get_monitor_value(logs)
        if current is None or epoch < self.start_from_epoch:
            logging.debug("No monitor value found or still in warm-up stage.")
            return

        if self.restore_best_weights and self.best_weights is None:
            # If best weights have never been set, initialize them with current weights
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            # Restart wait if we beat the baseline or our previous best
            if self.baseline is None or self._is_improvement(current, self.baseline):
                self.wait = 0
            logging.debug("Improvement found at epoch %d", epoch)
            return

        if self.wait >= self.patience and epoch > 0:
            # Stop training if patience threshold is exceeded
            self.stopped_epoch = epoch
            self.model.stop_training = True
            logging.debug("Patience exceeded, stopping training at epoch %d", epoch)

    def on_train_end(self, logs=None):
        """Restore weights and log when training ends"""
        if self.stopped_epoch > 0 and self.verbose > 0:
            io_utils.print_msg(f"Epoch {self.stopped_epoch + 1}: early stopping")
        if self.restore_best_weights and self.best_weights is not None:
            if self.verbose > 0:
                io_utils.print_msg(
                    "Restoring model weights from "
                    f"the best epoch: {self.best_epoch + 1}."
                )
            self.model.set_weights(self.best_weights)

    def get_monitor_value(self, logs):
        """Get the value of the monitored metric"""
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                f"Early stopping conditioned on metric `{self.monitor}` "
                "which is not available. "
                f"Available metrics are: {','.join(list(logs.keys()))}",
                stacklevel=2,
            )
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        """Check if there is an improvement"""
        if reference_value is None:
            return True
        return self.monitor_op(monitor_value - self.min_delta, reference_value)