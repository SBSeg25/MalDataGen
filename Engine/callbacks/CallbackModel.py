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
    import time
    import json
    import numpy

    import logging
    import tensorflow

    from tensorflow.keras.callbacks import Callback

except ImportError as error:
    print(error)
    sys.exit(-1)


class ModelMonitorCallback(Callback):

    def __init__(self, filename, k_fold):
        """
        Custom callback to log training progress and metrics to a JSON file.

        Args:
            filename (str): Name of the JSON file to save the training logs.
        """
        super(ModelMonitorCallback, self).__init__()
        self.filename = '{}/monitor_model_{}_fold.json'.format(filename, k_fold)
        self.batch_times = None
        self.epoch_start_time = None
        self.batch_start_time = None
        self.data = {
            "start_time": None,
            "elapsed_time": None,
            "expected_finish": None,
            "epochs": [],
            "avg_epoch_time": [],
            "avg_batch_time": [],
            "total_time": None,
            "status": "Normal"
        }

    def on_train_begin(self, logs=None):
        """
        Called at the beginning of training.

        Args:
            logs (dict): Currently no data.
        """
        self.data["start_time"] = time.time()
        self.batch_times = []
        logging.info("Training started...")
        print("Training started...")

    def on_epoch_begin(self, epoch, logs=None):
        """
        Called at the beginning of each epoch.

        Args:
            epoch (int): Current epoch number.
            logs (dict): Currently no data.
        """
        self.epoch_start_time = time.time()
        logging.info(f"Epoch {epoch + 1}/{self.params['epochs']} started...")
        print(f"Epoch {epoch + 1}/{self.params['epochs']} started...")

    def on_batch_begin(self, batch, logs=None):
        """
        Called at the beginning of each batch.

        Args:
            batch (int): Current batch number.
            logs (dict): Currently no data.
        """
        self.batch_start_time = time.time()
        logging.debug(f"Batch {batch} started...")

    def on_batch_end(self, batch, logs=None):
        """
        Called at the end of each batch.

        Args:
            batch (int): Current batch number.
            logs (dict): Currently no data.
        """
        batch_time = time.time() - self.batch_start_time
        self.batch_times.append(batch_time)
        logging.debug(f"Batch {batch} ended. Time taken: {batch_time:.4f}s")


    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch.

        Args:
            epoch (int): Current epoch number.
            logs (dict): Dictionary containing metrics for the current epoch.
        """
        if logs is None:
            logs = {}

        if not hasattr(self, "epoch_start_time") or self.epoch_start_time is None:
            logging.warning("Epoch start time not set. Using current time.")
            self.epoch_start_time = time.time()

        epoch_time = time.time() - self.epoch_start_time

        avg_batch_time = numpy.mean(self.batch_times) if self.batch_times else 0.0

        self.data["epochs"].append({
            "epoch": epoch + 1,
            "time": epoch_time,
            "avg_batch_time": avg_batch_time,
            "metrics": logs
        })
        self.batch_times = []

        elapsed_time = time.time() - self.data.get("start_time", time.time())
        self.data["elapsed_time"] = elapsed_time

        avg_epoch_time = numpy.mean([e["time"] for e in self.data["epochs"]]) if self.data["epochs"] else epoch_time

        remaining_epochs = max(self.params["epochs"] - (epoch + 1), 0)
        expected_finish = time.time() + remaining_epochs * avg_epoch_time
        self.data["expected_finish"] = expected_finish

        logging.info(
            f"Epoch {epoch + 1} ended. Time: {epoch_time:.4f}s, "
            f"Avg batch time: {avg_batch_time:.4f}s, "
            f"Expected finish: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(expected_finish))}"
        )

    def on_train_end(self, logs=None):
        """
        Called at the end of training.

        Args:
            logs (dict): Currently no data.
        """
        total_time = time.time() - self.data["start_time"]
        avg_epoch_time = numpy.mean([epoch["time"] for epoch in self.data["epochs"]])
        avg_batch_time = numpy.mean([epoch["avg_batch_time"] for epoch in self.data["epochs"]])
        self.data["total_time"] = total_time
        self.data["avg_epoch_time"] = avg_epoch_time
        self.data["avg_batch_time"] = avg_batch_time

        logging.info(f"Training ended. Total time: {total_time:.4f}s, Avg epoch time: {avg_epoch_time:.4f}s")

        # By default, assume the training finishes successfully
        with open(self.filename, "w") as f:
            json.dump(self.data, f, indent=4)

        logging.info(f"Training logs saved to {self.filename}")
