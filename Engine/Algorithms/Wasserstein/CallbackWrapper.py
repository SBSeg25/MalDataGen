#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Safe Callback Wrapper for WGAN-GP Training

This wrapper ensures callbacks work properly even if they expect
different interfaces (e.g., Keras-style callbacks).
"""

import time
from typing import Dict, Any, Optional


class CallbackWrapper:
    """
    Wrapper to make callbacks compatible with PyTorch training loop.
    """

    def __init__(self, callback):
        """
        Initialize the callback wrapper.

        Args:
            callback: The original callback object
        """
        self.callback = callback
        self.start_time = None
        self.epoch_start_time = None

    def on_train_begin(self):
        """Called at the beginning of training."""
        self.start_time = time.time()

        # Try to initialize callback data if it has a 'data' attribute
        if hasattr(self.callback, 'data'):
            if self.callback.data is None:
                self.callback.data = {}
            self.callback.data['start_time'] = self.start_time

        # Call the callback's on_train_begin if it exists
        if hasattr(self.callback, 'on_train_begin'):
            try:
                self.callback.on_train_begin()
            except Exception as e:
                print(f"Warning: Error in callback on_train_begin: {e}")

    def on_epoch_begin(self, epoch: int):
        """Called at the beginning of each epoch."""
        self.epoch_start_time = time.time()

        if hasattr(self.callback, 'on_epoch_begin'):
            try:
                self.callback.on_epoch_begin(epoch)
            except Exception as e:
                print(f"Warning: Error in callback on_epoch_begin: {e}")

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """Called at the end of each epoch."""
        # Add timing information
        if self.start_time:
            logs['start_time'] = self.start_time
        if self.epoch_start_time:
            logs['epoch_time'] = time.time() - self.epoch_start_time

        # Update callback data if it exists
        if hasattr(self.callback, 'data'):
            if self.callback.data is None:
                self.callback.data = {}
            self.callback.data.update(logs)
            if self.start_time:
                self.callback.data['start_time'] = self.start_time

        # Call the callback's on_epoch_end
        if hasattr(self.callback, 'on_epoch_end'):
            try:
                self.callback.on_epoch_end(epoch, logs)
            except TypeError:
                # Try without logs argument if it fails
                try:
                    self.callback.on_epoch_end(epoch)
                except Exception as e:
                    print(f"Warning: Error in callback on_epoch_end: {e}")
            except Exception as e:
                print(f"Warning: Error in callback on_epoch_end: {e}")

    def on_train_end(self):
        """Called at the end of training."""
        if hasattr(self.callback, 'on_train_end'):
            try:
                self.callback.on_train_end()
            except Exception as e:
                print(f"Warning: Error in callback on_train_end: {e}")


def wrap_callbacks(callbacks):
    """
    Wrap a list of callbacks to ensure compatibility.

    Args:
        callbacks: List of callback objects or None

    Returns:
        List of wrapped callbacks or None
    """
    if callbacks is None:
        return None

    wrapped = []
    for callback in callbacks:
        if callback is not None:
            wrapped.append(CallbackWrapper(callback))

    return wrapped if wrapped else None