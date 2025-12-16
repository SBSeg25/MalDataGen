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

DEFAULT_EARLY_STOP_USE = False
DEFAULT_EARLY_STOP_MONITOR = "val_loss"
DEFAULT_STOP_MIN_DELTA = 0.0001
DEFAULT_EARLY_STOP_PATIENCE = 50
DEFAULT_EARLY_STOP_MODE = "auto"
DEFAULT_EARLY_STOP_BASELINE = None
DEFAULT_EARLY_STOP_RESTORE_BEST_WEIGHTS = None
DEFAULT_EARLY_STOP_START_FROM_EPOCH = 100


def add_argument_early_stop(parser):
    parser.add_argument("--use_early_stop", type=bool, default=DEFAULT_EARLY_STOP_USE,
                        help="Metric to be monitored. Default is 'val_loss'.")

    parser.add_argument("--early_stop_monitor", type=str, default=DEFAULT_EARLY_STOP_MONITOR,
                        help="Metric to be monitored. Default is 'val_loss'.")

    parser.add_argument("--early_stop_min_delta", type=float, default=DEFAULT_STOP_MIN_DELTA,
                        help="Minimum change in the monitored metric to qualify as an improvement. Default is 0.")

    parser.add_argument("--early_stop_patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE,
                        help="Number of epochs with no improvement after which training will be stopped. Default is 0.")

    parser.add_argument("--early_stop_mode", type=str, choices=["auto", "min", "max"], default=DEFAULT_EARLY_STOP_MODE,
                        help="Mode for monitoring: 'min' stops when metric decreases, 'max' when it increases,"
                             " 'auto' infers mode based on metric.")

    parser.add_argument("--early_stop_baseline", type=float, default=DEFAULT_EARLY_STOP_BASELINE,
                        help="Baseline value for the monitored metric. Training stops if it doesn't improve. Default is None.")

    parser.add_argument("--early_stop_restore_best_weights", action=DEFAULT_EARLY_STOP_RESTORE_BEST_WEIGHTS,
                        help="If set, restores model weights from the epoch with the best monitored value. Default is False.")

    parser.add_argument("--early_stop_start_from_epoch", type=int, default=DEFAULT_EARLY_STOP_START_FROM_EPOCH,
                        help="Epoch number from which to start monitoring. Default is 0.")

    return parser
