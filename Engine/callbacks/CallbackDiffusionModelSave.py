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
    import os
    import sys

    import tensorflow

except ImportError as error:
    print(error)
    sys.exit(-1)

class SaveModelCallback(tensorflow.keras.callbacks.Callback):
    """
    A custom callback for saving the diffusion model and its optimizers' states at the end of each epoch during training.

    Attributes:
        @diffusion_model: A trained diffusion model to be saved.
        @save_dir: The directory where the model and optimizers' states will be saved.
    """

    def __init__(self, diffusion_model, save_dir):
        """
        Initializes the callback with the diffusion model and save directory.

        Args:
            diffusion_model: A trained diffusion model that should be saved.
            save_dir: Directory where the model and optimizers' states will be stored.
        """
        super().__init__()
        self.diffusion_model = diffusion_model
        self.save_dir = save_dir

    def on_epoch_end(self, epoch, logs=None):
        """
        This method is called at the end of each epoch. It creates the save directory (if it doesn't exist)
        and saves the model and optimizers' states for the current epoch.

        Args:
            epoch: The index of the epoch that just ended.
            logs: A dictionary containing logs of the current epoch's performance (optional).
        """
        os.makedirs(self.save_dir, exist_ok=True)  # Create the directory if it doesn't exist
        self.diffusion_model.save_models(self.save_dir, epoch)  # Save the model and optimizers' states
        print(f"Models and optimizers' states saved for epoch {epoch}.")