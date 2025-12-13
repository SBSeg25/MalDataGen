#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/07'
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
    import logging

    framework = os.getenv("ML_FRAMEWORK", "Tensorflow").lower()

    if framework == "pytorch":
        # Import the CLASS, not the module
        from Engine.Algorithms.DenoisingDiffusion.Torch.GaussianDenoisingDiffusionTorch import \
            GaussianDiffusionTorch as GaussianDenoisingDiffusionBase
    else:
        # Import the CLASS, not the module
        from Engine.Algorithms.DenoisingDiffusion.Tensorflow.GaussianDenoisingDiffusionTensorflow import \
            GaussianDenoisingDiffusionTensorflow as GaussianDenoisingDiffusionBase

except ImportError as error:
    logging.error(f"Import error: {error}")
    sys.exit(-1)


class GaussianDenoisingDiffusion(GaussianDenoisingDiffusionBase):
    """
    Framework-agnostic Gaussian Denoising Diffusion wrapper.

    Automatically selects the appropriate implementation (TensorFlow or PyTorch)
    based on the ML_FRAMEWORK environment variable.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._framework = framework

    @property
    def framework(self):
        """Get the current framework being used."""
        return self._framework