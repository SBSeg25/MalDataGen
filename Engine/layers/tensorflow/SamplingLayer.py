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

    from tensorflow.keras.layers import Layer

except ImportError as error:
    print(error)
    sys.exit(-1)

class LayerSampling:

    class Sampling(Layer):
        """
        A custom Keras layer that performs the reparameterization trick for Variational Autoencoders (VAEs).

        This layer generates a latent vector by sampling from a Gaussian distribution
        parameterized by the mean (`z_mean`) and the logarithm of the variance (`z_log_var`).
        The reparameterization trick allows gradients to be backpropagated through
        the sampling process during training.

        This method is inspired by the work of Kingma and Welling in their paper:

        Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. In Proceedings of the 2nd International
        Conference on Learning Representations (ICLR).

        Methods:
            @call(inputs):
                Generates a latent vector using the reparameterization trick.

        Static Methods:
            @call(inputs):
                Receives the mean and log-variance of a latent Gaussian distribution,
                samples from it using random noise, and returns the sampled latent vector.

        Attributes:
            None explicitly defined for this layer.
        """

        @staticmethod
        def call(inputs):
            """
            Performs the reparameterization trick to sample a latent vector.

            Args:
                inputs (tuple): A tuple containing:
                    - z_mean (Tensor): The mean of the latent Gaussian distribution.
                    - z_log_var (Tensor): The logarithm of the variance of the latent Gaussian distribution.

            Returns:
                Tensor: A tensor representing the sampled latent vector.

            """
            # Unpack the input tuple into mean and log variance vectors
            z_mean, z_log_var = inputs

            # Get the batch size (number of samples) from the mean vector shape
            batch = tensorflow.shape(z_mean)[0]

            # Get the latent space dimension from the mean vector shape
            dimension = tensorflow.shape(z_mean)[1]

            # Generate random normal noise with same shape as input (batch_size × latent_dim)
            # This epsilon serves as the stochastic component of the sampling
            epsilon = tensorflow.random.normal(shape=(batch, dimension))

            # Apply the reparameterization trick:
            # 1. Scale the log variance (0.5*exp converts log_var to std dev)
            # 2. Multiply by random noise (epsilon)
            # 3. Add to the mean vector
            # This allows backpropagation through the random sampling operation

            return z_mean + tensorflow.exp(0.5 * z_log_var) * epsilon
