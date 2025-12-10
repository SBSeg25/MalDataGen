#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

from Engine.Architectures.LatentDiffusion.Torch.VanillaDecoderDiffusionTorch import VanillaDecoderDiffusionTorch
from Engine.Architectures.LatentDiffusion.Torch.VanillaEncoderDiffusionTorch import VanillaEncoderDiffusionTorch

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
    import torch.nn as nn

except ImportError as error:
    print(error)
    sys.exit(-1)


class VariationalModelDiffusionTorch(VanillaDecoderDiffusionTorch, VanillaEncoderDiffusionTorch):
    """
    A Variational Model that integrates both VanillaEncoder and VanillaDecoder
    functionalities. This class enables flexible configuration of encoder and
    decoder parameters, facilitating variational-based learning tasks.
    """

    def __init__(self,
                 latent_dimension,
                 output_shape,
                 activation_function,
                 initializer_mean,
                 initializer_deviation,
                 dropout_decay_encoder,
                 dropout_decay_decoder,
                 last_layer_activation,
                 number_neurons_encoder,
                 number_neurons_decoder,
                 dataset_type=numpy.float32,
                 number_samples_per_class=None):
        """
        Initializes the VariationalModel with user-defined encoder and decoder configurations.
        """
        # Initialize nn.Module ONCE
        nn.Module.__init__(self)

        # Manually initialize encoder attributes (without calling __init__)
        self._encoder_latent_dimension = latent_dimension
        self._encoder_output_shape = output_shape
        self._encoder_activation_function = activation_function
        self._encoder_last_layer_activation = last_layer_activation
        self._encoder_dropout_decay_rate_encoder = dropout_decay_encoder
        self._encoder_dataset_type = dataset_type
        self._encoder_initializer_mean = initializer_mean
        self._encoder_initializer_deviation = initializer_deviation
        self._encoder_number_neurons_encoder = number_neurons_encoder
        self._encoder_number_samples_per_class = number_samples_per_class
        self._encoder_model = None

        # Manually initialize decoder attributes (without calling __init__)
        self._decoder_latent_dimension = latent_dimension
        self._decoder_output_shape = output_shape
        self._decoder_intermediary_activation_function = activation_function
        self._decoder_last_layer_activation = last_layer_activation
        self._decoder_dropout_decay_rate_decoder = dropout_decay_decoder
        self._decoder_dataset_type = dataset_type
        self._decoder_initializer_mean = initializer_mean
        self._decoder_initializer_deviation = initializer_deviation
        self._decoder_number_neurons_decoder = number_neurons_decoder
        self._decoder_number_samples_per_class = number_samples_per_class
        self._decoder_model = None

        # Build both encoder and decoder
        VanillaEncoderDiffusionTorch._build_encoder(self)
        VanillaDecoderDiffusionTorch._build_decoder(self)

    def get_encoder(self):
        """Returns a wrapper that specifically calls the encoder forward method."""

        class EncoderWrapper(nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.parent = parent

            def forward(self, neural_model_inputs, label_input):
                """Forward pass through encoder only."""
                return VanillaEncoderDiffusionTorch.forward(self.parent, neural_model_inputs, label_input)

            def parameters(self, recurse=True):
                """Return encoder parameters only."""
                params = []
                for name, param in self.parent.named_parameters(recurse=recurse):
                    if 'encoder' in name or 'latent_mean' in name or 'latent_log_var' in name or 'final_layer' in name:
                        params.append(param)
                return iter(params)

            def __getattr__(self, name):
                if name in ['parent', '_modules', '_parameters', '_buffers']:
                    return super().__getattr__(name)
                return getattr(self.parent, name)

        return EncoderWrapper(self)

    def get_decoder(self):
        """Returns a wrapper that specifically calls the decoder forward method."""

        class DecoderWrapper(nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.parent = parent

            def forward(self, neural_model_inputs, label_input):
                """Forward pass through decoder only."""
                return VanillaDecoderDiffusionTorch.forward(self.parent, neural_model_inputs, label_input)

            def parameters(self, recurse=True):
                """Return decoder parameters only."""
                params = []
                for name, param in self.parent.named_parameters(recurse=recurse):
                    if 'decoder' in name or 'output_layer' in name:
                        params.append(param)
                return iter(params)

            def __getattr__(self, name):
                if name in ['parent', '_modules', '_parameters', '_buffers']:
                    return super().__getattr__(name)
                return getattr(self.parent, name)

        return DecoderWrapper(self)

    def get_encoder_trained(self):
        """Returns the trained encoder model (wrapper)."""
        return self.get_encoder()

    def get_decoder_trained(self):
        """Returns the trained decoder model (wrapper)."""
        return self.get_decoder()

    # Keep the existing setter methods unchanged
    def latent_dimension(self, latent_dimension):
        """Sets the latent dimension for both encoder and decoder."""
        self._encoder_latent_dimension = latent_dimension
        self._decoder_latent_dimension = latent_dimension

    def output_shape(self, output_shape):
        """Configures the output shape for both encoder and decoder."""
        self._encoder_output_shape = output_shape
        self._decoder_output_shape = output_shape

    def intermediary_activation_function(self, activation_function):
        """Configures the activation function for intermediary layers."""
        self._encoder_activation_function = activation_function
        self._decoder_activation_function = activation_function

    def last_layer_activation(self, last_layer_activation):
        """Configures the activation function for the last layer."""
        self._encoder_last_layer_activation = last_layer_activation
        self._decoder_last_layer_activation = last_layer_activation