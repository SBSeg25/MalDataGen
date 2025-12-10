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
    import tensorflow

    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.models import Model

    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Concatenate

    from tensorflow.keras.initializers import RandomNormal
    from Engine.Activations.Activations import Activations

except ImportError as error:
    print(error)
    sys.exit(-1)


class VanillaDecoderDiffusion(Activations):
    """
      VanillaDecoder

      This class implements a conditional fully connected decoder network, which reconstructs data from a
      latent representation. It extends from the `Activations` base class to inherit common activation
      utilities. The decoder can be conditioned on class labels or other side information, making it
      suitable for use in conditional autoencoders or conditional generative models.

      The architecture consists of a sequence of fully connected layers, each followed by an
      activation function and optional dropout for regularization. The final output layer can use
      a customizable activation function (e.g., sigmoid, tanh) depending on the desired output format.

      Attributes
      ----------
      @decoder_latent_dimension : int
          Dimensionality of the latent space (input to the decoder).

      @decoder_output_shape : int
          Dimensionality of the output data (reconstructed data).

      @decoder_intermediary_activation_function : Callable
          Activation function applied to intermediate layers.

      @decoder_last_layer_activation : Callable
          Activation function applied to the final layer.

      @decoder_dropout_decay_rate_decoder : float
          Dropout rate applied to the dense layers for regularization.

      @decoder_dataset_type : type
          Data type for inputs and outputs (default: numpy.float32).

      @decoder_initializer_mean : float
          Mean of the normal distribution used for weight initialization.

      @decoder_initializer_deviation : float
          Standard deviation of the normal distribution used for weight initialization.

      @decoder_number_neurons_decoder : List[int]
          List specifying the number of neurons in each dense layer.

      @decoder_number_samples_per_class : Optional[Dict[str, int]]
          Dictionary containing class metadata (e.g., number of samples per class). This allows
          the decoder to incorporate label conditioning if provided.

      @decoder_model : Optional[Model]
          Placeholder for the actual compiled Keras model (built later).

      Methods
      -------
      (The methods are defined elsewhere in the class and would typically include model building,
      forward pass, and any required conditional logic for class conditioning.)

      Notes
      -----
      This decoder is typically used in conditional autoencoder architectures, where the input
      to the decoder includes both a latent vector (encoding the data) and optional class
      information (one-hot encoded). This allows the decoder to reconstruct samples that
      match a particular class, improving generative flexibility.

      Raises
      ------
      ValueError
          Raised during initialization if any of the provided hyperparameters (e.g., layer sizes,
          dropout rates) are invalid.

      Examples
      --------
      >>> decoder = VanillaDecoderDiffusion(
      ...     latent_dimension=128,
      ...     output_shape=784,
      ...     activation_function='relu',
      ...     initializer_mean=0.0,
      ...     initializer_deviation=0.02,
      ...     dropout_decay_decoder=0.3,
      ...     last_layer_activation='sigmoid',
      ...     number_neurons_decoder=[256, 128, 64],
      ...     dataset_type=numpy.float32,
      ...     number_samples_per_class={"number_classes": 10}
      ... )
      >>> decoder.build_model()

      References
      ----------
      Kingma, D.P., & Welling, M. (2013). Auto-Encoding Variational Bayes.
      https://arxiv.org/abs/1312.6114
      """

    def __init__(self,
                 latent_dimension,
                 output_shape,
                 activation_function,
                 initializer_mean,
                 initializer_deviation,
                 dropout_decay_decoder,
                 last_layer_activation,
                 number_neurons_decoder,
                 dataset_type=numpy.float32,
                 number_samples_per_class=None):
        """
        Initializes the VanillaDecoder with the given hyperparameters.

        Parameters
        ----------
        @latent_dimension : int
            Dimensionality of the input latent space.

        @output_shape : int
            Dimensionality of the output (typically the same as the input to the encoder).

        @activation_function : str or callable
            Activation function for intermediate layers.

        @initializer_mean : float
            Mean of the normal distribution for weight initialization.

        @initializer_deviation : float
            Standard deviation of the normal distribution for weight initialization.

        @dropout_decay_decoder : float
            Dropout rate applied to intermediate layers.

        @last_layer_activation : str or callable
            Activation function for the final layer (e.g., 'sigmoid' for normalized outputs).

        @number_neurons_decoder : list of int
            Number of neurons in each fully connected layer.

        @dataset_type : type, optional
            Data type for inputs and outputs (default: numpy.float32).

        @number_samples_per_class : dict, optional
            Optional metadata dictionary containing information about class counts
            (used for label conditioning, if applicable).

        Raises
        ------
        ValueError
            If any provided parameter has an invalid value (e.g., non-positive layer sizes,
            invalid dropout rates).
        """

        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")

        if not isinstance(output_shape, int) or output_shape <= 0:
            raise ValueError("output_shape must be a positive integer.")

        if not isinstance(activation_function, (str, callable)):
            raise ValueError("activation_function must be a string or a callable function.")

        if not isinstance(initializer_mean, (float, int)):
            raise ValueError("initializer_mean must be a float or an integer.")

        if not isinstance(initializer_deviation, (float, int)) or initializer_deviation <= 0:
            raise ValueError("initializer_deviation must be a positive float or integer.")

        if not isinstance(dropout_decay_decoder, (float, int)) or not (0 <= dropout_decay_decoder <= 1):
            raise ValueError("dropout_decay_decoder must be a float between 0 and 1.")

        if not isinstance(last_layer_activation, (str, callable)):
            raise ValueError("last_layer_activation must be a string or a callable function.")

        if not isinstance(number_neurons_decoder, list) or not all(isinstance(n, int) and n > 0 for n in number_neurons_decoder):
            raise ValueError("number_neurons_decoder must be a list of positive integers.")

        if not isinstance(dataset_type, type):
            raise ValueError("dataset_type must be a valid data type.")

        if number_samples_per_class is not None:
            if not isinstance(number_samples_per_class, dict) or "number_classes" not in number_samples_per_class:
                raise ValueError("number_samples_per_class must be a dictionary containing the key 'number_classes'.")

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

    def get_decoder_trained(self):

        return self._decoder_model

    def get_decoder(self):
        """
        Builds and returns the decoder model.

        The model is a fully connected neural network that accepts both latent vectors and conditional inputs
        (e.g., class labels). It uses dropout and specified activation functions for regularization and non-linearity.

        Returns:
            tensorflow.keras.Model: The decoder model.
        """

        initialization = RandomNormal(mean=self._decoder_initializer_mean, stddev=self._decoder_initializer_deviation)

        # Input layers
        neural_model_inputs = Input(shape=(self._decoder_latent_dimension,), dtype=self._decoder_dataset_type)
        label_input = Input(shape=(self._decoder_number_samples_per_class["number_classes"],), dtype=self._decoder_dataset_type)

        # Concatenate latent vector with conditional labels
        concatenate_input = Concatenate()([neural_model_inputs, label_input])
        conditional_decoder = Dense(self._decoder_number_neurons_decoder[0],
                                    kernel_initializer=initialization)(concatenate_input)
        conditional_decoder = self._add_activation_layer(conditional_decoder, self._decoder_intermediary_activation_function)
        conditional_decoder = Dropout(self._decoder_dropout_decay_rate_decoder)(conditional_decoder)

        # Hidden layers
        for number_filters in self._decoder_number_neurons_decoder[1:]:
            conditional_decoder = Dense(number_filters, kernel_initializer=initialization)(conditional_decoder)
            conditional_decoder = Dropout(self._decoder_dropout_decay_rate_decoder)(conditional_decoder)
            conditional_decoder = self._add_activation_layer(conditional_decoder, self._decoder_intermediary_activation_function)

        # Output layer
        conditional_decoder = Dense(self._decoder_output_shape, kernel_initializer=initialization)(conditional_decoder)
        conditional_decoder = self._add_activation_layer(conditional_decoder, self._decoder_last_layer_activation)

        self._decoder_model = Model([neural_model_inputs, label_input], conditional_decoder, name="Decoder")

        return self._decoder_model

    @property
    def dropout_decay_rate_decoder(self):
        """float: Gets or sets the dropout decay rate for the decoder."""
        return self._decoder_dropout_decay_rate_decoder

    @property
    def number_filters_decoder(self):
        """list[int]: Gets the number of neurons for each layer in the decoder."""
        return self._decoder_number_neurons_decoder

    @dropout_decay_rate_decoder.setter
    def dropout_decay_rate_decoder(self, dropout_decay_rate_discriminator):
        """
        Sets the dropout decay rate for the decoder.

        Args:
            dropout_decay_rate_discriminator (float): New dropout decay rate.
        """
        self._decoder_dropout_decay_rate_decoder = dropout_decay_rate_discriminator
