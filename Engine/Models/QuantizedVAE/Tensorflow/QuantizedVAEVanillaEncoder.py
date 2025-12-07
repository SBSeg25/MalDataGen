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

    from typing import Any
    from typing import Dict
    from typing import Optional
    
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.models import Model

    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Flatten

    from tensorflow.keras.layers import Concatenate

    from Engine.Activations.Activations import Activations
    from tensorflow.keras.initializers import RandomNormal

except ImportError as error:
    print(error)
    sys.exit(-1)


class QuantizedVAEVanillaEncoder(Activations):
    """
    VanillaEncoder

    A class representing a Vanilla Encoder model for deep learning applications. The encoder
    is designed to process inputs and labels, apply a series of dense layers with activations
    and dropout, and output a latent representation of the input data. This model is typically
    used in applications such as autoencoders, variational autoencoders, or other generative models.

    Attributes:
        @encoder_latent_dimension (int):
            The dimensionality of the latent space that the model will output.
        @encoder_output_shape (tuple):
            The desired output shape of the encoder, defining the shape of the encoded representation.
        @encoder_activation_function (str):
            The activation function applied to each layer of the encoder (e.g., 'ReLU', 'LeakyReLU').
        @encoder_last_layer_activation (str):
            The activation function applied to the final output layer.
        @encoder_dropout_decay_rate_encoder (float):
            The rate of dropout applied during encoding to improve generalization (must be between 0 and 1).
        @encoder_number_neurons_encoder (list):
            A list specifying the number of neurons (or units) in each layer of the encoder network.
        @encoder_dataset_type (dtype):
            The data type of the input dataset, default is numpy.float32.
        @encoder_initializer_mean (float):
            The mean for the normal distribution used to initialize the weights.
        @encoder_initializer_deviation (float):
            The standard deviation for the normal distribution used to initialize the weights.
        @encoder_number_samples_per_class (Optional[dict]):
            An optional dictionary containing metadata about the number of samples per class.

    Raises:
        ValueError:
            Raised when the following invalid arguments are passed during initialization:
            - `latent_dimension` is not a positive integer.
            - `initializer_mean` or `initializer_deviation` is not a number.
            - `dropout_decay_encoder` is outside the valid range [0, 1].
            - `number_neurons_encoder` is not a non-empty list or contains non-positive integers.
            - `number_samples_per_class` is provided but is not a dictionary.

    Example:
        >>> encoder = VanillaEncoder(
        ...     latent_dimension=128,
        ...     output_shape=(64, 64, 1),
        ...     activation_function='ReLU',
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_encoder=0.5,
        ...     last_layer_activation='sigmoid',
        ...     number_neurons_encoder=[512, 256, 128],
        ...     dataset_type=numpy.float32,
        ...     number_samples_per_class={"number_classes": 10}
        ... )
    """

    def __init__(self, latent_dimension: int, output_shape: tuple, activation_function: str, initializer_mean: float,
                 initializer_deviation: float, dropout_decay_encoder: float, last_layer_activation: str,
                 number_neurons_encoder: list, dataset_type: Any = numpy.float32,
                 number_samples_per_class: Optional[Dict[str, Any]] = None):
        """
        Initializes the VanillaEncoder with the provided parameters.

        Args:
            latent_dimension (int): The dimension of the latent space.
            output_shape (tuple): The desired output shape of the encoder.
            activation_function (str): The activation function to use for the layers.
            initializer_mean (float): The mean for weight initialization.
            initializer_deviation (float): The standard deviation for weight initialization.
            dropout_decay_encoder (float): The rate of dropout applied during encoding.
            last_layer_activation (str): The activation function for the last layer.
            number_neurons_encoder (list): List specifying the number of neurons in each encoder layer.
            dataset_type (dtype, optional): The data type of the input dataset. Defaults to numpy.float32.
            number_samples_per_class (dict, optional): Specifies the number of samples per class.
        """


        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")

        if not isinstance(initializer_mean, (int, float)):
            raise ValueError("initializer_mean must be a number.")

        if not isinstance(initializer_deviation, (int, float)):
            raise ValueError("initializer_deviation must be a number.")

        if not isinstance(dropout_decay_encoder, (int, float)) or not (0 <= dropout_decay_encoder <= 1):
            raise ValueError("dropout_decay_encoder must be a float between 0 and 1.")

        if not isinstance(number_neurons_encoder, list) or len(number_neurons_encoder) == 0:
            raise ValueError("number_neurons_encoder must be a non-empty list.")

        for neurons in number_neurons_encoder:
            if not isinstance(neurons, int) or neurons <= 0:
                raise ValueError("Each element in number_neurons_encoder must be a positive integer.")

        if number_samples_per_class is not None:

            if not isinstance(number_samples_per_class, dict):
                raise ValueError("number_samples_per_class must be a dictionary.")

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

    def get_encoder_model(self, input_shape: int) -> Model:
        """
        Creates and returns the encoder model.

        This method constructs the neural network by stacking dense layers with the provided
        configurations (neurons, dropout, and activation). It also concatenates the input data
        and labels before passing through the layers.

        Args:
            input_shape (tuple): The shape of the input data.

        Returns:
            keras.Model: The encoder model which takes input data and labels and outputs the
                          encoded latent representation and labels.
        """

        if not isinstance(input_shape, int) or input_shape <= 0:
            raise ValueError(f"Invalid value for output_shape: {input_shape}. It must be a positive integer.")

        # Initialize weights with a normal distribution
        initialization = RandomNormal(mean=self._encoder_initializer_mean, stddev=self._encoder_initializer_deviation)

        # Define input layers
        neural_model_inputs = Input(shape=(input_shape,), dtype=self._encoder_dataset_type, name="first_input")
        label_input = Input(shape=(self._encoder_number_samples_per_class["number_classes"],),
                            dtype=self._encoder_dataset_type, name="second_input")

        # Concatenate inputs
        concatenate_input = Concatenate()([neural_model_inputs, label_input])

        conditional_encoder = Dense(self._encoder_number_neurons_encoder[0],
                                    kernel_initializer=initialization)(concatenate_input)
        conditional_encoder = Dropout(self._encoder_dropout_decay_rate_encoder)(conditional_encoder)
        conditional_encoder = self._add_activation_layer(conditional_encoder, self._encoder_activation_function)

        # Iterate over specified dense layers
        for number_neurons in self._encoder_number_neurons_encoder[1:]:
            conditional_encoder = Dense(number_neurons, kernel_initializer=initialization)(conditional_encoder)
            conditional_encoder = Dropout(self._encoder_dropout_decay_rate_encoder)(conditional_encoder)
            conditional_encoder = self._add_activation_layer(conditional_encoder, self._encoder_activation_function)

        # Map to the latent space
        conditional_encoder = Dense(self._encoder_latent_dimension, kernel_initializer=initialization)(
            conditional_encoder)
        conditional_encoder = self._add_activation_layer(conditional_encoder, self._encoder_last_layer_activation)

        # Return the encoder model
        return Model([neural_model_inputs, label_input], [conditional_encoder, label_input], name="Encoder")

    @property
    def dropout_decay_rate_encoder(self) -> float:
        """
        Gets the rate of dropout decay for the encoder layers.

        Returns:
            float: The rate of dropout decay applied to the encoder layers.
        """
        return self._encoder_dropout_decay_rate_encoder

    @property
    def number_filters_encoder(self) -> list:
        """
        Gets the number of neurons for each encoder layer.

        Returns:
            list: A list specifying the number of neurons in each encoder layer.
        """
        return self._encoder_number_neurons_encoder

    @dropout_decay_rate_encoder.setter
    def dropout_decay_rate_encoder(self, dropout_decay_rate_generator: float) -> None:
        """
        Sets the rate of dropout decay for the encoder layers.

        Args:
            dropout_decay_rate_generator (float): The new dropout decay rate.

        Raises:
            ValueError: If the value is not a float between 0 and 1.
        """
        if not (0 <= dropout_decay_rate_generator <= 1):
            raise ValueError("dropout_decay_rate_encoder must be a float between 0 and 1.")

        self._encoder_dropout_decay_rate_encoder = dropout_decay_rate_generator
