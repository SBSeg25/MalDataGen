#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/10/29'
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
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import GRU
    from tensorflow.keras.layers import Bidirectional
    from tensorflow.keras.models import Model

    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Concatenate

    from Engine.Activations.Activations import Activations
    from tensorflow.keras.initializers import RandomNormal

except ImportError as error:
    print(error)
    sys.exit(-1)


class TimeSeriesEncoder(Activations):
    """
    TimeSeriesEncoder

    A class representing a Time Series Encoder model for deep learning applications with temporal data.
    The encoder is designed to process sequential inputs (Time, Features) and labels, apply a series of
    recurrent layers (LSTM/GRU) with activations and dropout, and output a latent representation of the
    time series data. This model is typically used in applications such as time series autoencoders,
    forecasting models, anomaly detection, or other temporal generative models.

    Attributes:
        @encoder_latent_dimension (int):
            The dimensionality of the latent space that the model will output.
        @encoder_output_shape (tuple):
            The desired output shape of the encoder, defining the shape of the encoded representation.
        @encoder_activation_function (str):
            The activation function applied to each layer of the encoder (e.g., 'tanh', 'ReLU').
        @encoder_last_layer_activation (str):
            The activation function applied to the final output layer.
        @encoder_dropout_decay_rate_encoder (float):
            The rate of dropout applied during encoding to improve generalization (must be between 0 and 1).
        @encoder_number_units_encoder (list):
            A list specifying the number of units in each recurrent layer of the encoder network.
        @encoder_recurrent_layer_type (str):
            Type of recurrent layer to use: 'LSTM', 'GRU', or 'BiLSTM' (Bidirectional LSTM).
        @encoder_use_bidirectional (bool):
            Whether to use bidirectional recurrent layers.
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
            - `number_units_encoder` is not a non-empty list or contains non-positive integers.
            - `recurrent_layer_type` is not one of ['LSTM', 'GRU', 'BiLSTM'].
            - `number_samples_per_class` is provided but is not a dictionary.

    Example:
        >>> encoder = TimeSeriesEncoder(
        ...     latent_dimension=128,
        ...     output_shape=(100, 10),
        ...     activation_function='tanh',
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_encoder=0.3,
        ...     last_layer_activation='sigmoid',
        ...     number_units_encoder=[256, 128, 64],
        ...     recurrent_layer_type='LSTM',
        ...     use_bidirectional=False,
        ...     dataset_type=numpy.float32,
        ...     number_samples_per_class={"number_classes": 10}
        ... )
    """

    def __init__(self, latent_dimension: int, output_shape: tuple, activation_function: str, initializer_mean: float,
                 initializer_deviation: float, dropout_decay_encoder: float, last_layer_activation: str,
                 number_units_encoder: list, recurrent_layer_type: str = 'LSTM', use_bidirectional: bool = False,
                 dataset_type: Any = numpy.float32, number_samples_per_class: Optional[Dict[str, Any]] = None):
        """
        Initializes the TimeSeriesEncoder with the provided parameters.

        Args:
            latent_dimension (int): The dimension of the latent space.
            output_shape (tuple): The desired output shape of the encoder (timesteps, features).
            activation_function (str): The activation function to use for the layers.
            initializer_mean (float): The mean for weight initialization.
            initializer_deviation (float): The standard deviation for weight initialization.
            dropout_decay_encoder (float): The rate of dropout applied during encoding.
            last_layer_activation (str): The activation function for the last layer.
            number_units_encoder (list): List specifying the number of units in each recurrent layer.
            recurrent_layer_type (str, optional): Type of recurrent layer ('LSTM', 'GRU', 'BiLSTM'). Defaults to 'LSTM'.
            use_bidirectional (bool, optional): Whether to use bidirectional layers. Defaults to False.
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

        if not isinstance(number_units_encoder, list) or len(number_units_encoder) == 0:
            raise ValueError("number_units_encoder must be a non-empty list.")

        for units in number_units_encoder:
            if not isinstance(units, int) or units <= 0:
                raise ValueError("Each element in number_units_encoder must be a positive integer.")

        if recurrent_layer_type not in ['LSTM', 'GRU', 'BiLSTM']:
            raise ValueError("recurrent_layer_type must be one of ['LSTM', 'GRU', 'BiLSTM'].")

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
        self._encoder_number_units_encoder = number_units_encoder
        self._encoder_recurrent_layer_type = recurrent_layer_type
        self._encoder_use_bidirectional = use_bidirectional
        self._encoder_number_samples_per_class = number_samples_per_class

    def _create_recurrent_layer(self, units: int, return_sequences: bool, initialization):
        """
        Creates a recurrent layer based on the specified type.

        Args:
            units (int): Number of units in the recurrent layer.
            return_sequences (bool): Whether to return sequences or just the last output.
            initialization: Weight initialization strategy.

        Returns:
            A Keras recurrent layer (LSTM, GRU, or Bidirectional LSTM).
        """
        if self._encoder_recurrent_layer_type == 'LSTM':
            layer = LSTM(units, return_sequences=return_sequences, 
                        kernel_initializer=initialization,
                        recurrent_initializer=initialization)
        elif self._encoder_recurrent_layer_type == 'GRU':
            layer = GRU(units, return_sequences=return_sequences,
                       kernel_initializer=initialization,
                       recurrent_initializer=initialization)
        elif self._encoder_recurrent_layer_type == 'BiLSTM':
            layer = Bidirectional(LSTM(units, return_sequences=return_sequences,
                                      kernel_initializer=initialization,
                                      recurrent_initializer=initialization))
        
        if self._encoder_use_bidirectional and self._encoder_recurrent_layer_type != 'BiLSTM':
            return Bidirectional(layer)
        return layer

    def get_encoder(self, input_shape: tuple) -> Model:
        """
        Creates and returns the time series encoder model.

        This method constructs the neural network by stacking recurrent layers with the provided
        configurations (units, dropout, and activation). It processes sequential data (Time, Features)
        and concatenates labels before encoding to latent space.

        Args:
            input_shape (tuple): The shape of the input time series data (timesteps, features).

        Returns:
            keras.Model: The encoder model which takes input time series data and labels and outputs
                         the encoded latent representation and labels.
        """

        # Initialize the weights using a normal distribution with specified mean and deviation
        initialization = RandomNormal(mean=self._encoder_initializer_mean, 
                                     stddev=self._encoder_initializer_deviation)

        # Define input layers for time series data and labels
        neural_model_inputs = Input(shape=input_shape, dtype=self._encoder_dataset_type, name="timeseries_input")
        label_input = Input(shape=(self._encoder_number_samples_per_class["number_classes"],),
                           dtype=self._encoder_dataset_type, name="label_input")

        # Process time series through recurrent layers
        conditional_encoder = neural_model_inputs

        # Apply recurrent layers
        for idx, number_units in enumerate(self._encoder_number_units_encoder):
            # Return sequences for all layers except the last one
            return_sequences = (idx < len(self._encoder_number_units_encoder) - 1)
            
            conditional_encoder = self._create_recurrent_layer(
                number_units, return_sequences, initialization
            )(conditional_encoder)
            conditional_encoder = Dropout(self._encoder_dropout_decay_rate_encoder)(conditional_encoder)
            
            # Apply activation if specified
            if self._encoder_activation_function:
                conditional_encoder = self._add_activation_layer(
                    conditional_encoder, self._encoder_activation_function
                )

        # Flatten the output from recurrent layers if needed
        if len(conditional_encoder.shape) > 2:
            conditional_encoder = Flatten()(conditional_encoder)

        # Concatenate encoded time series with labels
        concatenate_encoded = Concatenate()([conditional_encoder, label_input])

        # Map to the latent space through dense layers
        latent_representation = Dense(self._encoder_latent_dimension, 
                                     kernel_initializer=initialization)(concatenate_encoded)
        latent_representation = self._add_activation_layer(
            latent_representation, self._encoder_last_layer_activation
        )

        # Return the encoder model
        return Model([neural_model_inputs, label_input], [latent_representation, label_input], 
                    name="TimeSeriesEncoder")

    @property
    def dropout_decay_rate_encoder(self) -> float:
        """
        Gets the rate of dropout decay for the encoder layers.

        Returns:
            float: The rate of dropout decay applied to the encoder layers.
        """
        return self._encoder_dropout_decay_rate_encoder

    @property
    def number_units_encoder(self) -> list:
        """
        Gets the number of units for each encoder recurrent layer.

        Returns:
            list: A list specifying the number of units in each encoder layer.
        """
        return self._encoder_number_units_encoder

    @property
    def recurrent_layer_type(self) -> str:
        """
        Gets the type of recurrent layer used in the encoder.

        Returns:
            str: The recurrent layer type ('LSTM', 'GRU', or 'BiLSTM').
        """
        return self._encoder_recurrent_layer_type

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

    @recurrent_layer_type.setter
    def recurrent_layer_type(self, layer_type: str) -> None:
        """
        Sets the type of recurrent layer for the encoder.

        Args:
            layer_type (str): The recurrent layer type ('LSTM', 'GRU', or 'BiLSTM').

        Raises:
            ValueError: If the layer type is not one of the supported types.
        """
        if layer_type not in ['LSTM', 'GRU', 'BiLSTM']:
            raise ValueError("recurrent_layer_type must be one of ['LSTM', 'GRU', 'BiLSTM'].")

        self._encoder_recurrent_layer_type = layer_type
