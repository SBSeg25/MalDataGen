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

    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import GRU
    from tensorflow.keras.layers import Bidirectional
    from tensorflow.keras.layers import RepeatVector
    from tensorflow.keras.layers import TimeDistributed
    from tensorflow.keras.models import Model

    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Reshape
    from tensorflow.keras.layers import Concatenate

    from tensorflow.keras.initializers import RandomNormal
    from Engine.Activations.Activations import Activations

except ImportError as error:
    print(error)
    sys.exit(-1)


class TimeSeriesDecoder(Activations):
    """
    TimeSeriesDecoder

    A class representing a conditional Time Series Decoder model for deep learning applications with
    temporal data. The decoder is designed to process a latent representation and labels, apply a series
    of recurrent layers (LSTM/GRU) with activations and dropout, and output a reconstructed time series
    in the format (Time, Features). This class is typically used in tasks such as time series autoencoders,
    forecasting models, sequence-to-sequence models, and temporal generative models.

    Attributes:
        @decoder_latent_dimension (int):
            The dimensionality of the latent space input, which the decoder will use to generate outputs.
        @decoder_output_shape (tuple):
            The shape of the output time series (timesteps, features).
        @decoder_activation_function (str):
            The activation function applied to each layer of the decoder (e.g., 'tanh', 'ReLU').
        @decoder_last_layer_activation (str):
            The activation function applied to the final output layer.
        @decoder_dropout_decay_rate_decoder (float):
            The rate of dropout applied during decoding to improve generalization (must be between 0 and 1).
        @decoder_number_units_decoder (list):
            A list specifying the number of units in each recurrent layer of the decoder network.
        @decoder_recurrent_layer_type (str):
            Type of recurrent layer to use: 'LSTM', 'GRU', or 'BiLSTM' (Bidirectional LSTM).
        @decoder_use_bidirectional (bool):
            Whether to use bidirectional recurrent layers.
        @decoder_dataset_type (dtype):
            The data type of the input dataset, default is numpy.float32.
        @decoder_initializer_mean (float):
            The mean for the normal distribution used to initialize the weights.
        @decoder_initializer_deviation (float):
            The standard deviation for the normal distribution used to initialize the weights.
        @decoder_number_samples_per_class (Optional[dict]):
            An optional dictionary containing metadata about the number of classes for label input.

    Raises:
        ValueError:
            Raised when the following invalid arguments are passed during initialization:
            - `latent_dimension` is not a positive integer.
            - `output_shape` is not a tuple of two positive integers (timesteps, features).
            - `activation_function`, `last_layer_activation` are not strings.
            - `initializer_mean` or `initializer_deviation` are not numbers.
            - `dropout_decay_decoder` is outside the valid range [0, 1].
            - `number_units_decoder` is not a list of positive integers.
            - `recurrent_layer_type` is not one of ['LSTM', 'GRU', 'BiLSTM'].
            - `dataset_type` is not a valid type.
            - `number_samples_per_class` is provided but is not a dictionary containing 'number_classes'.

    Example:
        >>> decoder = TimeSeriesDecoder(
        ...     latent_dimension=64,
        ...     output_shape=(100, 10),  # 100 timesteps, 10 features
        ...     activation_function='tanh',
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_decoder=0.3,
        ...     last_layer_activation='sigmoid',
        ...     number_units_decoder=[32, 64, 128],
        ...     recurrent_layer_type='LSTM',
        ...     use_bidirectional=False,
        ...     dataset_type=numpy.float32,
        ...     number_samples_per_class={"number_classes": 5}
        ... )
    """

    def __init__(self, latent_dimension: int, output_shape: tuple, activation_function: str, initializer_mean: float,
                 initializer_deviation: float, dropout_decay_decoder: float, last_layer_activation: str,
                 number_units_decoder: list[int], recurrent_layer_type: str = 'LSTM', use_bidirectional: bool = False,
                 dataset_type: type = numpy.float32, number_samples_per_class: dict = None):
        """
        Initializes the TimeSeriesDecoder class with the given configuration.

        Args:
            latent_dimension (int): Dimensionality of the latent space input.
            output_shape (tuple): Shape of the output time series (timesteps, features).
            activation_function (str): Activation function name.
            initializer_mean (float): Mean for the initializer.
            initializer_deviation (float): Standard deviation for the initializer.
            dropout_decay_decoder (float): Dropout rate for decoder layers.
            last_layer_activation (str): Activation function for the output layer.
            number_units_decoder (list[int]): Number of units in decoder recurrent layers.
            recurrent_layer_type (str, optional): Type of recurrent layer ('LSTM', 'GRU', 'BiLSTM'). Defaults to 'LSTM'.
            use_bidirectional (bool, optional): Whether to use bidirectional layers. Defaults to False.
            dataset_type (type): Data type for inputs/outputs (default is numpy.float32).
            number_samples_per_class (dict, optional): Number of classes for label input.

        Raises:
            ValueError: If any of the provided parameters are invalid.
        """

        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError(f"Invalid value for latent_dimension: {latent_dimension}. It must be a positive integer.")
        
        if not isinstance(output_shape, tuple) or len(output_shape) != 2 or not all(isinstance(x, int) and x > 0 for x in output_shape):
            raise ValueError(f"Invalid value for output_shape: {output_shape}. It must be a tuple of two positive integers (timesteps, features).")
        
        if not isinstance(activation_function, str):
            raise ValueError(f"Invalid value for activation_function: {activation_function}. It must be a string.")
        
        if not isinstance(initializer_mean, (int, float)):
            raise ValueError(f"Invalid value for initializer_mean: {initializer_mean}. It must be a number.")
        
        if not isinstance(initializer_deviation, (int, float)):
            raise ValueError(f"Invalid value for initializer_deviation: {initializer_deviation}. It must be a number.")
        
        if not isinstance(dropout_decay_decoder, (int, float)) or not (0 <= dropout_decay_decoder <= 1):
            raise ValueError(f"Invalid value for dropout_decay_decoder: {dropout_decay_decoder}. It must be a number between 0 and 1.")
        
        if not isinstance(last_layer_activation, str):
            raise ValueError(f"Invalid value for last_layer_activation: {last_layer_activation}. It must be a string.")
        
        if not isinstance(number_units_decoder, list) or not all(isinstance(x, int) and x > 0 for x in number_units_decoder):
            raise ValueError(f"Invalid value for number_units_decoder: {number_units_decoder}. It must be a list of positive integers.")
        
        if recurrent_layer_type not in ['LSTM', 'GRU', 'BiLSTM']:
            raise ValueError(f"Invalid value for recurrent_layer_type: {recurrent_layer_type}. It must be one of ['LSTM', 'GRU', 'BiLSTM'].")
        
        if not isinstance(dataset_type, type):
            raise ValueError(f"Invalid value for dataset_type: {dataset_type}. It must be a valid type.")
        
        if number_samples_per_class is not None and (not isinstance(number_samples_per_class, dict) or "number_classes" not in number_samples_per_class):
            raise ValueError(f"Invalid value for number_samples_per_class: {number_samples_per_class}. It must be a dictionary with 'number_classes'.")

        self._decoder_latent_dimension = latent_dimension
        self._decoder_output_shape = output_shape
        self._decoder_activation_function = activation_function
        self._decoder_last_layer_activation = last_layer_activation
        self._decoder_dropout_decay_rate_decoder = dropout_decay_decoder
        self._decoder_dataset_type = dataset_type
        self._decoder_initializer_mean = initializer_mean
        self._decoder_initializer_deviation = initializer_deviation
        self._decoder_number_units_decoder = number_units_decoder
        self._decoder_recurrent_layer_type = recurrent_layer_type
        self._decoder_use_bidirectional = use_bidirectional
        self._decoder_number_samples_per_class = number_samples_per_class

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
        if self._decoder_recurrent_layer_type == 'LSTM':
            layer = LSTM(units, return_sequences=return_sequences, 
                        kernel_initializer=initialization,
                        recurrent_initializer=initialization)
        elif self._decoder_recurrent_layer_type == 'GRU':
            layer = GRU(units, return_sequences=return_sequences,
                       kernel_initializer=initialization,
                       recurrent_initializer=initialization)
        elif self._decoder_recurrent_layer_type == 'BiLSTM':
            layer = Bidirectional(LSTM(units, return_sequences=return_sequences,
                                      kernel_initializer=initialization,
                                      recurrent_initializer=initialization))
        
        if self._decoder_use_bidirectional and self._decoder_recurrent_layer_type != 'BiLSTM':
            return Bidirectional(layer)
        return layer

    def get_decoder(self, output_shape: tuple):
        """
        Constructs and returns the time series decoder model.

        This method constructs the neural network by processing the latent representation concatenated
        with labels through dense layers, then expanding to a sequence using RepeatVector, and finally
        processing through recurrent layers to output the reconstructed time series.

        Args:
            output_shape (tuple): The output shape of the time series (timesteps, features).

        Returns:
            keras.Model: The constructed decoder model.

        Raises:
            ValueError: If the output shape is invalid.
        """
        if not isinstance(output_shape, tuple) or len(output_shape) != 2 or not all(isinstance(x, int) and x > 0 for x in output_shape):
            raise ValueError(f"Invalid value for output_shape: {output_shape}. It must be a tuple of two positive integers (timesteps, features).")

        timesteps, features = output_shape

        # Initialize weights using a normal distribution
        initialization = RandomNormal(mean=self._decoder_initializer_mean, stddev=self._decoder_initializer_deviation)

        # Define input layers for the latent space and labels
        neural_model_inputs = Input(shape=(self._decoder_latent_dimension,), dtype=self._decoder_dataset_type, name="latent_input")
        label_input = Input(shape=(self._decoder_number_samples_per_class["number_classes"],),
                            dtype=self._decoder_dataset_type, name="label_input")

        # Concatenate latent space and labels
        concatenate_input = Concatenate()([neural_model_inputs, label_input])

        # Process through initial dense layer
        conditional_decoder = Dense(self._decoder_number_units_decoder[0], kernel_initializer=initialization)(
            concatenate_input)
        conditional_decoder = Dropout(self._decoder_dropout_decay_rate_decoder)(conditional_decoder)
        conditional_decoder = self._add_activation_layer(conditional_decoder, self._decoder_activation_function)

        # Repeat the latent vector for each timestep
        conditional_decoder = RepeatVector(timesteps)(conditional_decoder)

        # Apply recurrent layers
        for idx, number_units in enumerate(self._decoder_number_units_decoder[1:]):
            # All recurrent layers should return sequences for time series generation
            conditional_decoder = self._create_recurrent_layer(
                number_units, return_sequences=True, initialization=initialization
            )(conditional_decoder)
            conditional_decoder = Dropout(self._decoder_dropout_decay_rate_decoder)(conditional_decoder)
            
            if self._decoder_activation_function:
                conditional_decoder = self._add_activation_layer(
                    conditional_decoder, self._decoder_activation_function
                )

        # TimeDistributed Dense layer to generate features at each timestep
        conditional_decoder = TimeDistributed(
            Dense(features, kernel_initializer=initialization, name="Output_TimeDistributed")
        )(conditional_decoder)
        conditional_decoder = self._add_activation_layer(conditional_decoder, self._decoder_last_layer_activation)

        # Return the constructed model
        return Model([neural_model_inputs, label_input], conditional_decoder, name="TimeSeriesDecoder")

    @property
    def dropout_decay_rate_decoder(self) -> float:
        """float: Gets or sets the dropout decay rate for decoder layers."""
        return self._decoder_dropout_decay_rate_decoder

    @property
    def number_units_decoder(self) -> list[int]:
        """list[int]: Gets the number of units in decoder recurrent layers."""
        return self._decoder_number_units_decoder

    @property
    def recurrent_layer_type(self) -> str:
        """str: Gets the type of recurrent layer used in the decoder."""
        return self._decoder_recurrent_layer_type

    @property
    def output_shape(self) -> tuple:
        """tuple: Gets the output shape of the time series (timesteps, features)."""
        return self._decoder_output_shape

    @dropout_decay_rate_decoder.setter
    def dropout_decay_rate_decoder(self, dropout_decay_rate_discriminator: float) -> None:
        """
        Sets the dropout decay rate for the decoder layers.

        Args:
            dropout_decay_rate_discriminator (float): The dropout rate for the decoder layers (between 0 and 1).

        Raises:
            ValueError: If the dropout rate is not a valid number between 0 and 1.
        """
        if not isinstance(dropout_decay_rate_discriminator, (int, float)) or not (0 <= dropout_decay_rate_discriminator <= 1):
            raise ValueError(f"Invalid value for dropout_decay_rate_discriminator: {dropout_decay_rate_discriminator}. It must be a number between 0 and 1.")
        self._decoder_dropout_decay_rate_decoder = dropout_decay_rate_discriminator

    @recurrent_layer_type.setter
    def recurrent_layer_type(self, layer_type: str) -> None:
        """
        Sets the type of recurrent layer for the decoder.

        Args:
            layer_type (str): The recurrent layer type ('LSTM', 'GRU', or 'BiLSTM').

        Raises:
            ValueError: If the layer type is not one of the supported types.
        """
        if layer_type not in ['LSTM', 'GRU', 'BiLSTM']:
            raise ValueError(f"Invalid value for recurrent_layer_type: {layer_type}. It must be one of ['LSTM', 'GRU', 'BiLSTM'].")
        self._decoder_recurrent_layer_type = layer_type
