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

    from typing import List
    from typing import Tuple
    from typing import Optional

    from tensorflow.keras.models import Model

    from Engine.Architectures.Autoencoder.Tensorflow.TimeSeriesEncoderTensorflow import TimeSeriesEncoder
    from Engine.Architectures.Autoencoder.Tensorflow.TimeSeriesDecoderTensorflow import TimeSeriesDecoder

except ImportError as error:
    print(error)
    sys.exit(-1)


DEFAULT_AUTOENCODER_LATENT_DIMENSION = 64
DEFAULT_AUTOENCODER_ACTIVATION = "tanh"
DEFAULT_AUTOENCODER_DROPOUT_DECAY_RATE_ENCODER = 0.25
DEFAULT_AUTOENCODER_DROPOUT_DECAY_RATE_DECODER = 0.25
DEFAULT_AUTOENCODER_RECURRENT_UNITS_ENCODER = [128, 64, 32]
DEFAULT_AUTOENCODER_RECURRENT_UNITS_DECODER = [32, 64, 128]
DEFAULT_AUTOENCODER_LAST_ACTIVATION_LAYER = "sigmoid"
DEFAULT_AUTOENCODER_INITIALIZER_MEAN = 0.0
DEFAULT_AUTOENCODER_INITIALIZER_DEVIATION = 0.125
DEFAULT_AUTOENCODER_RECURRENT_LAYER_TYPE = "LSTM"
DEFAULT_AUTOENCODER_USE_BIDIRECTIONAL = False


class TimeSeriesAutoencoderModel(TimeSeriesEncoder, TimeSeriesDecoder):
    """
    TimeSeriesAutoencoderModel

    This class implements a Time Series Autoencoder model by inheriting from the TimeSeriesEncoder and
    TimeSeriesDecoder classes. It constructs an autoencoder architecture specifically designed for temporal
    data by combining both an encoder and a decoder with customizable recurrent layers (LSTM/GRU/BiLSTM).
    
    The autoencoder is typically used for tasks such as:
    - Time series dimensionality reduction
    - Anomaly detection in sequential data
    - Feature learning from temporal patterns
    - Denoising of time series data
    - Time series forecasting
    - Sequence-to-sequence learning

    Attributes:
        @latent_dimension (int):
            The dimensionality of the latent space (encoding space).
        @output_shape (Tuple[int, int]):
            The desired shape of the output time series (timesteps, features).
        @activation_function (str):
            The activation function used throughout the encoder and decoder (e.g., 'tanh', 'relu').
        @initializer_mean (float):
            The mean for weight initialization.
        @initializer_deviation (float):
            The standard deviation for weight initialization.
        @dropout_decay_encoder (float):
            The rate of dropout for the encoder layers.
        @dropout_decay_decoder (float):
            The rate of dropout for the decoder layers.
        @last_layer_activation (str):
            The activation function for the output layer.
        @number_units_encoder (List[int]):
            A list specifying the number of units for each recurrent layer of the encoder.
        @number_units_decoder (List[int]):
            A list specifying the number of units for each recurrent layer of the decoder.
        @recurrent_layer_type (str):
            Type of recurrent layer: 'LSTM', 'GRU', or 'BiLSTM'.
        @use_bidirectional (bool):
            Whether to use bidirectional recurrent layers.
        @dataset_type (numpy.dtype):
            The data type for the dataset (default is numpy.float32).
        @number_samples_per_class (Optional[dict]):
            Dictionary containing metadata about classes (optional).

    Raises:
        ValueError:
            Raised if invalid arguments are passed during initialization, such as:
            - Non-positive `latent_dimension`
            - Invalid `output_shape` (must be tuple of 2 positive integers)
            - Invalid `activation_function`, `initializer_mean`, or `initializer_deviation`
            - Invalid `dropout_decay_encoder` or `dropout_decay_decoder` (must be in [0, 1])
            - Mismatched types for `number_units_encoder` or `number_units_decoder`
            - Invalid `recurrent_layer_type` (must be 'LSTM', 'GRU', or 'BiLSTM')
            - Invalid `last_layer_activation`

    Example:
        >>> autoencoder_model = TimeSeriesAutoencoderModel(
        ...     latent_dimension=64,
        ...     output_shape=(100, 10),  # 100 timesteps, 10 features
        ...     activation_function="tanh",
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_encoder=0.3,
        ...     dropout_decay_decoder=0.3,
        ...     last_layer_activation="sigmoid",
        ...     number_units_encoder=[128, 64, 32],
        ...     number_units_decoder=[32, 64, 128],
        ...     recurrent_layer_type="LSTM",
        ...     use_bidirectional=False,
        ...     dataset_type=numpy.float32,
        ...     number_samples_per_class={"number_classes": 5}
        ... )
        >>> # Build the complete autoencoder
        >>> encoder = autoencoder_model.get_encoder(input_shape=(100, 10))
        >>> decoder = autoencoder_model.get_decoder(output_shape=(100, 10))
        >>> # Or get the full autoencoder model
        >>> full_model = autoencoder_model.get_autoencoder(input_shape=(100, 10))
    """

    def __init__(self,
                 latent_dimension: int = DEFAULT_AUTOENCODER_LATENT_DIMENSION,
                 output_shape: Tuple[int, int] = (100, 10),
                 activation_function: str = DEFAULT_AUTOENCODER_ACTIVATION,
                 initializer_mean: float = DEFAULT_AUTOENCODER_INITIALIZER_MEAN,
                 initializer_deviation: float = DEFAULT_AUTOENCODER_INITIALIZER_DEVIATION,
                 dropout_decay_encoder: float = DEFAULT_AUTOENCODER_DROPOUT_DECAY_RATE_ENCODER,
                 dropout_decay_decoder: float = DEFAULT_AUTOENCODER_DROPOUT_DECAY_RATE_DECODER,
                 last_layer_activation: str = DEFAULT_AUTOENCODER_LAST_ACTIVATION_LAYER,
                 number_units_encoder: Optional[List[int]] = None,
                 number_units_decoder: Optional[List[int]] = None,
                 recurrent_layer_type: str = DEFAULT_AUTOENCODER_RECURRENT_LAYER_TYPE,
                 use_bidirectional: bool = DEFAULT_AUTOENCODER_USE_BIDIRECTIONAL,
                 dataset_type: numpy.dtype = numpy.float32,
                 number_samples_per_class: Optional[dict] = None):
        """
        Initializes the TimeSeriesAutoencoderModel by setting up both the encoder and decoder with the provided parameters.

        Args:
            latent_dimension (int): The dimensionality of the latent space (encoding space).
            output_shape (Tuple[int, int]): The shape of the time series output (timesteps, features).
            activation_function (str): The activation function to be used throughout the encoder and decoder.
            initializer_mean (float): The mean for weight initialization.
            initializer_deviation (float): The standard deviation for weight initialization.
            dropout_decay_encoder (float): The dropout rate for the encoder layers.
            dropout_decay_decoder (float): The dropout rate for the decoder layers.
            last_layer_activation (str): The activation function for the output layer.
            number_units_encoder (List[int], optional): Number of units for each encoder recurrent layer.
            number_units_decoder (List[int], optional): Number of units for each decoder recurrent layer.
            recurrent_layer_type (str): Type of recurrent layer ('LSTM', 'GRU', 'BiLSTM').
            use_bidirectional (bool): Whether to use bidirectional recurrent layers.
            dataset_type (numpy.dtype): The data type for the dataset.
            number_samples_per_class (dict, optional): Dictionary with class metadata.

        Raises:
            ValueError: If any of the provided parameters are invalid.
        """
        if number_units_decoder is None:
            number_units_decoder = DEFAULT_AUTOENCODER_RECURRENT_UNITS_DECODER

        if number_units_encoder is None:
            number_units_encoder = DEFAULT_AUTOENCODER_RECURRENT_UNITS_ENCODER

        # Validation
        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")

        if not isinstance(output_shape, tuple) or len(output_shape) != 2 or not all(isinstance(x, int) and x > 0 for x in output_shape):
            raise ValueError("output_shape must be a tuple of two positive integers (timesteps, features).")

        if not isinstance(activation_function, str):
            raise ValueError("activation_function must be a string.")

        if not isinstance(initializer_mean, (float, int)):
            raise ValueError("initializer_mean must be a float or integer.")

        if not isinstance(initializer_deviation, (float, int)):
            raise ValueError("initializer_deviation must be a float or integer.")

        if not isinstance(dropout_decay_encoder, (float, int)) or not (0 <= dropout_decay_encoder <= 1):
            raise ValueError("dropout_decay_encoder must be a float or integer between 0 and 1.")

        if not isinstance(dropout_decay_decoder, (float, int)) or not (0 <= dropout_decay_decoder <= 1):
            raise ValueError("dropout_decay_decoder must be a float or integer between 0 and 1.")

        if not isinstance(last_layer_activation, str):
            raise ValueError("last_layer_activation must be a string.")

        if not isinstance(number_units_encoder, list) or not all(isinstance(x, int) and x > 0 for x in number_units_encoder):
            raise ValueError("number_units_encoder must be a list of positive integers.")

        if not isinstance(number_units_decoder, list) or not all(isinstance(x, int) and x > 0 for x in number_units_decoder):
            raise ValueError("number_units_decoder must be a list of positive integers.")

        if recurrent_layer_type not in ['LSTM', 'GRU', 'BiLSTM']:
            raise ValueError("recurrent_layer_type must be one of ['LSTM', 'GRU', 'BiLSTM'].")

        # Initialize Decoder
        TimeSeriesDecoder.__init__(self,
                                   latent_dimension,
                                   output_shape,
                                   activation_function,
                                   initializer_mean,
                                   initializer_deviation,
                                   dropout_decay_decoder,
                                   last_layer_activation,
                                   number_units_decoder,
                                   recurrent_layer_type,
                                   use_bidirectional,
                                   dataset_type,
                                   number_samples_per_class)

        # Initialize Encoder
        TimeSeriesEncoder.__init__(self,
                                   latent_dimension,
                                   output_shape,
                                   activation_function,
                                   initializer_mean,
                                   initializer_deviation,
                                   dropout_decay_encoder,
                                   last_layer_activation,
                                   number_units_encoder,
                                   recurrent_layer_type,
                                   use_bidirectional,
                                   dataset_type,
                                   number_samples_per_class)

    def get_timeseries_encoder_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Returns the time series encoder model.

        Args:
            input_shape (Tuple[int, int]): Shape of the input time series (timesteps, features).

        Returns:
            tensorflow.keras.Model: The model representing the encoder.
        """
        return self.get_encoder(input_shape)

    def get_timeseries_decoder_model(self, output_shape: Tuple[int, int]) -> Model:
        """
        Returns the time series decoder model.

        Args:
            output_shape (Tuple[int, int]): Shape of the output time series (timesteps, features).

        Returns:
            tensorflow.keras.Model: The model representing the decoder.
        """
        return self.get_decoder(output_shape)

    def get_autoencoder(self, input_shape: Tuple[int, int]) -> Model:
        """
        Constructs and returns the complete autoencoder model by connecting the encoder and decoder.

        Args:
            input_shape (Tuple[int, int]): Shape of the input time series (timesteps, features).

        Returns:
            tensorflow.keras.Model: The complete autoencoder model.
        """
        # Get encoder and decoder
        encoder = self.get_encoder(input_shape)
        decoder = self.get_decoder(input_shape)

        # Connect encoder output to decoder input
        # Encoder outputs: [latent_representation, label_input]
        # Decoder inputs: [latent_input, label_input]
        timeseries_input = encoder.input[0]
        label_input = encoder.input[1]
        
        latent_representation = encoder.output[0]
        
        # Decoder takes latent representation and labels
        reconstructed_output = decoder([latent_representation, label_input])

        # Create the full autoencoder model
        autoencoder = Model([timeseries_input, label_input], reconstructed_output, name="TimeSeriesAutoencoder")
        
        return autoencoder

    def set_latent_dimension(self, latent_dimension: int) -> None:
        """
        Sets the latent dimension for both the encoder and decoder.

        Args:
            latent_dimension (int): The dimensionality of the latent space.

        Raises:
            ValueError: If latent_dimension is not a positive integer.
        """
        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")

        self._encoder_latent_dimension = latent_dimension
        self._decoder_latent_dimension = latent_dimension

    def set_output_shape(self, output_shape: Tuple[int, int]) -> None:
        """
        Sets the output shape for both the encoder and decoder.

        Args:
            output_shape (Tuple[int, int]): The shape of the time series (timesteps, features).

        Raises:
            ValueError: If output_shape is not a valid tuple.
        """
        if not isinstance(output_shape, tuple) or len(output_shape) != 2 or not all(isinstance(x, int) and x > 0 for x in output_shape):
            raise ValueError("output_shape must be a tuple of two positive integers (timesteps, features).")

        self._encoder_output_shape = output_shape
        self._decoder_output_shape = output_shape

    def set_activation_function(self, activation_function: str) -> None:
        """
        Sets the activation function for both the encoder and decoder.

        Args:
            activation_function (str): The activation function to be applied.

        Raises:
            ValueError: If activation_function is not a string.
        """
        if not isinstance(activation_function, str):
            raise ValueError("activation_function must be a string.")

        self._encoder_activation_function = activation_function
        self._decoder_activation_function = activation_function

    def set_last_layer_activation(self, last_layer_activation: str) -> None:
        """
        Sets the activation function for the last layer of both the encoder and decoder.

        Args:
            last_layer_activation (str): The activation function to be used in the last layer.

        Raises:
            ValueError: If last_layer_activation is not a string.
        """
        if not isinstance(last_layer_activation, str):
            raise ValueError("last_layer_activation must be a string.")

        self._encoder_last_layer_activation = last_layer_activation
        self._decoder_last_layer_activation = last_layer_activation

    def set_recurrent_layer_type(self, recurrent_layer_type: str) -> None:
        """
        Sets the type of recurrent layer for both encoder and decoder.

        Args:
            recurrent_layer_type (str): The recurrent layer type ('LSTM', 'GRU', or 'BiLSTM').

        Raises:
            ValueError: If recurrent_layer_type is not valid.
        """
        if recurrent_layer_type not in ['LSTM', 'GRU', 'BiLSTM']:
            raise ValueError("recurrent_layer_type must be one of ['LSTM', 'GRU', 'BiLSTM'].")

        self._encoder_recurrent_layer_type = recurrent_layer_type
        self._decoder_recurrent_layer_type = recurrent_layer_type

    @property
    def latent_dimension(self) -> int:
        """int: Gets the latent dimension."""
        return self._encoder_latent_dimension

    @property
    def output_shape(self) -> Tuple[int, int]:
        """Tuple[int, int]: Gets the output shape (timesteps, features)."""
        return self._encoder_output_shape

    @property
    def recurrent_layer_type(self) -> str:
        """str: Gets the recurrent layer type."""
        return self._encoder_recurrent_layer_type
