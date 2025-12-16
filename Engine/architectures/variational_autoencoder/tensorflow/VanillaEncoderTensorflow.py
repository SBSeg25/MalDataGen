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

    from typing import List, Dict, Union, Optional
    from tensorflow import keras
    from tensorflow.keras.layers import Dense, Input, Dropout, Concatenate
    from tensorflow.keras.models import Model
    from tensorflow.keras.initializers import RandomNormal
    from Engine.activations.Activations import Activations
    from Engine.layers.tensorflow.SamplingLayer import LayerSampling

except ImportError as error:
    print(error)
    sys.exit(-1)


# ==================== ENCODER ====================

class VanillaEncoderTensorflow(Activations, LayerSampling):
    """
    VanillaEncoder with Concatenation

    Implements a fully connected conditional variational encoder (CVAE) using simple
    concatenation for label conditioning.

    Attributes:
        @encoder_latent_dimension (int):
            Dimensionality of the latent space, defining the encoded feature representation.
        @encoder_output_shape (int):
            Dimensionality of the input data that will be encoded.
        @encoder_activation_function (str):
            Activation function applied to all hidden layers (e.g., ReLU, Tanh, LeakyReLU).
        @encoder_last_layer_activation (str):
            Activation function applied to the final layer of the encoder.
        @encoder_dropout_decay_rate_encoder (float):
            Dropout rate applied to dense layers to improve generalization (must be between 0 and 1).
        @encoder_dataset_type (Union[numpy.dtype, type]):
            Data type of the input tensors (default: numpy.float32).
        @encoder_initializer_mean (float):
            Mean of the normal distribution used for weight initialization.
        @encoder_initializer_deviation (float):
            Standard deviation of the normal distribution used for weight initialization.
        @encoder_number_neurons_encoder (List[int]):
            List of integers specifying the number of units per dense layer, defining model complexity.
        @encoder_number_samples_per_class (Optional[Dict[str, int]]):
            Dictionary specifying the number of samples per class in conditional scenarios.
        @encoder_label_embed_dim (int):
            Embedding dimension for label conditioning.
        @encoder_model (Optional[Model]):
            Placeholder for the compiled Keras Model after build().
    """

    def __init__(self,
                 latent_dimension: int,
                 output_shape: int,
                 activation_function: str,
                 initializer_mean: float,
                 initializer_deviation: float,
                 dropout_decay_encoder: float,
                 last_layer_activation: str,
                 number_neurons_encoder: List[int],
                 dataset_type: Union[numpy.dtype, type] = numpy.float32,
                 number_samples_per_class: Optional[Dict[str, int]] = None,
                 label_embed_dim: int = 128) -> None:
        """
        Initializes the VanillaEncoder with concatenation.

        Args:
            @latent_dimension (int): Dimensionality of the latent space.
            @output_shape (int): Dimensionality of the input data.
            @activation_function (str): Activation function for hidden layers.
            @initializer_mean (float): Mean for weight initialization.
            @initializer_deviation (float): Std deviation for weight initialization.
            @dropout_decay_encoder (float): Dropout rate for regularization.
            @last_layer_activation (str): Activation for final layer.
            @number_neurons_encoder (List[int]): Neurons per encoder layer.
            @dataset_type (Union[numpy.dtype, type], optional): Input data type.
            @number_samples_per_class (Optional[Dict[str, int]], optional): Class metadata.
            @label_embed_dim (int, optional): Embedding dimension for labels.
        """

        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("Latent dimension must be a positive integer.")

        if not isinstance(output_shape, int) or output_shape <= 0:
            raise ValueError("Output shape must be a positive integer.")

        if not isinstance(activation_function, str):
            raise ValueError("Activation function must be a string.")

        if not isinstance(initializer_mean, (int, float)):
            raise ValueError("Initializer mean must be a numerical value.")

        if not isinstance(initializer_deviation, (int, float)) or initializer_deviation <= 0:
            raise ValueError("Initializer deviation must be a positive numerical value.")

        if not (0.0 <= dropout_decay_encoder <= 1.0):
            raise ValueError("Dropout decay rate must be between 0 and 1.")

        if not isinstance(last_layer_activation, str):
            raise ValueError("Last layer activation must be a string.")

        if not isinstance(number_neurons_encoder, list) or not all(
                isinstance(n, int) and n > 0 for n in number_neurons_encoder):
            raise ValueError("Number of neurons per encoder layer must be a list of positive integers.")

        if dataset_type not in [numpy.float16, numpy.float32, numpy.float64]:
            raise ValueError("Datasets type must be one of numpy.float16, numpy.float32, or numpy.float64.")

        if not isinstance(label_embed_dim, int) or label_embed_dim <= 0:
            raise ValueError("label_embed_dim must be a positive integer.")

        # Initialize instance variables
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
        self._encoder_label_embed_dim = label_embed_dim
        self._encoder_model = None

    def get_encoder(self, input_shape=None) -> Model:
        """
        Constructs and returns the encoder model with concatenation.

        The encoder uses simple concatenation to condition input features on label information.

        Args:
            input_shape: Optional input shape (for API compatibility, not used)

        Returns:
            Model: The constructed encoder model with concatenation.
        """
        # Ensure the number of classes is provided
        if not self._encoder_number_samples_per_class or "number_classes" not in self._encoder_number_samples_per_class:
            raise ValueError("The number of classes must be specified in 'number_samples_per_class'.")

        # Initialize weights with normal distribution
        initialization = RandomNormal(mean=self._encoder_initializer_mean,
                                      stddev=self._encoder_initializer_deviation)

        # Input layer for feature data
        neural_model_inputs = Input(shape=(self._encoder_output_shape,),
                                    dtype=self._encoder_dataset_type,
                                    name="first_input")

        # Input layer for class labels
        label_input = Input(shape=(self._encoder_number_samples_per_class["number_classes"],),
                            dtype=self._encoder_dataset_type,
                            name="second_input")

        # Label embedding
        label_input_embedding = Dense(self._encoder_label_embed_dim,
                                      activation='relu',
                                      kernel_initializer=initialization,
                                      name='label_embedding')(label_input)

        # Concatenate input features and label embedding
        concatenated = Concatenate(name='concatenate')([neural_model_inputs, label_input_embedding])

        # Build encoder layers with dense and dropout
        conditional_encoder = Dense(self._encoder_number_neurons_encoder[0],
                                    kernel_initializer=initialization)(concatenated)
        conditional_encoder = Dropout(self._encoder_dropout_decay_rate_encoder)(conditional_encoder)
        conditional_encoder = self._add_activation_layer(conditional_encoder,
                                                         self._encoder_activation_function)

        # Add additional dense layers based on configuration
        for number_neurons in self._encoder_number_neurons_encoder[1:]:
            conditional_encoder = Dense(number_neurons,
                                        kernel_initializer=initialization)(conditional_encoder)
            conditional_encoder = Dropout(self._encoder_dropout_decay_rate_encoder)(conditional_encoder)
            conditional_encoder = self._add_activation_layer(conditional_encoder,
                                                             self._encoder_activation_function)

        # Add final dense layer with specified activation function
        conditional_encoder = Dense(self._encoder_latent_dimension,
                                    activation=self._encoder_last_layer_activation,
                                    kernel_initializer=initialization)(conditional_encoder)

        # Generate latent mean and log variance layers
        latent_mean = Dense(self._encoder_latent_dimension, name="z_mean")(conditional_encoder)
        latent_log_var = Dense(self._encoder_latent_dimension, name="z_log_var")(conditional_encoder)

        # Sampling layer for latent representation
        latent = self.Sampling()([latent_mean, latent_log_var])

        # Compile the encoder model
        self._encoder_model = Model([neural_model_inputs, label_input],
                                    [latent_mean, latent_log_var, latent, label_input],
                                    name="Encoder_Concatenation")

        return self._encoder_model

    @property
    def dropout_decay_rate_encoder(self) -> float:
        """float: Dropout rate for encoder regularization."""
        return self._encoder_dropout_decay_rate_encoder

    @property
    def number_filters_encoder(self) -> List[int]:
        """List[int]: Number of neurons in each encoder layer."""
        return self._encoder_number_neurons_encoder

    @property
    def label_embed_dim(self) -> int:
        """int: Gets the embedding dimension for label conditioning."""
        return self._encoder_label_embed_dim

    @dropout_decay_rate_encoder.setter
    def dropout_decay_rate_encoder(self, dropout_decay_rate_encoder: float) -> None:
        """
        Set the dropout rate for encoder regularization.

        Args:
            dropout_decay_rate_encoder (float): Dropout rate to set.
        """
        if not (0.0 <= dropout_decay_rate_encoder <= 1.0):
            raise ValueError("Dropout decay rate must be between 0 and 1.")

        self._encoder_dropout_decay_rate_encoder = dropout_decay_rate_encoder


# ==================== DECODER ====================

class VanillaDecoderTensorflow(Activations):
    """
      VanillaDecoder with Concatenation

      This class implements a conditional fully connected decoder network using
      simple concatenation for label conditioning.

      The architecture consists of:
      1. Label embedding layer
      2. Concatenation of latent vector and label embedding
      3. Sequence of fully connected layers with activation and dropout
      4. Final output layer

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
          Dictionary containing class metadata (e.g., number of samples per class).
      @decoder_label_embed_dim : int
          Embedding dimension for label conditioning.
      @decoder_model : Optional[Model]
          Placeholder for the actual compiled Keras model (built later).
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
                 number_samples_per_class=None,
                 label_embed_dim=128):
        """
        Initializes the VanillaDecoder with concatenation.

        Parameters
        ----------
        @latent_dimension : int
            Dimensionality of the input latent space.
        @output_shape : int
            Dimensionality of the output.
        @activation_function : str or callable
            Activation function for intermediate layers.
        @initializer_mean : float
            Mean of the normal distribution for weight initialization.
        @initializer_deviation : float
            Standard deviation for weight initialization.
        @dropout_decay_decoder : float
            Dropout rate applied to intermediate layers.
        @last_layer_activation : str or callable
            Activation function for the final layer.
        @number_neurons_decoder : list of int
            Number of neurons in each fully connected layer.
        @dataset_type : type, optional
            Data type for inputs and outputs (default: numpy.float32).
        @number_samples_per_class : dict, optional
            Dictionary with class information (must contain 'number_classes' key).
        @label_embed_dim : int, optional
            Embedding dimension for label conditioning (default: 128).
        """

        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")

        if not isinstance(output_shape, int) or output_shape <= 0:
            raise ValueError("output_shape must be a positive integer.")

        if not isinstance(activation_function, (str, callable)):
            raise ValueError("activation_function must be a string or a callable function.")

        if not isinstance(last_layer_activation, (str, callable)):
            raise ValueError("last_layer_activation must be a string or a callable function.")

        if not isinstance(initializer_mean, (int, float)):
            raise ValueError("initializer_mean must be a numeric value.")

        if not isinstance(initializer_deviation, (int, float)) or initializer_deviation <= 0:
            raise ValueError("initializer_deviation must be a positive numeric value.")

        if not isinstance(dropout_decay_decoder, (int, float)) or not (0.0 <= dropout_decay_decoder <= 1.0):
            raise ValueError("dropout_decay_decoder must be a float between 0 and 1.")

        if not isinstance(number_neurons_decoder, list) or not all(
                isinstance(n, int) and n > 0 for n in number_neurons_decoder):
            raise ValueError("number_neurons_decoder must be a list of positive integers.")

        if number_samples_per_class is not None:
            if not isinstance(number_samples_per_class, dict):
                raise ValueError("number_samples_per_class must be a dictionary or None.")

            if "number_classes" not in number_samples_per_class or not isinstance(
                    number_samples_per_class["number_classes"], int):
                raise ValueError("number_samples_per_class must contain a key 'number_classes' with an integer value.")

        if not isinstance(label_embed_dim, int) or label_embed_dim <= 0:
            raise ValueError("label_embed_dim must be a positive integer.")

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
        self._decoder_label_embed_dim = label_embed_dim
        self._decoder_model = None

    def get_decoder(self, input_shape=None):
        """
        Builds and returns the decoder model with concatenation.

        The model concatenates the latent vector with the label embedding.

        Args:
            input_shape: Optional input shape (for API compatibility, not used)

        Returns:
            tensorflow.keras.Model: The decoder model with concatenation conditioning.
        """
        initialization = RandomNormal(mean=self._decoder_initializer_mean,
                                      stddev=self._decoder_initializer_deviation)

        # Input layers
        neural_model_inputs = Input(shape=(self._decoder_latent_dimension,),
                                    dtype=self._decoder_dataset_type,
                                    name='latent_input')
        label_input = Input(shape=(self._decoder_number_samples_per_class["number_classes"],),
                            dtype=self._decoder_dataset_type,
                            name='label_input')

        # Label embedding
        label_input_embedding = Dense(self._decoder_label_embed_dim,
                                      activation='relu',
                                      kernel_initializer=initialization,
                                      name='label_embedding')(label_input)

        # Concatenate latent vector and label embedding
        concatenated = Concatenate(name='concatenate')([neural_model_inputs, label_input_embedding])

        # First dense layer
        conditional_decoder = Dense(self._decoder_number_neurons_decoder[0],
                                    kernel_initializer=initialization)(concatenated)
        conditional_decoder = self._add_activation_layer(conditional_decoder,
                                                         self._decoder_intermediary_activation_function)
        conditional_decoder = Dropout(self._decoder_dropout_decay_rate_decoder)(conditional_decoder)

        # Hidden layers
        for number_filters in self._decoder_number_neurons_decoder[1:]:
            conditional_decoder = Dense(number_filters,
                                        kernel_initializer=initialization)(conditional_decoder)
            conditional_decoder = Dropout(self._decoder_dropout_decay_rate_decoder)(conditional_decoder)
            conditional_decoder = self._add_activation_layer(conditional_decoder,
                                                             self._decoder_intermediary_activation_function)

        # Output layer
        conditional_decoder = Dense(self._decoder_output_shape,
                                    kernel_initializer=initialization)(conditional_decoder)
        conditional_decoder = self._add_activation_layer(conditional_decoder,
                                                         self._decoder_last_layer_activation)

        self._decoder_model = Model([neural_model_inputs, label_input],
                                    conditional_decoder,
                                    name="Decoder_Concatenation")

        return self._decoder_model

    @property
    def dropout_decay_rate_decoder(self):
        """float: Gets or sets the dropout decay rate for the decoder."""
        return self._decoder_dropout_decay_rate_decoder

    @property
    def number_filters_decoder(self):
        """list[int]: Gets the number of neurons for each layer in the decoder."""
        return self._decoder_number_neurons_decoder

    @property
    def label_embed_dim(self):
        """int: Gets the embedding dimension for label conditioning."""
        return self._decoder_label_embed_dim

    @dropout_decay_rate_decoder.setter
    def dropout_decay_rate_decoder(self, dropout_decay_rate_discriminator):
        """
        Sets the dropout decay rate for the decoder.

        Args:
            dropout_decay_rate_discriminator (float): New dropout decay rate.
        """
        self._decoder_dropout_decay_rate_decoder = dropout_decay_rate_discriminator