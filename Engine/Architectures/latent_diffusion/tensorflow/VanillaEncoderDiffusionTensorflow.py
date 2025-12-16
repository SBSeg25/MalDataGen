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

    from typing import List
    from typing import Dict
    from typing import Union
    from typing import Optional

    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.models import Model

    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Concatenate

    from Engine.activations.Activations import Activations
    from tensorflow.keras.initializers import RandomNormal

    from Engine.layers.tensorflow.SamplingLayer import LayerSampling

except ImportError as error:
    print(error)
    sys.exit(-1)


class VanillaEncoderDiffusion(Activations, LayerSampling):
    """
    VanillaEncoder

    Implements a fully connected conditional variational encoder (CVAE) model designed
    for probabilistic generative tasks. This encoder maps input data to a structured
    latent space while incorporating conditional information, enhancing control over
    latent representations. The model supports various activation functions,
    dropout-based regularization, and custom weight initialization.

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
            Dictionary specifying the number of samples per class in conditional scenarios, if provided.
        @encoder_model (Optional[Model]):
            Placeholder for the compiled Keras Model after build().

    Raises:
        ValueError:
            Raised if invalid arguments are passed during initialization, such as:
            - Non-positive `latent_dimension` or `output_shape`
            - Dropout rate outside the range [0, 1]
            - Empty or invalid `number_neurons_encoder`

    References:
        - Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv preprint arXiv:1312.6114.
          Available at: https://arxiv.org/abs/1312.6114

    Example:
        >>> encoder = VanillaEncoderDiffusion(
        ...     latent_dimension=64,
        ...     output_shape=784,
        ...     activation_function='relu',
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_encoder=0.3,
        ...     last_layer_activation='linear',
        ...     number_neurons_encoder=[512, 256, 128],
        ...     dataset_type=numpy.float32,
        ...     number_samples_per_class={"number_classes": 10}
        ... )
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
                 number_samples_per_class: Optional[Dict[str, int]] = None) -> None:
        """
        Initializes the VanillaEncoder with user-defined hyperparameters.

        Args:
            @latent_dimension (int): Dimensionality of the latent space, defining
             the encoded feature representation.
            @output_shape (int): Dimensionality of the input data that will be encoded.
            @activation_function (str): Non-linear activation function applied in
             each encoder layer (e.g., ReLU, Tanh, LeakyReLU).
            @initializer_mean (float): Mean value for the Gaussian distribution
             used in weight initialization.
            @initializer_deviation (float): Standard deviation for the Gaussian
             distribution used in weight initialization.
            @dropout_decay_encoder (float): Dropout rate applied for regularization,
             preventing overfitting (must be between 0 and 1).
            @last_layer_activation (str): Activation function applied to the final
             layer of the encoder, defining latent space properties.
            @number_neurons_encoder (List[int]): List specifying the number of neurons
             per encoder layer, defining model complexity.
            @dataset_type (Union[numpy.dtype, type], optional): Data type of the input
             tensors (default: numpy.float32).
            @number_samples_per_class (Optional[Dict[str, int]], optional): Dictionary
             specifying the number of samples
            per class in conditional scenarios.

        Raises:
            ValueError: If latent_dimension, output_shape, or dropout_decay_encoder have invalid values.
        """


        # Validate inputs to ensure valid model configuration
        if latent_dimension <= 0:
            raise ValueError("Latent dimension must be greater than 0.")

        if output_shape <= 0:
            raise ValueError("Output shape must be greater than 0.")

        if not (0.0 <= dropout_decay_encoder <= 1.0):
            raise ValueError("Dropout decay rate must be between 0 and 1.")

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

        if not isinstance(last_layer_activation, (str, callable)):
            raise ValueError("last_layer_activation must be a string or a callable function.")

        if not isinstance(dataset_type, type):
            raise ValueError("dataset_type must be a valid data type.")

        if number_samples_per_class is not None:

            if not isinstance(number_samples_per_class, dict) or "number_classes" not in number_samples_per_class:
                raise ValueError("number_samples_per_class must be a dictionary containing the key 'number_classes'.")


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
        self._encoder_model = None

    def get_encoder_trained(self):

        return self._encoder_model

    def create_embedding(self, data):
        """
        Generates latent space embeddings using the trained encoder.

        Args:
            data (ndarray): Input data to encode.

        Returns:
            ndarray: Latent space representations.
        """
        return self._encoder_model.predict(data, batch_size=32)


    def get_encoder(self) -> Model:
        """
        Constructs and returns the encoder model.

        The encoder combines input features and labels into a conditional representation
        that maps to a latent space. The model uses variational layers for mean and log variance,
        enabling sampling in the latent space.

        Returns:
            Model: The constructed encoder model.

        Raises:
            ValueError: If the number of classes is not specified in number_samples_per_class.
        """
        # Ensure the number of classes is provided in the configuration
        if not self._encoder_number_samples_per_class or "number_classes" not in self._encoder_number_samples_per_class:
            raise ValueError("The number of classes must be specified in 'number_samples_per_class'.")

        # Initialize weights with normal distribution
        initialization = RandomNormal(mean=self._encoder_initializer_mean, stddev=self._encoder_initializer_deviation)

        # Input layer for feature data
        neural_model_inputs = Input(shape=(self._encoder_output_shape,), dtype=self._encoder_dataset_type, name="first_input")

        # Input layer for class labels
        label_input = Input(shape=(self._encoder_number_samples_per_class["number_classes"],),
                            dtype=self._encoder_dataset_type, name="second_input")

        # Concatenate feature and label inputs
        concatenate_input = Concatenate()([neural_model_inputs, label_input])

        # Build encoder layers with dense and dropout
        conditional_encoder = Dense(self._encoder_number_neurons_encoder[0], kernel_initializer=initialization)(concatenate_input)
        conditional_encoder = Dropout(self._encoder_dropout_decay_rate_encoder)(conditional_encoder)
        conditional_encoder = self._add_activation_layer(conditional_encoder, self._encoder_activation_function)

        # Add additional dense layers based on configuration
        for number_neurons in self._encoder_number_neurons_encoder[1:]:
            conditional_encoder = Dense(number_neurons, kernel_initializer=initialization)(conditional_encoder)
            conditional_encoder = Dropout(self._encoder_dropout_decay_rate_encoder)(conditional_encoder)
            conditional_encoder = self._add_activation_layer(conditional_encoder, self._encoder_activation_function)

        # Add final dense layer with specified activation function
        conditional_encoder = Dense(self._encoder_latent_dimension, activation=self._encoder_last_layer_activation,
                                    kernel_initializer=initialization)(conditional_encoder)

        # Generate latent mean and log variance layers
        latent_mean = Dense(self._encoder_latent_dimension, name="z_mean")(conditional_encoder)
        latent_log_var = Dense(self._encoder_latent_dimension, name="z_log_var")(conditional_encoder)

        # Sampling layer for latent representation
        latent = self.Sampling()([latent_mean, latent_log_var])

        # Compile the encoder model
        self._encoder_model = Model([neural_model_inputs, label_input],
                                    [latent_mean, latent_log_var, latent, label_input], name="Encoder")

        return self._encoder_model

    @property
    def dropout_decay_rate_encoder(self) -> float:
        """float: Dropout rate for encoder regularization."""
        return self._encoder_dropout_decay_rate_encoder

    @property
    def number_filters_encoder(self) -> List[int]:
        """List[int]: Number of neurons in each encoder layer."""
        return self._encoder_number_neurons_encoder

    @dropout_decay_rate_encoder.setter
    def dropout_decay_rate_encoder(self, dropout_decay_rate_encoder: float) -> None:
        """
        Set the dropout rate for encoder regularization.

        Args:
            dropout_decay_rate_encoder (float): Dropout rate to set.

        Raises:
            ValueError: If the dropout rate is not between 0 and 1.
        """
        if not (0.0 <= dropout_decay_rate_encoder <= 1.0):
            raise ValueError("Dropout decay rate must be between 0 and 1.")

        self._encoder_dropout_decay_rate_encoder = dropout_decay_rate_encoder
