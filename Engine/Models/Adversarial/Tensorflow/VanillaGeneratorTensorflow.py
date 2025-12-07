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

    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Model

    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dropout

    from tensorflow.keras.layers import Concatenate

    from tensorflow.keras.initializers import RandomNormal
    from Engine.Activations.Activations import Activations

except ImportError as error:
    print(error)
    sys.exit(-1)


class VanillaGenerator(Activations):
    """
    VanillaGenerator

    Implements a dense generator model for generating synthetic data using a
    vanilla architecture. This class is designed for generating synthetic data
    from a latent space using a fully connected neural network. It supports
    flexible configurations for the generator layers, activations, and dropout
    rates, with the option for conditional generation based on the number of
    samples per class.

    Attributes:
        @latent_dimension (int):
            The dimensionality of the latent space, which serves as the input to the generator.
        @output_shape (int):
            The desired dimension of the generated output data.
        @activation_function (str):
            The activation function used in intermediate layers (e.g., 'ReLU', 'LeakyReLU').
        @initializer_mean (float):
            The mean for the weight initialization.
        @initializer_deviation (float):
            The standard deviation for the weight initialization.
        @dropout_decay_rate_g (float):
            The rate at which the dropout is applied in generator layers, should be between 0.0 and 1.0.
        @last_layer_activation (str):
            The activation function to be applied in the last layer (e.g., 'sigmoid' or 'tanh').
        @dense_layer_sizes_g (list):
            A list of integers representing the number of units in each dense layer of the generator.
        @dataset_type (type):
            The data type for the input tensors (default is numpy.float32).
        @number_samples_per_class (dict | None):
            An optional dictionary indicating the number of samples per class for conditional data generation.

    Raises:
        ValueError:
            Raised if invalid arguments are passed during initialization, such as:
            - Non-positive `latent_dimension` or `output_shape`
            - `dropout_decay_rate_g` outside the range [0.0, 1.0]
            - Invalid values in `dense_layer_sizes_g`

    Example:
        >>> generator = VanillaGenerator(
        ...     latent_dimension=100,
        ...     output_shape=784,
        ...     activation_function="ReLU",
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_rate_g=0.3,
        ...     last_layer_activation="sigmoid",
        ...     dense_layer_sizes_g=[128, 256, 512],
        ...     number_samples_per_class={"class_1": 50, "class_2": 100}
        ... )
    """

    def __init__(self,
                 latent_dimension: int,
                 output_shape: int,
                 activation_function: str,
                 initializer_mean: float,
                 initializer_deviation: float,
                 dropout_decay_rate_g: float,
                 last_layer_activation: str,
                 dense_layer_sizes_g: list,
                 dataset_type: type = numpy.float32,
                 number_samples_per_class: dict | None = None):
        """
        Initializes the VanillaGenerator class with the specified parameters.

        Args:
            latent_dimension (int): Dimension of the latent space.
            output_shape (int): Dimension of the output data.
            activation_function (str): Activation function for intermediate layers.
            initializer_mean (float): Mean for weight initialization.
            initializer_deviation (float): Standard deviation for weight initialization.
            dropout_decay_rate_g (float): Dropout rate for generator layers.
            last_layer_activation (str): Activation function for the output layer.
            dense_layer_sizes_g (list): List of dense layer sizes.
            dataset_type (type): Data type for the input tensors.
            number_samples_per_class (dict, optional): Dictionary with the number of samples per class for conditional generation.

        Raises:
            ValueError: If `latent_dimension`, `output_shape`, or `dropout_decay_rate_g` are invalid.
        """

        if not isinstance(latent_dimension, int) or latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer.")

        if not isinstance(output_shape, int) or output_shape <= 0:
            raise ValueError("output_shape must be a positive integer.")

        if not isinstance(activation_function, str):
            raise ValueError("activation_function must be a string.")

        if not isinstance(initializer_mean, (float, int)):
            raise ValueError("initializer_mean must be a float or an integer.")

        if not isinstance(initializer_deviation, (float, int)) or initializer_deviation <= 0:
            raise ValueError("initializer_deviation must be a positive float or integer.")

        if not isinstance(last_layer_activation, str):
            raise ValueError("last_layer_activation must be a string.")

        if number_samples_per_class is not None and not isinstance(number_samples_per_class, dict):
            raise ValueError("number_samples_per_class must be a dictionary if provided.")

        # super().__init__()
        self._generator_number_samples_per_class = number_samples_per_class
        self._generator_latent_dimension = latent_dimension
        self._generator_output_shape = output_shape
        self._generator_activation_function = activation_function
        self._generator_last_layer_activation = last_layer_activation
        self._generator_dropout_decay_rate_g = dropout_decay_rate_g
        self._generator_dense_layer_sizes_g = dense_layer_sizes_g
        self._generator_dataset_type = dataset_type
        self._generator_initializer_mean = initializer_mean
        self._generator_initializer_deviation = initializer_deviation
        self._generator_model_dense = None

    def get_generator(self) -> Model:
        """
        Builds and returns the generator model.

        Returns:
            Model: A Keras model with inputs for latent vectors and conditional labels, and an output for generated data.

        Raises:
            ValueError: If `number_samples_per_class` is not provided for conditional generation.
        """
        if not self._generator_number_samples_per_class or "number_classes" not in self._generator_number_samples_per_class:
            raise ValueError(
                "`number_samples_per_class` must include a 'number_classes' key for conditional generation.")

        initialization = RandomNormal(mean=self._generator_initializer_mean, stddev=self._generator_initializer_deviation)
        neural_model_inputs = Input(shape=(self._generator_latent_dimension,), dtype=self._generator_dataset_type)
        latent_input = Input(shape=(self._generator_latent_dimension,))
        label_input = Input(shape=(self._generator_number_samples_per_class["number_classes"],), dtype=self._generator_dataset_type)

        # Build dense generator model
        generator_model = Dense(self._generator_dense_layer_sizes_g[0], kernel_initializer=initialization)(neural_model_inputs)
        generator_model = Dropout(self._generator_dropout_decay_rate_g)(generator_model)
        generator_model = self._add_activation_layer(generator_model, self._generator_activation_function)

        for layer_size in self._generator_dense_layer_sizes_g[1:]:
            generator_model = Dense(layer_size, kernel_initializer=initialization)(generator_model)
            generator_model = Dropout(self._generator_dropout_decay_rate_g)(generator_model)
            generator_model = self._add_activation_layer(generator_model, self._generator_activation_function)

        generator_model = Dense(self._generator_output_shape, kernel_initializer=initialization)(generator_model)
        generator_model = self._add_activation_layer(generator_model, self._generator_last_layer_activation)
        generator_model = Model(neural_model_inputs, generator_model, name="Dense_Generator")
        self._generator_model_dense = generator_model

        # Concatenate latent input with label input for conditional generation
        concatenate_output = Concatenate()([latent_input, label_input])
        label_embedding = Flatten()(concatenate_output)
        model_input = Dense(self._generator_latent_dimension)(label_embedding)
        model_input = self._add_activation_layer(model_input, self._generator_activation_function)
        generator_output_flow = generator_model(model_input)

        return Model([latent_input, label_input], generator_output_flow, name="Generator")

    def get_dense_generator_model(self) -> Model | None:
        """
        Returns the standalone dense generator model.

        Returns:
            Model | None: A Keras model without label conditioning, or None if not built.
        """
        return self._generator_model_dense

    def set_dropout_decay_rate_generator(self, dropout_decay_rate_generator: float) -> None:
        """
        Updates the dropout rate of the generator.

        Args:
            dropout_decay_rate_generator (float): New dropout rate.

        Raises:
            ValueError: If the dropout rate is not between 0.0 and 1.0.
        """
        if not (0.0 <= dropout_decay_rate_generator <= 1.0):
            raise ValueError("`dropout_decay_rate_generator` must be between 0.0 and 1.0.")
        self._generator_dropout_decay_rate_g = dropout_decay_rate_generator

    def set_dense_layer_sizes_generator(self, dense_layer_sizes_generator: list) -> None:
        """
        Updates the dense layer sizes of the generator.

        Args:
            dense_layer_sizes_generator (list): New dense layer sizes.

        Raises:
            ValueError: If the list is empty or contains non-positive integers.
        """
        if not dense_layer_sizes_generator or any(size <= 0 for size in dense_layer_sizes_generator):
            raise ValueError("`dense_layer_sizes_generator` must be a list of positive integers.")
        self._generator_dense_layer_sizes_g = dense_layer_sizes_generator
