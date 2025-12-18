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

    from tensorflow.keras.layers import Layer
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model

    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Reshape
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import UpSampling1D

    from tensorflow.keras.layers import Concatenate

    from tensorflow.keras.initializers import RandomNormal
    from Engine.activations.Activations import Activations

except ImportError as error:
    print(error)
    sys.exit(-1)


class VanillaGeneratorTensorflow(Activations):
    """
    VanillaGenerator

    Implements a fully connected (dense) generator model for use in generative models,
    such as GANs. This generator is designed to work with label conditioning and
    supports customization of activation functions, layer sizes, initialization, and
    other hyperparameters.

    This class now supports both dense and convolutional (Conv1D) architectures through
    the optimizer parameter.

    Attributes:
        @generator_latent_dimension (int):
            Dimensionality of the input latent space.
        @generator_output_shape (int):
            Dimensionality of the generated output data.
        @generator_activation_function (Callable):
            Activation function applied to all hidden layers.
        @generator_last_layer_activation (Callable):
            Activation function applied to the final output layer.
        @generator_dropout_decay_rate_g (float):
            Dropout rate applied to dense layers to improve generalization.
        @generator_dense_layer_sizes_g (List[int]):
            List of integers specifying the number of units in each dense layer.
        @generator_dataset_type (type):
            Data type of the input dataset (default: numpy.float32).
        @generator_initializer_mean (float):
            Mean of the normal distribution used for weight initialization.
        @generator_initializer_deviation (float):
            Standard deviation of the normal distribution used for weight initialization.
        @generator_number_samples_per_class (Optional[Dict[str, int]]):
            Optional dictionary containing metadata about class distribution.
            Must include a key "number_classes" if provided.
        @generator_optimizer (str):
            Type of architecture to use: 'dense' (default) or 'convolutional' for Conv1D.
        @generator_model_dense (Optional[Model]):
            Placeholder for the compiled Keras Model after build().

    Raises:
        ValueError:
            Raised if invalid arguments are passed during initialization, such as:
            - Non-positive `latent_dimension` or `output_shape`
            - Dropout rate outside the range [0, 1]
            - Empty or invalid `dense_layer_sizes_g`
            - Missing required key "number_classes" in `number_samples_per_class`, if provided

    References:
        - Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014).
          Generative adversarial Networks. arXiv preprint arXiv:1406.2661.
          Available at: https://arxiv.org/abs/1406.2661

    Example:
        >>> generator = VanillaGenerator(
        ...     latent_dimension=100,
        ...     output_shape=784,
        ...     activation_function=leaky_relu,
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_rate_g=0.3,
        ...     last_layer_activation=tanh,
        ...     dense_layer_sizes_g=[256, 512, 1024],
        ...     dataset_type=numpy.float32,
        ...     number_samples_per_class={"number_classes": 10},
        ...     optimizer='convolutional'
        ... )
        >>> generator.build()  # Example method call if present
    """

    def __init__(self, latent_dimension: int,
                 output_shape: int,
                 activation_function: callable,
                 initializer_mean: float,
                 initializer_deviation: float,
                 dropout_decay_rate_g: float,
                 last_layer_activation: callable,
                 dense_layer_sizes_g: list[int],
                 dataset_type: type = numpy.float32,
                 number_samples_per_class: dict | None = None,
                 optimizer: str = 'dense'):
        """
        Initializes the VanillaGenerator class with the provided parameters.

        Args:
            @latent_dimension (int):
                Dimensionality of the latent space.
            @output_shape (int):
                Dimensionality of the generated output data.
            @activation_function (Callable):
                Activation function for all hidden layers.
            @initializer_mean (float):
                Mean of the normal distribution used to initialize weights.
            @initializer_deviation (float):
                Standard deviation of the normal distribution used to initialize weights.
            @dropout_decay_rate_g (float):
                Dropout rate applied to dense layers (0 to 1).
            @last_layer_activation (Callable):
                Activation function applied to the final output layer.
            @dense_layer_sizes_g (List[int]):
                List of integers specifying the number of units per dense layer.
            @dataset_type (type, optional):
                Data type of the input data (default: numpy.float32).
            @number_samples_per_class (Optional[Dict[str, int]], optional):
                Optional dictionary containing the number of samples per class. If provided, it must contain the key "number_classes".
            @optimizer (str, optional):
                Type of architecture: 'dense' for fully-connected (default) or 'convolutional' for Conv1D architecture.

        Raises:
            ValueError:
                If `latent_dimension` or `output_shape` is <= 0.
                If `dropout_decay_rate_g` is not within [0, 1].
                If `dense_layer_sizes_g` is empty or contains non-positive values.
                If `number_samples_per_class` is provided but does not contain the key "number_classes".

        """

        if latent_dimension <= 0:
            raise ValueError("Latent dimension must be a positive integer.")

        if output_shape <= 0:
            raise ValueError("Output shape must be a positive integer.")

        if initializer_mean < 0:
            raise ValueError("Initializer mean must be non-negative.")

        if initializer_deviation <= 0:
            raise ValueError("Initializer deviation must be positive.")

        if dropout_decay_rate_g < 0 or dropout_decay_rate_g > 1:
            raise ValueError("Dropout decay rate must be in the range [0, 1].")

        if not dense_layer_sizes_g or any(size <= 0 for size in dense_layer_sizes_g):
            raise ValueError("Dense layer sizes must be a list of positive integers.")

        self._generator_latent_dimension = latent_dimension
        self._generator_output_shape = output_shape
        self._generator_activation_function = activation_function
        self._generator_last_layer_activation = last_layer_activation
        self._generator_dropout_decay_rate_g = dropout_decay_rate_g
        self._generator_dense_layer_sizes_g = dense_layer_sizes_g
        self._generator_dataset_type = dataset_type
        self._generator_initializer_mean = initializer_mean
        self._generator_initializer_deviation = initializer_deviation
        self._generator_number_samples_per_class = number_samples_per_class
        self._generator_optimizer = optimizer
        self._generator_model_dense = None

    def get_generator(self) -> Model:
        """
        Constructs and returns the generator model using either dense or Conv1D layers.

        Returns:
        --------
        Model : keras.Model
            A Keras model implementing the generator with latent and label inputs.

        Raises:
        -------
        ValueError
            If number_samples_per_class is not properly defined.
        """
        if not self._generator_number_samples_per_class or "number_classes" not in self._generator_number_samples_per_class:
            raise ValueError("Number of samples per class must include 'number_classes'.")

        # Define inputs
        neural_model_inputs = Input(shape=(self._generator_latent_dimension,), dtype=self._generator_dataset_type)
        latent_input = Input(shape=(self._generator_latent_dimension,))
        label_input = Input(shape=(self._generator_number_samples_per_class["number_classes"],),
                            dtype=self._generator_dataset_type)

        # Build generator based on optimizer type
        if self._generator_optimizer == 'convolutional':
            generator_model = self._build_convolutional_generator(neural_model_inputs)
        else:
            generator_model = self._build_dense_generator(neural_model_inputs)

        self._generator_model_dense = generator_model

        # Concatenate label information
        concatenate_output = Concatenate()([latent_input, label_input])
        label_embedding = Flatten()(concatenate_output)
        model_input = Dense(self._generator_latent_dimension)(label_embedding)
        model_input = self._add_activation_layer(model_input, self._generator_activation_function)
        generator_output_flow = generator_model(model_input)

        return Model([latent_input, label_input], generator_output_flow, name="Generator")

    def _build_dense_generator(self, input_layer) -> Model:
        """
        Build a fully-connected (dense) generator architecture.

        Args:
            input_layer: Input layer for the generator.

        Returns:
            Model: A Keras Model with dense layers.
        """
        initialization = RandomNormal(mean=self._generator_initializer_mean,
                                      stddev=self._generator_initializer_deviation)

        generator_model = Dense(self._generator_dense_layer_sizes_g[0],
                                kernel_initializer=initialization)(input_layer)
        generator_model = Dropout(self._generator_dropout_decay_rate_g)(generator_model)
        generator_model = self._add_activation_layer(generator_model, self._generator_activation_function)

        for layer_size in self._generator_dense_layer_sizes_g[1:]:
            generator_model = Dense(layer_size, kernel_initializer=initialization)(generator_model)
            generator_model = Dropout(self._generator_dropout_decay_rate_g)(generator_model)
            generator_model = self._add_activation_layer(generator_model, self._generator_activation_function)

        generator_model = Dense(self._generator_output_shape, kernel_initializer=initialization)(generator_model)
        generator_model = self._add_activation_layer(generator_model, self._generator_last_layer_activation)

        return Model(input_layer, generator_model, name="Dense_Generator")

    def _build_convolutional_generator(self, input_layer) -> Model:
        """
        Build a 1D convolutional generator architecture.

        Args:
            input_layer: Input layer for the generator.

        Returns:
            Model: A Keras Model with Conv1D layers.
        """
        initialization = RandomNormal(mean=self._generator_initializer_mean,
                                      stddev=self._generator_initializer_deviation)

        # Start with a dense layer to project latent dimension to initial size
        initial_size = self._generator_output_shape // (2 ** len(self._generator_dense_layer_sizes_g))
        initial_size = max(4, initial_size)  # Minimum initial size

        generator_model = Dense(initial_size * self._generator_dense_layer_sizes_g[0],
                               kernel_initializer=initialization)(input_layer)
        generator_model = self._add_activation_layer(generator_model, self._generator_activation_function)
        generator_model = Dropout(self._generator_dropout_decay_rate_g)(generator_model)

        # Reshape to (batch, timesteps, features) for Conv1D
        generator_model = Reshape((initial_size, self._generator_dense_layer_sizes_g[0]))(generator_model)

        # Build convolutional layers with upsampling
        for i, filters in enumerate(self._generator_dense_layer_sizes_g[1:]):
            kernel_size = 3  # Fixed kernel size for upsampling

            generator_model = UpSampling1D(size=2)(generator_model)
            generator_model = Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                strides=1,
                padding='same',
                kernel_initializer=initialization
            )(generator_model)
            generator_model = self._add_activation_layer(generator_model, self._generator_activation_function)
            generator_model = Dropout(self._generator_dropout_decay_rate_g)(generator_model)

        # Upsample if needed to reach target output shape
        current_size = initial_size * (2 ** len(self._generator_dense_layer_sizes_g[1:]))
        while current_size < self._generator_output_shape:
            generator_model = UpSampling1D(size=2)(generator_model)
            current_size *= 2

        # Final Conv1D to get single feature dimension and flatten
        generator_model = Conv1D(
            filters=1,
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_initializer=initialization
        )(generator_model)
        generator_model = self._add_activation_layer(generator_model, self._generator_activation_function)

        # Flatten to match output shape
        generator_model = Flatten()(generator_model)

        # Final dense layer to ensure exact output shape
        generator_model = Dense(self._generator_output_shape, kernel_initializer=initialization)(generator_model)
        generator_model = self._add_activation_layer(generator_model, self._generator_last_layer_activation)

        return Model(input_layer, generator_model, name="Convolutional_Generator")

    @property
    def dense_generator_model(self) -> Model | None:
        """Property that retrieves the dense generator submodel without label conditioning."""
        return self._generator_model_dense

    @property
    def dropout_decay_rate_generator(self) -> float:
        """Property to get the dropout decay rate for the generator."""
        return self._generator_dropout_decay_rate_g

    @property
    def dense_layer_sizes_generator(self) -> list[int]:
        """Property to get the dense layer sizes for the generator."""
        return self._generator_dense_layer_sizes_g

    @property
    def optimizer(self) -> str:
        """Property to get the current optimizer/architecture type."""
        return self._generator_optimizer

    @dropout_decay_rate_generator.setter
    def dropout_decay_rate_generator(self, dropout_decay_rate_generator: float):
        """Property to set the dropout decay rate for the generator."""

        if dropout_decay_rate_generator < 0 or dropout_decay_rate_generator > 1:
            raise ValueError("Dropout decay rate must be in the range [0, 1].")

        self._generator_dropout_decay_rate_g = dropout_decay_rate_generator

    @dense_layer_sizes_generator.setter
    def dense_layer_sizes_generator(self, dense_layer_sizes_generator: list[int]):
        """Property to set the dense layer sizes for the generator."""

        if not dense_layer_sizes_generator or any(size <= 0 for size in dense_layer_sizes_generator):
            raise ValueError("Dense layer sizes must be a list of positive integers.")

        self._generator_dense_layer_sizes_g = dense_layer_sizes_generator

    @optimizer.setter
    def optimizer(self, optimizer: str):
        """Property to set the optimizer/architecture type."""
        if optimizer not in ['dense', 'convolutional']:
            raise ValueError("optimizer must be either 'dense' or 'convolutional'.")
        self._generator_optimizer = optimizer