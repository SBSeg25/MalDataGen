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

    from tensorflow.keras.layers import Concatenate

    from tensorflow.keras.initializers import RandomNormal
    from Engine.Activations.Activations import Activations

except ImportError as error:
    print(error)
    sys.exit(-1)


class VanillaGenerator(Activations):
    """
    VanillaGenerator

    Implements a fully connected (dense) generator model for use in generative models,
    such as GANs. This generator is designed to work with label conditioning and
    supports customization of activation functions, layer sizes, initialization, and
    other hyperparameters.

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
          Generative Adversarial Networks. arXiv preprint arXiv:1406.2661.
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
        ...     number_samples_per_class={"number_classes": 10}
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
                 number_samples_per_class: dict | None = None):
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
        self._generator_model_dense = None

    def get_generator(self) -> Model:
        """
        Constructs and returns the generator model, including the latent space input and label conditioning.

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

        initialization = RandomNormal(mean=self._generator_initializer_mean,
                                      stddev=self._generator_initializer_deviation)

        # Define inputs
        neural_model_inputs = Input(shape=(self._generator_latent_dimension,), dtype=self._generator_dataset_type)
        latent_input = Input(shape=(self._generator_latent_dimension,))
        label_input = Input(shape=(self._generator_number_samples_per_class["number_classes"],),
                            dtype=self._generator_dataset_type)

        # Dense generator model
        generator_model = Dense(self._generator_dense_layer_sizes_g[0],
                                kernel_initializer=initialization)(neural_model_inputs)
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

        # Concatenate label information
        concatenate_output = Concatenate()([latent_input, label_input])
        label_embedding = Flatten()(concatenate_output)
        model_input = Dense(self._generator_latent_dimension)(label_embedding)
        model_input = self._add_activation_layer(model_input, self._generator_activation_function)
        generator_output_flow = generator_model(model_input)

        return Model([latent_input, label_input], generator_output_flow, name="Generator")

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
