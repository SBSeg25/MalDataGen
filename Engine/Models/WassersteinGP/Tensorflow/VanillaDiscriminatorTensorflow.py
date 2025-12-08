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
    
    from typing import Dict
    from typing import List
    from typing import Tuple
    from typing import Optional
    from typing import Callable

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



class VanillaDiscriminator(Activations):
    """
    VanillaDiscriminator

    Implements a fully-connected (dense) discriminator network for use in generative models,
    such as GANs or WGANs. This class supports fully customizable layer sizes, activation
    functions, dropout rates, and initialization schemes, allowing it to be adapted to
    various tasks requiring a critic or discriminator network.

    This class does not implement training or loss computation directly, focusing instead
    on the architecture definition and construction.

    Attributes:
        @discriminator_latent_dimension (int):
            Dimensionality of the latent space used by the model.
        @discriminator_output_shape (Tuple[int, ...]):
            Shape of the expected output data (e.g., for image discrimination, this
            could be (28, 28, 1) for grayscale images).
        @discriminator_activation_function (Callable):
            Activation function applied to all hidden layers.
        @discriminator_last_layer_activation (Callable):
            Activation function applied to the final output layer.
        @discriminator_dropout_decay_rate_d (float):
            Dropout rate applied to dense layers to improve generalization.
        @discriminator_dense_layer_sizes_d (List[int]):
            List of integers specifying the number of units in each dense layer.
        @discriminator_dataset_type (type):
            Data type of the input dataset (default: numpy.float32).
        @discriminator_initializer_mean (float):
            Mean of the normal distribution used for weight initialization.
        @discriminator_initializer_deviation (float):
            Standard deviation of the normal distribution used for weight initialization.
        @discriminator_number_samples_per_class (Optional[Dict[str, int]]):
            Optional dictionary containing metadata about class distribution.
            Must include a key "number_classes" if provided.
        @discriminator_model_dense (Optional[Model]):
            Placeholder for the compiled Keras Model after build().

    Raises:
        ValueError:
            Raised if invalid arguments are passed during initialization, such as:
            - Non-positive `latent_dimension`
            - Dropout rate outside the range [0, 1]
            - Empty or invalid `dense_layer_sizes_d`
            - Missing required key "number_classes" in `number_samples_per_class`, if provided

    References:
        - Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014).
          Generative Adversarial Networks. arXiv preprint arXiv:1406.2661.
          Available at: https://arxiv.org/abs/1406.2661

    Example:
        >>> discriminator = VanillaDiscriminator(
        ...     latent_dimension=100,
        ...     output_shape=(28, 28, 1),
        ...     activation_function=tf.nn.leaky_relu,
        ...     initializer_mean=0.0,
        ...     initializer_deviation=0.02,
        ...     dropout_decay_rate_d=0.3,
        ...     last_layer_activation=tf.nn.sigmoid,
        ...     dense_layer_sizes_d=[512, 256, 128],
        ...     dataset_type=numpy.float32,
        ...     number_samples_per_class={"number_classes": 10}
        ... )
        >>> discriminator.build()  # Example method call if present
    """

    def __init__(
            self,
            latent_dimension: int,
            output_shape: Tuple[int, ...],
            activation_function: Callable,
            initializer_mean: float,
            initializer_deviation: float,
            dropout_decay_rate_d: float,
            last_layer_activation: Callable,
            dense_layer_sizes_d: List[int],
            dataset_type: type = numpy.float32,
            number_samples_per_class: Optional[Dict[str, int]] = None) -> None:
        """
        Initializes the VanillaDiscriminator.

        This constructor sets up all internal attributes related to the discriminator
        architecture, including layer sizes, activation functions, initializers, and
        optional class distribution metadata.

        Args:
            @latent_dimension (int): Dimensionality of the latent space.
            @output_shape (Tuple[int, ...]): Shape of the output data.
            @activation_function (Callable): Activation function for all hidden layers.
            @initializer_mean (float): Mean of the normal distribution used to initialize weights.
            @initializer_deviation (float): Standard deviation of the normal distribution used to initialize weights.
            @dropout_decay_rate_d (float): Dropout rate applied to dense layers (0 to 1).
            @last_layer_activation (Callable): Activation function applied to the final output layer.
            @dense_layer_sizes_d (List[int]): List of integers specifying the number of units per dense layer.
            @dataset_type (type, optional):Data type of the input data (default: numpy.float32).
            @number_samples_per_class (Optional[Dict[str, int]], optional): Optional dictionary containing
            the number of samples per class. If provided, it must contain the key "number_classes".

        Raises:
            ValueError:
                If `latent_dimension` is <= 0.
                If `dropout_decay_rate_d` is not within [0, 1].
                If `dense_layer_sizes_d` is empty or contains non-positive values.
                If `number_samples_per_class` is provided but does not contain the key "number_classes".

        """

        if latent_dimension <= 0:
            raise ValueError("The latent_dimension must be a positive integer.")

        if dropout_decay_rate_d < 0 or dropout_decay_rate_d > 1:
            raise ValueError("The dropout_decay_rate_d must be between 0 and 1.")

        if not dense_layer_sizes_d or not all(isinstance(x, int) and x > 0 for x in dense_layer_sizes_d):
            raise ValueError("dense_layer_sizes_d must be a non-empty list of positive integers.")

        if number_samples_per_class and "number_classes" not in number_samples_per_class:
            raise ValueError("number_samples_per_class must include a 'number_classes' key if provided.")

        self._discriminator_latent_dimension: int = latent_dimension
        self._discriminator_output_shape: Tuple[int, ...] = output_shape
        self._discriminator_activation_function: Callable = activation_function
        self._discriminator_last_layer_activation: Callable = last_layer_activation
        self._discriminator_dropout_decay_rate_d: float = dropout_decay_rate_d
        self._discriminator_dense_layer_sizes_d: List[int] = dense_layer_sizes_d
        self._discriminator_dataset_type: type = dataset_type
        self._discriminator_initializer_mean: float = initializer_mean
        self._discriminator_initializer_deviation: float = initializer_deviation
        self._discriminator_number_samples_per_class: Optional[Dict[str, int]] = number_samples_per_class
        self._discriminator_model_dense: Optional[Model] = None

    def get_discriminator(self) -> Model:
        """
        Constructs the discriminator model using dense layers, dropout, and activation functions.

        Returns:
            Model: A Keras Model instance representing the discriminator.

        Raises:
            ValueError: If number_samples_per_class or its "number_classes" key is not properly set.
        """
        if not self._discriminator_number_samples_per_class or "number_classes" not in self._discriminator_number_samples_per_class:
            raise ValueError("number_samples_per_class with a 'number_classes' key must be provided to construct the model.")

        # Initialize the weights using a normal distribution with specified mean and deviation
        initialization = RandomNormal(mean=self._discriminator_initializer_mean, stddev=self._discriminator_initializer_deviation)

        neural_model_input = Input(shape=(self._discriminator_output_shape,), dtype=self._discriminator_dataset_type)
        discriminator_shape_input = Input(shape=(self._discriminator_output_shape,))
        label_input = Input(shape=(self._discriminator_number_samples_per_class["number_classes"],),
                            dtype=self._discriminator_dataset_type)

        discriminator_model = Dense(self._discriminator_dense_layer_sizes_d[0],
                                    kernel_initializer=initialization)(neural_model_input)
        discriminator_model = Dropout(self._discriminator_dropout_decay_rate_d)(discriminator_model)
        discriminator_model = self._add_activation_layer(discriminator_model, self._discriminator_activation_function)

        for layer_size in self._discriminator_dense_layer_sizes_d[1:]:
            discriminator_model = Dense(layer_size,
                                        kernel_initializer=initialization)(discriminator_model)
            discriminator_model = Dropout(self._discriminator_dropout_decay_rate_d)(discriminator_model)
            discriminator_model = self._add_activation_layer(discriminator_model, self._discriminator_activation_function)

        discriminator_model = Dense(1)(discriminator_model)
        discriminator_model = self._add_activation_layer(discriminator_model, self._discriminator_last_layer_activation)

        discriminator_model = Model(inputs=neural_model_input, outputs=discriminator_model, name="Dense_Discriminator")
        self._discriminator_model_dense = discriminator_model

        concatenate_output = Concatenate()([discriminator_shape_input, label_input])
        label_embedding = Flatten()(concatenate_output)
        model_input = Dense(self._discriminator_output_shape, kernel_initializer=initialization)(label_embedding)

        validity = discriminator_model(model_input)

        return Model(inputs=[discriminator_shape_input, label_input], outputs=validity, name="Discriminator")


    def dense_discriminator_model(self) -> Optional[Model]:
        """Returns the dense part of the discriminator model."""
        return self._discriminator_model_dense
      
    def dropout_decay_rate_discriminator(self) -> float:
        """Gets the dropout rate for the discriminator."""
        return self._discriminator_dropout_decay_rate_d

    def dense_layer_sizes_discriminator(self) -> List[int]:
        """Gets the sizes of the dense layers for the discriminator."""
        return self._discriminator_dense_layer_sizes_d
      
    def dropout_decay_rate_discriminator(self, dropout_decay_rate_discriminator: float) -> None:
        """Sets the dropout rate for the discriminator."""
        if dropout_decay_rate_discriminator < 0 or dropout_decay_rate_discriminator > 1:
            raise ValueError("The dropout_decay_rate_discriminator must be between 0 and 1.")
        self._discriminator_dropout_decay_rate_d = dropout_decay_rate_discriminator

    def dense_layer_sizes_discriminator(self, dense_layer_sizes_discriminator: List[int]) -> None:
        """Sets the sizes of the dense layers for the discriminator."""
        if not dense_layer_sizes_discriminator or not all(isinstance(x, int) and x > 0 for x in dense_layer_sizes_discriminator):
            raise ValueError("dense_layer_sizes_discriminator must be a non-empty list of positive integers.")
        self._discriminator_dense_layer_sizes_d = dense_layer_sizes_discriminator
