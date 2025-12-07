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

    from typing import List
    from typing import Dict
    from typing import Union

    from typing import Optional

    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Model

    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dropout

    from tensorflow.keras.layers import Concatenate

    from Engine.Activations.Activations import Activations
    from tensorflow.keras.initializers import RandomNormal


except ImportError as error:
    print(error)
    sys.exit(-1)


class VanillaDiscriminator(Activations):
    """
     VanillaDiscriminator

     Implements a fully-connected (dense) discriminator network for use in generative models,
     such as Generative Adversarial Networks (GANs). This class provides flexibility in the design
     of the architecture, including customizable latent dimensions, output shapes, activation functions,
     dropout rates, and layer sizes. It allows easy adaptation to various GAN tasks where a discriminator
     or critic network is required.

     This class focuses on defining the model architecture and does not directly handle training
     or loss computation.

     Attributes:
         @discriminator_latent_dimension (int):
             Dimensionality of the input latent space for the discriminator network.
         @discriminator_output_shape (int):
             The output shape of the network, typically used to define the shape of input data like images.
         @discriminator_activation_function (str):
             The activation function applied to all hidden layers (e.g., 'relu', 'leaky_relu').
         @discriminator_last_layer_activation (str):
             The activation function applied to the last layer (e.g., 'sigmoid').
         @discriminator_dropout_decay_rate_d (float):
             Dropout rate applied to layers in the network to help prevent overfitting.
         @discriminator_dense_layer_sizes_d (List[int]):
             List of integers defining the number of units in each dense layer.
         @discriminator_dataset_type (numpy.dtype):
             The data type of the dataset (default: numpy.float32).
         @discriminator_initializer_mean (float):
             Mean of the normal distribution used for weight initialization.
         @discriminator_initializer_deviation (float):
             Standard deviation of the normal distribution used for weight initialization.
         @discriminator_number_samples_per_class (Optional[Dict[str, int]]):
             Optional dictionary containing the number of samples per class.
         @discriminator_model_dense (Optional[Model]):
             Placeholder for the compiled Keras model after building the network.

     Raises:
         ValueError:
             Raised if invalid arguments are passed during initialization, such as:
             - Non-positive `latent_dimension`
             - Dropout rate outside the range [0, 1]
             - Empty or invalid `dense_layer_sizes_d`
             - Missing required key "number_classes" in `number_samples_per_class`, if provided

     Example:
         >>> discriminator = VanillaDiscriminator(
         ...     latent_dimension=100,
         ...     output_shape=(28, 28, 1),
         ...     activation_function='leaky_relu',
         ...     initializer_mean=0.0,
         ...     initializer_deviation=0.02,
         ...     dropout_decay_rate_d=0.3,
         ...     last_layer_activation='sigmoid',
         ...     dense_layer_sizes_d=[512, 256, 128],
         ...     dataset_type=numpy.float32,
         ...     number_samples_per_class={"number_classes": 10}
         ... )
         >>> discriminator.build()  # Example method call if present
     """

    def __init__(self,
                 latent_dimension: int,
                 output_shape: int,
                 activation_function: str,
                 initializer_mean: float,
                 initializer_deviation: float,
                 dropout_decay_rate_d: float,
                 last_layer_activation: str,
                 dense_layer_sizes_d: List[int],
                 dataset_type: numpy.dtype = numpy.float32,
                 number_samples_per_class: Optional[Dict[str, int]] = None):
        """
        Initializes the VanillaDiscriminator class with the provided parameters.

        This constructor sets up the architecture of the discriminator, including the latent
        dimension, output shape, activation functions, weight initializers, dropout rates,
        and any additional information like class distribution metadata.

        Args:
            latent_dimension (int):
                The dimensionality of the input latent space.
            output_shape (int):
                The shape of the expected output data (e.g., image size).
            activation_function (str):
                The activation function to apply to all hidden layers.
            initializer_mean (float):
                The mean for weight initialization.
            initializer_deviation (float):
                The standard deviation for weight initialization.
            dropout_decay_rate_d (float):
                Dropout rate for dense layers (should be between 0 and 1).
            last_layer_activation (str):
                The activation function for the last output layer.
            dense_layer_sizes_d (List[int]):
                A list of integers specifying the number of units in each dense layer.
            dataset_type (numpy.dtype, optional):
                The data type of the input data (default is numpy.float32).
            number_samples_per_class (Optional[Dict[str, int]], optional):
                A dictionary containing metadata about class distribution. It should
                include the key "number_classes" if provided.

        Raises:
            ValueError:
                If `latent_dimension` is <= 0.
                If `dropout_decay_rate_d` is not within the range [0, 1].
                If `dense_layer_sizes_d` is empty or contains invalid values.
                If `number_samples_per_class` is provided but does not contain the key "number_classes".
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

        if not isinstance(dropout_decay_rate_d, (float, int)) or not (0 <= dropout_decay_rate_d <= 1):
            raise ValueError("dropout_decay_rate_d must be a float between 0 and 1.")

        if not isinstance(last_layer_activation, str):
            raise ValueError("last_layer_activation must be a string.")

        if not isinstance(dense_layer_sizes_d, list) or not all(isinstance(n, int) and n > 0 for n in dense_layer_sizes_d):
            raise ValueError("dense_layer_sizes_d must be a list of positive integers.")

        if number_samples_per_class is not None and not isinstance(number_samples_per_class, dict):
            raise ValueError("number_samples_per_class must be a dictionary if provided.")


        self._discriminator_number_samples_per_class = number_samples_per_class
        self._discriminator_latent_dimension = latent_dimension
        self._discriminator_output_shape = output_shape
        self._discriminator_activation_function = activation_function
        self._discriminator_last_layer_activation = last_layer_activation
        self._discriminator_dropout_decay_rate_d = dropout_decay_rate_d
        self._discriminator_dense_layer_sizes_d = dense_layer_sizes_d
        self._discriminator_dataset_type = dataset_type
        self._discriminator_initializer_mean = initializer_mean
        self._discriminator_initializer_deviation = initializer_deviation
        self._discriminator_model_dense = None

    def get_discriminator(self) -> Model:
        """
        Build and return the complete discriminator model.

        This method constructs a neural network model using dense layers with dropout and activation functions
        as specified during initialization. The model is built for the purpose of classifying inputs as real or fake.

        Returns:
            Model: A Keras Model instance representing the discriminator.
        """
        # Define the input layers
        neural_model_input = Input(shape=(self._discriminator_output_shape,), dtype=self._discriminator_dataset_type)
        discriminator_shape_input = Input(shape=(self._discriminator_output_shape,))
        label_input = Input(shape=(self._discriminator_number_samples_per_class["number_classes"],), dtype=self._discriminator_dataset_type)

        # Build the discriminator model
        discriminator_model = Dense(self._discriminator_dense_layer_sizes_d[0])(neural_model_input)
        discriminator_model = Dropout(self._discriminator_dropout_decay_rate_d)(discriminator_model)
        discriminator_model = self._add_activation_layer(discriminator_model, self._discriminator_activation_function)

        # Add additional dense layers with dropout and activations
        for layer_size in self._discriminator_dense_layer_sizes_d[1:]:
            discriminator_model = Dense(layer_size)(discriminator_model)
            discriminator_model = Dropout(self._discriminator_dropout_decay_rate_d)(discriminator_model)
            discriminator_model = self._add_activation_layer(discriminator_model, self._discriminator_activation_function)

        # Final output layer with specified activation function
        discriminator_model = Dense(1)(discriminator_model)
        discriminator_model = self._add_activation_layer(discriminator_model, self._discriminator_last_layer_activation)
        discriminator_model = Model(inputs=neural_model_input, outputs=discriminator_model)

        # Save the discriminator model for later use
        self._discriminator_model_dense = discriminator_model

        # Concatenate the input label and shape input, then process with a dense layer
        concatenate_output = Concatenate()([discriminator_shape_input, label_input])
        label_embedding = Flatten()(concatenate_output)
        model_input = Dense(self._discriminator_output_shape)(label_embedding)
        model_input = self._add_activation_layer(model_input, self._discriminator_activation_function)

        # Get the final output of the discriminator model
        validity = discriminator_model(model_input)

        return Model(inputs=[discriminator_shape_input, label_input], outputs=validity, name='Discriminator')

    def get_dense_discriminator_model(self) -> Optional[Model]:
        """
        Retrieve the dense discriminator model.

        Returns:
            Optional[Model]: The dense discriminator model, or None if not set.
        """
        return self._discriminator_model_dense

    def set_dropout_decay_rate_discriminator(self, dropout_decay_rate_discriminator: float):
        """
        Set the dropout decay rate for the discriminator network.

        Args:
            dropout_decay_rate_discriminator (float): The new dropout decay rate.
        """
        self._discriminator_dropout_decay_rate_d = dropout_decay_rate_discriminator

    def set_dense_layer_sizes_discriminator(self, dense_layer_sizes_discriminator: List[int]):
        """
        Set the sizes for the dense layers in the discriminator network.

        Args:
            dense_layer_sizes_discriminator (List[int]): A list of integers specifying the layer sizes.
        """
        self._discriminator_dense_layer_sizes_d = dense_layer_sizes_discriminator
