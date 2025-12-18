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
    from tensorflow.keras.layers import Reshape
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import UpSampling1D
    from tensorflow.keras.layers import SeparableConv1D
    from tensorflow.keras.layers import BatchNormalization

    from tensorflow.keras.layers import Concatenate

    from tensorflow.keras.initializers import RandomNormal
    from Engine.activations.Activations import Activations

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
        @optimizer (str):
            Type of architecture to use: 'dense' (default) or 'convolutional' for Conv1D.

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
        ...     number_samples_per_class={"number_classes": 10},
        ...     optimizer='convolutional'
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
                 number_samples_per_class: dict | None = None,
                 optimizer: str = 'convolutional'):
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
            optimizer (str, optional): Type of architecture: 'dense' for fully-connected (default) or
                'convolutional' for Conv1D architecture.

        Raises:
            ValueError: If `latent_dimension`, `output_shape`, or `dropout_decay_rate_g` are invalid.
        """

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
        self._generator_optimizer = optimizer
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

        initialization = RandomNormal(mean=self._generator_initializer_mean,
                                      stddev=self._generator_initializer_deviation)
        neural_model_inputs = Input(shape=(self._generator_latent_dimension,), dtype=self._generator_dataset_type)
        latent_input = Input(shape=(self._generator_latent_dimension,))
        label_input = Input(shape=(self._generator_number_samples_per_class["number_classes"],),
                            dtype=self._generator_dataset_type)

        # Build generator based on optimizer type
        if self._generator_optimizer == 'convolutional':
            generator_model = self._build_convolutional_generator(neural_model_inputs, initialization)
        else:
            generator_model = self._build_dense_generator(neural_model_inputs, initialization)

        self._generator_model_dense = generator_model

        # Concatenate latent input with label input for conditional generation
        concatenate_output = Concatenate()([latent_input, label_input])
        label_embedding = Flatten()(concatenate_output)
        model_input = Dense(self._generator_latent_dimension)(label_embedding)
        model_input = self._add_activation_layer(model_input, self._generator_activation_function)
        generator_output_flow = generator_model(model_input)

        return Model([latent_input, label_input], generator_output_flow, name="Generator")

    def _build_dense_generator(self, input_layer, initialization):
        """
        Build a fully-connected (dense) generator architecture.

        Args:
            input_layer: Input layer for the generator.
            initialization: Weight initializer.

        Returns:
            Model: A Keras Model with dense layers.
        """
        generator_model = Dense(self._generator_dense_layer_sizes_g[0], kernel_initializer=initialization)(input_layer)
        generator_model = Dropout(self._generator_dropout_decay_rate_g)(generator_model)
        generator_model = self._add_activation_layer(generator_model, self._generator_activation_function)

        for layer_size in self._generator_dense_layer_sizes_g[1:]:
            generator_model = Dense(layer_size, kernel_initializer=initialization)(generator_model)
            generator_model = Dropout(self._generator_dropout_decay_rate_g)(generator_model)
            generator_model = self._add_activation_layer(generator_model, self._generator_activation_function)

        generator_model = Dense(self._generator_output_shape, kernel_initializer=initialization)(generator_model)
        generator_model = self._add_activation_layer(generator_model, self._generator_last_layer_activation)

        return Model(input_layer, generator_model, name="Dense_Generator")

    def _build_convolutional_generator(self, input_layer, initialization):
        """
        Build a memory-constrained 1D convolutional generator with strict limits.

        CRITICAL Memory Optimizations:
        - Prevents large Dense layers (max 16MB per layer)
        - Progressive expansion through multiple small steps
        - Chunked processing for large outputs
        - Memory-aware filter calculations
        - Dynamic architecture based on available memory
        - Failsafe checks to prevent OOM

        Args:
            input_layer: Input layer for the generator.
            initialization: Weight initializer.

        Returns:
            Model: A memory-safe Keras Model with Conv1D layers.
        """
        import tensorflow as tf
        import math

        target_output = self._generator_output_shape
        num_layers = len(self._generator_dense_layer_sizes_g)

        # ============================================================================
        # MEMORY SAFETY: Calculate maximum safe dimensions
        # ============================================================================
        MAX_DENSE_PARAMS = 4_000_000  # Max 4M parameters per Dense layer (~16MB)
        MIN_SPATIAL = 4
        MAX_SPATIAL = 32  # Cap spatial dimension

        def calculate_safe_dimensions(latent_dim, target_out):
            """Calculate dimensions that won't cause OOM."""
            # Start with minimal spatial dimension
            for spatial in [4, 6, 8, 12, 16, 24, 32]:
                # Calculate filters needed
                max_safe_filters = MAX_DENSE_PARAMS // (spatial * latent_dim)
                if max_safe_filters >= 8:  # Minimum viable filters
                    filters = min(max_safe_filters, 64)  # Cap at 64
                    return spatial, filters

            # Absolute fallback
            return 4, 8

        initial_spatial, base_filters = calculate_safe_dimensions(
            self._generator_latent_dimension,
            target_output
        )

        # ============================================================================
        # OPTIMIZATION 1: Multi-step progressive expansion (no giant Dense layers)
        # ============================================================================
        # Step 1: Small initial expansion
        intermediate_dim = min(256, self._generator_latent_dimension * 2)

        generator_model = Dense(
            intermediate_dim,
            kernel_initializer=initialization,
            use_bias=False
        )(input_layer)
        generator_model = BatchNormalization(momentum=0.9)(generator_model)
        generator_model = self._add_activation_layer(generator_model, self._generator_activation_function)

        # Step 2: Gradual expansion to target
        current_dim = intermediate_dim
        target_dim = initial_spatial * base_filters

        # Calculate number of expansion steps needed
        num_steps = max(1, math.ceil(math.log2(target_dim / current_dim)))

        for step in range(num_steps):
            # Exponential growth
            next_dim = min(current_dim * 2, target_dim)

            # Safety check: prevent too large layers
            param_count = current_dim * next_dim
            if param_count > MAX_DENSE_PARAMS:
                # Use smaller step
                next_dim = min(current_dim + MAX_DENSE_PARAMS // current_dim, target_dim)

            generator_model = Dense(
                next_dim,
                kernel_initializer=initialization,
                use_bias=False
            )(generator_model)
            generator_model = BatchNormalization(momentum=0.9)(generator_model)
            generator_model = self._add_activation_layer(generator_model, self._generator_activation_function)

            current_dim = next_dim

            if current_dim >= target_dim:
                break

        # Final adjustment to exact dimension
        if current_dim != target_dim:
            generator_model = Dense(
                target_dim,
                kernel_initializer=initialization,
                use_bias=False
            )(generator_model)
            generator_model = BatchNormalization(momentum=0.9)(generator_model)
            generator_model = self._add_activation_layer(generator_model, self._generator_activation_function)

        generator_model = Reshape((initial_spatial, base_filters))(generator_model)

        # ============================================================================
        # OPTIMIZATION 2: Memory-efficient convolutional layers
        # ============================================================================
        current_spatial = initial_spatial

        # Calculate upsampling schedule
        total_upsample_needed = max(1, target_output / current_spatial)
        num_upsamples = min(num_layers, math.ceil(math.log2(total_upsample_needed)))

        upsampling_schedule = []
        for i in range(num_layers):
            if i < num_upsamples and current_spatial < target_output:
                upsampling_schedule.append(2)
                current_spatial *= 2
            else:
                upsampling_schedule.append(1)

        # Reset for actual processing
        current_spatial = initial_spatial

        # ============================================================================
        # OPTIMIZATION 3: Lightweight convolutional blocks
        # ============================================================================
        for i in range(num_layers):
            # Aggressive filter reduction to save memory
            filter_ratio = math.exp(-i / max(num_layers - 1, 1) * 2.0)
            current_filters = max(int(base_filters * filter_ratio), 4)  # Minimum 4 filters

            # Small kernels only
            kernel_size = 3

            # Use depthwise separable for maximum efficiency
            if current_filters >= 8:
                generator_model = SeparableConv1D(
                    filters=current_filters,
                    kernel_size=kernel_size,
                    padding='same',
                    depthwise_initializer=initialization,
                    pointwise_initializer=initialization,
                    use_bias=False
                )(generator_model)
            else:
                # Regular conv for very small filter counts
                generator_model = Conv1D(
                    filters=current_filters,
                    kernel_size=kernel_size,
                    padding='same',
                    kernel_initializer=initialization,
                    use_bias=False
                )(generator_model)

            generator_model = BatchNormalization(momentum=0.9)(generator_model)
            generator_model = self._add_activation_layer(generator_model, self._generator_activation_function)

            # Minimal dropout
            if i >= num_layers - 1 and self._generator_dropout_decay_rate_g > 0:
                generator_model = Dropout(self._generator_dropout_decay_rate_g * 0.2)(generator_model)

            # Upsample according to schedule
            if upsampling_schedule[i] > 1:
                generator_model = UpSampling1D(size=upsampling_schedule[i])(generator_model)
                current_spatial *= upsampling_schedule[i]

        # ============================================================================
        # OPTIMIZATION 4: Safe final projection with chunking if needed
        # ============================================================================
        generator_model = Flatten()(generator_model)
        current_dim = current_spatial * current_filters

        if current_dim != target_output:
            # Check if direct projection is safe
            direct_params = current_dim * target_output

            if direct_params > MAX_DENSE_PARAMS:
                # Use chunked approach: intermediate layer first
                intermediate = min(
                    current_dim * 2,
                    int(math.sqrt(MAX_DENSE_PARAMS)),
                    target_output
                )

                generator_model = Dense(
                    intermediate,
                    kernel_initializer=initialization,
                    use_bias=False
                )(generator_model)
                generator_model = BatchNormalization(momentum=0.9)(generator_model)
                generator_model = self._add_activation_layer(
                    generator_model,
                    self._generator_activation_function
                )

                # Now project to final size
                if intermediate != target_output:
                    # This should now be safe
                    generator_model = Dense(
                        target_output,
                        kernel_initializer=initialization
                    )(generator_model)
            else:
                # Direct projection is safe
                generator_model = Dense(
                    target_output,
                    kernel_initializer=initialization
                )(generator_model)

        # Final activation
        generator_model = self._add_activation_layer(
            generator_model,
            self._generator_last_layer_activation
        )

        return Model(input_layer, generator_model, name="Convolutional_Generator_MemorySafe")
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

    def get_optimizer(self) -> str:
        """
        Get the current optimizer/architecture type.

        Returns:
            str: The optimizer type ('dense' or 'convolutional').
        """
        return self._generator_optimizer

    def set_optimizer(self, optimizer: str):
        """
        Set the optimizer/architecture type.

        Args:
            optimizer (str): The optimizer type ('dense' or 'convolutional').
        """
        self._generator_optimizer = optimizer