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
    from tensorflow.keras.layers import Reshape
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import MaxPooling1D

    from tensorflow.keras.layers import Concatenate

    from Engine.activations.Activations import Activations
    from tensorflow.keras.initializers import RandomNormal


except ImportError as error:
    print(error)
    sys.exit(-1)


class VanillaDiscriminator(Activations):
    """
     VanillaDiscriminator

     Implements a fully-connected (dense) discriminator network for use in generative models,
     such as Generative adversarial Networks (GANs). This class provides flexibility in the design
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
         @discriminator_optimizer (str):
             Type of architecture to use: 'dense' (default) or 'convolutional' for Conv1D.
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
         ...     number_samples_per_class={"number_classes": 10},
         ...     optimizer='convolutional'
         ... )
         >>> discriminator.build()  # Example method call if present
     """

    def __init__(self, latent_dimension: int, output_shape: int, activation_function: str, initializer_mean: float,
                 initializer_deviation: float, dropout_decay_rate_d: float, last_layer_activation: str,
                 dense_layer_sizes_d: List[int], dataset_type: numpy.dtype = numpy.float32,
                 number_samples_per_class: Optional[Dict[str, int]] = None, optimizer: str = 'convolutional'):
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
            optimizer (str, optional):
                Type of architecture: 'dense' for fully-connected (default) or
                'convolutional' for Conv1D architecture.

        Raises:
            ValueError:
                If `latent_dimension` is <= 0.
                If `dropout_decay_rate_d` is not within the range [0, 1].
                If `dense_layer_sizes_d` is empty or contains invalid values.
                If `number_samples_per_class` is provided but does not contain the key "number_classes".
        """

        super().__init__()
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
        self._discriminator_optimizer = optimizer
        self._discriminator_model_dense = None

    def set_model(self, model):
        self._discriminator_model_dense = model

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

    def get_optimizer(self) -> str:
        """
        Get the current optimizer/architecture type.

        Returns:
            str: The optimizer type ('dense' or 'convolutional').
        """
        return self._discriminator_optimizer

    def set_optimizer(self, optimizer: str):
        """
        Set the optimizer/architecture type.

        Args:
            optimizer (str): The optimizer type ('dense' or 'convolutional').
        """
        self._discriminator_optimizer = optimizer

    def _build_dense_discriminator(self, input_layer):
        """
        Build a fully-connected (dense) discriminator architecture.

        Args:
            input_layer: Input layer for the discriminator.

        Returns:
            Model: A Keras Model with dense layers.
        """
        discriminator_model = Dense(self._discriminator_dense_layer_sizes_d[0])(input_layer)
        discriminator_model = Dropout(self._discriminator_dropout_decay_rate_d)(discriminator_model)
        discriminator_model = self._add_activation_layer(discriminator_model, self._discriminator_activation_function)

        # Add additional dense layers with dropout and activations
        for layer_size in self._discriminator_dense_layer_sizes_d[1:]:
            discriminator_model = Dense(layer_size)(discriminator_model)
            discriminator_model = Dropout(self._discriminator_dropout_decay_rate_d)(discriminator_model)
            discriminator_model = self._add_activation_layer(discriminator_model,
                                                             self._discriminator_activation_function)

        # Final output layer with specified activation function
        discriminator_model = Dense(1)(discriminator_model)
        discriminator_model = self._add_activation_layer(discriminator_model, self._discriminator_last_layer_activation)

        return Model(inputs=input_layer, outputs=discriminator_model)

    # ALTERNATIVE: Ultra-lightweight version for extreme memory constraints
    def _build_ultra_lightweight_discriminator(self, input_layer):
        """
        Ultra-lightweight convolutional discriminator for extreme memory constraints.

        Sacrifices some discriminative power for minimal memory usage.
        """
        from tensorflow.keras.layers import GlobalAveragePooling1D, DepthwiseConv1D

        discriminator_model = Reshape((self._discriminator_output_shape, 1))(input_layer)

        # Immediate massive downsampling
        if self._discriminator_output_shape > 1000:
            discriminator_model = MaxPooling1D(pool_size=16, padding='same')(discriminator_model)
        elif self._discriminator_output_shape > 500:
            discriminator_model = MaxPooling1D(pool_size=8, padding='same')(discriminator_model)
        elif self._discriminator_output_shape > 100:
            discriminator_model = MaxPooling1D(pool_size=4, padding='same')(discriminator_model)

        # Single lightweight convolutional block
        # Use DepthwiseConv1D: processes each channel separately (minimal params)
        discriminator_model = DepthwiseConv1D(
            kernel_size=3,
            strides=2,
            padding='same',
            depthwise_initializer=RandomNormal(
                mean=self._discriminator_initializer_mean,
                stddev=self._discriminator_initializer_deviation
            )
        )(discriminator_model)
        discriminator_model = self._add_activation_layer(
            discriminator_model,
            self._discriminator_activation_function
        )

        # Pointwise convolution to mix channels (1x1 conv)
        discriminator_model = Conv1D(32, kernel_size=1, padding='same')(discriminator_model)
        discriminator_model = self._add_activation_layer(
            discriminator_model,
            self._discriminator_activation_function
        )
        discriminator_model = Dropout(self._discriminator_dropout_decay_rate_d)(discriminator_model)

        # Global pooling to collapse spatial dimensions
        discriminator_model = GlobalAveragePooling1D()(discriminator_model)

        # Minimal dense path to output
        discriminator_model = Dense(16)(discriminator_model)
        discriminator_model = self._add_activation_layer(
            discriminator_model,
            self._discriminator_activation_function
        )

        # Final output
        discriminator_model = Dense(1)(discriminator_model)
        discriminator_model = self._add_activation_layer(
            discriminator_model,
            self._discriminator_last_layer_activation
        )

        return Model(inputs=input_layer, outputs=discriminator_model, name="Ultra_Lightweight_Discriminator")

    def _build_convolutional_discriminator(self, input_layer):
        """
        Build a memory-safe 1D convolutional discriminator with strict memory controls.

        CRITICAL Memory Optimizations:
        - Progressive downsampling strategy based on input size
        - Minimal filter counts with intelligent scaling
        - SeparableConv1D for 9x parameter reduction
        - GlobalAveragePooling to eliminate massive Flatten
        - Adaptive architecture based on input dimension
        - Early pooling for large inputs (>1000 dims)
        - Memory-aware dense layers

        Args:
            input_layer: Input layer for the discriminator.

        Returns:
            Model: A memory-safe Keras Model with Conv1D layers.
        """
        from tensorflow.keras.layers import GlobalAveragePooling1D, SeparableConv1D, BatchNormalization
        import math

        # ============================================================================
        # MEMORY SAFETY: Calculate optimal architecture parameters
        # ============================================================================
        MAX_FILTERS = 64  # Conservative maximum
        MIN_FILTERS = 8
        TARGET_SPATIAL_FINAL = 8  # Target spatial dimension before pooling

        input_dim = self._discriminator_output_shape

        # Calculate required downsampling
        total_downsample_needed = max(1, input_dim / TARGET_SPATIAL_FINAL)
        num_downsample_ops = math.ceil(math.log2(total_downsample_needed))

        # ============================================================================
        # OPTIMIZATION 1: Intelligent initial downsampling
        # ============================================================================
        discriminator_model = Reshape((input_dim, 1))(input_layer)
        current_spatial = input_dim

        # Aggressive initial pooling based on input size
        if input_dim > 4096:
            # Extreme: reduce by 16x immediately
            discriminator_model = MaxPooling1D(pool_size=16, padding='same')(discriminator_model)
            current_spatial = (current_spatial + 15) // 16
        elif input_dim > 2048:
            # Very large: reduce by 8x
            discriminator_model = MaxPooling1D(pool_size=8, padding='same')(discriminator_model)
            current_spatial = (current_spatial + 7) // 8
        elif input_dim > 1024:
            # Large: reduce by 4x
            discriminator_model = MaxPooling1D(pool_size=4, padding='same')(discriminator_model)
            current_spatial = (current_spatial + 3) // 4
        elif input_dim > 512:
            # Medium: reduce by 2x
            discriminator_model = MaxPooling1D(pool_size=2, padding='same')(discriminator_model)
            current_spatial = (current_spatial + 1) // 2

        # ============================================================================
        # OPTIMIZATION 2: Calculate optimal number of conv layers
        # ============================================================================
        # Fewer layers for larger inputs (already downsampled heavily)
        if input_dim > 2048:
            num_conv_layers = 2
        elif input_dim > 1024:
            num_conv_layers = 3
        elif input_dim > 512:
            num_conv_layers = 3
        else:
            num_conv_layers = min(4, len(self._discriminator_dense_layer_sizes_d))

        # ============================================================================
        # OPTIMIZATION 3: Memory-efficient filter progression
        # ============================================================================
        # Start small, grow moderately
        filter_progression = []
        for i in range(num_conv_layers):
            # Logarithmic growth, capped at MAX_FILTERS
            filters = min(MIN_FILTERS * (2 ** i), MAX_FILTERS)
            filter_progression.append(filters)

        # ============================================================================
        # OPTIMIZATION 4: Convolutional blocks with adaptive downsampling
        # ============================================================================
        for i, filters in enumerate(filter_progression):
            # Adaptive kernel size (smaller for better memory)
            if current_spatial < 16:
                kernel_size = 3
            elif current_spatial < 64:
                kernel_size = 3
            else:
                kernel_size = 5  # Slightly larger for very large dims

            # Calculate stride based on current spatial dimension
            # More aggressive striding for larger dimensions
            if current_spatial > 64 and i < 2:
                stride = 2
            elif current_spatial > 32 and i < 1:
                stride = 2
            else:
                stride = 1

            # Use SeparableConv1D for maximum efficiency (9x fewer params)
            if filters >= 16 and current_spatial >= 8:
                discriminator_model = SeparableConv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding='same',
                    depthwise_initializer=RandomNormal(
                        mean=self._discriminator_initializer_mean,
                        stddev=self._discriminator_initializer_deviation
                    ),
                    pointwise_initializer=RandomNormal(
                        mean=self._discriminator_initializer_mean,
                        stddev=self._discriminator_initializer_deviation
                    ),
                    use_bias=False
                )(discriminator_model)
            else:
                # Regular Conv1D for very small dimensions or filter counts
                discriminator_model = Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding='same',
                    kernel_initializer=RandomNormal(
                        mean=self._discriminator_initializer_mean,
                        stddev=self._discriminator_initializer_deviation
                    ),
                    use_bias=False
                )(discriminator_model)

            # BatchNorm (more efficient than heavy dropout)
            discriminator_model = BatchNormalization(momentum=0.9)(discriminator_model)
            discriminator_model = self._add_activation_layer(
                discriminator_model,
                self._discriminator_activation_function
            )

            # Minimal dropout
            if i >= num_conv_layers - 1:
                discriminator_model = Dropout(self._discriminator_dropout_decay_rate_d * 0.3)(discriminator_model)

            # Update spatial dimension after stride
            current_spatial = (current_spatial + stride - 1) // stride

            # Additional pooling if still too large
            if current_spatial > 64 and i < num_conv_layers - 1:
                pool_size = 2
                discriminator_model = MaxPooling1D(pool_size=pool_size, padding='same')(discriminator_model)
                current_spatial = (current_spatial + pool_size - 1) // pool_size

        # ============================================================================
        # OPTIMIZATION 5: Global Average Pooling (eliminates Flatten memory spike)
        # ============================================================================
        # This converts (batch, spatial, filters) -> (batch, filters)
        # No parameters, no memory issues!
        discriminator_model = GlobalAveragePooling1D()(discriminator_model)

        # ============================================================================
        # OPTIMIZATION 6: Minimal dense pathway to output
        # ============================================================================
        # Use very small dense layer (fixed 32 units)
        dense_size = 32
        discriminator_model = Dense(
            dense_size,
            kernel_initializer=RandomNormal(
                mean=self._discriminator_initializer_mean,
                stddev=self._discriminator_initializer_deviation
            )
        )(discriminator_model)
        discriminator_model = self._add_activation_layer(
            discriminator_model,
            self._discriminator_activation_function
        )
        discriminator_model = Dropout(self._discriminator_dropout_decay_rate_d * 0.5)(discriminator_model)

        # Final binary classification output
        discriminator_model = Dense(1)(discriminator_model)
        discriminator_model = self._add_activation_layer(
            discriminator_model,
            self._discriminator_last_layer_activation
        )

        return Model(inputs=input_layer, outputs=discriminator_model, name="Discriminator_ConvMemorySafe")

    def get_discriminator(self) -> Model:
        """
        Build and return the complete discriminator model with memory-safe embedding.

        CRITICAL FIX: Eliminates massive Dense layers that cause OOM.
        Uses progressive expansion and intelligent embedding strategies.

        Returns:
            Model: A memory-safe Keras Model instance.
        """
        import tensorflow as tf
        from tensorflow.keras.layers import Lambda

        MAX_DENSE_PARAMS = 4_000_000  # Max 4M parameters per layer (~16MB)

        # Define input layers
        neural_model_input = Input(
            shape=(self._discriminator_output_shape,),
            dtype=self._discriminator_dataset_type
        )
        discriminator_shape_input = Input(shape=(self._discriminator_output_shape,))
        label_input = Input(
            shape=(self._discriminator_number_samples_per_class["number_classes"],),
            dtype=self._discriminator_dataset_type
        )

        # Build discriminator based on optimizer type
        if self._discriminator_optimizer == 'convolutional':
            discriminator_model = self._build_convolutional_discriminator(neural_model_input)
        else:
            discriminator_model = self._build_dense_discriminator(neural_model_input)

        self._discriminator_model_dense = discriminator_model

        # ============================================================================
        # OPTIMIZATION: Memory-safe label conditioning
        # ============================================================================
        num_classes = self._discriminator_number_samples_per_class["number_classes"]
        output_shape = self._discriminator_output_shape

        # Concatenate inputs
        concatenate_output = Concatenate()([discriminator_shape_input, label_input])
        label_embedding = Flatten()(concatenate_output)

        # Calculate safe embedding size
        input_size = output_shape + num_classes

        # Strategy 1: Use small embedding first
        initial_embedding_size = min(128, input_size // 4)
        initial_embedding_size = max(32, initial_embedding_size)

        model_input = Dense(
            initial_embedding_size,
            kernel_initializer=RandomNormal(
                mean=self._discriminator_initializer_mean,
                stddev=self._discriminator_initializer_deviation
            )
        )(label_embedding)
        model_input = self._add_activation_layer(model_input, self._discriminator_activation_function)

        # Strategy 2: Progressive expansion to output_shape
        current_dim = initial_embedding_size
        target_dim = output_shape

        # Check if we need to expand
        if current_dim < target_dim:
            # Calculate safe expansion steps
            expansion_ratio = target_dim / current_dim

            if expansion_ratio > 8:
                # Large expansion needed - use multiple steps
                intermediate_sizes = []

                # Calculate intermediate steps
                num_steps = math.ceil(math.log2(expansion_ratio))
                for step in range(1, num_steps):
                    intermediate = min(
                        int(current_dim * (2 ** step)),
                        target_dim
                    )

                    # Check if this step is safe
                    params = current_dim * intermediate
                    if params <= MAX_DENSE_PARAMS:
                        intermediate_sizes.append(intermediate)
                        current_dim = intermediate
                    else:
                        # Use smaller step
                        safe_size = min(
                            current_dim + MAX_DENSE_PARAMS // current_dim,
                            target_dim
                        )
                        intermediate_sizes.append(safe_size)
                        current_dim = safe_size

                    if current_dim >= target_dim:
                        break

                # Apply intermediate layers
                for size in intermediate_sizes:
                    model_input = Dense(
                        size,
                        kernel_initializer=RandomNormal(
                            mean=self._discriminator_initializer_mean,
                            stddev=self._discriminator_initializer_deviation
                        )
                    )(model_input)
                    model_input = self._add_activation_layer(
                        model_input,
                        self._discriminator_activation_function
                    )

            # Final expansion to exact output_shape
            if current_dim != target_dim:
                # Check if direct expansion is safe
                params = current_dim * target_dim

                if params <= MAX_DENSE_PARAMS:
                    # Direct expansion is safe
                    model_input = Dense(
                        target_dim,
                        kernel_initializer=RandomNormal(
                            mean=self._discriminator_initializer_mean,
                            stddev=self._discriminator_initializer_deviation
                        )
                    )(model_input)
                    model_input = self._add_activation_layer(
                        model_input,
                        self._discriminator_activation_function
                    )
                else:
                    # Use tile/repeat strategy (no parameters!)
                    def expand_with_tiling(x):
                        # Calculate repeat factor needed
                        repeat_factor = (target_dim + current_dim - 1) // current_dim
                        # Tile the tensor
                        repeated = tf.tile(x, [1, repeat_factor])
                        # Crop to exact size
                        return repeated[:, :target_dim]

                    model_input = Lambda(
                        expand_with_tiling,
                        output_shape=(target_dim,)
                    )(model_input)

        # Get discriminator output
        validity = discriminator_model(model_input)

        return Model(
            inputs=[discriminator_shape_input, label_input],
            outputs=validity,
            name='Discriminator_MemorySafe'
        )