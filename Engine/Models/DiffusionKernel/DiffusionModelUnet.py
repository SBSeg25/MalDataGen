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
    import tensorflow

    from tensorflow.keras.layers import Add
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Layer

    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Reshape

    from tensorflow.keras.layers import Concatenate

    from Engine.Activations.Activations import Activations
    from Engine.Layers.Tensorflow.ClusteringLayer import ClusteringLayer

    from Engine.Layers.Tensorflow.TimeEmbeddingLayer import TimeEmbedding
    from Engine.Layers.Tensorflow.AttentionBlockLayer import AttentionBlock

except ImportError as error:
    print(error)
    sys.exit(-1)

class SqueezeLayer(Layer):
    def __init__(self, axis=1, **kwargs):
        super(SqueezeLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tensorflow.squeeze(inputs, axis=self.axis)



class UNetModelKernel(Activations):
    """
    UNetModel

    Implements a deep learning architecture designed for image processing tasks such as image segmentation or generation.
    The model follows the U-Net style with modifications, including attention blocks, time embedding, and description embeddings.
    The architecture is flexible and configurable, supporting various numbers of layers, attention mechanisms, residual
    blocks, and normalization strategies.

    Attributes:
        @embedding_dimension (int):
            The size of the input image dimensions.
        @embedding_channels (int):
            The number of channels in the input image.
        @list_neurons_per_level (List[int]):
            The number of neurons or filters at each level of the network.
        @list_attentions (List[bool]):
            Indicates whether attention mechanisms should be applied at each level of the network.
        @number_residual_blocks (int):
            The number of residual blocks to apply at each level of the network.
        @normalization_groups (int):
            The number of groups used for normalization in residual blocks.
        @intermediary_activation_function (str):
            The activation function to use for intermediate layers (e.g., 'ReLU', 'LeakyReLU').
        @intermediary_activation_alpha (float):
            The alpha parameter for activation functions like LeakyReLU.
        @last_layer_activation (str):
            The activation function to use for the final output layer.
        @number_samples_per_class (Dict[str, int]):
            Contains metadata about the dataset, including the "number_classes" key to specify the number of classes.

    Raises:
        ValueError:
            Raised if invalid arguments are passed during initialization, such as:
            - Non-positive `embedding_dimension` or `embedding_channels`
            - Mismatched length of `list_neurons_per_level` and `list_attentions`
            - Non-positive `number_residual_blocks` or invalid `normalization_groups`
            - Invalid activation function names or unrecognized `last_layer_activation`
            - Missing or incorrect `number_classes` in `number_samples_per_class`

    Example:
        >>> unet_model = UNetModel(
        ...     embedding_dimension=256,
        ...     embedding_channels=3,
        ...     list_neurons_per_level=[64, 128, 256],
        ...     list_attentions=[True, False, True],
        ...     number_residual_blocks=2,
        ...     normalization_groups=4,
        ...     intermediary_activation_function="LeakyReLU",
        ...     intermediary_activation_alpha=0.2,
        ...     last_layer_activation="sigmoid",
        ...     number_samples_per_class={"number_classes": 10}
        ... )
    """

    def __init__(self,
                 embedding_dimension,
                 embedding_channels,
                 list_neurons_per_level,
                 list_attentions,
                 number_residual_blocks,
                 normalization_groups,
                 intermediary_activation_function,
                 intermediary_activation_alpha,
                 last_layer_activation, number_samples_per_class):
        """
        Initializes the UNetModel class with the provided parameters.

        This constructor sets up all internal attributes related to the U-Net architecture, including
        input image dimensions, network depth, attention mechanisms, and activation functions for all layers.

        Args:
            @embedding_dimension (int):
                The dimension of the input image.
            @embedding_channels (int):
                The number of channels in the input image (e.g., 3 for RGB images).
            @list_neurons_per_level (List[int]):
                A list specifying the number of neurons/filters at each level of the network.
            @list_attentions (List[bool]):
                A list indicating where attention blocks should be applied (True or False for each level).
            @number_residual_blocks (int):
                The number of residual blocks to apply at each level.
            @normalization_groups (int):
                The number of groups for normalization in residual blocks.
            @intermediary_activation_function (str):
                The activation function for intermediate layers (e.g., 'ReLU', 'LeakyReLU').
            @intermediary_activation_alpha (float):
                The alpha parameter for activation functions such as LeakyReLU.
            @last_layer_activation (str):
                The activation function for the last layer of the model.
            @number_samples_per_class (Dict[str, int]):
                A dictionary containing metadata about the dataset, including the key "number_classes" to define the number of classes.

        Raises:
            ValueError:
                If `embedding_dimension` or `embedding_channels` is non-positive.
                If the length of `list_neurons_per_level` does not match `list_attentions`.
                If `number_residual_blocks` or `normalization_groups` is non-positive.
                If the `intermediary_activation_function` or `last_layer_activation` is invalid.
                If `number_samples_per_class` is missing the key "number_classes".
        """

        self._embedding_dimension = embedding_dimension
        self._embedding_channels = embedding_channels
        self._list_neurons_per_level = list_neurons_per_level
        self._list_attention = list_attentions
        self._last_layer_activation = last_layer_activation
        self._number_residual_blocks = number_residual_blocks
        self._normalization_groups = normalization_groups
        self._intermediary_activation_function = intermediary_activation_function
        self._intermediary_activation_alpha = intermediary_activation_alpha
        self._number_samples_per_class = number_samples_per_class

    def _down_sample(self, width):
        """
        Downsamples the input by reducing its dimensionality.

        Args:
            width (int): The target width for the downsampling.

        Returns:
            Function: A function that applies the downsampling operation to a given input tensor.
        """
        def apply(down_sample_flow):
            original_shape = down_sample_flow.shape
            down_sample_flow = Flatten()(down_sample_flow)
            down_sample_flow = Dense(original_shape[1] // 2 * width)(down_sample_flow)
            self._add_activation_layer(down_sample_flow, self._intermediary_activation_function)

            down_sample_flow = Reshape((original_shape[1] // 2, width))(down_sample_flow)

            return down_sample_flow

        return apply

    def _up_sample(self, width):
        """
        Upsamples the input by increasing its dimensionality.

        Args:
            width (int): The target width for the upsampling.

        Returns:
            Function: A function that applies the upsampling operation to a given input tensor.
        """
        def apply(up_sample_flow):
            original_shape = up_sample_flow.shape
            up_sample_flow = Flatten()(up_sample_flow)
            up_sample_flow = Dense(original_shape[1] * 2 * width)(up_sample_flow)
            up_sample_flow = self._add_activation_layer(up_sample_flow,
                                                                    self._intermediary_activation_function)
            up_sample_flow = Reshape((original_shape[1] * 2, width))(up_sample_flow)

            return up_sample_flow

        return apply

    def _time_MLP(self, units):
        """
        Creates a Multi-Layer Perceptron (MLP) to process time embeddings.

        Args:
            units (int): The number of units for the dense layers in the MLP.

        Returns:
            Function: A function that applies the MLP transformation to a given input.
        """
        def apply(inputs):
            time_embedding = Dense(units)(Dense(units)(inputs))

            time_embedding = self._add_activation_layer(time_embedding,
                                                                    self._intermediary_activation_function)

            return time_embedding


        return apply

    def _residual_block(self, number_filters, groups=1):
        """
        Builds a residual block for the network, which includes convolution, normalization,
        and embedding layers for time and description inputs. The block follows the
        residual learning framework by applying a skip connection that adds the
        original input to the transformed output, facilitating gradient flow and
        improving training stability.

        The residual block performs the following operations:
        - If the input width matches the number of filters, the residual block directly passes the input.
        - Otherwise, it reshapes the input, applies a dense transformation, and adds activation.
        - Time and description embeddings are applied to match the number of filters and added to the output.
        - A final skip connection is added to the transformed output, which improves learning by retaining
        original input features.

        Args:
            number_filters (int): The number of filters used in the convolutional layers of the residual block.
            groups (int, optional): The number of normalization groups (default is 1). This could be used for group
            normalization in more advanced versions.

        Returns:
            Function: A function that applies the residual block transformation to a given input.
                    The function accepts a list of inputs and returns the transformed tensor with the residual connection applied.
        """

        def apply(inputs):
            # Extract the inputs
            residual_block_flow, time_embedding, description_embedding, clustering_kernels = inputs
            input_width = residual_block_flow.shape[-1]

            # If input width matches the number of filters, use the original input as the residual
            if input_width == number_filters:
                residual = residual_block_flow
            else:
                # Reshape the input if the widths don't match and apply a dense transformation
                reshaped_input = Reshape((-1,))(residual_block_flow)
                transformed = Dense(number_filters * residual_block_flow.shape[1])(reshaped_input)
                transformed = self._add_activation_layer(transformed, self._intermediary_activation_function)
                residual = Reshape((residual_block_flow.shape[1], number_filters))(transformed)

            # Apply the time embedding transformation
            time_embedding = Dense(number_filters)(time_embedding)[:, None, :]
            time_embedding = self._add_activation_layer(time_embedding, self._intermediary_activation_function)

            # Apply the description embedding transformation
            description_embedding = Dense(number_filters)(description_embedding)[:, None, :]
            description_embedding = self._add_activation_layer(description_embedding,
                                                               self._intermediary_activation_function)
            # Apply the description embedding transformation
            clustering_kernels_embedding = Dense(number_filters)(clustering_kernels)[:, None, :]
            clustering_kernels_embedding = SqueezeLayer(axis=1)(clustering_kernels_embedding)
            clustering_kernels_embedding = Flatten()(clustering_kernels_embedding)
            number_neurons = residual_block_flow.shape[1]
            clustering_kernels_embedding = Dense(number_neurons * number_filters)(clustering_kernels_embedding)

            clustering_kernels_embedding = Reshape((number_neurons, number_filters))(clustering_kernels_embedding)

            clustering_kernels_embedding = self._add_activation_layer(clustering_kernels_embedding,
                                                               self._intermediary_activation_function)

            # Flatten and apply a dense transformation to the residual block flow
            number_neurons = residual_block_flow.shape[1]
            residual_block_flow = Flatten()(residual_block_flow)
            residual_block_flow = Dense(number_neurons * number_filters)(residual_block_flow)
            residual_block_flow = self._add_activation_layer(residual_block_flow,
                                                             self._intermediary_activation_function)

            # Reshape the transformed flow and add the embeddings
            residual_block_flow = Reshape((number_neurons, number_filters))(residual_block_flow)
            residual_block_flow = Add()([residual_block_flow,
                                         time_embedding,
                                         description_embedding,
                                         clustering_kernels_embedding])

            # Flatten the residual block flow and apply a final dense layer
            original_shape = residual_block_flow.shape
            residual_block_flow = Flatten()(residual_block_flow)
            residual_block_flow = Dense(original_shape[1] * original_shape[2])(residual_block_flow)
            residual_block_flow = self._add_activation_layer(residual_block_flow,
                                                             self._intermediary_activation_function)

            # Reshape back to the original residual block shape
            residual_block_flow = Reshape((original_shape[1], original_shape[2]))(residual_block_flow)

            # Add the original residual input to the transformed output
            residual_block_flow = Add()([residual_block_flow, residual])

            return residual_block_flow

        return apply

    def build_model(self):
        """
        Constructs the U-Net model, integrating all components like downsampling,
        upsampling, residual blocks, and attention mechanisms. The model is designed
        to process inputs such as images, time embeddings, and description embeddings
        to produce an output that is reshaped back into an image-like structure.

        The model architecture consists of:
        - Initial convolution and dense layers to process the image input.
        - Time embedding and description embedding layers to process time and description inputs.
        - Residual blocks with optional attention mechanisms at each level.
        - Skip connections to preserve information at each level of the network.
        - Downsampling and upsampling blocks for maintaining spatial resolution.
        - Final reshaping and dense layers to output a processed image with specified dimensions.

        The final output is a model ready for training with image, time, and description inputs.

        Returns:
            Model: A compiled U-Net model configured with the provided architecture.
        """
        # Define the input layers
        image_input = Input(shape=(self._embedding_dimension, self._embedding_channels), name="image_input")
        time_input = Input(shape=(), dtype=tensorflow.int32, name="time_input")
        description_input = Input(shape=(self._number_samples_per_class["number_classes"],), dtype=tensorflow.float32,
                                  name="description_input")

        # Initial convolutional processing
        first_conv_channels = self._list_neurons_per_level[0]
        clustering_kernels = ClusteringLayer(5,
                                             self._embedding_dimension,
                                             1000,
                                             alpha=0.5)([image_input, time_input])

        network_flow = Flatten()(image_input)
        network_flow = Dense(self._embedding_dimension)(network_flow)

        # Apply intermediary activation function
        network_flow = self._add_activation_layer(network_flow, self._intermediary_activation_function)

        # Reshape the network flow to match the embedding dimensions
        network_flow = Reshape((self._embedding_dimension, self._embedding_channels))(network_flow)

        # Time and description embeddings
        time_embedding = TimeEmbedding(first_conv_channels * 4)(time_input)
        time_embedding = self._time_MLP(first_conv_channels * 4)(time_embedding)
        description_embedding = self._time_MLP(first_conv_channels * 4)(description_input)

        # Initialize skip connections
        skip_connection_flow = [network_flow]

        # U-Net architecture loop: downsampling and residual blocks with attention
        for number_neurons in range(len(self._list_neurons_per_level)):

            # Add residual blocks
            for _ in range(self._number_residual_blocks):
                network_flow = self._residual_block(self._list_neurons_per_level[number_neurons])(
                    [network_flow, time_embedding, description_embedding, clustering_kernels])

                # Optionally apply attention mechanism
                if self._list_attention[number_neurons]:
                    network_flow = AttentionBlock(self._list_neurons_per_level[number_neurons])(network_flow)

                # Append to skip connections
                skip_connection_flow.append(network_flow)

            # Downsample if not at the last level
            if self._list_neurons_per_level[number_neurons] != self._list_neurons_per_level[-1]:
                network_flow = self._down_sample(self._list_neurons_per_level[number_neurons])(network_flow)
                skip_connection_flow.append(network_flow)

        # Final residual block and attention mechanism at the last level
        network_flow = self._residual_block(self._list_neurons_per_level[-1])(
            [network_flow, time_embedding, description_embedding, clustering_kernels])
        network_flow = AttentionBlock(self._list_neurons_per_level[-1])(network_flow)
        network_flow = self._residual_block(self._list_neurons_per_level[-1])(
            [network_flow, time_embedding, description_embedding, clustering_kernels])

        # U-Net architecture loop: upsampling and residual blocks with attention
        for number_neurons in reversed(range(len(self._list_neurons_per_level))):

            for _ in range(self._number_residual_blocks + 1):
                # Concatenate with skip connections
                network_flow = Concatenate(axis=-1)([network_flow, skip_connection_flow.pop()])
                network_flow = self._residual_block(self._list_neurons_per_level[number_neurons],
                                                    self._normalization_groups)(
                    [network_flow, time_embedding, description_embedding, clustering_kernels])

                # Apply attention mechanism if specified
                if self._list_attention[number_neurons]:
                    network_flow = AttentionBlock(self._list_neurons_per_level[number_neurons],
                                                  self._normalization_groups)(network_flow)

            # Upsample if not at the first level
            if number_neurons != 0:
                network_flow = self._up_sample(self._list_neurons_per_level[number_neurons])(network_flow)

        # Final output processing: flatten, dense, and reshape
        network_flow = Flatten()(network_flow)
        network_flow = Dense(self._embedding_dimension)(network_flow)
        # network_flow = self._add_activation_layer(network_flow, self._last_layer_activation)
        network_flow = Reshape((self._embedding_dimension, self._embedding_channels))(network_flow)

        # Create the model instance
        unet_model_instance = Model([image_input, time_input, description_input], network_flow, name="UnetModel")

        return unet_model_instance
