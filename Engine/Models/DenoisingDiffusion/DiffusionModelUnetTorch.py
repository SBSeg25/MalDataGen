#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

import math
import warnings

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
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from Engine.Activations.Activations import Activations
    from Engine.Layers.Pytorch.TimeEmbeddingLayer import TimeEmbedding
    from Engine.Layers.Pytorch.AttentionBlockLayer import AttentionBlock
    from Engine.Layers.Pytorch.CrossAttentionLayer import CrossAttentionBlock

except ImportError as error:
    print(error)
    sys.exit(-1)

DEFAULT_DIFFUSION_UNET_LAST_LAYER_ACTIVATION = 'linear'
DEFAULT_DIFFUSION_LATENT_DIMENSION = 64
DEFAULT_DIFFUSION_UNET_NUMBER_EMBEDDING_CHANNELS = 1
DEFAULT_DIFFUSION_UNET_CHANNELS_PER_LEVEL = [1, 2, 4]
DEFAULT_DIFFUSION_UNET_ATTENTION_MODE = [False, True, True]
DEFAULT_DIFFUSION_UNET_NUMBER_RESIDUAL_BLOCKS = 2
DEFAULT_DIFFUSION_UNET_GROUP_NORMALIZATION = 1
DEFAULT_DIFFUSION_UNET_INTERMEDIARY_ACTIVATION = 'swish'
DEFAULT_DIFFUSION_UNET_INTERMEDIARY_ACTIVATION_ALPHA = 0.05


class UNetDenoisingModelTorch(nn.Module, Activations):
    """
    UNetModel

    Implements a deep learning architecture designed for image processing tasks such
    as image segmentation or generation. The model follows the U-Net style with
    modifications, including attention blocks, time embedding, and description embeddings.
    The architecture is flexible and configurable, supporting various numbers of layers,
    attention mechanisms, residual blocks, and normalization strategies.

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
        >>> unet_model = UNetDenoisingModel(
        ...    output_shape=256,
        ...    embedding_channels=3,
        ...    list_neurons_per_level=[64, 128, 256],
        ...    list_attentions=[True, False, True],
        ...    number_residual_blocks=2,
        ...    normalization_groups=4,
        ...    intermediary_activation_function="LeakyReLU",
        ...    intermediary_activation_alpha=0.2,
        ...    last_layer_activation="sigmoid",
        ...    number_samples_per_class={"number_classes": 10}
        ... )
    """

    def __init__(self,
                 output_shape: int = 128,
                 embedding_channels: int = DEFAULT_DIFFUSION_UNET_NUMBER_EMBEDDING_CHANNELS,
                 list_neurons_per_level=None,
                 list_attentions=None,
                 number_residual_blocks: int = DEFAULT_DIFFUSION_UNET_NUMBER_RESIDUAL_BLOCKS,
                 normalization_groups: int = DEFAULT_DIFFUSION_UNET_GROUP_NORMALIZATION,
                 intermediary_activation_function: int = DEFAULT_DIFFUSION_UNET_INTERMEDIARY_ACTIVATION,
                 intermediary_activation_alpha: str = DEFAULT_DIFFUSION_UNET_INTERMEDIARY_ACTIVATION_ALPHA,
                 last_layer_activation: str = DEFAULT_DIFFUSION_UNET_LAST_LAYER_ACTIVATION,
                 number_samples_per_class=None):
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
        super(UNetDenoisingModelTorch, self).__init__()

        if list_neurons_per_level is None:
            list_neurons_per_level = DEFAULT_DIFFUSION_UNET_CHANNELS_PER_LEVEL

        if list_attentions is None:
            list_attentions = DEFAULT_DIFFUSION_UNET_ATTENTION_MODE

        if not isinstance(output_shape, int) or output_shape <= 0:
            raise ValueError("output_shape must be a positive integer.")

        if not isinstance(embedding_channels, int) or embedding_channels <= 0:
            raise ValueError("embedding_channels must be a positive integer.")

        if not isinstance(list_neurons_per_level, list) or not all(
                isinstance(n, int) and n > 0 for n in list_neurons_per_level):
            raise ValueError("list_neurons_per_level must be a list of positive integers.")

        if not isinstance(list_attentions, list) or not all(isinstance(a, bool) for a in list_attentions):
            raise ValueError("list_attentions must be a list of boolean values.")

        if not isinstance(number_residual_blocks, int) or number_residual_blocks <= 0:
            raise ValueError("number_residual_blocks must be a positive integer.")

        if not isinstance(normalization_groups, int) or normalization_groups <= 0:
            raise ValueError("normalization_groups must be a positive integer.")

        if not isinstance(intermediary_activation_function, str):
            raise ValueError("intermediary_activation_function must be a string.")

        if not isinstance(intermediary_activation_alpha, (float, int)) or intermediary_activation_alpha < 0:
            raise ValueError("intermediary_activation_alpha must be a non-negative float or integer.")

        if not isinstance(last_layer_activation, str):
            raise ValueError("last_layer_activation must be a string.")

        if not isinstance(number_samples_per_class, dict) or "number_classes" not in number_samples_per_class:
            raise ValueError("number_samples_per_class must be a dictionary containing the key 'number_classes'.")

        self._embedding_channels = embedding_channels
        self._list_neurons_per_level = list_neurons_per_level
        self._list_attention = list_attentions
        self._last_layer_activation = last_layer_activation
        self._number_residual_blocks = number_residual_blocks
        self._normalization_groups = normalization_groups
        self._intermediary_activation_function = intermediary_activation_function
        self._intermediary_activation_alpha = intermediary_activation_alpha
        self._number_samples_per_class = number_samples_per_class
        self._output_shape = self._adjust_output_shape_for_downsampling(output_shape, len(self._list_neurons_per_level))

        # Build the model architecture
        self._build_model()

    @staticmethod
    def _adjust_output_shape_for_downsampling(shape: int, number_downsamples: int) -> int:
        """
        Ensures the output shape is divisible by 2 exactly `number_downsamples` times without remainder.

        This is necessary to support successive downsampling operations in the U-Net architecture.
        If the condition is not met, the shape is automatically adjusted (padded) to the smallest
        value that satisfies this constraint. A warning is issued to inform the user.

        Args:
            shape (int): The initial spatial dimension (height or width) of the input.
            number_downsamples (int): The number of required downsampling steps (i.e., divisions by 2).

        Returns:
            int: A valid shape that can be divided by 2 `num_downsamples` times without producing a fraction.

        Raises:
            ValueError: If the input `shape` is not a positive integer.
        """
        if not isinstance(shape, int) or shape <= 0:
            raise ValueError("Input `shape` must be a positive integer.")

        original_shape = shape
        success = True

        for _ in range(number_downsamples):
            if shape % 2 != 0:
                success = False
                break
            shape = shape // 2

        if success:
            return original_shape  # No padding required

        # Compute the next closest number divisible by 2 `num_downsamples` times
        required_multiple = 2 ** number_downsamples
        padded_shape = math.ceil(original_shape / required_multiple) * required_multiple

        warnings.warn(
            f"The provided `output_shape` ({original_shape}) cannot be evenly divided by 2 "
            f"{number_downsamples} times. It has been automatically adjusted to {padded_shape} "
            f"to ensure compatibility with the network's downsampling path.",
            UserWarning
        )

        return padded_shape

    def _build_model(self):
        """
        Constructs the U-Net model architecture as PyTorch modules.
        """
        first_conv_channels = self._list_neurons_per_level[0]

        # Initial convolution
        self.first_dense = nn.Linear(self._output_shape, self._output_shape)

        # Time embedding layers
        self.time_embedding = TimeEmbedding(first_conv_channels * 4)
        self.time_mlp = self._time_MLP(first_conv_channels * 4)

        # Label embedding
        self.label_mlp = self._label_embedding_MLP(self._number_samples_per_class["number_classes"])

        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        for level_idx, num_neurons in enumerate(self._list_neurons_per_level):
            level_blocks = nn.ModuleList()

            for _ in range(self._number_residual_blocks):
                level_blocks.append(self._create_residual_block(num_neurons))

                if self._list_attention[level_idx]:
                    level_blocks.append(CrossAttentionBlock(num_neurons))

            self.down_blocks.append(level_blocks)

            if level_idx != len(self._list_neurons_per_level) - 1:
                self.down_samples.append(self._create_down_sample(num_neurons))

        # Middle blocks
        self.mid_block1 = self._create_residual_block(self._list_neurons_per_level[-1])
        self.mid_attn = CrossAttentionBlock(self._list_neurons_per_level[-1])
        self.mid_block2 = self._create_residual_block(self._list_neurons_per_level[-1])

        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for level_idx in reversed(range(len(self._list_neurons_per_level))):
            num_neurons = self._list_neurons_per_level[level_idx]
            level_blocks = nn.ModuleList()

            for _ in range(self._number_residual_blocks + 1):
                level_blocks.append(self._create_residual_block(num_neurons))

                if self._list_attention[level_idx]:
                    level_blocks.append(CrossAttentionBlock(num_neurons))

            self.up_blocks.append(level_blocks)

            if level_idx != 0:
                self.up_samples.append(self._create_up_sample(num_neurons))

        # Final output layer
        self.final_dense = nn.Linear(self._output_shape, self._output_shape)

    def _create_down_sample(self, width):
        """Creates a downsampling module."""
        return nn.ModuleDict({
            'dense': nn.Linear(self._output_shape // 2 * width, self._output_shape // 2 * width)
        })

    def _create_up_sample(self, width):
        """Creates an upsampling module."""
        return nn.ModuleDict({
            'dense': nn.Linear(self._output_shape * 2 * width, self._output_shape * 2 * width)
        })

    def _time_MLP(self, units):
        """Creates a Multi-Layer Perceptron for time embeddings."""
        return nn.Sequential(
            nn.Linear(units, units),
            nn.SiLU(),
            nn.Linear(units, units)
        )

    def _label_embedding_MLP(self, units):
        """Creates a Multi-Layer Perceptron for label embeddings."""
        return nn.Sequential(
            nn.Linear(units, units),
            nn.SiLU(),
            nn.Linear(units, self._output_shape)
        )

    def _create_residual_block(self, number_filters):
        """Creates a residual block module."""
        return ResidualBlock(number_filters, self._intermediary_activation_function)

    def forward(self, image_input, time_input, description_input):
        """
        Forward pass through the U-Net model.

        Args:
            image_input: Input images (batch_size, seq_len, channels)
            time_input: Time step indices (batch_size,)
            description_input: Class labels (batch_size, num_classes)

        Returns:
            Output tensor (batch_size, seq_len, channels)
        """
        # Initial processing
        batch_size = image_input.shape[0]
        x = image_input.view(batch_size, -1)
        x = self.first_dense(x)
        x = self._get_activation(self._intermediary_activation_function)(x)
        x = x.view(batch_size, self._output_shape, self._embedding_channels)

        # Time and label embeddings
        t_emb = self.time_embedding(time_input)
        t_emb = self.time_mlp(t_emb)
        l_emb = self.label_mlp(description_input)

        # Skip connections
        skip_connections = [x]

        # Encoder
        for level_idx, level_blocks in enumerate(self.down_blocks):
            for block in level_blocks:
                if isinstance(block, ResidualBlock):
                    x = block(x, t_emb)
                else:  # CrossAttentionBlock
                    x = block(x, l_emb)
                skip_connections.append(x)

            if level_idx < len(self.down_samples):
                x = self._down_sample_forward(x, self.down_samples[level_idx])
                skip_connections.append(x)

        # Middle
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x, l_emb)
        x = self.mid_block2(x, t_emb)

        # Decoder
        for level_idx, level_blocks in enumerate(self.up_blocks):
            for block_idx, block in enumerate(level_blocks):
                if block_idx == 0:
                    skip = skip_connections.pop()
                    x = torch.cat([x, skip], dim=-1)

                if isinstance(block, ResidualBlock):
                    x = block(x, t_emb)
                else:  # CrossAttentionBlock
                    x = block(x, l_emb)

            if level_idx < len(self.up_samples):
                x = self._up_sample_forward(x, self.up_samples[level_idx])

        # Final output
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.final_dense(x)
        x = x.view(batch_size, self._output_shape, self._embedding_channels)

        return x

    def _down_sample_forward(self, x, module):
        """Applies downsampling."""
        batch_size, seq_len, width = x.shape
        x = x.view(batch_size, -1)
        x = module['dense'](x)
        x = self._get_activation(self._intermediary_activation_function)(x)
        x = x.view(batch_size, seq_len // 2, width)
        return x

    def _up_sample_forward(self, x, module):
        """Applies upsampling."""
        batch_size, seq_len, width = x.shape
        x = x.view(batch_size, -1)
        x = module['dense'](x)
        x = self._get_activation(self._intermediary_activation_function)(x)
        x = x.view(batch_size, seq_len * 2, width)
        return x

    def _get_activation(self, name):
        """Returns the activation function."""
        activations = {
            'swish': nn.SiLU(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(self._intermediary_activation_alpha),
            'linear': nn.Identity()
        }
        return activations.get(name.lower(), nn.SiLU())

    @property
    def embedding_dimension(self):
        return self._output_shape

    @embedding_dimension.setter
    def embedding_dimension(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("embedding_dimension must be a positive integer.")
        self._output_shape = value

    @property
    def embedding_channels(self):
        return self._embedding_channels

    @embedding_channels.setter
    def embedding_channels(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("embedding_channels must be a positive integer.")
        self._embedding_channels = value

    @property
    def list_neurons_per_level(self):
        return self._list_neurons_per_level

    @list_neurons_per_level.setter
    def list_neurons_per_level(self, value):
        if not isinstance(value, list) or not all(isinstance(n, int) and n > 0 for n in value):
            raise ValueError("list_neurons_per_level must be a list of positive integers.")
        self._list_neurons_per_level = value

    @property
    def list_attention(self):
        return self._list_attention

    @list_attention.setter
    def list_attention(self, value):
        if not isinstance(value, list) or not all(isinstance(a, bool) for a in value):
            raise ValueError("list_attentions must be a list of boolean values.")
        self._list_attention = value

    @property
    def last_layer_activation(self):
        return self._last_layer_activation

    @last_layer_activation.setter
    def last_layer_activation(self, value):
        if not isinstance(value, str):
            raise ValueError("last_layer_activation must be a string.")
        self._last_layer_activation = value

    @property
    def number_residual_blocks(self):
        return self._number_residual_blocks

    @number_residual_blocks.setter
    def number_residual_blocks(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("number_residual_blocks must be a positive integer.")
        self._number_residual_blocks = value

    @property
    def normalization_groups(self):
        return self._normalization_groups

    @normalization_groups.setter
    def normalization_groups(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("normalization_groups must be a positive integer.")
        self._normalization_groups = value

    @property
    def intermediary_activation_function(self):
        return self._intermediary_activation_function

    @intermediary_activation_function.setter
    def intermediary_activation_function(self, value):
        if not isinstance(value, str):
            raise ValueError("intermediary_activation_function must be a string.")
        self._intermediary_activation_function = value

    @property
    def intermediary_activation_alpha(self):
        return self._intermediary_activation_alpha

    @intermediary_activation_alpha.setter
    def intermediary_activation_alpha(self, value):
        if not isinstance(value, (float, int)) or value < 0:
            raise ValueError("intermediary_activation_alpha must be a non-negative float or integer.")
        self._intermediary_activation_alpha = value

    @property
    def number_samples_per_class(self):
        return self._number_samples_per_class

    @number_samples_per_class.setter
    def number_samples_per_class(self, value):
        if not isinstance(value, dict) or "number_classes" not in value:
            raise ValueError("number_samples_per_class must be a dictionary containing the key 'number_classes'.")
        self._number_samples_per_class = value


class ResidualBlock(nn.Module):
    """Residual block for the UNet model."""

    def __init__(self, num_filters, activation='swish'):
        super(ResidualBlock, self).__init__()
        self.num_filters = num_filters
        self.activation = activation

    def forward(self, x, t_emb):
        input_width = x.shape[-1]
        batch_size, seq_len, _ = x.shape

        # Residual connection
        if input_width == self.num_filters:
            residual = x
        else:
            residual = nn.Linear(seq_len * input_width, seq_len * self.num_filters).to(x.device)(
                x.view(batch_size, -1)
            ).view(batch_size, seq_len, self.num_filters)

        # Time embedding
        t_emb_proj = nn.Linear(t_emb.shape[-1], self.num_filters).to(x.device)(t_emb).unsqueeze(1)

        # Main path
        x_flat = x.view(batch_size, -1)
        x = nn.Linear(x_flat.shape[-1], seq_len * self.num_filters).to(x.device)(x_flat)
        x = F.silu(x) if self.activation == 'swish' else x
        x = x.view(batch_size, seq_len, self.num_filters)

        # Add embeddings
        x = x + t_emb_proj

        # Second transformation
        x_flat = x.view(batch_size, -1)
        x = nn.Linear(x_flat.shape[-1], seq_len * self.num_filters).to(x.device)(x_flat)
        x = F.silu(x) if self.activation == 'swish' else x
        x = x.view(batch_size, seq_len, self.num_filters)

        # Residual connection
        return x + residual