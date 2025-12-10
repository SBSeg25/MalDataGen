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
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from Engine.Layers.Torch.Activations import Activations
    from Engine.Layers.Torch.TimeEmbedding import TimeEmbedding
    from Engine.Layers.Torch.CrossAttentionBlock import CrossAttentionBlock

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


class UNetModelTorch(nn.Module, Activations):
    """
    UNetModel - PyTorch Implementation

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

    Example:
        >>> unet_model = UNetModelTorch(
        ...     embedding_dimension=256,
        ...     embedding_channels=3,
        ...     list_neurons_per_level=[64, 128, 256],
        ...     list_attentions=[True, False, True],
        ...     number_residual_blocks=2,
        ...     normalization_groups=4,
        ...     intermediary_activation_function="leaky_relu",
        ...     intermediary_activation_alpha=0.2,
        ...     last_layer_activation="sigmoid",
        ...     number_samples_per_class={"number_classes": 10}
        ... )
    """

    def __init__(self,
                 embedding_dimension: int = DEFAULT_DIFFUSION_LATENT_DIMENSION,
                 embedding_channels: int = DEFAULT_DIFFUSION_UNET_NUMBER_EMBEDDING_CHANNELS,
                 list_neurons_per_level=None,
                 list_attentions=None,
                 number_residual_blocks: int = DEFAULT_DIFFUSION_UNET_NUMBER_RESIDUAL_BLOCKS,
                 normalization_groups: int = DEFAULT_DIFFUSION_UNET_GROUP_NORMALIZATION,
                 intermediary_activation_function: str = DEFAULT_DIFFUSION_UNET_INTERMEDIARY_ACTIVATION,
                 intermediary_activation_alpha: float = DEFAULT_DIFFUSION_UNET_INTERMEDIARY_ACTIVATION_ALPHA,
                 last_layer_activation: str = DEFAULT_DIFFUSION_UNET_LAST_LAYER_ACTIVATION,
                 number_samples_per_class=None):
        """
        Initializes the UNetModel class with the provided parameters.

        Args:
            @embedding_dimension (int): The dimension of the input image.
            @embedding_channels (int): The number of channels in the input image.
            @list_neurons_per_level (List[int]): Number of neurons/filters at each level.
            @list_attentions (List[bool]): Attention blocks at each level.
            @number_residual_blocks (int): Number of residual blocks per level.
            @normalization_groups (int): Groups for normalization.
            @intermediary_activation_function (str): Activation for intermediate layers.
            @intermediary_activation_alpha (float): Alpha parameter for activation.
            @last_layer_activation (str): Activation for the last layer.
            @number_samples_per_class (Dict[str, int]): Dataset metadata with "number_classes".
        """
        super(UNetModelTorch, self).__init__()

        if list_neurons_per_level is None:
            list_neurons_per_level = DEFAULT_DIFFUSION_UNET_CHANNELS_PER_LEVEL

        if list_attentions is None:
            list_attentions = DEFAULT_DIFFUSION_UNET_ATTENTION_MODE

        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be a positive integer.")

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

        # Build model
        self._build_model()

    def _build_model(self):
        """Constructs the U-Net model architecture."""
        first_conv_channels = self._list_neurons_per_level[0]

        # Initial convolution
        self.first_dense = nn.Linear(self._embedding_dimension, self._embedding_dimension)

        # Time embedding
        self.time_embedding = TimeEmbedding(first_conv_channels * 4)
        self.time_mlp = self._time_MLP(first_conv_channels * 4)

        # Description embedding
        self.description_mlp = self._label_embedding_MLP(self._number_samples_per_class["number_classes"])

        # Encoder blocks
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        for level_idx, num_neurons in enumerate(self._list_neurons_per_level):
            level_blocks = nn.ModuleList()

            for _ in range(self._number_residual_blocks):
                level_blocks.append(ResidualBlockTorch(num_neurons, self._intermediary_activation_function))

                if self._list_attention[level_idx]:
                    level_blocks.append(CrossAttentionBlock(num_neurons))

            self.down_blocks.append(level_blocks)

            if level_idx != len(self._list_neurons_per_level) - 1:
                self.down_samples.append(DownSampleBlock(num_neurons))

        # Middle blocks
        self.mid_block1 = ResidualBlockTorch(self._list_neurons_per_level[-1], self._intermediary_activation_function)
        self.mid_attn = CrossAttentionBlock(self._list_neurons_per_level[-1])
        self.mid_block2 = ResidualBlockTorch(self._list_neurons_per_level[-1], self._intermediary_activation_function)

        # Decoder blocks
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for level_idx in reversed(range(len(self._list_neurons_per_level))):
            num_neurons = self._list_neurons_per_level[level_idx]

            level_blocks = nn.ModuleList()

            for _ in range(self._number_residual_blocks + 1):
                level_blocks.append(ResidualBlockTorch(num_neurons, self._intermediary_activation_function))

                if self._list_attention[level_idx]:
                    level_blocks.append(CrossAttentionBlock(num_neurons))

            self.up_blocks.append(level_blocks)

            if level_idx != 0:
                self.up_samples.append(UpSampleBlock(num_neurons))

        # Final output
        self.final_dense = nn.Linear(self._embedding_dimension, self._embedding_dimension)

    def _time_MLP(self, units):
        """Creates time embedding MLP."""
        return nn.Sequential(
            nn.Linear(units, units),
            nn.SiLU(),
            nn.Linear(units, units)
        )

    def _label_embedding_MLP(self, units):
        """Creates label embedding MLP."""
        return nn.Sequential(
            nn.Linear(units, units),
            nn.SiLU(),
            nn.Linear(units, self._embedding_dimension)
        )

    def forward(self, image_input, time_input, description_input):
        """
        Forward pass through the U-Net model.

        Args:
            image_input (Tensor): Image input [batch, embedding_dimension, embedding_channels]
            time_input (Tensor): Time step input [batch]
            description_input (Tensor): Class description [batch, number_classes]

        Returns:
            Tensor: Output [batch, embedding_dimension, embedding_channels]
        """
        # Initial processing
        x = image_input.view(image_input.shape[0], -1)
        x = self.first_dense(x)
        x = self._get_activation(self._intermediary_activation_function)(x)
        x = x.view(image_input.shape[0], self._embedding_dimension, self._embedding_channels)

        # Time and description embeddings
        t_emb = self.time_embedding(time_input)
        t_emb = self.time_mlp(t_emb)
        d_emb = self.description_mlp(description_input)

        # Skip connections
        skip_connections = [x]

        # Encoder
        for level_idx, level_blocks in enumerate(self.down_blocks):
            for block in level_blocks:
                if isinstance(block, ResidualBlockTorch):
                    x = block(x, t_emb)
                else:  # CrossAttentionBlock
                    x = block(x, d_emb)

            skip_connections.append(x)

            if level_idx < len(self.down_samples):
                x = self.down_samples[level_idx](x)
                skip_connections.append(x)

        # Middle
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x, d_emb)
        x = self.mid_block2(x, t_emb)

        # Decoder
        for level_idx, level_blocks in enumerate(self.up_blocks):
            for _ in range(self._number_residual_blocks + 1):
                skip = skip_connections.pop()
                x = torch.cat([x, skip], dim=-1)

                # Project concatenated features
                batch_size, seq_len, channels = x.shape
                x = x.view(batch_size, -1)
                proj_layer = nn.Linear(seq_len * channels, seq_len * self._list_neurons_per_level[
                    len(self._list_neurons_per_level) - 1 - level_idx]).to(x.device)
                x = proj_layer(x)
                x = self._get_activation(self._intermediary_activation_function)(x)
                x = x.view(batch_size, seq_len,
                           self._list_neurons_per_level[len(self._list_neurons_per_level) - 1 - level_idx])

                for block in level_blocks:
                    if isinstance(block, ResidualBlockTorch):
                        x = block(x, t_emb)
                    else:  # CrossAttentionBlock
                        x = block(x, d_emb)

            if level_idx < len(self.up_samples):
                x = self.up_samples[level_idx](x)

        # Final output
        x = x.view(x.shape[0], -1)
        x = self.final_dense(x)
        x = x.view(x.shape[0], self._embedding_dimension, self._embedding_channels)

        return x

    def _get_activation(self, name):
        """Get activation function by name."""
        activations = {
            'swish': nn.SiLU(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(self._intermediary_activation_alpha),
            'elu': nn.ELU(self._intermediary_activation_alpha),
            'linear': nn.Identity(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
        }
        return activations.get(name.lower(), nn.SiLU())

    def build_model(self):
        """Returns the model (for compatibility with original interface)."""
        return self

    # Properties
    @property
    def embedding_dimension(self):
        return self._embedding_dimension

    @embedding_dimension.setter
    def embedding_dimension(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("embedding_dimension must be a positive integer.")
        self._embedding_dimension = value

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


class ResidualBlockTorch(nn.Module):
    """Residual block for U-Net."""

    def __init__(self, num_filters, activation='swish'):
        super(ResidualBlockTorch, self).__init__()
        self.num_filters = num_filters
        self.activation = activation
        self._layers_cache = {}

    def forward(self, x, t_emb):
        """Forward pass through residual block."""
        input_width = x.shape[-1]
        batch_size, seq_len, _ = x.shape

        # Residual connection
        if input_width == self.num_filters:
            residual = x
        else:
            key = f'residual_{seq_len}_{input_width}'
            if key not in self._layers_cache:
                self._layers_cache[key] = nn.Linear(seq_len * input_width, seq_len * self.num_filters).to(x.device)
            layer = self._layers_cache[key]
            residual = layer(x.view(batch_size, -1)).view(batch_size, seq_len, self.num_filters)

        # Time embedding
        key_time = f'time_{t_emb.shape[-1]}'
        if key_time not in self._layers_cache:
            self._layers_cache[key_time] = nn.Linear(t_emb.shape[-1], self.num_filters).to(x.device)
        t_layer = self._layers_cache[key_time]
        t_emb_proj = t_layer(t_emb).unsqueeze(1)

        # Main transformation
        key_main = f'main_{seq_len}_{input_width}'
        if key_main not in self._layers_cache:
            self._layers_cache[key_main] = nn.Linear(seq_len * input_width, seq_len * self.num_filters).to(x.device)
        main_layer = self._layers_cache[key_main]
        x = main_layer(x.view(batch_size, -1))
        x = self._get_activation(x)
        x = x.view(batch_size, seq_len, self.num_filters)

        # Add embeddings
        x = x + t_emb_proj

        # Second transformation
        key_main2 = f'main2_{seq_len}'
        if key_main2 not in self._layers_cache:
            self._layers_cache[key_main2] = nn.Linear(seq_len * self.num_filters, seq_len * self.num_filters).to(
                x.device)
        main2_layer = self._layers_cache[key_main2]
        x = main2_layer(x.view(batch_size, -1))
        x = self._get_activation(x)
        x = x.view(batch_size, seq_len, self.num_filters)

        return x + residual

    def _get_activation(self, x):
        """Apply activation function."""
        if self.activation.lower() == 'swish':
            return F.silu(x)
        elif self.activation.lower() == 'relu':
            return F.relu(x)
        elif self.activation.lower() == 'leaky_relu':
            return F.leaky_relu(x, 0.2)
        else:
            return x


class DownSampleBlock(nn.Module):
    """Downsampling block."""

    def __init__(self, width):
        super(DownSampleBlock, self).__init__()
        self.width = width

    def forward(self, x):
        batch_size, seq_len, width = x.shape
        x = x.view(batch_size, -1)
        dense = nn.Linear(seq_len * width, (seq_len // 2) * width).to(x.device)
        x = dense(x)
        x = F.silu(x)
        x = x.view(batch_size, seq_len // 2, width)
        return x


class UpSampleBlock(nn.Module):
    """Upsampling block."""

    def __init__(self, width):
        super(UpSampleBlock, self).__init__()
        self.width = width

    def forward(self, x):
        batch_size, seq_len, width = x.shape
        x = x.view(batch_size, -1)
        dense = nn.Linear(seq_len * width, (seq_len * 2) * width).to(x.device)
        x = dense(x)
        x = F.silu(x)
        x = x.view(batch_size, seq_len * 2, width)
        return x