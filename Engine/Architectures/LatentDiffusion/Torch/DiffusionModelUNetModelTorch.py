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

try:
    import sys
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from Engine.Layers.Torch.Activations import Activations
    from Engine.Layers.Torch.TimeEmbedding import TimeEmbedding

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


class DiffusionModelUNetModelTorch(nn.Module, Activations):
    """
    UNetModel - PyTorch Implementation with Dynamic Projection Layers

    This fixed version properly handles:
    - 3D embeddings (batch, seq_len, channels)
    - Dynamically created projection layers for variable tensor sizes
    - Proper encoder-decoder skip connection matching
    - EMA-compatible layer management
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
        """
        super(DiffusionModelUNetModelTorch, self).__init__()

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
        """Constructs the U-Net model architecture with dynamic projection layers."""
        first_conv_channels = self._list_neurons_per_level[0]

        # Initial convolution - handles flattened input
        input_features = self._embedding_dimension * self._embedding_channels
        output_features = self._embedding_dimension * first_conv_channels

        print(f"[UNet Build] Creating first_dense: input={input_features}, output={output_features}")
        print(
            f"[UNet Build] embedding_dimension={self._embedding_dimension}, embedding_channels={self._embedding_channels}")
        print(f"[UNet Build] first_conv_channels={first_conv_channels}")

        self.first_dense = nn.Linear(input_features, output_features)

        # Time embedding
        self.time_embedding = TimeEmbedding(first_conv_channels * 4)
        self.time_mlp = self._time_MLP(first_conv_channels * 4)

        # Description embedding
        self.description_mlp = self._label_embedding_MLP(self._number_samples_per_class["number_classes"])

        # ========================================================================
        # ENCODER BLOCKS
        # ========================================================================
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

        # ========================================================================
        # MIDDLE BLOCKS
        # ========================================================================
        self.mid_block1 = ResidualBlockTorch(self._list_neurons_per_level[-1], self._intermediary_activation_function)
        self.mid_attn = CrossAttentionBlock(self._list_neurons_per_level[-1])
        self.mid_block2 = ResidualBlockTorch(self._list_neurons_per_level[-1], self._intermediary_activation_function)

        # ========================================================================
        # DECODER BLOCKS
        # ========================================================================
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

        # ========================================================================
        # DECODER PROJECTION CACHE (DYNAMIC CREATION - FIX)
        # ========================================================================
        # Use a cache dictionary to store dynamically created projection layers
        # This allows EMA to work while handling variable tensor sizes
        self.decoder_projections = {}

        # ========================================================================
        # FINAL OUTPUT LAYER
        # ========================================================================
        self.final_dense = nn.Linear(
            self._embedding_dimension * first_conv_channels,
            self._embedding_dimension * self._embedding_channels
        )

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
        Forward pass through the U-Net model with fixed shape handling.

        Args:
            image_input (Tensor): Image input [batch, embedding_dimension, embedding_channels]
            time_input (Tensor): Time step input [batch]
            description_input (Tensor): Class description [batch, number_classes]

        Returns:
            Tensor: Output [batch, embedding_dimension, embedding_channels]
        """
        batch_size = image_input.shape[0]
        first_conv_channels = self._list_neurons_per_level[0]

        # ========================================================================
        # INITIAL PROCESSING
        # ========================================================================
        # Flatten and process initial input
        x = image_input.view(batch_size, -1)  # (batch, embedding_dimension * embedding_channels)
        x = self.first_dense(x)  # (batch, embedding_dimension * first_conv_channels)
        x = self._get_activation(self._intermediary_activation_function)(x)
        x = x.view(batch_size, self._embedding_dimension, first_conv_channels)

        # Time and description embeddings
        t_emb = self.time_embedding(time_input)
        t_emb = self.time_mlp(t_emb)
        d_emb = self.description_mlp(description_input)

        # Skip connections storage
        skip_connections = []

        # ========================================================================
        # ENCODER PATH
        # ========================================================================
        for level_idx, level_blocks in enumerate(self.down_blocks):
            # Process all blocks at this level
            for block in level_blocks:
                if isinstance(block, ResidualBlockTorch):
                    x = block(x, t_emb)
                else:  # CrossAttentionBlock
                    x = block(x, d_emb)

            # Save skip connection BEFORE downsampling
            skip_connections.append(x.clone())

            # Downsample for next level (except last level)
            if level_idx < len(self.down_samples):
                x = self.down_samples[level_idx](x)

        # ========================================================================
        # BOTTLENECK (Middle)
        # ========================================================================
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x, d_emb)
        x = self.mid_block2(x, t_emb)

        # ========================================================================
        # DECODER PATH WITH DYNAMIC PROJECTIONS (FIX)
        # ========================================================================
        num_levels = len(self._list_neurons_per_level)

        for level_idx in range(num_levels):
            # Upsample FIRST (except for first decoder level which is at bottleneck)
            if level_idx > 0:
                upsampler_idx = level_idx - 1
                if upsampler_idx < len(self.up_samples):
                    x = self.up_samples[upsampler_idx](x)

            # Get corresponding skip connection (pop from end = reverse order)
            skip = skip_connections.pop()

            # Ensure skip and x have matching sequence lengths
            if x.shape[1] != skip.shape[1]:
                if x.shape[1] < skip.shape[1]:
                    # Upsample x to match skip's sequence length
                    ratio = skip.shape[1] // x.shape[1]
                    x = x.repeat_interleave(ratio, dim=1)
                else:
                    # This shouldn't happen in a properly designed U-Net
                    x = x[:, :skip.shape[1], :]

            # Concatenate along channel dimension
            x_concat = torch.cat([x, skip], dim=-1)

            # Get target channels for this decoder level
            decoder_level = num_levels - 1 - level_idx
            target_channels = self._list_neurons_per_level[decoder_level]

            # Project concatenated features using dynamically created projection layer
            batch_sz, seq_len, concat_channels = x_concat.shape
            x_flat = x_concat.view(batch_sz, -1)

            # Create a unique key for this projection based on actual dimensions
            proj_key = f'decoder_proj_{seq_len}_{concat_channels}_{target_channels}'

            # Create projection layer if it doesn't exist
            if proj_key not in self.decoder_projections:
                self.decoder_projections[proj_key] = nn.Linear(
                    seq_len * concat_channels,
                    seq_len * target_channels
                ).to(x_flat.device)

            # Use the dynamically created projection
            x = self.decoder_projections[proj_key](x_flat)
            x = self._get_activation(self._intermediary_activation_function)(x)
            x = x.view(batch_sz, seq_len, target_channels)

            # Process through all residual and attention blocks at this level
            level_blocks = self.up_blocks[level_idx]
            for block in level_blocks:
                if isinstance(block, ResidualBlockTorch):
                    x = block(x, t_emb)
                else:  # CrossAttentionBlock
                    x = block(x, d_emb)

        # ========================================================================
        # FINAL OUTPUT
        # ========================================================================
        x = x.view(batch_size, -1)  # Flatten
        x = self.final_dense(x)  # Project back to original space
        x = x.view(batch_size, self._embedding_dimension, self._embedding_channels)

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

    # ========================================================================
    # PROPERTIES
    # ========================================================================
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


# ============================================================================
# HELPER CLASSES
# ============================================================================

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
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Kayuã Oleques']

# MIT License
#
# Copyright (c) 2025 Synthetic Ocean AI

try:
    import sys
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

except ImportError as error:
    print(error)
    sys.exit(-1)


class CrossAttentionBlock(nn.Module):
    """
    A custom PyTorch module that implements a Cross-Attention mechanism.

    This layer computes cross-attention between two sets of input sequences:
    queries and key-value pairs. The cross-attention mechanism computes attention
    scores between the queries and keys and uses the resulting attention weights
    to perform weighted aggregation of the values. The output of the attention is
    projected into a desired number of units. It also incorporates a residual
    connection between the input and the attention output.

    References:
        - Vaswani, A., et al. (2017). Attention is all you need.
          Advances in Neural Information Processing Systems, 30.
          URL: https://arxiv.org/abs/1706.03762
    """

    def __init__(self, units):
        """
        Initializes the CrossAttentionBlock layer.

        Args:
            units (int): The number of output units for the attention block.
        """
        super(CrossAttentionBlock, self).__init__()
        self.units = units

        # Initialize the weight matrices for query, key, value, and output projection
        # Note: PyTorch Linear layers will be created dynamically in forward pass
        # to handle variable input dimensions
        self.query_weights = None
        self.key_weights = None
        self.value_weights = None
        self.projection_weights = nn.Linear(units, units)

    def forward(self, query_inputs, key_value_inputs):
        """
        Performs the forward pass of the CrossAttentionBlock layer.

        Args:
            query_inputs (torch.Tensor): The query tensor of shape (batch_size, seq_len, num_channels).
            key_value_inputs (torch.Tensor): The key-value tensor, expected shape (batch_size, embedding_dim).

        Returns:
            torch.Tensor: The resulting tensor of shape (batch_size, seq_len, units),
                         which is the sum of the original query inputs and the attention output.
        """
        # Get the dimensions for batch size, sequence length, and number of channels
        batch_size, seq_len, num_channels = query_inputs.shape

        # FIX: Ensure key_value_inputs is 2D (batch_size, embedding_dim)
        # Squeeze out any extra dimensions that may have been added
        while key_value_inputs.dim() > 2:
            key_value_inputs = key_value_inputs.squeeze(1)

        # Now key_value_inputs should be (batch_size, embedding_dim)
        embedding_dim = key_value_inputs.shape[-1]

        # Expand key_value_inputs to match query shape: (batch_size, seq_len, embedding_dim)
        # We broadcast across the sequence length dimension
        key_value_inputs = key_value_inputs.unsqueeze(1).expand(batch_size, seq_len, embedding_dim)

        # Initialize linear layers if not already created
        # Use embedding_dim instead of num_channels for key/value projections
        if self.query_weights is None:
            self.query_weights = nn.Linear(num_channels, self.units).to(query_inputs.device)
            self.key_weights = nn.Linear(embedding_dim, self.units).to(query_inputs.device)
            self.value_weights = nn.Linear(embedding_dim, self.units).to(query_inputs.device)

        # Calculate scaling factor for attention scores
        scale = self.units ** -0.5

        # Apply linear projections to queries, keys, and values
        query = self.query_weights(query_inputs)  # (batch_size, seq_len, units)
        key = self.key_weights(key_value_inputs)  # (batch_size, seq_len, units)
        value = self.value_weights(key_value_inputs)  # (batch_size, seq_len, units)

        # Compute attention scores
        # query: (batch_size, seq_len, units)
        # key.transpose(-2, -1): (batch_size, units, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale  # (batch_size, seq_len, seq_len)

        # Apply softmax to obtain attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, seq_len, seq_len)

        # Compute attention output by applying attention weights to values
        attention_output = torch.matmul(attention_weights, value)  # (batch_size, seq_len, units)

        # Project the attention output to the desired dimensionality
        attention_output = self.projection_weights(attention_output)  # (batch_size, seq_len, units)

        # Apply residual connection by adding the input query values to the attention output
        # Note: If query_inputs has different dimensions than attention_output, we need to project it
        if query_inputs.shape[-1] != self.units:
            residual_proj = nn.Linear(query_inputs.shape[-1], self.units).to(query_inputs.device)
            return residual_proj(query_inputs) + attention_output

        return query_inputs + attention_output