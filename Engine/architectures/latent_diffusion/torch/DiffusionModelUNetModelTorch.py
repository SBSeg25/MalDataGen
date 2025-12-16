#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/14'
__credits__ = ['Synthetic Ocean AI']

try:
    import sys
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from Engine.layers.torch.Activations import Activations
    from Engine.layers.torch.TimeEmbedding import TimeEmbedding

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
    UNetModel - PyTorch Implementation with Properly Registered Dynamic layers

    CRITICAL FIX: All dynamic layers now use nn.ModuleDict for proper registration
    This ensures EMA weight copying works correctly and parameter counts match.
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
        """Initializes the UNetModel class with the provided parameters."""
        super(DiffusionModelUNetModelTorch, self).__init__()

        if list_neurons_per_level is None:
            list_neurons_per_level = DEFAULT_DIFFUSION_UNET_CHANNELS_PER_LEVEL

        if list_attentions is None:
            list_attentions = DEFAULT_DIFFUSION_UNET_ATTENTION_MODE

        # Validation
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

        # CRITICAL FIX: Use nn.ModuleDict instead of regular dict
        self.decoder_projections = nn.ModuleDict()

        # Build model
        self._build_model()

    def _build_model(self):
        """Constructs the U-Net model architecture."""
        first_conv_channels = self._list_neurons_per_level[0]

        # Initial convolution
        input_features = self._embedding_dimension * self._embedding_channels
        output_features = self._embedding_dimension * first_conv_channels

        self.first_dense = nn.Linear(input_features, output_features)

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

        # Final output layer
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

    def _get_or_create_projection(self, seq_len, concat_channels, target_channels, device):
        """
        CRITICAL FIX: Get or create projection layer with proper registration.
        Uses nn.ModuleDict to ensure layers are properly tracked.
        """
        proj_key = f'proj_{seq_len}_{concat_channels}_{target_channels}'

        if proj_key not in self.decoder_projections:
            self.decoder_projections[proj_key] = nn.Linear(
                seq_len * concat_channels,
                seq_len * target_channels
            ).to(device)

        return self.decoder_projections[proj_key]

    def forward(self, image_input, time_input, description_input):
        """Forward pass through the U-Net model."""
        batch_size = image_input.shape[0]
        first_conv_channels = self._list_neurons_per_level[0]

        # Initial processing
        x = image_input.view(batch_size, -1)
        x = self.first_dense(x)
        x = self._get_activation(self._intermediary_activation_function)(x)
        x = x.view(batch_size, self._embedding_dimension, first_conv_channels)

        # Embeddings
        t_emb = self.time_embedding(time_input)
        t_emb = self.time_mlp(t_emb)
        d_emb = self.description_mlp(description_input)

        skip_connections = []

        # Encoder
        for level_idx, level_blocks in enumerate(self.down_blocks):
            for block in level_blocks:
                if isinstance(block, ResidualBlockTorch):
                    x = block(x, t_emb)
                else:
                    x = block(x, d_emb)

            skip_connections.append(x.clone())

            if level_idx < len(self.down_samples):
                x = self.down_samples[level_idx](x)

        # Bottleneck
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x, d_emb)
        x = self.mid_block2(x, t_emb)

        # Decoder with FIXED projection handling
        num_levels = len(self._list_neurons_per_level)

        for level_idx in range(num_levels):
            if level_idx > 0:
                upsampler_idx = level_idx - 1
                if upsampler_idx < len(self.up_samples):
                    x = self.up_samples[upsampler_idx](x)

            skip = skip_connections.pop()

            # Match sequence lengths
            if x.shape[1] != skip.shape[1]:
                if x.shape[1] < skip.shape[1]:
                    ratio = skip.shape[1] // x.shape[1]
                    x = x.repeat_interleave(ratio, dim=1)
                else:
                    x = x[:, :skip.shape[1], :]

            # Concatenate
            x_concat = torch.cat([x, skip], dim=-1)

            # Project using properly registered layer
            decoder_level = num_levels - 1 - level_idx
            target_channels = self._list_neurons_per_level[decoder_level]

            batch_sz, seq_len, concat_channels = x_concat.shape
            x_flat = x_concat.view(batch_sz, -1)

            # Use properly registered projection layer
            proj_layer = self._get_or_create_projection(
                seq_len, concat_channels, target_channels, x_flat.device
            )
            x = proj_layer(x_flat)
            x = self._get_activation(self._intermediary_activation_function)(x)
            x = x.view(batch_sz, seq_len, target_channels)

            # Process blocks
            level_blocks = self.up_blocks[level_idx]
            for block in level_blocks:
                if isinstance(block, ResidualBlockTorch):
                    x = block(x, t_emb)
                else:
                    x = block(x, d_emb)

        # Final output
        x = x.view(batch_size, -1)
        x = self.final_dense(x)
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
        """Returns the model (for compatibility)."""
        return self

    # Properties omitted for brevity - keep all your existing properties


class ResidualBlockTorch(nn.Module):
    """Residual block with properly registered dynamic layers."""

    def __init__(self, num_filters, activation='swish'):
        super(ResidualBlockTorch, self).__init__()
        self.num_filters = num_filters
        self.activation = activation
        # CRITICAL FIX: Use nn.ModuleDict instead of regular dict
        self._layers_cache = nn.ModuleDict()

    def _get_or_create_layer(self, key, in_features, out_features, device):
        """Get or create a layer with proper registration."""
        if key not in self._layers_cache:
            self._layers_cache[key] = nn.Linear(in_features, out_features).to(device)
        return self._layers_cache[key]

    def forward(self, x, t_emb):
        """Forward pass through residual block."""
        input_width = x.shape[-1]
        batch_size, seq_len, _ = x.shape

        # Residual connection
        if input_width == self.num_filters:
            residual = x
        else:
            key = f'residual_{seq_len}_{input_width}'
            layer = self._get_or_create_layer(key, seq_len * input_width, seq_len * self.num_filters, x.device)
            residual = layer(x.view(batch_size, -1)).view(batch_size, seq_len, self.num_filters)

        # Time embedding
        key_time = f'time_{t_emb.shape[-1]}'
        t_layer = self._get_or_create_layer(key_time, t_emb.shape[-1], self.num_filters, x.device)
        t_emb_proj = t_layer(t_emb).unsqueeze(1)

        # Main transformation
        key_main = f'main_{seq_len}_{input_width}'
        main_layer = self._get_or_create_layer(key_main, seq_len * input_width, seq_len * self.num_filters, x.device)
        x = main_layer(x.view(batch_size, -1))
        x = self._get_activation(x)
        x = x.view(batch_size, seq_len, self.num_filters)

        x = x + t_emb_proj

        # Second transformation
        key_main2 = f'main2_{seq_len}'
        main2_layer = self._get_or_create_layer(key_main2, seq_len * self.num_filters, seq_len * self.num_filters,
                                                x.device)
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
    """Downsampling block with proper layer registration."""

    def __init__(self, width):
        super(DownSampleBlock, self).__init__()
        self.width = width
        self._layer_cache = nn.ModuleDict()

    def forward(self, x):
        batch_size, seq_len, width = x.shape
        x = x.view(batch_size, -1)

        key = f'down_{seq_len}_{width}'
        if key not in self._layer_cache:
            self._layer_cache[key] = nn.Linear(seq_len * width, (seq_len // 2) * width).to(x.device)

        x = self._layer_cache[key](x)
        x = F.silu(x)
        x = x.view(batch_size, seq_len // 2, width)
        return x


class UpSampleBlock(nn.Module):
    """Upsampling block with proper layer registration."""

    def __init__(self, width):
        super(UpSampleBlock, self).__init__()
        self.width = width
        self._layer_cache = nn.ModuleDict()

    def forward(self, x):
        batch_size, seq_len, width = x.shape
        x = x.view(batch_size, -1)

        key = f'up_{seq_len}_{width}'
        if key not in self._layer_cache:
            self._layer_cache[key] = nn.Linear(seq_len * width, (seq_len * 2) * width).to(x.device)

        x = self._layer_cache[key](x)
        x = F.silu(x)
        x = x.view(batch_size, seq_len * 2, width)
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-Attention with proper layer registration."""

    def __init__(self, units):
        super(CrossAttentionBlock, self).__init__()
        self.units = units
        # Initialize as None, will be created properly in forward
        self.query_weights = None
        self.key_weights = None
        self.value_weights = None
        self.projection_weights = nn.Linear(units, units)
        self._residual_proj = nn.ModuleDict()

    def forward(self, query_inputs, key_value_inputs):
        batch_size, seq_len, num_channels = query_inputs.shape

        while key_value_inputs.dim() > 2:
            key_value_inputs = key_value_inputs.squeeze(1)

        embedding_dim = key_value_inputs.shape[-1]
        key_value_inputs = key_value_inputs.unsqueeze(1).expand(batch_size, seq_len, embedding_dim)

        # Initialize layers properly on first forward pass
        if self.query_weights is None:
            self.query_weights = nn.Linear(num_channels, self.units).to(query_inputs.device)
            self.key_weights = nn.Linear(embedding_dim, self.units).to(query_inputs.device)
            self.value_weights = nn.Linear(embedding_dim, self.units).to(query_inputs.device)

        scale = self.units ** -0.5

        query = self.query_weights(query_inputs)
        key = self.key_weights(key_value_inputs)
        value = self.value_weights(key_value_inputs)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)
        attention_output = self.projection_weights(attention_output)

        # Handle residual connection with proper registration
        if query_inputs.shape[-1] != self.units:
            key = f'residual_{num_channels}_to_{self.units}'
            if key not in self._residual_proj:
                self._residual_proj[key] = nn.Linear(query_inputs.shape[-1], self.units).to(query_inputs.device)
            return self._residual_proj[key](query_inputs) + attention_output

        return query_inputs + attention_output