#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

from Engine.Layers.Torch.Activations import Activations
from Engine.Layers.Torch.TimeEmbedding import TimeEmbedding

try:
    import sys
    import torch
    import math
    import warnings
    import torch.nn as nn
    import torch.nn.functional as F

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


class UNetDenoisingModelTorch(nn.Module, Activations):
    """
    UNetModel - Fixed version with dynamic skip projections
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

        # FIXED: Calculate actual number of downsamples (not number of levels)
        # Number of downsamples is one less than number of levels
        number_downsamples = len(self._list_neurons_per_level) - 1
        self._output_shape = self._adjust_output_shape_for_downsampling(output_shape, number_downsamples)
        self._initial_output_shape = self._output_shape

        # Initialize skip projection cache for dynamic creation
        self._skip_projections = nn.ModuleDict()

        # Build the model architecture
        self._build_model()

    @staticmethod
    def _adjust_output_shape_for_downsampling(shape: int, number_downsamples: int) -> int:
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
            return original_shape

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
        """Constructs the U-Net model architecture as PyTorch modules."""
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

        current_seq_len = self._output_shape

        for level_idx, num_neurons in enumerate(self._list_neurons_per_level):
            level_blocks = nn.ModuleList()

            for _ in range(self._number_residual_blocks):
                level_blocks.append(self._create_residual_block(num_neurons))

                if self._list_attention[level_idx]:
                    level_blocks.append(CrossAttentionBlock(num_neurons))

            self.down_blocks.append(level_blocks)

            if level_idx != len(self._list_neurons_per_level) - 1:
                self.down_samples.append(self._create_down_sample(num_neurons, current_seq_len))
                current_seq_len = current_seq_len // 2

        # Middle blocks
        self.mid_block1 = self._create_residual_block(self._list_neurons_per_level[-1])
        self.mid_attn = CrossAttentionBlock(self._list_neurons_per_level[-1])
        self.mid_block2 = self._create_residual_block(self._list_neurons_per_level[-1])

        # Decoder (upsampling path)
        # Skip projections will be created dynamically during forward pass
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
                self.up_samples.append(self._create_up_sample(num_neurons, current_seq_len))
                current_seq_len = current_seq_len * 2

        # Final output layer
        self.final_dense = nn.Linear(
            self._output_shape * self._list_neurons_per_level[0],
            self._output_shape * self._embedding_channels
        )

    def _get_skip_projection(self, level_idx, input_features, output_features, device):
        """
        Get or create a skip projection layer for the given level with the specified dimensions.
        This allows for dynamic creation based on actual runtime dimensions.
        """
        proj_key = f'level_{level_idx}_{input_features}_{output_features}'

        if proj_key not in self._skip_projections:
            self._skip_projections[proj_key] = nn.Linear(
                input_features,
                output_features
            ).to(device)

        return self._skip_projections[proj_key]

    def _create_down_sample(self, width, seq_len):
        input_features = seq_len * width
        output_features = (seq_len // 2) * width

        return nn.ModuleDict({
            'dense': nn.Linear(input_features, output_features)
        })

    def _create_up_sample(self, width, seq_len):
        input_features = seq_len * width
        output_features = (seq_len * 2) * width

        return nn.ModuleDict({
            'dense': nn.Linear(input_features, output_features)
        })

    def _time_MLP(self, units):
        return nn.Sequential(
            nn.Linear(units, units),
            nn.SiLU(),
            nn.Linear(units, units)
        )

    def _label_embedding_MLP(self, units):
        return nn.Sequential(
            nn.Linear(units, units),
            nn.SiLU(),
            nn.Linear(units, self._output_shape)
        )

    def _create_residual_block(self, number_filters):
        return ResidualBlock(number_filters, self._intermediary_activation_function)

    def forward(self, image_input, time_input, description_input):
        """
        Forward pass through the U-Net model.
        FIXED: Properly handles both padding and truncation to restore original shape
        """
        # Store original dimensions before any modifications
        batch_size = image_input.shape[0]
        original_seq_len = image_input.shape[1]

        # Check if input has channel dimension
        if len(image_input.shape) == 2:
            # Input is [batch, seq] - add channel dimension
            image_input = image_input.unsqueeze(-1)
            input_had_no_channels = True
        else:
            input_had_no_channels = False

        original_channels = image_input.shape[2]

        # Pad or truncate input if necessary to match internal output_shape
        if original_seq_len < self._output_shape:
            padding_needed = self._output_shape - original_seq_len
            image_input = F.pad(image_input, (0, 0, 0, padding_needed), mode='constant', value=0)
        elif original_seq_len > self._output_shape:
            image_input = image_input[:, :self._output_shape, :]

        # Initial processing
        x = image_input.view(batch_size, -1)
        x = self.first_dense(x)
        x = self._get_activation(self._intermediary_activation_function)(x)
        x = x.view(batch_size, self._output_shape, self._embedding_channels)

        # Time and label embeddings
        t_emb = self.time_embedding(time_input)
        t_emb = self.time_mlp(t_emb)
        l_emb = self.label_mlp(description_input)

        # Skip connections
        skip_connections = []

        # Encoder
        for level_idx, level_blocks in enumerate(self.down_blocks):
            for block in level_blocks:
                if isinstance(block, ResidualBlock):
                    x = block(x, t_emb)
                else:  # CrossAttentionBlock
                    current_seq_len = x.shape[1]
                    if l_emb.shape[1] != current_seq_len:
                        l_emb_resized = F.interpolate(
                            l_emb.unsqueeze(1),
                            size=current_seq_len,
                            mode='linear',
                            align_corners=False
                        ).squeeze(1)
                    else:
                        l_emb_resized = l_emb

                    x = block(x, l_emb_resized)

            # Save skip connection at the end of each level
            skip_connections.append(x)

            if level_idx < len(self.down_samples):
                x = self._down_sample_forward(x, self.down_samples[level_idx])

        # Middle
        x = self.mid_block1(x, t_emb)
        current_seq_len = x.shape[1]
        if l_emb.shape[1] != current_seq_len:
            l_emb_resized = F.interpolate(
                l_emb.unsqueeze(1),
                size=current_seq_len,
                mode='linear',
                align_corners=False
            ).squeeze(1)
        else:
            l_emb_resized = l_emb
        x = self.mid_attn(x, l_emb_resized)
        x = self.mid_block2(x, t_emb)

        # Decoder - FIXED with dynamic skip projections
        for level_idx, level_blocks in enumerate(self.up_blocks):
            # Determine the original encoder level this corresponds to
            original_level_idx = len(self._list_neurons_per_level) - 1 - level_idx

            # Upsample first (if not the first decoder level)
            if level_idx > 0:
                x = self._up_sample_forward(x, self.up_samples[level_idx - 1])

            # Get skip connection
            skip = skip_connections.pop()

            # Ensure skip and x have the same sequence length
            if skip.shape[1] != x.shape[1]:
                skip = F.interpolate(
                    skip.transpose(1, 2),
                    size=x.shape[1],
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)

            # Concatenate along channel dimension
            x = torch.cat([x, skip], dim=-1)

            # FIXED: Create projection dynamically based on actual dimensions
            current_seq_len = x.shape[1]
            current_channels = x.shape[2]
            target_channels = self._list_neurons_per_level[original_level_idx]

            input_features = current_seq_len * current_channels
            output_features = current_seq_len * target_channels

            # Get or create the skip projection with correct dimensions
            proj_layer = self._get_skip_projection(
                original_level_idx,
                input_features,
                output_features,
                x.device
            )

            # Apply projection
            x_flat = x.view(batch_size, -1)
            x = proj_layer(x_flat)
            x = self._get_activation(self._intermediary_activation_function)(x)
            x = x.view(batch_size, current_seq_len, target_channels)

            # Process through blocks
            for block in level_blocks:
                if isinstance(block, ResidualBlock):
                    x = block(x, t_emb)
                else:  # CrossAttentionBlock
                    current_seq_len = x.shape[1]
                    if l_emb.shape[1] != current_seq_len:
                        l_emb_resized = F.interpolate(
                            l_emb.unsqueeze(1),
                            size=current_seq_len,
                            mode='linear',
                            align_corners=False
                        ).squeeze(1)
                    else:
                        l_emb_resized = l_emb
                    x = block(x, l_emb_resized)

        # Final output
        x = x.view(batch_size, -1)
        x = self.final_dense(x)
        x = x.view(batch_size, self._output_shape, self._embedding_channels)

        # Restore original sequence length if it differs from internal output_shape
        if original_seq_len < self._output_shape:
            # We padded the input, so truncate the output
            x = x[:, :original_seq_len, :]
        elif original_seq_len > self._output_shape:
            # We truncated the input, so pad the output to restore original length
            padding_needed = original_seq_len - self._output_shape
            x = F.pad(x, (0, 0, 0, padding_needed), mode='constant', value=0)

        # Remove channel dimension if input didn't have one originally
        if input_had_no_channels:
            x = x.squeeze(-1)

        return x

    def _down_sample_forward(self, x, module):
        batch_size, seq_len, width = x.shape
        x = x.view(batch_size, -1)
        x = module['dense'](x)
        x = self._get_activation(self._intermediary_activation_function)(x)
        new_seq_len = seq_len // 2
        x = x.view(batch_size, new_seq_len, width)
        return x

    def _up_sample_forward(self, x, module):
        batch_size, seq_len, width = x.shape
        x = x.view(batch_size, -1)
        x = module['dense'](x)
        x = self._get_activation(self._intermediary_activation_function)(x)
        new_seq_len = seq_len * 2
        x = x.view(batch_size, new_seq_len, width)
        return x

    def _get_activation(self, name):
        activations = {
            'swish': nn.SiLU(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(self._intermediary_activation_alpha),
            'elu': nn.ELU(self._intermediary_activation_alpha),
            'linear': nn.Identity()
        }
        return activations.get(name.lower(), nn.SiLU())

    # Properties
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
    """Residual block for the UNet model with proper layer caching."""

    def __init__(self, num_filters, activation='swish'):
        super(ResidualBlock, self).__init__()
        self.num_filters = num_filters
        self.activation = activation
        self._layer_cache = {}

    def _get_or_create_layer(self, key, input_size, output_size, device):
        if key not in self._layer_cache:
            self._layer_cache[key] = nn.Linear(input_size, output_size).to(device)
        return self._layer_cache[key]

    def forward(self, x, t_emb):
        input_width = x.shape[-1]
        batch_size, seq_len, _ = x.shape

        # Create residual projection if needed
        if input_width == self.num_filters:
            residual = x
        else:
            key = f'residual_{seq_len}_{input_width}'
            layer = self._get_or_create_layer(
                key,
                seq_len * input_width,
                seq_len * self.num_filters,
                x.device
            )
            residual = layer(x.view(batch_size, -1)).view(batch_size, seq_len, self.num_filters)

        # Time embedding projection
        key_time = f'time_{t_emb.shape[-1]}'
        t_layer = self._get_or_create_layer(
            key_time,
            t_emb.shape[-1],
            self.num_filters,
            x.device
        )
        t_emb_proj = t_layer(t_emb).unsqueeze(1)

        # Main path - first transformation
        key_main1 = f'main1_{seq_len}_{input_width}'
        main1_layer = self._get_or_create_layer(
            key_main1,
            seq_len * input_width,
            seq_len * self.num_filters,
            x.device
        )
        x = main1_layer(x.view(batch_size, -1))
        x = self._get_activation(x)
        x = x.view(batch_size, seq_len, self.num_filters)

        # Add time embedding
        x = x + t_emb_proj

        # Second transformation
        key_main2 = f'main2_{seq_len}'
        main2_layer = self._get_or_create_layer(
            key_main2,
            seq_len * self.num_filters,
            seq_len * self.num_filters,
            x.device
        )
        x = main2_layer(x.view(batch_size, -1))
        x = self._get_activation(x)
        x = x.view(batch_size, seq_len, self.num_filters)

        return x + residual

    def _get_activation(self, x):
        if self.activation.lower() == 'swish':
            return F.silu(x)
        elif self.activation.lower() == 'relu':
            return F.relu(x)
        elif self.activation.lower() == 'leaky_relu':
            return F.leaky_relu(x, 0.2)
        elif self.activation.lower() == 'elu':
            return F.elu(x, 0.05)
        else:
            return x