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

    from typing import Optional

    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Layer
    from Engine.Layers.Tensorflow.RouterLayer import Router

except ImportError as error:
    print(error)
    sys.exit(-1)

class Switch(Layer):

    """A Switch layer implementing conditional computation via expert routing, based on the
    Switch Transformer architecture.

    This layer implements the expert routing mechanism described in:
    "Switch Transformers: Scaling to Trillion Parameter Architectures with Simple and Efficient
    Sparsity" by Fedus et al. (2022). arXiv:2101.03961

    Mathematical Formulation:
        For an input token x:
            1. Routing probabilities: p(x) = Softmax(W_r * x + b_r)
            2. Selected expert: e = argmax(p(x))
            3. Expert output: y = f_e(x) where f_e is the e-th expert network
            4. Final output: z = p_e(x) * y  (weighted by routing probability)

    The layer implements capacity-aware routing where each expert has a fixed capacity
    (number of tokens it can process per batch). Tokens exceeding capacity are dropped.

    Reference:
        William Fedus, Barret Zoph, Noam Shazeer. "Switch Transformers: Scaling to
        Trillion Parameter Architectures with Simple and Efficient Sparsity".
        arXiv:2101.03961, 2022.

    Example:
        >>>  python
        ...     # Create a Switch layer with 4 experts processing 512-dim embeddings
        ...     switch_layer = Switch(
        ...     number_experts=4,
        ...     embedding_dimension=512,
        ...     number_tokens_per_batch=2048,
        ...     capacity_factor=1.25
        ...     )
        ...     # Example input (batch_size=2, sequence_length=10, embed_dim=512)
        ...     inputs = tf.random.normal([2, 10, 512])
        ...     # Process through switch layer
        ...     outputs = switch_layer(inputs)
        ...     print(outputs.shape)  # (2, 10, 512)
        ...     # The layer can be used in a Transformer block:
        ...     class TransformerBlock(layers.Layer):
        ...         def __init__(self):
        ...             self.attention = layers.MultiHeadAttention(4, 512)
        ...             self.switch = Switch(4, 512, 2048)
        ...             self.norm1 = layers.LayerNormalization()
        ...             self.norm2 = layers.LayerNormalization()
        ...
        ...     def call(self, inputs):
        ...         x = self.norm1(inputs + self.attention(inputs, inputs))
        ...         return self.norm2(x + self.switch(x))
        >>>
    """

    def __init__(self,
                 number_experts: int,
                 embedding_dim: int,
                 number_tokens_per_batch: int,
                 capacity_factor: float = 1.0,
                 **kwargs):
        """Initializes the Switch layer with multiple experts.

        Args:
            number_experts: Number of expert networks in the layer (must be ≥ 1)
            embedding_dim: Dimensionality of input token embeddings (must be ≥ 1)
            number_tokens_per_batch: Total tokens across all examples in batch (must be ≥ 1)
            capacity_factor: Multiplier for expert capacity calculation (default: 1.0)
                            Higher values provide more buffer against token overflow
            **kwargs: Additional base layer arguments (e.g., name, dtype)

        Raises:
            ValueError: If any parameter fails validation
        """

        # Parameter validation
        if number_experts < 1:
            raise ValueError(f"num_experts must be ≥ 1, got {number_experts}")

        if embedding_dim < 1:
            raise ValueError(f"embed_dim must be ≥ 1, got {embedding_dim}")

        if number_tokens_per_batch < 1:
            raise ValueError(f"num_tokens_per_batch must be ≥ 1, got {number_tokens_per_batch}")

        if capacity_factor <= 0:
            raise ValueError(f"capacity_factor must be > 0, got {capacity_factor}")

        super().__init__(**kwargs)
        self.number_experts = number_experts
        self.embedding_dimension = embedding_dim
        self.capacity_factor = capacity_factor

        # Create expert networks
        self.experts = [self._create_feedforward_network(embedding_dim) for _ in range(number_experts)]

        # Calculate expert capacity (minimum 1 token per expert)
        self.expert_capacity = max(1, int((number_tokens_per_batch * capacity_factor) / number_experts))

        # Initialize router for token-to-expert assignment
        self.router = Router(num_experts=self.number_experts, expert_capacity=self.expert_capacity)

    def call(self, inputs: tensorflow.Tensor) -> tensorflow.Tensor:
        """Processes input tokens through expert routing and computation.

        The forward pass consists of:

            1. Routing tokens to experts using learned routing weights
            2. Processing tokens by their assigned experts
            3. Combining expert outputs back into original token order

        Args:
            inputs: Input tensor of shape [batch_size, num_tokens, embed_dim]

        Returns:
            Output tensor of same shape as inputs after expert processing

        Raises:
            ValueError: If input tensor is invalid or routing fails
            RuntimeError: If expert computation fails
        """

        if not isinstance(inputs, tensorflow.Tensor):
            raise TypeError(f"Inputs must be tf.Tensor, got {type(inputs)}")

        if len(inputs.shape) != 3:
            raise ValueError(f"Inputs must be rank 3 [batch, tokens, embed_dim], got shape {inputs.shape}")

        if inputs.shape[-1] != self.embedding_dimension:
            raise ValueError(f"Last dimension must match embed_dim {self.embedding_dimension}, got {inputs.shape[-1]}")

        try:
            # Preserve original shape for output
            batch_size = tensorflow.shape(inputs)[0]
            number_tokens = tensorflow.shape(inputs)[1]
            original_shape = [batch_size, number_tokens, self.embedding_dimension]

            # 1. Route tokens to experts
            dispatch_tensor, combine_tensor = self.router(inputs)

            # 2. Process tokens through experts
            expert_inputs = self._route_to_experts(inputs, dispatch_tensor)
            expert_outputs = self._process_experts(expert_inputs)

            # 3. Combine expert outputs
            combined_outputs = self._combine_outputs(expert_outputs, combine_tensor)

            # Restore original shape
            return tensorflow.reshape(combined_outputs, original_shape)

        except Exception as e:
            raise RuntimeError(f"Switch layer computation failed: {str(e)}") from e

    @staticmethod
    def _route_to_experts(inputs: tensorflow.Tensor, dispatch_tensor: tensorflow.Tensor) -> tensorflow.Tensor:
        """Distributes input tokens to their assigned experts.

        Args:
            inputs: Original input tokens [batch, tokens, embed_dim]
            dispatch_tensor: Routing tensor from router [batch, tokens, experts, capacity]

        Returns:
            Tensor of expert inputs shaped [batch, experts, capacity, embed_dim]
            where each expert's inputs are padded to expert_capacity
        """
        return tensorflow.einsum("bte,btec->bec", inputs, dispatch_tensor)

    def _process_experts(self, expert_inputs: tensorflow.Tensor) -> list[tensorflow.Tensor]:
        """Processes routed tokens through their assigned experts.

        Args:
            expert_inputs: Tensor of expert inputs [batch, experts, capacity, embed_dim]

        Returns:
            List of expert output tensors, one per expert
        """
        # Transpose to [experts, batch, capacity, embed_dim] for parallel processing
        expert_inputs = tensorflow.transpose(expert_inputs, perm=[1, 0, 2, 3])

        # Process each expert's inputs in parallel
        return [expert(tensorflow.reshape(expert_inputs[i],
                                          [-1, self.embedding_dimension])) for i, expert in enumerate(self.experts)]

    @staticmethod
    def _combine_outputs(expert_outputs: list[tensorflow.Tensor],
                         combine_tensor: tensorflow.Tensor) -> tensorflow.Tensor:
        """Recombines expert outputs into original token order.

        Args:
            expert_outputs: List of expert output tensors
            combine_tensor: Combination weights from router [batch, tokens, experts, capacity]

        Returns:
            Combined output tensor [batch, tokens, embed_dim]
        """
        # Stack expert outputs along expert dimension
        stacked_outputs = tensorflow.stack(expert_outputs, axis=1)

        # Combine using learned weights
        return tensorflow.einsum("btec,beci->bti", combine_tensor, stacked_outputs)

    @staticmethod
    def _create_feedforward_network(ff_dim: int, name: Optional[str] = None) -> keras.Sequential:
        """Creates a simple feedforward neural network with two dense layers.

        Args:
            ff_dim (int): The number of units (neurons) in each dense layer.
            name (Optional[str], optional): The name of the model. Default is None.

        Returns:
            keras.Sequential: A feedforward neural network model consisting of two dense layers.

        Raises:
            ValueError: If the parameters are invalid.
        """
        # Validate parameters
        if not isinstance(ff_dim, int) or ff_dim <= 0:
            raise ValueError("ff_dim must be a positive integer.")

        try:
            # Define a Sequential model with two dense layers of ReLU activations
            return keras.Sequential([
                Dense(ff_dim, activation="relu"),
                Dense(ff_dim, activation="relu")], name=name)

        except Exception as e:
            raise ValueError(f"Error creating the feedforward network: {e}")


    def get_config(self) -> dict:
        """Gets the layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "num_experts": self.number_experts,
            "embed_dim": self.embedding_dimension,
            "num_tokens_per_batch": int(self.num_tokens_per_batch),
            "capacity_factor": self.capacity_factor,
        })
        return config