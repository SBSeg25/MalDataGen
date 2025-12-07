#!/usr/bin/env python3
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

    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Layer

except ImportError as error:
    print(error)
    sys.exit(-1)


class Router(Layer):
    """Router layer implementing token-to-expert assignment with load balancing, based on the
    Switch Transformer architecture.

    This layer implements the routing mechanism described in:
    "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient
    Sparsity" by Fedus et al. (2022). arXiv:2101.03961

    Mathematical Formulation:

        1. Routing logits: h(x) = W_r * x + b_r
        2. Routing probabilities: p(x) = softmax(h(x))
        3. Expert selection: e = argmax(p(x))
        4. Mask creation: m = one_hot(e) * 1{position < capacity}

        5. Load balancing loss: L_aux = N/E * sum(f_i * P_i) where:
            - f_i is fraction of tokens routed to expert i
            - P_i is average routing probability for expert i
            - N is number of tokens, E is number of experts

    The layer implements capacity-aware routing with:
        - Top-1 expert selection
        - Load balancing via auxiliary loss
        - Expert capacity constraints
        - Stochastic routing during training

    Reference:
        William Fedus, Barret Zoph, Noam Shazeer. "Switch Transformers: Scaling to Trillion Parameter
        Models with Simple and Efficient Sparsity". arXiv:2101.03961, 2022.

    Example:
        >>>     python
        ...     # Create a router for 8 experts with capacity 32 tokens/expert
        ...     router = Router(num_experts=8,
        ...     expert_capacity=32
        ...     )
        ...
        ...     # Example input (batch_size=4, sequence_length=64, embed_dim=512)
        ...     inputs = tensorflow.random.normal([4, 64, 512])
        ...
        ...     # Get routing masks
        ...     dispatch_tensor, combine_tensor = router(inputs, training=True)
        ...
        ...     # Tensors will have shape:
        ...     # dispatch_tensor: [4, 64, 8, 32] (one-hot expert assignment)
        ...     # combine_tensor: [4, 64, 8, 32] (weighted combination)
        ...
        ...     # The router can be used in a Switch layer:
        ...     class Switch(layers.Layer):
        ...         def __init__(self):
        ...             self.router = Router(num_experts=8, expert_capacity=32)
        ...             self.experts = [FeedForwardNetwork() for _ in range(8)]
        ...
        ...         def call(self, inputs):
        ...             dispatch, combine = self.router(inputs)
        ...             expert_inputs = tensorflow.einsum('bte,btec->bec', inputs, dispatch)
        ...             expert_outputs = [expert(expert_inputs[:,i]) for i,expert in enumerate(self.experts)]
        ...             return tensorflow.einsum('bec,btec->bte', tensorflow.stack(expert_outputs, 1), combine)
        >>>
    """

    def __init__(self, number_experts: int, expert_capacity: int, **kwargs):
        """Initializes the Router layer with expert routing capabilities.

        Args:
            number_experts: Number of experts to route to (must be ≥ 1)
            expert_capacity: Maximum tokens each expert can handle per batch (must be ≥ 1)
            **kwargs: Additional base layer arguments (e.g., name, dtype)

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if number_experts < 1:
            raise ValueError(f"number_experts must be ≥ 1, got {number_experts}")

        if expert_capacity < 1:
            raise ValueError(f"expert_capacity must be ≥ 1, got {expert_capacity}")

        super().__init__(**kwargs)
        self.number_experts = number_experts
        self.expert_capacity = expert_capacity

        # Routing layer learns to assign tokens to experts
        self.route = Dense(
            units=number_experts,
            kernel_initializer='glorot_uniform',
            name='router_weights'
        )

    def call(self, inputs: tensorflow.Tensor,
             training: bool = False) -> tuple[tensorflow.Tensor, tensorflow.Tensor]:
        """Computes expert routing assignments with load balancing.

        Implements the routing algorithm from Section 3 of the paper:
            1. Compute routing logits
            2. Apply noise during training (optional)
            3. Convert to probabilities via softmax
            4. Select top-1 expert per token
            5. Enforce expert capacity constraints
            6. Compute load balancing auxiliary loss

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, hidden_dim]
            training: Whether in training mode (enables stochastic routing)

        Returns:
            Tuple containing:
            - dispatch_tensor: [batch_size, seq_len, num_experts, expert_capacity]
                            One-hot tensor indicating expert assignments
            - combine_tensor: [batch_size, seq_len, num_experts, expert_capacity]
                            Weighted combination tensor for gradient flow

        Raises:
            ValueError: If input tensor is invalid
            RuntimeError: If routing computation fails
        """
        if not isinstance(inputs, tensorflow.Tensor):
            raise TypeError(f"Inputs must be tf.Tensor, got {type(inputs)}")

        if len(inputs.shape) != 3:
            raise ValueError(f"Inputs must be rank 3 [batch, seq_len, hidden_dim], got {inputs.shape}")

        try:
            # 1. Compute routing logits
            router_logits = self.route(inputs)  # [batch, seq_len, num_experts]

            # 2. Add noise during training (Section 3.2 of paper)
            if training:
                noise = tensorflow.random.normal(shape=tensorflow.shape(router_logits), mean=0.0, stddev=1.0)
                router_logits += noise

            # 3. Convert to probabilities
            router_probs = tensorflow.nn.softmax(router_logits, axis=-1)

            # 4. Expert selection (top-1 routing)
            expert_gate, expert_index = tensorflow.math.top_k(router_probs, k=1)
            expert_mask = tensorflow.one_hot(expert_index, depth=self.number_experts)

            # 5. Compute load balancing auxiliary loss (Eq. 4 in paper)
            self.add_loss(self._load_balanced_loss(router_probs, expert_mask))

            # 6. Enforce expert capacity constraints
            position_in_expert = tensorflow.cumsum(expert_mask, axis=1, exclusive=True)
            capacity_mask = tensorflow.cast(position_in_expert < self.expert_capacity, tensorflow.float32)
            expert_mask *= capacity_mask

            # Create dispatch and combine tensors
            dispatch_tensor = tensorflow.einsum('bse,bsec->bsec',
                                                tensorflow.squeeze(expert_gate, -1),
                                                tensorflow.one_hot(position_in_expert, depth=self.expert_capacity))

            combine_tensor = tensorflow.einsum('bse,bsec->bsec',
                                               tensorflow.squeeze(router_probs, -1),
                                               tensorflow.one_hot(position_in_expert, depth=self.expert_capacity))

            return dispatch_tensor, combine_tensor

        except Exception as e:
            raise RuntimeError(f"Routing computation failed: {str(e)}") from e

    @staticmethod
    def _load_balanced_loss(router_probs: tensorflow.Tensor, expert_mask: tensorflow.Tensor) -> tensorflow.Tensor:
        """Computes the load-balanced loss for routing experts in the mixture of experts model.

        Args:
            router_probs (tensorflow.Tensor): The probabilities produced by the router for each expert.
            expert_mask (tensorflow.Tensor): The mask indicating which expert was chosen for each token.

        Returns:
            tensorflow.Tensor: A scalar tensor representing the load-balanced loss.

        Raises:
            ValueError: If any parameter has an incorrect shape or type.
        """
        # Validate parameters
        if not isinstance(router_probs, tensorflow.Tensor) or not isinstance(expert_mask, tensorflow.Tensor):
            raise ValueError("Both router_probs and expert_mask must be tensors.")

        try:
            num_experts = tensorflow.shape(expert_mask)[-1]
            density = tensorflow.reduce_mean(expert_mask, axis=0)  # Density of active experts
            density_proxy = tensorflow.reduce_mean(router_probs, axis=0)  # Proxy for density based on routing probabilities

            # Calculate the load-balanced loss
            loss = tensorflow.reduce_mean(density_proxy * density) * tensorflow.cast((num_experts ** 2),
                                                                                     tensorflow.float32)
            return loss
        except Exception as e:
            raise ValueError(f"Error calculating the load-balanced loss: {e}")

    def get_config(self) -> dict:
        """Returns the layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'num_experts': self.number_experts,
            'expert_capacity': self.expert_capacity
        })
        return config
