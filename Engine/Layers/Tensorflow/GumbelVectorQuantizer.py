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

    from keras import Model
    from keras import Layer

    from tensorflow.keras.layers import Dense

except ImportError as error:
    print(error)
    sys.exit(-1)


class GumbelVectorQuantizer(Layer):
    """
    A Gumbel-Softmax based Vector Quantizer for neural networks, typically used for
    discretization of continuous representations in the context of generative models.

    This class implements the vector quantization process where the continuous hidden
    states of a neural network are quantized using a Gumbel-Softmax relaxation.
    It includes a learnable codebook, and the codebook is updated during
    training via backpropagation.

    **Reference:**
    A reference to the original paper can be found here:
    - "Neural Discrete Representation Learning" by Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu.
     [https://arxiv.org/abs/1711.00937]

    Attributes:
        @number_groups (int): The number of groups for the codebook vectors.
        @number_vectors (int): The number of vectors per group.
        @code_vector_size (int): The size of each codebook vector after division by number of groups.
        @temperature (float): The temperature parameter used in the Gumbel-Softmax distribution.
        @linear (tf.keras.layers.Dense): A dense layer used to project the hidden states to the codebook space.
        @code_book (tf.Tensor): The learnable codebook that contains the code vectors.
    """

    def __init__(self):
        """
        Initializes the GumbelVectorQuantizer model with the configuration parameters.

        Args:
            config (object): A configuration object containing the hyperparameters for the vector quantizer.
                The configuration should include:
                    - num_code_vector_groups (int): Number of groups for codebook vectors.
                    - num_code_vectors_per_group (int): Number of code vectors per group.
                    - code_vector_size (int): The size of each codebook vector.
                    - gumbel_init_temperature (float): Initial temperature for the Gumbel-Softmax distribution.
        """
        super().__init__()
        self.number_groups = 4
        self.number_vectors = 16
        self.code_vector_size = 16 // 4
        self.temperature = 0.2

        self.linear = Dense(self.number_groups * self.number_vectors)
        self.code_book = self.add_weight(
            shape=(1, self.number_groups, self.number_vectors, self.code_vector_size),
            initializer="random_normal", trainable=True
        )

    @staticmethod
    def _compute_perplexity(probs, lengths):
        """
        Computes the perplexity of the Gumbel-Softmax distribution.

        Perplexity is a measure of uncertainty in the distribution and is used
        to evaluate how diverse the sampled code vectors are.

        Args:
            probs (tf.Tensor): The probability distribution over codebook vectors. Shape: (B, L, G, V)
            lengths (tf.Tensor): Lengths of the sequences in the batch. Shape: (B,)

        Returns:
            tf.Tensor: Perplexity for each group and vector, shape (G, V).
        """
        mask = tensorflow.sequence_mask(lengths, maxlen=tensorflow.shape(probs)[1], dtype=probs.dtype)
        mask = tensorflow.reshape(mask, (-1, 1, 1))
        masked_probs = probs * mask
        num_values = tensorflow.reduce_sum(mask)
        perplexity = tensorflow.reduce_sum(masked_probs, axis=0) / num_values

        return perplexity

    def call(self, hidden_states, lengths):
        """
        Performs the forward pass of the Gumbel-Softmax vector quantization.

        The function takes the continuous hidden states as input and quantizes them
        using the Gumbel-Softmax technique. It returns the quantized code vectors
        and the perplexity of the codebook's distribution.

        Args:
            hidden_states (tf.Tensor): The input hidden states to be quantized. Shape: (B, L, D1)
            lengths (tf.Tensor): The lengths of the sequences in the batch. Shape: (B,)

        Returns:
            Tuple:
                - code_vectors (tf.Tensor): Quantized code vectors. Shape: (B, L, D2)
                - perplexity (tf.Tensor): Perplexity of the codebook. Shape: (G, V)
        """
        batch_size, length, _ = tensorflow.shape(hidden_states)

        # Project hidden states to the codebook space
        hidden_states = self.linear(hidden_states)
        hidden_states = tensorflow.reshape(hidden_states, (-1, self.number_vectors))

        # Sample codebook vector probabilities using Gumbel-Softmax
        code_vector_probs = tensorflow.nn.softmax(tensorflow.random.uniform(tensorflow.shape(hidden_states))
                                          + hidden_states / self.temperature, axis=-1)

        # Apply hard quantization (straight-through estimator)
        code_vector_probs_hard = tensorflow.one_hot(tensorflow.argmax(code_vector_probs, axis=-1), depth=self.number_vectors)
        code_vector_probs = tensorflow.stop_gradient(code_vector_probs_hard - code_vector_probs) + code_vector_probs

        # Reshape and apply codebook lookup
        code_vector_probs = tensorflow.reshape(code_vector_probs, (batch_size * length, self.number_groups, -1, 1))
        code_vectors = code_vector_probs * self.code_book

        # Sum along the group dimension
        code_vectors = tensorflow.reduce_sum(code_vectors, axis=-2)
        code_vectors = tensorflow.reshape(code_vectors, (batch_size, length, -1))

        # Compute soft distribution over the codebook vectors for perplexity
        code_vector_soft_distance = tensorflow.nn.softmax(tensorflow.reshape(hidden_states,
                                                                         (batch_size, length,
                                                                          self.number_groups, -1)), axis=-1)

        perplexity = self._compute_perplexity(code_vector_soft_distance, lengths)

        return code_vectors, perplexity
