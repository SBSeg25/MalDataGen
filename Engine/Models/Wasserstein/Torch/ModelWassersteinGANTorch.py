#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/07'
__credits__ = ['Synthetic Ocean AI']

from Engine.Models.Wasserstein.Torch.VanillaDiscriminatorTorch import VanillaDiscriminatorTorch
from Engine.Models.Wasserstein.Torch.VanillaGeneratorTorch import VanillaGeneratorTorch

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
    import numpy

    from typing import List
    from typing import Tuple
    from typing import Union
    from typing import Optional
    from typing import Dict

except ImportError as error:
    print(error)
    print()
    sys.exit(-1)

DEFAULT_WASSERSTEIN_GAN_LATENT_DIMENSION = 128
DEFAULT_WASSERSTEIN_GAN_ACTIVATION = "leaky_relu"
DEFAULT_WASSERSTEIN_GAN_DROPOUT_DECAY_RATE_G = 0.2
DEFAULT_WASSERSTEIN_GAN_DROPOUT_DECAY_RATE_D = 0.4
DEFAULT_WASSERSTEIN_GAN_DENSE_LAYERS_SETTINGS_GENERATOR = [128]
DEFAULT_WASSERSTEIN_GAN_DENSE_LAYERS_SETTINGS_DISCRIMINATOR = [128]
DEFAULT_WASSERSTEIN_GAN_LAST_ACTIVATION_LAYER = "sigmoid"
DEFAULT_WASSERSTEIN_GAN_INITIALIZER_MEAN = 0.0
DEFAULT_WASSERSTEIN_GAN_INITIALIZER_DEVIATION = 0.125


class WassersteinModelTorch(VanillaDiscriminatorTorch, VanillaGeneratorTorch):
    """
    WassersteinGP Generative Adversarial Network (WGAN-GP) with Gradient Penalty (PyTorch version).

    This class implements a Wasserstein GAN, a type of Generative Adversarial
    Network designed to improve training stability and provide a more meaningful
    loss metric by approximating the Earth Mover's Distance (Wasserstein-1 Distance)
    between real and generated data distributions.

    The model integrates both the **generator** (which synthesizes new data samples)
    and the **critic** (which scores the realism of samples) into a single interface,
    ensuring consistency across architectural configuration and training routines.

    Unlike traditional GANs, the discriminator (referred to as "critic" in WGANs)
    does not classify inputs as "real" or "fake." Instead, it assigns a scalar score,
    which is optimized to approximate the Wasserstein distance between the true data
    distribution and the distribution induced by the generator.

    To enforce the Lipschitz continuity condition required by the WGAN framework,
    this model supports **Gradient Penalty (GP)**, which penalizes deviations from
    unit gradient norms during training, following the approach introduced in
    Gulrajani et al., 2017.

    References:
        - Arjovsky, M., Chintala, S., & Bottou, L. (2017).
          Wasserstein GAN. arXiv preprint arXiv:1701.07875.
          Available at: https://arxiv.org/abs/1701.07875

        - Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017).
          Improved Training of Wasserstein GANs. arXiv preprint arXiv:1704.00028.
          Available at: https://arxiv.org/abs/1704.00028

    Attributes:
        latent_dimension (int): Dimensionality of the latent space.
        output_shape (int): Dimensionality of the generated samples.
        activation_function (str): Activation function applied in hidden layers.
        initializer_mean (float): Mean of the normal distribution for weight initialization.
        initializer_deviation (float): Standard deviation for weight initialization.
        dropout_decay_rate_g (float): Dropout rate for the generator's dense layers.
        dropout_decay_rate_d (float): Dropout rate for the critic's dense layers.
        last_layer_activation (str): Activation function for the generator's output layer.
        dense_layer_sizes_g (List[int]): Sizes of dense layers in the generator.
        dense_layer_sizes_d (List[int]): Sizes of dense layers in the critic.
        dataset_type (torch.dtype): Data type for the training data.
        number_samples_per_class (Optional[Dict[str, int]]): Number of samples per class.

    Example:
        >>> wgan = WassersteinModelTorch(
        ...     latent_dimension=100,
        ...     output_shape=784,
        ...     activation_function='leaky_relu',
        ...     dropout_decay_rate_g=0.2,
        ...     dropout_decay_rate_d=0.4,
        ...     dense_layer_sizes_g=[256, 512, 1024],
        ...     dense_layer_sizes_d=[512, 256, 128],
        ...     number_samples_per_class={"number_classes": 10}
        ... )
        >>> generator = wgan.get_generator()
        >>> critic = wgan.get_discriminator()
    """

    def __init__(self, latent_dimension: int = DEFAULT_WASSERSTEIN_GAN_LATENT_DIMENSION,
                 output_shape: int = 128,
                 activation_function: str = DEFAULT_WASSERSTEIN_GAN_ACTIVATION,
                 initializer_mean: float = DEFAULT_WASSERSTEIN_GAN_INITIALIZER_MEAN,
                 initializer_deviation: float = DEFAULT_WASSERSTEIN_GAN_INITIALIZER_DEVIATION,
                 dropout_decay_rate_g: float = DEFAULT_WASSERSTEIN_GAN_DROPOUT_DECAY_RATE_G,
                 dropout_decay_rate_d: float = DEFAULT_WASSERSTEIN_GAN_DROPOUT_DECAY_RATE_D,
                 last_layer_activation: str = DEFAULT_WASSERSTEIN_GAN_LAST_ACTIVATION_LAYER,
                 dense_layer_sizes_g: Optional[List[int]] = None,
                 dense_layer_sizes_d: Optional[List[int]] = None,
                 dataset_type: torch.dtype = torch.float32,
                 number_samples_per_class: Optional[Dict[str, int]] = None):
        """
        Initializes a WassersteinModelTorch, combining the generator and critic components.

        This constructor sets up both the generator and the critic networks, applying
        the provided architectural and training configurations. The generator maps
        random noise vectors into the data space, while the critic evaluates how
        realistic those samples are relative to real data.

        Args:
            latent_dimension (int): Dimensionality of the latent space.
            output_shape (int): Dimensionality of the generated samples.
            activation_function (str): Activation function for hidden layers.
            initializer_mean (float): Mean for weight initialization.
            initializer_deviation (float): Standard deviation for weight initialization.
            dropout_decay_rate_g (float): Dropout rate for the generator.
            dropout_decay_rate_d (float): Dropout rate for the critic.
            last_layer_activation (str): Activation for generator's final layer.
            dense_layer_sizes_g (List[int]): Dense layer sizes in the generator.
            dense_layer_sizes_d (List[int]): Dense layer sizes in the critic.
            dataset_type (torch.dtype): Data type for the dataset.
            number_samples_per_class (Optional[Dict[str, int]]): Samples per class.

        Raises:
            ValueError: If any provided argument is invalid.
        """
        if dense_layer_sizes_d is None:
            dense_layer_sizes_d = DEFAULT_WASSERSTEIN_GAN_DENSE_LAYERS_SETTINGS_DISCRIMINATOR

        if dense_layer_sizes_g is None:
            dense_layer_sizes_g = DEFAULT_WASSERSTEIN_GAN_DENSE_LAYERS_SETTINGS_GENERATOR

        if latent_dimension <= 0:
            raise ValueError("Latent dimension must be a positive integer.")

        if not all(size > 0 for size in dense_layer_sizes_g):
            raise ValueError("All generator dense layer sizes must be positive integers.")

        if not all(size > 0 for size in dense_layer_sizes_d):
            raise ValueError("All discriminator dense layer sizes must be positive integers.")

        if dropout_decay_rate_g < 0 or dropout_decay_rate_g > 1:
            raise ValueError("Generator dropout decay rate must be between 0 and 1.")

        if dropout_decay_rate_d < 0 or dropout_decay_rate_d > 1:
            raise ValueError("Discriminator dropout decay rate must be between 0 and 1.")

        # CRITICAL FIX: Initialize nn.Module first to avoid MRO issues
        nn.Module.__init__(self)

        # Manually set all discriminator attributes
        self._discriminator_latent_dimension = latent_dimension
        self._discriminator_output_shape = output_shape
        self._discriminator_activation_function = activation_function
        self._discriminator_last_layer_activation = last_layer_activation
        self._discriminator_dropout_decay_rate_d = dropout_decay_rate_d
        self._discriminator_dense_layer_sizes_d = dense_layer_sizes_d
        self._discriminator_dataset_type = dataset_type
        self._discriminator_initializer_mean = initializer_mean
        self._discriminator_initializer_deviation = initializer_deviation
        self._discriminator_number_samples_per_class = number_samples_per_class
        self._discriminator_model_dense = None
        self._discriminator_model_with_labels = None

        # Manually set all generator attributes
        self._generator_latent_dimension = latent_dimension
        self._generator_output_shape = output_shape
        self._generator_activation_function = activation_function
        self._generator_last_layer_activation = last_layer_activation
        self._generator_dropout_decay_rate_g = dropout_decay_rate_g
        self._generator_dense_layer_sizes_g = dense_layer_sizes_g
        self._generator_dataset_type = dataset_type
        self._generator_initializer_mean = initializer_mean
        self._generator_initializer_deviation = initializer_deviation
        self._generator_number_samples_per_class = number_samples_per_class
        self._generator_model_dense = None
        self._generator_model_with_labels = None

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_gradient_penalty(self, real_samples: torch.Tensor,
                                 fake_samples: torch.Tensor,
                                 labels: torch.Tensor,
                                 lambda_gp: float = 10.0) -> torch.Tensor:
        """
        Computes the gradient penalty for WGAN-GP training.

        The gradient penalty enforces the Lipschitz constraint by penalizing
        the critic when the gradient norm deviates from 1 on interpolated samples.

        Args:
            real_samples: (batch, output_shape) real data samples
            fake_samples: (batch, output_shape) generated samples
            labels: (batch, number_classes) label conditioning
            lambda_gp: Weight for the gradient penalty term

        Returns:
            Gradient penalty loss value
        """
        if self._discriminator_model_with_labels is None:
            raise ValueError("Critic has not been built. Call get_discriminator() first.")

        batch_size = real_samples.size(0)
        device = real_samples.device

        # Random weight for interpolation
        alpha = torch.rand(batch_size, 1, device=device)
        alpha = alpha.expand_as(real_samples)

        # Interpolated samples
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)

        # Critic scores for interpolated samples
        critic_interpolates = self._discriminator_model_with_labels(interpolates, labels)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=critic_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(critic_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Flatten gradients
        gradients = gradients.view(batch_size, -1)

        # Compute gradient penalty
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()

        return gradient_penalty

    def train_step(self, real_samples: torch.Tensor, labels: torch.Tensor,
                   optimizer_g: torch.optim.Optimizer,
                   optimizer_d: torch.optim.Optimizer,
                   n_critic: int = 5,
                   lambda_gp: float = 10.0) -> Tuple[float, float, float]:
        """
        Performs one training step for the WGAN-GP.

        Args:
            real_samples: (batch, output_shape) real data samples
            labels: (batch, number_classes) label conditioning
            optimizer_g: Optimizer for the generator
            optimizer_d: Optimizer for the critic
            n_critic: Number of critic updates per generator update
            lambda_gp: Weight for gradient penalty

        Returns:
            Tuple of (critic_loss, generator_loss, gradient_penalty)
        """
        batch_size = real_samples.size(0)
        device = real_samples.device

        # Train Critic
        for _ in range(n_critic):
            optimizer_d.zero_grad()

            # Generate fake samples
            z = torch.randn(batch_size, self._generator_latent_dimension, device=device)
            fake_samples = self._generator_model_with_labels(z, labels)

            # Critic scores
            real_validity = self._discriminator_model_with_labels(real_samples, labels)
            fake_validity = self._discriminator_model_with_labels(fake_samples.detach(), labels)

            # Wasserstein loss
            critic_loss = fake_validity.mean() - real_validity.mean()

            # Gradient penalty
            gp = self.compute_gradient_penalty(real_samples, fake_samples, labels, lambda_gp)

            # Total critic loss
            total_critic_loss = critic_loss + gp
            total_critic_loss.backward()
            optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()

        # Generate fake samples
        z = torch.randn(batch_size, self._generator_latent_dimension, device=device)
        fake_samples = self._generator_model_with_labels(z, labels)

        # Generator loss (maximize critic score for fake samples)
        fake_validity = self._discriminator_model_with_labels(fake_samples, labels)
        generator_loss = -fake_validity.mean()

        generator_loss.backward()
        optimizer_g.step()

        return critic_loss.item(), generator_loss.item(), gp.item()

    def generate_samples(self, num_samples: int, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generates samples using the trained generator.

        Args:
            num_samples: Number of samples to generate
            labels: (num_samples, number_classes) label conditioning (optional)

        Returns:
            Generated samples of shape (num_samples, output_shape)
        """
        if self._generator_model_with_labels is None:
            raise ValueError("Generator has not been built. Call get_generator() first.")

        device = self.device

        # Generate random latent vectors
        z = torch.randn(num_samples, self._generator_latent_dimension, device=device)

        # If no labels provided, generate random labels
        if labels is None:
            if self._generator_number_samples_per_class is not None:
                num_classes = self._generator_number_samples_per_class["number_classes"]
                random_labels = torch.randint(0, num_classes, (num_samples,), device=device)
                labels = torch.eye(num_classes, device=device)[random_labels]
            else:
                raise ValueError("Labels must be provided if number_samples_per_class is not set.")

        # Generate samples
        self._generator_model_with_labels.eval()
        with torch.no_grad():
            generated = self._generator_model_with_labels(z, labels)

        return generated

    # Unified setters for both generator and critic
    def set_latent_dimension(self, latent_dimension: int) -> None:
        """Sets the latent dimension for both the discriminator and generator."""
        if latent_dimension <= 0:
            raise ValueError("Latent dimension must be a positive integer.")

        self._discriminator_latent_dimension = latent_dimension
        self._generator_latent_dimension = latent_dimension

    def set_output_shape(self, output_shape: int) -> None:
        """Sets the output shape for both the discriminator and generator."""
        if output_shape <= 0:
            raise ValueError("Output shape must be a positive integer.")

        self._discriminator_output_shape = output_shape
        self._generator_output_shape = output_shape

    def set_activation_function(self, activation_function: str) -> None:
        """Sets the activation function for both the discriminator and generator."""
        if not isinstance(activation_function, str):
            raise ValueError("Activation function must be a string.")

        self._discriminator_activation_function = activation_function
        self._generator_activation_function = activation_function

    def set_last_layer_activation(self, last_layer_activation: str) -> None:
        """Sets the last layer activation for both the discriminator and generator."""
        if not isinstance(last_layer_activation, str):
            raise ValueError("Last layer activation must be a string.")

        self._discriminator_last_layer_activation = last_layer_activation
        self._generator_last_layer_activation = last_layer_activation

    def set_dataset_type(self, dataset_type: torch.dtype) -> None:
        """Sets the data type for the input dataset for both networks."""
        self._discriminator_dataset_type = dataset_type
        self._generator_dataset_type = dataset_type

    def set_initializer_mean(self, initializer_mean: float) -> None:
        """Sets the mean value for the weights initializer for both networks."""
        if not isinstance(initializer_mean, (int, float)):
            raise ValueError("Initializer mean must be a numerical value.")

        self._discriminator_initializer_mean = initializer_mean
        self._generator_initializer_mean = initializer_mean

    def set_initializer_deviation(self, initializer_deviation: float) -> None:
        """Sets the deviation for the weights initializer for both networks."""
        if not isinstance(initializer_deviation, (int, float)) or initializer_deviation <= 0:
            raise ValueError("Initializer deviation must be a positive numerical value.")

        self._discriminator_initializer_deviation = initializer_deviation
        self._generator_initializer_deviation = initializer_deviation

    # Properties
    @property
    def generator(self) -> Optional[nn.Module]:
        """Returns the generator model."""
        return self._generator_model_with_labels

    @property
    def critic(self) -> Optional[nn.Module]:
        """Returns the critic (discriminator) model."""
        return self._discriminator_model_with_labels

    @property
    def wgan_latent_dimension(self) -> int:
        """Returns the latent dimension."""
        return self._generator_latent_dimension

    @property
    def wgan_output_shape(self) -> int:
        """Returns the output shape."""
        return self._generator_output_shape