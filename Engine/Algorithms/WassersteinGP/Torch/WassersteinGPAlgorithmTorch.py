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
    import os
    import sys
    import json
    import numpy

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from abc import ABC
    from typing import Any
    from typing import Callable

except ImportError as error:
    print(error)
    sys.exit(-1)


class WassersteinGPAlgorithmTorch(nn.Module):
    """
    A WassersteinGP Generative Adversarial Network (WassersteinGP GAN) model.

    This class represents a WassersteinGP GAN consisting of a generator and discriminator model.
    It implements the WassersteinGP loss with gradient penalty to improve the training of the discriminator and generator.

    Reference:
        Arjovsky, M., Chintala, S., & Bottou, L. (2017). WassersteinGP GAN.
        In Proceedings of the 34th International Conference on Machine Learning (ICML 2017) (Vol. 70, pp. 214-223).
        http://proceedings.mlr.press/v70/arjovsky17a.html

    Attributes:
        @_generator (nn.Module):
            The generator model responsible for generating synthetic data.
        @_discriminator (nn.Module):
            The discriminator model used to evaluate the authenticity of generated data.
        @_latent_dimension (int):
            The dimension of the latent space from which the generator takes input.
        @_generator_optimizer (Optimizer):
            Optimizer used for training the generator.
        @_discriminator_optimizer (Optimizer):
            Optimizer used for training the discriminator.
        @_generator_loss_fn (function):
            Loss function used for training the generator.
        @_discriminator_loss_fn (function):
            Loss function used for training the discriminator.
        @_latent_mean_distribution (float):
            Mean of the latent space distribution.
        @_latent_stander_deviation (float):
            Standard deviation of the latent space distribution.
        @_smoothing_rate (float):
            Rate for label smoothing applied to the discriminator's true labels.
        @_gradient_penalty_weight (float):
            Weight for the gradient penalty term in the WassersteinGP loss.
        @_discriminator_steps (int):
            Number of discriminator updates per generator update.
        @_file_name_discriminator (str):
            File name for saving/loading the discriminator model.
        @_file_name_generator (str):
            File name for saving/loading the generator model.
        @_models_saved_path (str):
            Path where the models are saved.

    Raises:
        ValueError:
            Raised if:
            - The latent dimension is non-positive.
            - The gradient penalty weight is non-positive.
            - The smoothing rate is outside the valid range (0, 1).
            - The number of discriminator steps is non-positive.

    Example:
        >>> generator = build_generator_model()
        >>> discriminator = build_discriminator_model()
        >>> wgan = WassersteinGPAlgorithm(
        ...     generator_model=generator,
        ...     discriminator_model=discriminator,
        ...     latent_dimension=100,
        ...     generator_loss_fn=generator_loss_fn,
        ...     discriminator_loss_fn=discriminator_loss_fn,
        ...     file_name_discriminator='discriminator_model.pth',
        ...     file_name_generator='generator_model.pth',
        ...     models_saved_path='./models/',
        ...     latent_mean_distribution=0.0,
        ...     latent_stander_deviation=1.0,
        ...     smoothing_rate=0.1,
        ...     gradient_penalty_weight=10.0,
        ...     discriminator_steps=5
        ... )
        >>> wgan.train_step(real_data, batch_size=64)
    """

    def __init__(self,
                 generator_model,
                 discriminator_model,
                 latent_dimension,
                 generator_loss_fn,
                 discriminator_loss_fn,
                 file_name_discriminator,
                 file_name_generator,
                 models_saved_path,
                 latent_mean_distribution,
                 latent_stander_deviation,
                 smoothing_rate,
                 gradient_penalty_weight,
                 discriminator_steps,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        # Initialize instance variables with provided or default values
        self._generator_optimizer = None
        self._discriminator_optimizer = None
        self._generator = generator_model
        self._discriminator = discriminator_model
        self._latent_dimension = latent_dimension
        self._discriminator_loss_fn = discriminator_loss_fn
        self._generator_loss_fn = generator_loss_fn
        self._gradient_penalty_weight = gradient_penalty_weight
        self._smooth_rate = smoothing_rate
        self._latent_mean_distribution = latent_mean_distribution
        self._latent_stander_deviation = latent_stander_deviation
        self._file_name_discriminator = file_name_discriminator
        self._file_name_generator = file_name_generator
        self._models_saved_path = models_saved_path
        self._discriminator_steps = discriminator_steps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compile(self, optimizer_generator, optimizer_discriminator,
                loss_generator, loss_discriminator, *args, **kwargs):
        """
        Compile the WassersteinGP Generative Adversarial Network (WGAN) with custom optimizers and loss functions.

        Args:
            optimizer_generator (torch.optim.Optimizer):
                The optimizer for the generator.
            optimizer_discriminator (torch.optim.Optimizer):
                The optimizer for the discriminator.
            loss_generator (callable):
                The loss function for the generator.
            loss_discriminator (callable):
                The loss function for the discriminator.
            *args:
                Additional positional arguments.
            **kwargs:
                Additional keyword arguments.

        This method compiles the GAN with custom optimizers and loss functions specified as arguments.
        It sets the optimizer and loss for both the generator and discriminator.
        """
        self._discriminator_optimizer = optimizer_discriminator
        self._generator_optimizer = optimizer_generator
        self._discriminator_loss_fn = loss_discriminator
        self._generator_loss_fn = loss_generator

    def gradient_penalty(self, batch_size, real_feature, real_label, synthetic_feature):
        """
        Compute the gradient penalty for the WassersteinGP GAN.

        The gradient penalty is used to enforce the Lipschitz constraint on the discriminator's output.

        Parameters:
            batch_size (int):
                The batch size of the input data.
            real_feature (torch.Tensor):
                Real data features.
            real_label (torch.Tensor):
                Real data labels.
            synthetic_feature (torch.Tensor):
                Synthetic (generated) data features.

        Returns:
            torch.Tensor: The gradient penalty value.
        """
        # Generate random noise for smoothing.
        random_smooth = torch.normal(0.0, 0.1, size=(batch_size, 1)).to(self.device)

        # Calculate the linear distance between real and synthetic features.
        linear_distance = synthetic_feature - real_feature

        # Interpolate between real and synthetic features using the random noise.
        interpolated_feature = real_feature + random_smooth * linear_distance
        interpolated_feature.requires_grad_(True)

        # Get discriminator's output for the interpolated features.
        labels_predicted = self._discriminator(interpolated_feature, real_label)

        # Calculate the gradient of the discriminator's output with respect to the interpolated features.
        gradients = torch.autograd.grad(
            outputs=labels_predicted,
            inputs=interpolated_feature,
            grad_outputs=torch.ones_like(labels_predicted),
            create_graph=True,
            retain_graph=True
        )[0]

        # Compute the gradient magnitude and normalize it.
        gradient_normalized = torch.sqrt(torch.sum(gradients ** 2, dim=1))

        # Calculate the final gradient penalty as the mean squared difference from 1.0 and return.
        gradient_penalty_final = torch.mean((gradient_normalized - 1.0) ** 2)

        return gradient_penalty_final

    def train_step(self, batch):
        """
        Executes one training step for the GAN model.

        This step updates both the discriminator and the generator.
        The discriminator is updated multiple times (controlled by self._discriminator_steps),
        while the generator is updated once.

        Args:
            batch (tuple): A tuple containing:
                - real_feature: A batch of real data samples (features).
                - real_samples_label: Corresponding class labels for each sample.

        Returns:
            dict: Dictionary containing the discriminator and generator loss for the current training step.
        """

        # Unpack batch into features and labels.
        real_feature, real_samples_label = batch
        real_feature = real_feature.to(self.device)
        real_samples_label = real_samples_label.to(self.device)

        batch_size = real_feature.shape[0]

        # Expand label dimensions if needed
        if len(real_samples_label.shape) == 1:
            real_samples_label = real_samples_label.unsqueeze(-1)

        # === Discriminator Training Loop ===
        for _ in range(self._discriminator_steps):
            self._discriminator_optimizer.zero_grad()

            # Generate random noise vectors for the latent space.
            latent_space = torch.normal(
                self._latent_mean_distribution,
                self._latent_stander_deviation,
                size=(batch_size, self._latent_dimension)
            ).to(self.device)

            # Generate synthetic samples from the generator using noise and labels.
            with torch.no_grad():
                synthetic_feature = self._generator(latent_space, real_samples_label)

            # Predict "real/fake" labels using the discriminator for real and synthetic samples.
            label_predicted_real = self._discriminator(real_feature, real_samples_label)
            label_predicted_synthetic = self._discriminator(synthetic_feature, real_samples_label)

            # Compute discriminator loss (real vs fake).
            discriminator_loss_result = self._discriminator_loss_fn(
                real_img=label_predicted_real, fake_img=label_predicted_synthetic)

            # Compute gradient penalty for improved stability (WGAN-GP, etc.).
            gradient_penalty = self.gradient_penalty(batch_size,
                                                     real_feature,
                                                     real_samples_label,
                                                     synthetic_feature)

            # Combine loss with gradient penalty.
            all_discriminator_loss = discriminator_loss_result + gradient_penalty * self._gradient_penalty_weight

            # Compute and apply gradients to update the discriminator's weights.
            all_discriminator_loss.backward()
            self._discriminator_optimizer.step()

        # === Generator Training Step ===
        self._generator_optimizer.zero_grad()

        # Generate fresh random noise vectors for the latent space.
        latent_space = torch.normal(
            self._latent_mean_distribution,
            self._latent_stander_deviation,
            size=(batch_size, self._latent_dimension)
        ).to(self.device)

        # Generate synthetic samples from the generator.
        synthetic_feature = self._generator(latent_space, real_samples_label)

        # Predict "real/fake" labels for synthetic samples using the discriminator.
        predicted_labels = self._discriminator(synthetic_feature, real_samples_label)

        # Compute generator loss (how well generator fools the discriminator).
        all_generator_loss = self._generator_loss_fn(predicted_labels)

        # Compute and apply gradients to update the generator's weights.
        all_generator_loss.backward()
        self._generator_optimizer.step()

        # Return the loss values for monitoring/tracking.
        return {"d_loss": all_discriminator_loss.item(), "g_loss": all_generator_loss.item()}

    def get_samples(self, number_samples_per_class):
        """
        Generates synthetic samples for each specified class using the trained generator.

        This method creates samples conditioned on class labels, using random noise vectors
        and the generator to produce the samples.

        Args:
            number_samples_per_class (dict): A dictionary containing:
                - "classes" (dict): Mapping of class labels to the number of samples to generate for each class.
                - "number_classes" (int): Total number of classes (used for one-hot encoding).

        Returns:
            dict: A dictionary where each key is a class label and the value is an array of generated samples.
        """

        # Dictionary to store generated samples for each class.
        generated_data = {}

        self._generator.eval()
        with torch.no_grad():
            # Loop through each class and the desired number of samples for that class.
            for label_class, number_instances in number_samples_per_class["classes"].items():
                # Create one-hot encoded labels for all samples of the current class.
                label_samples_generated = F.one_hot(
                    torch.tensor([label_class] * number_instances),
                    num_classes=number_samples_per_class["number_classes"]
                ).float().to(self.device)

                # Sample random noise vectors from a normal distribution.
                latent_noise = torch.normal(
                    self._latent_mean_distribution,
                    self._latent_stander_deviation,
                    size=(number_instances, self._latent_dimension)
                ).to(self.device)

                # Generate synthetic samples using the generator.
                generated_samples = self._generator(latent_noise, label_samples_generated)

                # Round the generated samples to integer values
                # (if samples are intended to be binary, e.g., images with pixel values 0 or 1).
                generated_samples = torch.round(generated_samples).cpu().numpy()

                # Store generated samples for the current class.
                generated_data[label_class] = generated_samples

        self._generator.train()
        # Return the dictionary containing generated samples for all requested classes.
        return generated_data

    def save_model(self, directory, file_name):
        """
        Save the generator and discriminator models.

        Args:
            directory (str): Directory where models will be saved.
            file_name (str): Base file name for saving models.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Construct file names for generator and discriminator models
        generator_file_name = os.path.join(directory, f"fold_{file_name}_generator.pth")
        discriminator_file_name = os.path.join(directory, f"fold_{file_name}_discriminator.pth")

        # Save generator model
        torch.save({
            'model_state_dict': self._generator.state_dict(),
            'model_config': str(self._generator)
        }, generator_file_name)

        # Save discriminator model
        torch.save({
            'model_state_dict': self._discriminator.state_dict(),
            'model_config': str(self._discriminator)
        }, discriminator_file_name)

    def load_models(self, directory, file_name):
        """
        Load the generator and discriminator models from a directory.

        Args:
            directory (str): Directory where models are stored.
            file_name (str): Base file name for loading models.
        """

        # Construct file names for generator and discriminator models
        generator_file_name = os.path.join(directory, f"{file_name}_generator.pth")
        discriminator_file_name = os.path.join(directory, f"{file_name}_discriminator.pth")

        # Load the generator and discriminator models
        generator_checkpoint = torch.load(generator_file_name, map_location=self.device)
        self._generator.load_state_dict(generator_checkpoint['model_state_dict'])

        discriminator_checkpoint = torch.load(discriminator_file_name, map_location=self.device)
        self._discriminator.load_state_dict(discriminator_checkpoint['model_state_dict'])

    @property
    def discriminator(self) -> Any:
        """Get the discriminator model instance.

        Returns:
            The discriminator model used in GAN training.
        """
        return self._discriminator

    @discriminator.setter
    def discriminator(self, value: Any) -> None:
        """Set the discriminator model instance.

        Args:
            value: The discriminator model to set.
        """
        self._discriminator = value

    @property
    def latent_dimension(self) -> int:
        """Get the dimension of the latent space.

        Returns:
            The size of the latent space dimension (positive integer).
        """
        return self._latent_dimension

    @latent_dimension.setter
    def latent_dimension(self, value: int) -> None:
        """Set the dimension of the latent space.

        Args:
            value: The latent dimension size (must be positive).

        Raises:
            ValueError: If value is not a positive integer.
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Latent dimension must be a positive integer")
        self._latent_dimension = value

    @property
    def discriminator_loss_fn(self) -> Callable:
        """Get the discriminator loss function.

        Returns:
            The loss function used for discriminator training.
        """
        return self._discriminator_loss_fn

    @discriminator_loss_fn.setter
    def discriminator_loss_fn(self, value: Callable) -> None:
        """Set the discriminator loss function.

        Args:
            value: The loss function to use for discriminator training.
        """
        self._discriminator_loss_fn = value

    @property
    def generator_loss_fn(self) -> Callable:
        """Get the generator loss function.

        Returns:
            The loss function used for generator training.
        """
        return self._generator_loss_fn

    @generator_loss_fn.setter
    def generator_loss_fn(self, value: Callable) -> None:
        """Set the generator loss function.

        Args:
            value: The loss function to use for generator training.
        """
        self._generator_loss_fn = value

    @property
    def gradient_penalty_weight(self) -> float:
        """Get the weight for gradient penalty in WGAN-GP.

        Returns:
            The weight factor for gradient penalty term.
        """
        return self._gradient_penalty_weight

    @gradient_penalty_weight.setter
    def gradient_penalty_weight(self, value: float) -> None:
        """Set the weight for gradient penalty in WGAN-GP.

        Args:
            value: The penalty weight (must be non-negative).

        Raises:
            ValueError: If value is negative.
        """
        if value < 0:
            raise ValueError("Gradient penalty weight cannot be negative")
        self._gradient_penalty_weight = value

    @property
    def smooth_rate(self) -> float:
        """Get the label smoothing rate.

        Returns:
            The rate used for one-sided label smoothing.
        """
        return self._smooth_rate

    @smooth_rate.setter
    def smooth_rate(self, value: float) -> None:
        """Set the label smoothing rate.

        Args:
            value: The smoothing rate (typically between 0 and 0.3).

        Raises:
            ValueError: If value is not between 0 and 1.
        """
        if not 0 <= value <= 1:
            raise ValueError("Smoothing rate must be between 0 and 1")
        self._smooth_rate = value

    @property
    def latent_mean_distribution(self) -> float:
        """Get the mean of the latent space distribution.

        Returns:
            The mean value used for latent space sampling.
        """
        return self._latent_mean_distribution

    @latent_mean_distribution.setter
    def latent_mean_distribution(self, value: float) -> None:
        """Set the mean of the latent space distribution.

        Args:
            value: The mean value for latent distribution.
        """
        self._latent_mean_distribution = value

    @property
    def latent_stander_deviation(self) -> float:
        """Get the standard deviation of the latent space distribution.

        Returns:
            The standard deviation used for latent space sampling.
        """
        return self._latent_stander_deviation

    @latent_stander_deviation.setter
    def latent_stander_deviation(self, value: float) -> None:
        """Set the standard deviation of the latent space distribution.

        Args:
            value: The standard deviation (must be positive).

        Raises:
            ValueError: If value is not positive.
        """
        if value <= 0:
            raise ValueError("Standard deviation must be positive")
        self._latent_stander_deviation = value

    @property
    def file_name_discriminator(self) -> str:
        """Get the discriminator model save filename.

        Returns:
            The filename pattern for saving discriminator models.
        """
        return self._file_name_discriminator

    @file_name_discriminator.setter
    def file_name_discriminator(self, value: str) -> None:
        """Set the discriminator model save filename.

        Args:
            value: The filename pattern to use.
        """
        self._file_name_discriminator = value

    @property
    def file_name_generator(self) -> str:
        """Get the generator model save filename.

        Returns:
            The filename pattern for saving generator models.
        """
        return self._file_name_generator

    @file_name_generator.setter
    def file_name_generator(self, value: str) -> None:
        """Set the generator model save filename.

        Args:
            value: The filename pattern to use.
        """
        self._file_name_generator = value

    @property
    def models_saved_path(self) -> str:
        """Get the path for saving models.

        Returns:
            The directory path where models are saved.
        """
        return self._models_saved_path

    @models_saved_path.setter
    def models_saved_path(self, value: str) -> None:
        """Set the path for saving models.

        Args:
            value: The directory path to use for saving models.
        """
        self._models_saved_path = value

    @property
    def discriminator_steps(self) -> int:
        """Get the number of discriminator steps per iteration.

        Returns:
            The number of discriminator training steps per GAN iteration.
        """
        return self._discriminator_steps

    @discriminator_steps.setter
    def discriminator_steps(self, value: int) -> None:
        """Set the number of discriminator steps per iteration.

        Args:
            value: The number of steps (must be positive integer).

        Raises:
            ValueError: If value is not a positive integer.
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Discriminator steps must be a positive integer")
        self._discriminator_steps = value
