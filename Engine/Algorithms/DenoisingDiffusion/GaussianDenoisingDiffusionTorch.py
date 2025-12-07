#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/07'
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
    import numpy as np
    import torch
    import torch.nn as nn

except ImportError as error:
    print(error)
    sys.exit(-1)

DEFAULT_DIFFUSION_GAUSSIAN_BETA_START = 1e-4
DEFAULT_DIFFUSION_GAUSSIAN_BETA_END = 0.02
DEFAULT_DIFFUSION_GAUSSIAN_TIME_STEPS = 1000
DEFAULT_DIFFUSION_GAUSSIAN_CLIP_MIN = -1.0
DEFAULT_DIFFUSION_GAUSSIAN_CLIP_MAX = 1.0


class GaussianDiffusionTorch:
    """
    A PyTorch implementation of the Gaussian diffusion process used in diffusion models
    for denoising and generative tasks.

    This implementation follows the method proposed by Ho et al. (2020), where a sequence of
    Gaussian noise is applied iteratively to a data sample, and a neural network is trained
    to reverse this process, allowing the generation of high-quality synthetic samples.

    Reference:
    ----------
        Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models."
        Advances in Neural Information Processing Systems (NeurIPS).

    Mathematical Formalism:
    -----------------------
        The diffusion process follows these key equations:

        1. Forward Process (q):
            q(x_t|x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_tI)
            where β_t is the noise schedule

        2. Cumulative Forward Process:
            q(x_t|x_0) = N(x_t; √(ᾱ_t)x_0, (1-ᾱ_t)I)
            where α_t = 1-β_t and ᾱ_t = ∏_{s=1}^t α_s

        3. Reverse Process (p_θ):
            p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t))

        4. Posterior Distribution (q):
            q(x_{t-1}|x_t,x_0) = N(x_{t-1}; μ̃_t(x_t,x_0), β̃_tI)
            where:
            μ̃_t(x_t,x_0) = (√ᾱ_{t-1}β_t)/(1-ᾱ_t)x_0 + (√α_t(1-ᾱ_{t-1}))/(1-ᾱ_t)x_t
            β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t)β_t

    Attributes:
    -----------
        @beta_start: float
            The starting value of the beta schedule, controlling noise variance.
        @beta_end: float
            The ending value of the beta schedule, defining the final noise level.
        @time_steps: int
            The number of discrete time steps for the diffusion process.
        @clip_min: float
            The minimum value to clip the output during denoising, ensuring numerical stability.
        @clip_max: float
            The maximum value to clip the output during denoising.
        @device: torch.device
            Device on which tensors are stored (CPU or CUDA).

    Derived Attributes:
    -------------------
        @betas : torch.Tensor
            The linearly spaced beta values used in the diffusion process.
        @alphas_cumulative_product: torch.Tensor
            The cumulative product of (1 - beta) over time steps, controlling noise accumulation.
        @alphas_cumulative_product_previous: torch.Tensor
            The cumulative product of (1 - beta) for the previous time step.
        @posterior_variance: torch.Tensor
            Variance of the posterior distribution for each time step, used in reverse diffusion.
        @posterior_log_variance_clipped: torch.Tensor
            Logarithm of the posterior variance, clipped to avoid numerical instability.
        @posterior_mean_first_coefficient: torch.Tensor
            Coefficient for the first term in the posterior mean computation.
        @posterior_mean_second_coefficient: torch.Tensor
            Coefficient for the second term in the posterior mean computation.

    Methods:
    --------
        extract(a, t, x_shape):
            Extracts and reshapes the corresponding values from a tensor for the given time step.
        q_mean_variance(x_start, t):
            Computes the mean, variance, and log variance of the forward process at a given time step.
        q_sample(x_start, t, noise):
            Samples a noisy version of x_start at time step t using Gaussian noise.
        predict_start_from_noise(x_t, t, noise):
            Predicts the original x_start from a noisy sample x_t and noise.
        q_posterior(x_start, x_t, t):
            Computes the mean, variance, and log variance of the posterior distribution.
        p_mean_variance(predicted_noise, x, t, clip_denoised=True):
            Predicts the mean and variance of the model's posterior distribution given predicted noise.
        p_sample(predicted_noise, x, t, clip_denoised=True):
            Samples from the model's posterior distribution at time step t.

    Example:
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> diffusion = GaussianDiffusionTorch(
        ...     beta_start=0.0001,
        ...     beta_end=0.02,
        ...     time_steps=1000,
        ...     clip_min=-1.0,
        ...     clip_max=1.0,
        ...     device=device
        ... )
        >>> noise = torch.randn(1, 32, 32, 3, device=device)
        >>> t = torch.tensor([10], device=device)
        >>> x_t = diffusion.q_sample(x_start=noise, t=t, noise=noise)
        >>> print(x_t.shape)  # Expected output: torch.Size([1, 32, 32, 3])
    """

    def __init__(self,
                 beta_start: float = DEFAULT_DIFFUSION_GAUSSIAN_BETA_START,
                 beta_end: float = DEFAULT_DIFFUSION_GAUSSIAN_BETA_END,
                 time_steps: int = DEFAULT_DIFFUSION_GAUSSIAN_TIME_STEPS,
                 clip_min: float = DEFAULT_DIFFUSION_GAUSSIAN_CLIP_MIN,
                 clip_max: float = DEFAULT_DIFFUSION_GAUSSIAN_CLIP_MAX,
                 device: torch.device = None):
        """
        Initializes the GaussianDiffusionTorch class with the given beta schedule and clipping values.

        Parameters:
        -----------
            beta_start : float
                Starting value for beta.
            beta_end : float
                Ending value for beta.
            time_steps : int
                Number of time steps in the diffusion process.
            clip_min : float
                Minimum value for clipping the denoised output.
            clip_max : float
                Maximum value for clipping the denoised output.
            device : torch.device, optional
                Device for tensor storage (CPU or CUDA). If None, uses CUDA if available.
        """

        self._beta_start = beta_start
        self._beta_end = beta_end
        self._time_steps = time_steps
        self._clip_min = clip_min
        self._clip_max = clip_max

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Generate linearly spaced beta values from beta_start to beta_end
        betas = np.linspace(
            beta_start,
            beta_end,
            time_steps,
            dtype=np.float32
        )

        self._number_time_steps = int(time_steps)

        # Calculate alpha values (1 - beta)
        alphas = 1.0 - betas

        # Compute cumulative product of alphas (ᾱ_t = ∏_{s=1}^t α_s)
        alphas_cumulative_product = np.cumprod(alphas, axis=0)

        # Compute cumulative product of alphas shifted by one time step (ᾱ_{t-1})
        alphas_cumulative_product_previous = np.append(1.0, alphas_cumulative_product[:-1])

        # Convert all numpy arrays to PyTorch tensors and move to device
        self._betas = torch.tensor(betas, dtype=torch.float32, device=self.device)

        self._alphas_cumulative_product = torch.tensor(
            alphas_cumulative_product, dtype=torch.float32, device=self.device
        )

        self._alphas_cumulative_product_previous = torch.tensor(
            alphas_cumulative_product_previous, dtype=torch.float32, device=self.device
        )

        self._sqrt_alphas_cumulative_product = torch.tensor(
            np.sqrt(alphas_cumulative_product), dtype=torch.float32, device=self.device
        )

        self._sqrt_one_minus_alphas_cumulative_product = torch.tensor(
            np.sqrt(1.0 - alphas_cumulative_product), dtype=torch.float32, device=self.device
        )

        self._log_one_minus_alphas_cumulative_product = torch.tensor(
            np.log(1.0 - alphas_cumulative_product), dtype=torch.float32, device=self.device
        )

        self._sqrt_recip_alphas_cumulative_product = torch.tensor(
            np.sqrt(1.0 / alphas_cumulative_product), dtype=torch.float32, device=self.device
        )

        self._sqrt_recipm1_alphas_cumulative_product = torch.tensor(
            np.sqrt(1.0 / alphas_cumulative_product - 1), dtype=torch.float32, device=self.device
        )

        # Calculate posterior distribution parameters
        _posterior_variance = (
                betas * (1.0 - alphas_cumulative_product_previous) / (1.0 - alphas_cumulative_product)
        )
        self._posterior_variance = torch.tensor(
            _posterior_variance, dtype=torch.float32, device=self.device
        )

        self._posterior_log_variance_clipped = torch.tensor(
            np.log(np.maximum(_posterior_variance, 1e-20)), dtype=torch.float32, device=self.device
        )

        self._posterior_mean_first_coefficient = torch.tensor(
            betas * np.sqrt(alphas_cumulative_product_previous) / (1.0 - alphas_cumulative_product),
            dtype=torch.float32, device=self.device
        )

        self._posterior_mean_second_coefficient = torch.tensor(
            (1.0 - alphas_cumulative_product_previous) * np.sqrt(alphas) / (1.0 - alphas_cumulative_product),
            dtype=torch.float32, device=self.device
        )

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """
        Extracts values from a tensor based on the time index and reshapes them to match the input batch.

        Parameters:
        -----------
            a : torch.Tensor
                Tensor from which values are extracted.
            t : torch.Tensor
                Time step indices.
            x_shape : torch.Size
                Shape of the input tensor.

        Returns:
        --------
            torch.Tensor
                Extracted and reshaped values.
        """
        batch_size = x_shape[0]
        out = a.gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_mean_variance(self, x_start: torch.Tensor, t: torch.Tensor):
        """
        Computes the mean, variance, and log variance of the forward diffusion process at a given time step.

        Parameters:
        -----------
            x_start : torch.Tensor
                Original input data.
            t : torch.Tensor
                Time step indices.

        Returns:
        --------
            tuple of torch.Tensor
                Mean, variance, and log variance of the forward process.
        """
        mean = self._extract(self._sqrt_alphas_cumulative_product, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self._alphas_cumulative_product, t, x_start.shape)
        log_variance = self._extract(self._log_one_minus_alphas_cumulative_product, t, x_start.shape)

        return mean, variance, log_variance

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Samples a noisy version of the input at a given time step.

        Parameters:
        -----------
            x_start : torch.Tensor
                Original input data.
            t : torch.Tensor
                Time step indices.
            noise : torch.Tensor
                Gaussian noise to add.

        Returns:
        --------
            torch.Tensor
                Noisy sample at time step t.
        """
        return (
                self._extract(self._sqrt_alphas_cumulative_product, t, x_start.shape) * x_start +
                self._extract(self._sqrt_one_minus_alphas_cumulative_product, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Predicts the original input x_start given a noisy sample x_t and noise.

        Parameters:
        -----------
            x_t : torch.Tensor
                Noisy input data at time step t.
            t : torch.Tensor
                Time step indices.
            noise : torch.Tensor
                Gaussian noise applied during diffusion.

        Returns:
        --------
            torch.Tensor
                Predicted x_start.
        """
        return (
                self._extract(self._sqrt_recip_alphas_cumulative_product, t, x_t.shape) * x_t -
                self._extract(self._sqrt_recipm1_alphas_cumulative_product, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """
        Computes the posterior mean and variance for the reverse diffusion process.

        Parameters:
        -----------
            x_start : torch.Tensor
                Original input data.
            x_t : torch.Tensor
                Noisy input at time step t.
            t : torch.Tensor
                Time step indices.

        Returns:
        --------
            tuple of torch.Tensor
                Posterior mean, variance, and log variance.
        """
        posterior_mean = (
                self._extract(self._posterior_mean_first_coefficient, t, x_t.shape) * x_start +
                self._extract(self._posterior_mean_second_coefficient, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self._posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self._posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, predicted_noise: torch.Tensor, x: torch.Tensor,
                        t: torch.Tensor, clip_denoised: bool = True):
        """
        Predicts the mean and variance of the model's posterior distribution at time step t.

        Parameters:
        -----------
            predicted_noise : torch.Tensor
                Noise predicted by the model.
            x : torch.Tensor
                Noisy input at time step t.
            t : torch.Tensor
                Time step indices.
            clip_denoised : bool, optional
                Whether to clip the denoised output.

        Returns:
        --------
            tuple of torch.Tensor
                Predicted mean, variance, and log variance.
        """
        x_recon = self.predict_start_from_noise(x, t=t, noise=predicted_noise)

        if clip_denoised:
            x_recon = torch.clamp(x_recon, self._clip_min, self._clip_max)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )

        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, predicted_noise: torch.Tensor, x: torch.Tensor,
                 t: torch.Tensor, clip_denoised: bool = True) -> torch.Tensor:
        """
        Samples from the model's posterior distribution at a given time step.

        Parameters:
        -----------
            predicted_noise : torch.Tensor
                Noise predicted by the model.
            x : torch.Tensor
                Noisy input at time step t.
            t : torch.Tensor
                Time step indices.
            clip_denoised : bool, optional
                Whether to clip the denoised output.

        Returns:
        --------
            torch.Tensor
                Sampled output at time step t.
        """
        model_mean, _, model_log_variance = self.p_mean_variance(
            predicted_noise, x=x, t=t, clip_denoised=clip_denoised
        )

        noise = torch.randn_like(x)

        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))

        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    # Properties with getters and setters
    @property
    def beta_start(self) -> float:
        """Get the starting value of beta for the noise schedule."""
        return self._beta_start

    @beta_start.setter
    def beta_start(self, value: float) -> None:
        """Set the starting value of beta for the noise schedule."""
        if value <= 0:
            raise ValueError("beta_start must be positive")
        if hasattr(self, '_beta_end') and value > self._beta_end:
            raise ValueError("beta_start must be less than or equal to beta_end")
        self._beta_start = value

    @property
    def beta_end(self) -> float:
        """Get the ending value of beta for the noise schedule."""
        return self._beta_end

    @beta_end.setter
    def beta_end(self, value: float) -> None:
        """Set the ending value of beta for the noise schedule."""
        if value <= 0:
            raise ValueError("beta_end must be positive")
        if hasattr(self, '_beta_start') and value < self._beta_start:
            raise ValueError("beta_end must be greater than or equal to beta_start")
        self._beta_end = value

    @property
    def time_steps(self) -> int:
        """Get the number of diffusion time steps."""
        return self._time_steps

    @time_steps.setter
    def time_steps(self, value: int) -> None:
        """Set the number of diffusion time steps."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("time_steps must be a positive integer")
        self._time_steps = value

    @property
    def clip_min(self) -> float:
        """Get the minimum clipping value for the output."""
        return self._clip_min

    @clip_min.setter
    def clip_min(self, value: float) -> None:
        """Set the minimum clipping value for the output."""
        if hasattr(self, '_clip_max') and value >= self._clip_max:
            raise ValueError("clip_min must be less than clip_max")
        self._clip_min = value

    @property
    def clip_max(self) -> float:
        """Get the maximum clipping value for the output."""
        return self._clip_max

    @clip_max.setter
    def clip_max(self, value: float) -> None:
        """Set the maximum clipping value for the output."""
        if hasattr(self, '_clip_min') and value <= self._clip_min:
            raise ValueError("clip_max must be greater than clip_min")
        self._clip_max = value

    def to(self, device: torch.device):
        """
        Move all tensors to the specified device.

        Args:
            device: Target device (e.g., 'cuda' or 'cpu').

        Returns:
            self: Returns self for method chaining.
        """
        self.device = device

        # Move all tensor attributes to the new device
        self._betas = self._betas.to(device)
        self._alphas_cumulative_product = self._alphas_cumulative_product.to(device)
        self._alphas_cumulative_product_previous = self._alphas_cumulative_product_previous.to(device)
        self._sqrt_alphas_cumulative_product = self._sqrt_alphas_cumulative_product.to(device)
        self._sqrt_one_minus_alphas_cumulative_product = self._sqrt_one_minus_alphas_cumulative_product.to(device)
        self._log_one_minus_alphas_cumulative_product = self._log_one_minus_alphas_cumulative_product.to(device)
        self._sqrt_recip_alphas_cumulative_product = self._sqrt_recip_alphas_cumulative_product.to(device)
        self._sqrt_recipm1_alphas_cumulative_product = self._sqrt_recipm1_alphas_cumulative_product.to(device)
        self._posterior_variance = self._posterior_variance.to(device)
        self._posterior_log_variance_clipped = self._posterior_log_variance_clipped.to(device)
        self._posterior_mean_first_coefficient = self._posterior_mean_first_coefficient.to(device)
        self._posterior_mean_second_coefficient = self._posterior_mean_second_coefficient.to(device)

        return self