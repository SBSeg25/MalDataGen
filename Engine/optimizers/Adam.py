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

    from tensorflow.keras import ops

    from tensorflow.keras.optimizers import Optimizer

except ImportError as error:
    print(error)
    sys.exit(-1)

class Adam(Optimizer):
    """
    Optimizer implementing the Adam (Adaptive Moment Estimation) algorithm.

    Adam is a gradient-based optimization algorithm that computes adaptive learning rates for each parameter
    by considering both the first moment (mean) and the second moment (uncentered variance) of the gradients.

    The update rule for parameters is as follows:

    1. Update the first moment estimate (m_t):
        m_t = β₁ * m_(t-1) + (1 - β₁) * g_t

    2. Update the second moment estimate (v_t):
        v_t = β₂ * v_(t-1) + (1 - β₂) * g_t²

    3. Bias correction for the first moment:
        m̂_t = m_t / (1 - β₁^t)

    4. Bias correction for the second moment:
        v̂_t = v_t / (1 - β₂^t)

    5. Update the parameters (θ):
        θ_t = θ_(t-1) - α * m̂_t / (√(v̂_t) + ε)

    Where:

        - g_t is the gradient at time step t.
        - m_t and v_t are the first and second moment estimates.
        - β₁ and β₂ are the decay rates for the first and second moment estimates.
        - α is the learning rate.
        - ε is a small constant for numerical stability.

    The main advantage of Adam is that it combines the best properties of AdaGrad and RMSProp, making it suitable
    for many deep learning problems, while being computationally efficient with low memory requirements.

    Args:
    ----
        learning_rate (float or callable): The learning rate. Default is 0.001.
        beta_1 (float or callable): The exponential decay rate for the first moment estimate. Default is 0.9.
        beta_2 (float or callable): The exponential decay rate for the second moment estimate. Default is 0.999.
        epsilon (float): Small constant for numerical stability. Default is 1e-7.
        amsgrad (bool): Whether to apply the AMSGrad variant of Adam. Default is False.

    Reference:
    ---------

        Kingma, D.P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization.
        *International Conference on Learning Representations (ICLR)*.
        URL: https://arxiv.org/abs/1412.6980
        # This implementation is adapted from the original Keras source code,
        # available at: https://github.com/keras-team/keras
        # It has been modified for customization and integration into this specific context.

    Example:
        >>> python3
        ...     optimizer = Adam(learning_rate=0.001)
        >>>     model.compile(optimizer=optimizer, loss='mse')

    """


    def __init__(
            self,
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=False,
            **kwargs,
    ):
        """
        Initialize the Adam optimizer with the given parameters.

        Args:
            learning_rate (float): The learning rate to be used during optimization.
            beta_1 (float): The decay rate for the first moment estimate.
            beta_2 (float): The decay rate for the second moment estimate.
            epsilon (float): Constant for numerical stability.
            amsgrad (bool): Whether to use the AMSGrad variant of Adam.
        """
        super().__init__(learning_rate=learning_rate, **kwargs)
        self.beta_1 = beta_1  # Decay rate for the first moment estimate
        self.beta_2 = beta_2  # Decay rate for the second moment estimate
        self.epsilon = epsilon  # Numerical stability constant
        self.amsgrad = amsgrad  # Whether to apply AMSGrad

    def build(self, var_list):
        """
        Initialize the optimizer's variables (moments and velocities).

        Args:
            var_list (list): List of model variables to be optimized.
        """
        if self.built:
            return
        super().build(var_list)
        # Initialize momentums (m_t) and velocities (v_t) for each variable
        self._momentums = [self.add_variable_from_reference(var, "momentum") for var in var_list]
        self._velocities = [self.add_variable_from_reference(var, "velocity") for var in var_list]
        if self.amsgrad:
            # If AMSGrad is enabled, also initialize velocity_hats (v̂_t)
            self._velocity_hats = [self.add_variable_from_reference(var, "velocity_hat") for var in var_list]

    def update_step(self, gradient, variable, learning_rate):
        """
        Perform a parameter update based on the computed gradient.

        Args:
            gradient (tensor): The computed gradient for the variable.
            variable (tensor): The model parameter to be updated.
            learning_rate (float): The current learning rate.
        """
        lr = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)

        # Current time step
        local_step = ops.cast(self.iterations + 1, variable.dtype)
        beta_1_power = ops.power(ops.cast(self.beta_1, variable.dtype), local_step)  # β₁^t
        beta_2_power = ops.power(ops.cast(self.beta_2, variable.dtype), local_step)  # β₂^t

        # Retrieve the first (m_t) and second (v_t) moment estimates for the variable
        momentum_list = self._momentums[self._get_variable_index(variable)]
        velocity_list = self._velocities[self._get_variable_index(variable)]

        # Compute the bias-corrected first moment (m̂_t)
        alpha = lr * ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        # Update the first moment estimate (m_t)
        self.assign_add(momentum_list, ops.multiply(ops.subtract(gradient, momentum_list), 1 - self.beta_1))

        # Update the second moment estimate (v_t)
        self.assign_add(velocity_list, ops.multiply(ops.subtract(ops.square(gradient), velocity_list), 1 - self.beta_2))

        # If AMSGrad is enabled, maintain the maximum of the second moment estimates
        if self.amsgrad:
            v_hat = self._velocity_hats[self._get_variable_index(variable)]
            self.assign(v_hat, ops.maximum(v_hat, velocity_list))  # Apply the AMSGrad variant
            velocity_list = v_hat

        # Update the parameter using the bias-corrected first and second moment estimates
        self.assign_sub(variable, ops.divide(ops.multiply(momentum_list, alpha), ops.add(ops.sqrt(velocity_list), self.epsilon)))

    def get_config(self):
        """
        Returns the configuration of the optimizer as a dictionary.

        The dictionary contains the parameters used to initialize the optimizer.
        """
        config = super().get_config()
        config.update({
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "epsilon": self.epsilon,
            "amsgrad": self.amsgrad,
        })
        return config

    @property
    def beta_1(self) -> float:
        """Get the exponential decay rate for the first moment estimates.

        This controls how quickly the moving average of gradients decays.
        Typical values are between 0.8 and 0.999.

        Returns:
            float: The decay rate for the first moment estimate.
        """
        return self._beta_1

    @beta_1.setter
    def beta_1(self, value: float) -> None:
        """Set the exponential decay rate for the first moment estimates.

        Args:
            value: The decay rate (must be between 0 and 1).

        Raises:
            ValueError: If value is not in range [0, 1).
        """
        if not isinstance(value, float) or not (0 <= value < 1):
            raise ValueError("beta_1 must be a float in the range [0, 1)")

        self._beta_1 = value

    @property
    def beta_2(self) -> float:
        """Get the exponential decay rate for the second moment estimates.

        This controls how quickly the moving average of squared gradients decays.
        Should be close to 1.0 for stability (typically > 0.9).

        Returns:
            float: The decay rate for the second moment estimate.
        """
        return self._beta_2

    @beta_2.setter
    def beta_2(self, value: float) -> None:
        """Set the exponential decay rate for the second moment estimates.

        Args:
            value: The decay rate (must be between 0 and 1).

        Raises:
            ValueError: If value is not in range [0, 1).
        """
        if not isinstance(value, float) or not (0 <= value < 1):
            raise ValueError("beta_2 must be a float in the range [0, 1)")

        self._beta_2 = value

    @property
    def epsilon(self) -> float:
        """Get the numerical stability constant.

        Small constant to prevent division by zero in the update rule.
        Doesn't need to be tuned but should be non-negative.

        Returns:
            float: The epsilon value for numerical stability.
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        """Set the numerical stability constant.

        Args:
            value: The epsilon value (must be positive).

        Raises:
            ValueError: If value is not positive.
        """
        if not isinstance(value, float) or value <= 0:
            raise ValueError("epsilon must be a positive float")

        self._epsilon = value

    @property
    def amsgrad(self) -> bool:
        """Get whether AMSGrad variant is enabled.

        AMSGrad is a modification of Adam that uses the maximum of past squared gradients
        rather than the exponential average, which can improve convergence.

        Returns:
            bool: True if AMSGrad is enabled, False otherwise.
        """
        return self._amsgrad

    @amsgrad.setter
    def amsgrad(self, value: bool) -> None:
        """Set whether to use the AMSGrad variant.

        Args:
            value: Whether to enable AMSGrad.
        """
        if not isinstance(value, bool):
            raise ValueError("amsgrad must be a boolean")

        self._amsgrad = value
