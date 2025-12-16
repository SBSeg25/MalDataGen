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

class RMSProp(Optimizer):
    """Optimizer that implements the RMSprop (Root Mean Square Propagation) algorithm.

    RMSprop is an adaptive learning rate optimization method that maintains a moving
    average of the squared gradients for each parameter. It divides the learning rate
    by the root of this average, which helps to scale the learning rate appropriately
    for each parameter.

    Mathematical Formulation:

        The standard RMSprop update rule is:

        v_t = ρ * v_{t-1} + (1 - ρ) * g_t^2
        w_{t+1} = w_t - lr * g_t / (sqrt(v_t) + ε)

    where:
        - w_t is the parameter at time step t
        - g_t is the gradient at time step t
        - v_t is the moving average of squared gradients
        - ρ is the discounting factor (rho)
        - lr is the learning rate
        - ε is a small constant for numerical stability

    When momentum is used (momentum > 0), the update becomes:

        m_t = momentum * m_{t-1} + lr * g_t / (sqrt(v_t) + ε)
        w_{t+1} = w_t - m_t

    For centered RMSprop (centered=True), the update includes a moving average of gradients:

        avg_grad_t = ρ * avg_grad_{t-1} + (1 - ρ) * g_t
        v_t = ρ * v_{t-1} + (1 - ρ) * g_t^2
        denominator = sqrt(v_t - (avg_grad_t)^2 + ε)
        w_{t+1} = w_t - lr * g_t / denominator

    Args:
        learning_rate:
            A float, a keras.optimizers.schedules.LearningRateSchedule instance, or a callable that returns
            the actual value for the learning rate. Default is 0.001.
        rho:
            Discounting factor for the old gradients (float between 0 and 1). Default is 0.9.
        momentum:
            Momentum factor (float between 0 and 1). If not 0.0, the optimizer tracks momentum value with
            decay rate equal to 1 - momentum. Default is 0.0.
        epsilon:
            Small constant for numerical stability. Default is 1e-7.
        centered:
            Boolean. If True, gradients are normalized by the estimated variance of the gradient
            (centered RMSprop). If False, by the uncentered second moment. Setting to True may help with
            training but is more computationally expensive. Default is False.
        weight_decay:
            Optional weight decay (L2 regularization) parameter. Default is None.
        clipnorm:
            Optional gradient clipping by norm. Default is None.
        clipvalue:
            Optional gradient clipping by value. Default is None.
        global_clipnorm:
            Optional global gradient clipping by norm. Default is None.
        use_ema:
            Boolean to enable Exponential Moving Average of gradients. Default is False.
        ema_momentum:
            Float, the momentum for EMA. Default is 0.99.
        ema_overwrite_frequency:
            Optional frequency for EMA overwrite. Default is None.
        loss_scale_factor:
            Optional scaling factor for the loss. Default is None.
        gradient_accumulation_steps:
            Optional number of gradient accumulation steps. Default is None.
        name:
            The name of the optimizer. Default is 'rmsprop'.

    References:
        - [Hinton, 2012] (http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
        - "Divide the gradient by a running average of its recent magnitude" - Hinton's original RMSprop proposal
        - "Neural Networks for Machine Learning" course (Coursera) by Geoffrey Hinton

# This implementation is adapted from the original Keras source code,
# available at: https://github.com/keras-team/keras
# It has been modified for customization and integration into this specific context.

    Example:
        >>> python3
        ...     #Standard RMSprop
        ...     optimizer = RMSprop(learning_rate=0.01, rho=0.9)
        ...     # RMSprop with momentum
        ...     optimizer = RMSprop(learning_rate=0.01, rho=0.9, momentum=0.9)
        ...     # Centered RMSprop
        ...     optimizer = RMSprop(learning_rate=0.01, rho=0.9, centered=True)
        ...     # Update model parameters
        ...     for gradient, variable in zip(gradients, variables):
        >>>     optimizer.update_step(gradient, variable, learning_rate)


    """

    def __init__(self,
                 learning_rate=0.001,
                 rho=0.9,
                 momentum=0.0,
                 epsilon=1e-7,
                 centered=False,
                 weight_decay=None,
                 clipnorm=None,
                 clipvalue=None,
                 global_clipnorm=None,
                 use_ema=False,
                 ema_momentum=0.99,
                 ema_overwrite_frequency=None,
                 loss_scale_factor=None,
                 gradient_accumulation_steps=None,
                 name="rmsprop",
                 **kwargs,):

        """Initialize the RMSprop optimizer.

        Args:
            learning_rate: A float or callable, the learning rate value.
            rho: Float, the discounting factor for old gradients (between 0 and 1).
            momentum: Float, the momentum factor (between 0 and 1).
            epsilon: Small positive float for numerical stability.
            centered: Boolean, whether to use centered RMSprop.
            weight_decay: Optional float for L2 regularization.
            clipnorm: Optional, gradient clipping by norm.
            clipvalue: Optional, gradient clipping by value.
            global_clipnorm: Optional, global gradient clipping.
            use_ema: Boolean, whether to use Exponential Moving Average of gradients.
            ema_momentum: Float, the momentum for EMA.
            ema_overwrite_frequency: Optional, the frequency of EMA overwriting.
            loss_scale_factor: Optional scaling factor for the loss.
            gradient_accumulation_steps: Optional number of gradient accumulation steps.
            name: The optimizer's name.
        """
        super().__init__(learning_rate=learning_rate,
                         weight_decay=weight_decay,
                         clipnorm=clipnorm,
                         clipvalue=clipvalue,
                         global_clipnorm=global_clipnorm,
                         use_ema=use_ema,
                         ema_momentum=ema_momentum,
                         ema_overwrite_frequency=ema_overwrite_frequency,
                         loss_scale_factor=loss_scale_factor,
                         gradient_accumulation_steps=gradient_accumulation_steps,
                         name=name, **kwargs,)

        # Validate hyperparameters
        if not isinstance(rho, float) or not (0 <= rho <= 1):
            raise ValueError("rho must be a float between [0, 1].")

        if not isinstance(momentum, float) or not (0 <= momentum <= 1):
            raise ValueError("momentum must be a float between [0, 1].")

        if not isinstance(epsilon, float) or epsilon <= 0:
            raise ValueError("epsilon must be a positive float.")

        self.rho = rho
        self.momentum = momentum
        self.epsilon = epsilon
        self.centered = centered

    def build(self, var_list):
        """Initialize optimizer variables.

        RMSprop requires:
            - velocities (moving average of squared gradients) for each parameter
            - momentums (if momentum > 0)
            - average gradients (if centered=True)

        Args:
            var_list: List of model variables to build RMSprop variables for.
        """
        if self.built:
            return

        super().build(var_list)

        # Initialize velocities (moving average of squared gradients)
        self._velocities = []

        for var in var_list:
            self._velocities.append(self.add_variable_from_reference(var, "velocity"))

        # Initialize momentums if momentum is used
        self._momentums = []

        if self.momentum > 0:

            for var in var_list:
                self._momentums.append(self.add_variable_from_reference(var, "momentum"))

        # Initialize average gradients if centered RMSprop is used
        self._average_gradients = []

        if self.centered:
            for var in var_list:
                self._average_gradients.append(self.add_variable_from_reference(var, "average_gradient"))

    def update_step(self, gradient, variable, learning_rate):
        """Perform the optimization step to update the model parameters.

        This method updates the model variable using the RMSprop algorithm,
        with optional momentum and centering.

        Args:
            gradient: The gradient of the loss function with respect to the variable.
            variable: The model variable being updated.
            learning_rate: The current learning rate.
        """
        lr = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)

        # Retrieve stored variables
        velocity = self._velocities[self._get_variable_index(variable)]
        momentum = None

        if self.momentum > 0:
            momentum = self._momentums[self._get_variable_index(variable)]

        average_grad = None

        if self.centered:
            average_grad = self._average_gradients[self._get_variable_index(variable)]

        rho = self.rho

        # Update velocity (exponentially weighted average of squared gradients)
        self.assign(velocity, ops.add(ops.multiply(rho, velocity), ops.multiply(1 - rho, ops.square(gradient)), ), )

        # Update average gradient if using centered RMSprop
        if self.centered:
            self.assign(average_grad, ops.add(ops.multiply(rho, average_grad), ops.multiply(1 - rho, gradient), ), )
            denominator = ops.subtract(velocity, ops.square(average_grad)) + self.epsilon

        else:
            denominator = ops.add(velocity, self.epsilon)

        # Compute the update increment
        increment = ops.divide(ops.multiply(lr, gradient), ops.sqrt(denominator))

        # Apply momentum if enabled
        if self.momentum > 0:
            self.assign(momentum, ops.add(ops.multiply(self.momentum, momentum), increment), )
            self.assign_sub(variable, momentum)

        else:
            self.assign_sub(variable, increment)

    def get_config(self):
        """Returns the configuration of the optimizer as a dictionary.

        The configuration is used to recreate the optimizer instance.

        Returns:
            A dictionary containing the configuration of the optimizer.
        """
        config = super().get_config()
        config.update(
            {
                "rho": self.rho,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "centered": self.centered,
            }
        )
        return config

    @property
    def rho(self) -> float:
        """Gets the discounting factor for the gradient moving average.

        Controls the decay rate of historical gradient information.
        Typical values are between 0.9 and 0.99.
        Higher values give more weight to past gradients.

        Returns:
            float: Current rho value between [0, 1]
        """
        return self._rho

    @rho.setter
    def rho(self, value: float) -> None:
        """Sets the discounting factor for gradient history.

        Args:
            value: New rho value (must be between 0 and 1)

        Raises:
            ValueError: If value is not in range [0, 1]
        """
        if not isinstance(value, float) or not (0 <= value <= 1):
            raise ValueError("rho must be a float in range [0, 1]")

        self._rho = value

    @property
    def momentum(self) -> float:
        """Gets the momentum factor for parameter updates.

        When > 0, accelerates updates in the direction of persistent gradient.
        Typical values are between 0.0 and 0.9.
        Set to 0.0 to disable momentum.

        Returns:
            float: Current momentum value between [0, 1]
        """
        return self._momentum

    @momentum.setter
    def momentum(self, value: float) -> None:
        """Sets the momentum factor for parameter updates.

        Args:
            value: New momentum value (must be between 0 and 1)

        Raises:
            ValueError: If value is not in range [0, 1]
        """
        if not isinstance(value, float) or not (0 <= value <= 1):
            raise ValueError("momentum must be a float in range [0, 1]")

        self._momentum = value

    @property
    def epsilon(self) -> float:
        """Gets the numerical stability constant.

        Small value added to prevent division by zero in updates.
        Typical values between 1e-7 and 1e-8.
        Usually doesn't need tuning.

        Returns:
            float: Current epsilon value (positive)
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        """Sets the numerical stability term.

        Args:
            value: New epsilon value (must be positive)

        Raises:
            ValueError: If value is not positive
        """
        if not isinstance(value, float) or value <= 0:
            raise ValueError("epsilon must be a positive float")

        self._epsilon = value

    @property
    def centered(self) -> bool:
        """Gets whether centered RMSprop is enabled.

        When True, normalizes gradients by estimated variance.
        When False, uses uncentered second moment.
        Centered version often works better but is more expensive.

        Returns:
            bool: True if centered RMSprop is enabled
        """
        return self._centered

    @centered.setter
    def centered(self, value: bool) -> None:
        """Enables/disables centered RMSprop.

        Args:
            value: Whether to use centered RMSprop

        Raises:
            ValueError: If value is not a boolean
        """
        if not isinstance(value, bool):
            raise ValueError("centered must be a boolean")

        self._centered = value
