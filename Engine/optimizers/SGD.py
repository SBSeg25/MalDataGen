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

    from tensorflow.keras import ops

    from tensorflow.keras.optimizers import Optimizer

except ImportError as error:
    print(error)
    sys.exit(-1)


class SGD(Optimizer):
    """Stochastic Gradient Descent (SGD) optimizer with optional momentum and Nesterov acceleration.

     The Stochastic Gradient Descent (SGD) algorithm is an optimization method used to minimize the
     loss function by iteratively adjusting the model parameters in the direction of the negative gradient.

     **Mathematical Formulation:**
     When momentum is 0 (standard SGD), the update rule for a parameter `w` at time `t` is:

     w_{t+1} = w_t - learning_rate * g_t

     where:
         - `w_t` is the parameter at time step `t`.
         - `g_t` is the gradient at time step `t`.
         - `learning_rate` is the step size.

     With momentum (momentum > 0), the update rule becomes:

        - v_{t+1} = momentum * v_t - learning_rate * g_t
        - w_{t+1} = w_t + v_{t+1}

     where:
         - `v_t` is the velocity (momentum term) at time step `t`.
         - `momentum` is the momentum coefficient, a float in the range [0, 1].

     If Nesterov momentum is enabled (`nesterov=True`), the update rule is modified as:

        - v_{t+1} = momentum * v_t - learning_rate * g_t
        - w_{t+1} = w_t + momentum * v_{t+1} - learning_rate * g_t

     Args:
         learning_rate:
            A float, a `keras.optimizers.schedules.LearningRateSchedule` instance, or a callable that returns
            the actual value for the learning rate. Default is 0.01.
         momentum:
            A float between 0 and 1 that accelerates gradient descent in the relevant direction and dampens
            oscillations. Default is 0.0.
         nesterov:
            Boolean indicating whether to apply Nesterov momentum. Default is False. weight_decay: Optional
            weight decay (L2 regularization) parameter. Default is None.
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
            The name of the optimizer. Default is 'SGD'.
# This implementation is adapted from the original Keras source code,
# available at: https://github.com/keras-team/keras
# It has been modified for customization and integration into this specific context.

     Example:
        >>> python3
        ...     # Instantiate an SGD optimizer with momentum and Nesterov acceleration
        ...     optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        ...     # Update the model parameters using the optimizer
        ...     for gradient, variable in zip(gradients, variables):
        >>>     optimizer.update_step(gradient, variable, learning_rate)
         ```

     References:
         - "A Method for Constrained Optimization Problems" by Polyak, B. T. (1964).
         - "On the importance of single directions in stochastic optimization" by Nesterov, Y. (1983).
     """

    def __init__(self,
                 learning_rate=0.01,
                 momentum=0.0,
                 nesterov=False,
                 weight_decay=None,
                 clipnorm=None,
                 clipvalue=None,
                 global_clipnorm=None,
                 use_ema=False,
                 ema_momentum=0.99,
                 ema_overwrite_frequency=None,
                 loss_scale_factor=None,
                 gradient_accumulation_steps=None,
                 name="SGD", **kwargs,):

        super().__init__(learning_rate=learning_rate,
                         name=name,
                         weight_decay=weight_decay,
                         clipnorm=clipnorm,
                         clipvalue=clipvalue,
                         global_clipnorm=global_clipnorm,
                         use_ema=use_ema,
                         ema_momentum=ema_momentum,
                         ema_overwrite_frequency=ema_overwrite_frequency,
                         loss_scale_factor=loss_scale_factor,
                         gradient_accumulation_steps=gradient_accumulation_steps,
                         **kwargs,)

        if not isinstance(momentum, float) or momentum < 0 or momentum > 1:
            raise ValueError("`momentum` must be a float between [0, 1].")

        self.momentum = momentum
        self.nesterov = nesterov

    def build(self, variables):
        """
        Initializes optimizer-specific variables.

        This method sets up the required variables for the SGD optimizer. If momentum
        is enabled (i.e., `self.momentum` > 0), it creates a momentum variable for each
        trainable model variable. These momentum variables help in accelerating gradient
        descent and reducing oscillations.

        Args:
            variables (list): A list of model variables (e.g., weights) that the optimizer
                will update during training.

        Example:
            >>> optimizer = SGD(learning_rate=0.01, momentum=0.9)
            >>> model_variables = [tf.Variable(1.0), tf.Variable(2.0)]
            >>> optimizer.build(model_variables)
            >>> print(len(optimizer.momentums))  # If momentum != 0, should match len(model_variables)
            2
        """
        # Check if the optimizer has already been built to avoid redundant initialization
        if self.built:
            return

        # Call the parent class's build method to handle generic initialization
        super().build(variables)

        # Initialize an empty list to store momentum variables (if applicable)
        self.momentums = []

        # Only create momentum variables if momentum is greater than zero
        if self.momentum != 0:

            # Create a new momentum variable linked to the corresponding model variable
            for variable in variables:
                self.momentums.append(self.add_variable_from_reference(reference_variable=variable, name="momentum"))

    def update_step(self, gradient, variable, learning_rate):
        """
        Performs a single step of parameter update for the given model variable using gradient descent (with optional momentum).

        The update rule follows one of the three options depending on the settings for momentum and nesterov:

        1. **Vanilla Gradient Descent (No Momentum)**:
            w = w - learning_rate * g

        2. **Gradient Descent with Momentum** (momentum > 0):
            v = momentum * v - learning_rate * g
            w = w + v

        3. **Nesterov Accelerated Gradient (if nesterov=True)**:
            v = momentum * v - learning_rate * g
            w = w + momentum * v - learning_rate * g

        Args:
            gradient: The computed gradient for the model variable.
            variable: The model variable to be updated.
            learning_rate: The learning rate, typically a scalar value that controls the size of the update.

        In this method:
            - If momentum is enabled, it updates the variable using the momentum mechanism.
            - If nesterov is enabled, the update uses the Nesterov accelerated gradient formula.
            - If momentum is disabled, it applies the vanilla gradient descent update.

        The method updates the variable in place by modifying the values stored in `self.momentums`
        if momentum is being used.

        """

        # Cast learning rate and gradient to the same data type as the variable
        learning_rate = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)

        # Initialize momentum value (m) to None
        momentum_vector = None

        # Check if momentum is not zero, and retrieve the momentum for the variable if applicable
        if self.momentum != 0: # Momentum is enabled
            momentum_vector = self.momentums[self._get_variable_index(variable)]

        # Momentum is enabled
        if momentum_vector is not None:

            # Cast momentum to the same type as the variable
            momentum = ops.cast(self.momentum, variable.dtype)

            # Update momentum: m = momentum * m - learning_rate * gradient
            self.assign(
                momentum_vector,
                ops.subtract(
                    ops.multiply(momentum_vector, momentum),
                    ops.multiply(gradient, learning_rate),
                ),
            )

            # If Nesterov momentum is enabled
            if self.nesterov:

                # Update variable with Nesterov acceleration: w = w + momentum * m - learning_rate * gradient
                self.assign_add(
                    variable,
                    ops.subtract(
                        ops.multiply(momentum_vector, momentum),
                        ops.multiply(gradient, learning_rate),
                    ),
                )
            else:
                # Standard momentum update: w = w + m
                self.assign_add(variable, momentum_vector)

        else: # No momentum, standard gradient descent

            # Vanilla gradient descent: w = w - learning_rate * gradient
            self.assign_sub(variable, ops.multiply(gradient, learning_rate))

    def get_config(self):
        """
        Returns the configuration of the optimizer as a dictionary.

        This method is useful for saving and restoring the optimizer state. It retrieves
        the base configuration from the parent class and updates it with additional
        hyperparameters specific to SGD, such as `momentum` and `nesterov`.

        Returns:
            dict: A dictionary containing the configuration of the optimizer, including:
                - "momentum" (float): The momentum coefficient used in updates.
                - "nesterov" (bool): Whether Nesterov momentum is enabled.

        Example:
            >>> optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
            >>> config = optimizer.get_config()
            >>> print(config)
            {'learning_rate': 0.01, 'momentum': 0.9, 'nesterov': True, ...}
        """
        # Retrieve the base optimizer configuration
        config = super().get_config()

        # Update the configuration with SGD-specific hyperparameters
        config.update(
            {
                "momentum": self.momentum,  # Stores the momentum value
                "nesterov": self.nesterov,  # Indicates whether Nesterov acceleration is used
            }
        )

        # Return the complete configuration dictionary
        return config

    @property
    def momentum(self) -> float:
        """Gets the momentum factor for parameter updates.

        Controls how much of the previous update vector is retained.
        Value of 0.0 corresponds to standard SGD without momentum.
        Typical values are between 0.5 and 0.99.

        Returns:
            float: Current momentum value between [0, 1)
        """
        return self._momentum

    @momentum.setter
    def momentum(self, value: float) -> None:
        """Sets the momentum factor for parameter updates.

        Args:
            value: New momentum value (must be between 0 and 1)

        Raises:
            ValueError: If value is not in range [0, 1)
        """
        if not isinstance(value, (float, int)) or not (0 <= value < 1):
            raise ValueError("momentum must be a float in range [0, 1)")

        self._momentum = float(value)

    @property
    def nesterov(self) -> bool:
        """Gets whether Nesterov momentum is enabled.

        When True, applies Nesterov accelerated gradient.
        When False, uses standard momentum.
        Nesterov momentum often provides better convergence.

        Returns:
            bool: True if Nesterov momentum is enabled
        """
        return self._nesterov

    @nesterov.setter
    def nesterov(self, value: bool) -> None:
        """Enables/disables Nesterov momentum.

        Args:
            value: Whether to use Nesterov momentum

        Raises:
            ValueError: If value is not a boolean
        """
        if not isinstance(value, bool):
            raise ValueError("nesterov must be a boolean")
        self._nesterov = value
