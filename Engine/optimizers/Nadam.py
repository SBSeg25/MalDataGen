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
    from tensorflow.keras import backend

    from tensorflow.keras.optimizers import Optimizer

except ImportError as error:
    print(error)
    sys.exit(-1)

class NADAM(Optimizer):
    """Optimizer that implements the Nadam algorithm (Nesterov-accelerated Adaptive Moment Estimation).

    Nadam combines the benefits of Adam (Adaptive Moment Estimation) with Nesterov momentum,
    typically leading to faster convergence and better optimization performance.

    Mathematical Formulation:

        The Nadam update rule is as follows:

            m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t
            v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2

            u_t = beta_1 * (1 - 0.5 * 0.96^{t})
            u_{t+1} = beta_1 * (1 - 0.5 * 0.96^{t+1})

            m_hat = (u_{t+1} * m_t) / (1 - prod_{i=1}^{t+1} u_i) + ((1 - u_t) * g_t) / (1 - prod_{i=1}^t u_i)

            v_hat = v_t / (1 - beta_2^t)

            w_{t+1} = w_t - lr * m_hat / (sqrt(v_hat) + epsilon)

    where:
        - w_t is the parameter at time step t
        - g_t is the gradient at time step t
        - m_t is the first moment (momentum) estimate
        - v_t is the second moment (velocity) estimate
        - beta_1, beta_2 are the exponential decay rates
        - epsilon is a small constant for numerical stability
        - lr is the learning rate

    Args:
        learning_rate:
            A float, a keras.optimizers.schedules.LearningRateSchedule instance, or a callable that returns
            the actual value for the learning rate. Default is 0.001.
        beta_1:
            The exponential decay rate for the 1st moment estimates. Should be in [0, 1). Default is 0.9.
        beta_2:
            The exponential decay rate for the 2nd moment estimates. Should be in [0, 1). Default is 0.999.
        epsilon:
            Small constant for numerical stability. Default is 1e-7.
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
            The name of the optimizer. Default is 'nadam'.
# This implementation is adapted from the original Keras source code,
# available at: https://github.com/keras-team/keras
# It has been modified for customization and integration into this specific context.

    Example:

        >>> python3
        ...     # Instantiate a Nadam optimizer
        ...     optimizer = Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        ...     # Update the model parameters using the optimizer
        ...     for gradient, variable in zip(gradients, variables):
        >>>     optimizer.update_step(gradient, variable, learning_rate)
        ```

    References:
        - [Dozat, 2015](http://cs229.stanford.edu/proj2015/054_report.pdf)
        - "Adam: A Method for Stochastic Optimization" by Kingma & Ba (2014)
        - "On the importance of initialization and momentum in deep learning" by Sutskever et al. (2013)
    """

    def __init__(self, learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 weight_decay=None,
                 clipnorm=None,
                 clipvalue=None,
                 global_clipnorm=None,
                 use_ema=False,
                 ema_momentum=0.99,
                 ema_overwrite_frequency=None,
                 loss_scale_factor=None,
                 gradient_accumulation_steps=None,
                 name="nadam",
                 **kwargs,):

        """Initialize the Nadam optimizer.

        Args:
            learning_rate: A float or callable, the learning rate value.
            beta_1: Float, the exponential decay rate for 1st moment estimates.
            beta_2: Float, the exponential decay rate for 2nd moment estimates.
            epsilon: Small constant for numerical stability.
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

        # Validate hyperparameters
        if not isinstance(beta_1, float) or not (0 <= beta_1 < 1):
            raise ValueError("beta_1 must be a float in [0, 1).")

        if not isinstance(beta_2, float) or not (0 <= beta_2 < 1):
            raise ValueError("beta_2 must be a float in [0, 1).")

        if not isinstance(epsilon, float) or epsilon <= 0:
            raise ValueError("epsilon must be a positive float.")

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def build(self, var_list):
        """Initialize optimizer variables.

        Nadam requires two sets of variables for each model parameter:
        - momentums (first moment estimates)
        - velocities (second moment estimates)
        Additionally maintains a product of momentum correction terms.

        Args:
            var_list: List of model variables to build Nadam variables for.
        """
        if self.built:
            return

        # Determine dtype from first variable if available
        if var_list:
            dtype = var_list[0].dtype

        else:
            dtype = backend.floatx()

        super().build(var_list)

        # Initialize momentums and velocities for each variable
        self._momentums = []
        self._velocities = []

        # Initialize product of momentum correction terms
        self._u_product = backend.Variable(1.0, dtype=dtype)

        for var in var_list:
            self._momentums.append(self.add_variable_from_reference(reference_variable=var, name="momentum"))
            self._velocities.append(self.add_variable_from_reference(reference_variable=var, name="velocity"))

    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Update the product of momentum correction terms.

        This is called before the main update step to maintain the running product
        of momentum correction terms needed for Nesterov acceleration.

        Args:
            grads: List of gradients for the variables.
            trainable_variables: List of variables to update.
            learning_rate: The current learning rate.
        """
        dtype = self._u_product.dtype
        self.assign(self._u_product, self._u_product * self.beta_1 * (1.0
                                                                      - 0.5 * ops.power(0.96,
                                                                                        ops.cast(self.iterations + 1,
                                                                                                 dtype))),)
        super()._backend_update_step(grads, trainable_variables, learning_rate)

    def update_step(self, gradient, variable, learning_rate):
        """Perform the optimization step to update the model parameters.

        This method updates the model variable using the Nadam algorithm,
        incorporating both momentum and adaptive learning rates with Nesterov
        acceleration.

        Args:
            gradient: The gradient of the loss function with respect to the variable.
            variable: The model variable being updated.
            learning_rate: The current learning rate.
        """
        var_dtype = variable.dtype
        learning_rate = ops.cast(learning_rate, var_dtype)
        gradient = ops.cast(gradient, var_dtype)

        # Calculate time-dependent terms
        local_step = ops.cast(self.iterations + 1, var_dtype)
        next_step = ops.cast(self.iterations + 2, var_dtype)
        decay = ops.cast(0.96, var_dtype)
        beta_1 = ops.cast(self.beta_1, var_dtype)
        beta_2 = ops.cast(self.beta_2, var_dtype)

        # Calculate momentum correction terms
        u_t = beta_1 * (1.0 - 0.5 * (ops.power(decay, local_step)))
        u_t_1 = beta_1 * (1.0 - 0.5 * (ops.power(decay, next_step)))
        u_product_t = ops.cast(self._u_product, var_dtype)
        u_product_t_1 = u_product_t * u_t_1
        beta_2_power = ops.power(beta_2, local_step)

        # Retrieve momentum and velocity for this variable
        momentum_vector = self._momentums[self._get_variable_index(variable)]
        velocity_vector = self._velocities[self._get_variable_index(variable)]

        # Update momentum (biased first moment estimate)
        self.assign_add(momentum_vector, ops.multiply(ops.subtract(gradient, momentum_vector), (1 - beta_1)))

        # Update velocity (biased second moment estimate)
        self.assign_add(velocity_vector, ops.multiply(ops.subtract(ops.square(gradient),
                                                                   velocity_vector), (1 - beta_2)))

        # Compute Nesterov-accelerated momentum estimate
        m_hat = ops.add(ops.divide(ops.multiply(u_t_1, momentum_vector), 1 - u_product_t_1),
                        ops.divide(ops.multiply(1 - u_t, gradient), 1 - u_product_t),)

        # Compute bias-corrected velocity estimate
        v_hat = ops.divide(velocity_vector, (1 - beta_2_power))

        # Apply the update
        self.assign_sub(variable, ops.divide(ops.multiply(m_hat, learning_rate),
                                             ops.add(ops.sqrt(v_hat), self.epsilon)),)

    def get_config(self):
        """Returns the configuration of the optimizer as a dictionary.

        The configuration is used to recreate the optimizer instance.

        Returns:
            A dictionary containing the configuration of the optimizer.
        """
        config = super().get_config()
        config.update(
            {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
            }
        )
        return config

    @property
    def beta_1(self) -> float:
        """Exponential decay rate for the first moment estimates (gradients).

        Controls how quickly the moving average of gradients decays.
        Typical values are between 0.8 and 0.999.
        Higher values give more weight to previous estimates.

        Returns:
            float: Current beta_1 value between [0, 1)
        """
        return self._beta_1

    @beta_1.setter
    def beta_1(self, value: float) -> None:
        """Sets the decay rate for the first moment estimate.

        Args:
            value: New value for beta_1 (must be between 0 and 1)

        Raises:
            ValueError: If value is not in range [0, 1)
        """
        if not isinstance(value, float) or not (0 <= value < 1):
            raise ValueError("beta_1 must be a float in range [0, 1)")
        self._beta_1 = value

    @property
    def beta_2(self) -> float:
        """Exponential decay rate for the second moment estimates (squared gradients).

        Controls how quickly the moving average of squared gradients decays.
        Typical values are between 0.9 and 0.9999.
        Should be greater than beta_1 for stability.

        Returns:
            float: Current beta_2 value between [0, 1)
        """
        return self._beta_2

    @beta_2.setter
    def beta_2(self, value: float) -> None:
        """Sets the decay rate for the second moment estimate.

        Args:
            value: New value for beta_2 (must be between 0 and 1)

        Raises:
            ValueError: If value is not in range [0, 1) or <= beta_1
        """
        if not isinstance(value, float) or not (0 <= value < 1):
            raise ValueError("beta_2 must be a float in range [0, 1)")
        if hasattr(self, '_beta_1') and value <= self._beta_1:
            raise ValueError("beta_2 must be greater than beta_1 for stability")
        self._beta_2 = value

    @property
    def epsilon(self) -> float:
        """Numerical stability term.

        Small constant added to prevent division by zero.
        Typical values between 1e-7 and 1e-8.
        Usually doesn't need tuning.

        Returns:
            float: Current epsilon value (positive)
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        """Sets the epsilon regularization term.

        Args:
            value: New value for epsilon (must be positive)

        Raises:
            ValueError: If value is not positive
        """
        if not isinstance(value, float) or value <= 0:
            raise ValueError("epsilon must be a positive float")
        self._epsilon = value
