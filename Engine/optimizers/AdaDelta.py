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


class AdaDelta(Optimizer):
    """
    Optimizer implementing the AdaDelta algorithm.

    AdaDelta is an extension of AdaGrad that seeks to reduce its aggressive, monotonically decreasing
    learning rate. Instead of accumulating all past squared gradients, AdaDelta restricts the window
    of accumulated past gradients to a fixed size, using an exponentially decaying average.

    The update rule for parameters is as follows:

        1. Update accumulated gradient (E[g²]):
            E[g²]_t = ρ * E[g²]_(t-1) + (1 - ρ) * g_t²

        2. Compute parameter update (Δθ):
            Δθ_t = - (RMS[Δθ]_(t-1) / RMS[g]_t) * g_t

        3. Update accumulated parameter updates (E[Δθ²]):
            E[Δθ²]_t = ρ * E[Δθ²]_(t-1) + (1 - ρ) * Δθ_t²

        4. Update parameters (θ):
            θ_t = θ_(t-1) + learning_rate * Δθ_t

    Where:

        - g_t is the gradient at time step t.
        - ρ is the decay rate for both accumulated gradients and parameter updates.
        - RMS[x] = √(E[x²] + ε) is the root mean squared value.
        - ε is a small constant for numerical stability.

    The key advantage of AdaDelta is that it eliminates the need for manually setting a learning rate,
    as it adapts the learning rate based on the historical gradient information.

    Args:
    ----
        learning_rate (float or callable): The learning rate. Default is 0.001.
        rho (float): Decay rate for both gradient and parameter update accumulations. Default is 0.95.
        epsilon (float): Small constant for numerical stability. Default is 1e-7.
        weight_decay (float, optional): Weight decay factor. Default is None.
        clipnorm (float, optional): Gradient clipping by norm. Default is None.
        clipvalue (float, optional): Gradient clipping by value. Default is None.
        global_clipnorm (float, optional): Global gradient clipping by norm. Default is None.
        use_ema (bool): Whether to use exponential moving average. Default is False.
        ema_momentum (float): Momentum for exponential moving average. Default is 0.99.
        ema_overwrite_frequency (int, optional): Frequency to overwrite variables with EMA values. Default is None.
        loss_scale_factor (float, optional): loss scale factor for mixed precision training. Default is None.
        gradient_accumulation_steps (int, optional): Number of steps for gradient accumulation. Default is None.
        name (str): Name of the optimizer. Default is "adadelta".

    Reference:
    ---------
        Zeiler, M. D. (2012). ADADELTA: An Adaptive Learning Rate Method.
        *arXiv preprint arXiv:1212.5701*.
        URL: https://arxiv.org/abs/1212.5701
        # This implementation is adapted from the original Keras source code,
        # available at: https://github.com/keras-team/keras
        # It has been modified for customization and integration into this specific context.

    Example:
        >>> python3
        ...     optimizer = AdaDelta(learning_rate=1.0, rho=0.95)
        >>>     model.compile(optimizer=optimizer, loss='mse')
    """
    def __init__(self,
                 learning_rate=0.001,
                 rho=0.95,
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
                 name="adadelta", **kwargs,):
        """
        Initialize the AdaDelta optimizer with the given parameters.

        Args:
            learning_rate (float): The base learning rate. Default is 0.001.
            rho (float): Decay factor for gradient and parameter update accumulations.
            epsilon (float): Small constant to avoid division by zero.
            weight_decay (float, optional): Optional weight decay factor.
            clipnorm (float, optional): Gradient norm clipping threshold.
            clipvalue (float, optional): Gradient value clipping threshold.
            global_clipnorm (float, optional): Global gradient norm clipping threshold.
            use_ema (bool): Whether to use exponential moving average.
            ema_momentum (float): Momentum factor for EMA.
            ema_overwrite_frequency (int, optional): Frequency to overwrite with EMA values.
            loss_scale_factor (float, optional): Scaling factor for loss.
            gradient_accumulation_steps (int, optional): Steps for gradient accumulation.
            name (str): Name of the optimizer.
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

        self.rho = rho
        self.epsilon = epsilon

    def build(self, var_list):
        """
        Initialize the optimizer's variables (accumulated gradients and delta vars).

        Args:
            var_list (list): List of model variables to be optimized.
        """

        if self.built:
            return

        super().build(var_list)

        self._accumulated_grads = []
        self._accumulated_delta_vars = []

        for var in var_list:
            self._accumulated_grads.append(self.add_variable_from_reference(var, "accumulated_grad"))
            self._accumulated_delta_vars.append(self.add_variable_from_reference(var, "accumulated_delta_var"))

    def update_step(self, gradient, variable, learning_rate):
        """
        Perform a parameter update based on the computed gradient.

        Args:
            gradient (tensor): The computed gradient for the variable.
            variable (tensor): The model parameter to be updated.
            learning_rate (float): The current learning rate.
        """

        learning_rate_converted = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)

        rho = self.rho
        accumulated_gradient = self._accumulated_grads[self._get_variable_index(variable)]
        accumulated_delta_var = self._accumulated_delta_vars[self._get_variable_index(variable)]

        def rms(x):
            """Compute root mean squared value with numerical stability."""
            return ops.sqrt(ops.add(x, self.epsilon))

        # Update accumulated gradients (E[g²])
        self.assign(accumulated_gradient, ops.add(rho * accumulated_gradient,
                                                  ops.multiply(1 - rho, ops.square(gradient))), )

        # Compute parameter update (Δθ)
        delta_var = ops.negative(ops.divide(ops.multiply(rms(accumulated_delta_var), gradient),
                                            rms(accumulated_gradient), ))

        # Update accumulated parameter updates (E[Δθ²])
        self.assign(accumulated_delta_var, ops.add(ops.multiply(rho, accumulated_delta_var),
                                                   ops.multiply(1 - rho, ops.square(delta_var)), ), )

        # Apply the update to the variable
        self.assign_add(variable, ops.multiply(learning_rate_converted, delta_var))

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "rho": self.rho,
                "epsilon": self.epsilon,
            }
        )
        return config
