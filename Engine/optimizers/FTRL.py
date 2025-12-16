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

    from tensorflow.keras import ops, backend
    from tensorflow.keras import initializers
    from tensorflow.keras.optimizers import Optimizer


except ImportError as error:
    print(error)
    sys.exit(-1)

class FTRL(Optimizer):
# This implementation is adapted from the original Keras source code,
# available at: https://github.com/keras-team/keras
# It has been modified for customization and integration into this specific context.

    def __init__(self,
                 learning_rate=0.001,
                 learning_rate_power=-0.5,
                 initial_accumulator_value=0.1,
                 l1_regularization_strength=0.0,
                 l2_regularization_strength=0.0,
                 l2_shrinkage_regularization_strength=0.0,
                 beta=0.0,
                 weight_decay=None,
                 clipnorm=None,
                 clipvalue=None,
                 global_clipnorm=None,
                 use_ema=False,
                 ema_momentum=0.99,
                 ema_overwrite_frequency=None,
                 loss_scale_factor=None,
                 gradient_accumulation_steps=None,
                 name="ftrl",
                 **kwargs,):

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

        if initial_accumulator_value < 0.0:
            raise ValueError("`initial_accumulator_value` needs to be positive or zero. "
                             "Received: initial_accumulator_value="
                             f"{initial_accumulator_value}.")
        if learning_rate_power > 0.0:
            raise ValueError(
                "`learning_rate_power` needs to be negative or zero. Received: "
                f"learning_rate_power={learning_rate_power}."
            )
        if l1_regularization_strength < 0.0:
            raise ValueError(
                "`l1_regularization_strength` needs to be positive or zero. "
                "Received: l1_regularization_strength="
                f"{l1_regularization_strength}."
            )
        if l2_regularization_strength < 0.0:
            raise ValueError(
                "`l2_regularization_strength` needs to be positive or zero. "
                "Received: l2_regularization_strength="
                f"{l2_regularization_strength}."
            )
        if l2_shrinkage_regularization_strength < 0.0:
            raise ValueError(
                "`l2_shrinkage_regularization_strength` needs to be positive "
                "or zero. Received: l2_shrinkage_regularization_strength"
                f"={l2_shrinkage_regularization_strength}."
            )

        self.learning_rate_power = learning_rate_power
        self.initial_accumulator_value = initial_accumulator_value
        self.l1_regularization_strength = l1_regularization_strength
        self.l2_regularization_strength = l2_regularization_strength
        self.l2_shrinkage_regularization_strength = (l2_shrinkage_regularization_strength)
        self.beta = beta

    def build(self, var_list):
        """Initialize optimizer variables.

        Args:
            var_list: list of model variables to build Ftrl variables on.
        """
        if self.built:
            return
        super().build(var_list)
        self._accumulators = []
        self._linears = []

        for var in var_list:

            self._accumulators.append(
                self.add_variable(shape=var.shape, dtype=var.dtype, name="accumulator",
                                  initializer=initializers.Constant(self.initial_accumulator_value,),))

            self._linears.append(self.add_variable_from_reference(reference_variable=var, name="linear"))

    def update_step(self, gradient, variable, learning_rate):
        """Update step given gradient and the associated model variable."""

        lr = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)

        accum = self._accumulators[self._get_variable_index(variable)]
        linear = self._linears[self._get_variable_index(variable)]

        lr_power = self.learning_rate_power
        l2_reg = self.l2_regularization_strength
        l2_reg = l2_reg + self.beta / (2.0 * lr)

        grad_to_use = ops.add(gradient, ops.multiply(2 * self.l2_shrinkage_regularization_strength, variable),)
        new_accum = ops.add(accum, ops.square(gradient))

        self.assign_add(linear, ops.subtract(grad_to_use, ops.multiply(ops.divide(
            ops.subtract(ops.power(new_accum, -lr_power), ops.power(accum, -lr_power), ), lr, ), variable, ), ), )

        quadratic = ops.add(ops.divide(ops.power(new_accum, (-lr_power)), lr), 2 * l2_reg)
        linear_clipped = ops.clip(linear, -self.l1_regularization_strength, self.l1_regularization_strength,)
        self.assign(variable, ops.divide(ops.subtract(linear_clipped, linear), quadratic), )
        self.assign(accum, new_accum)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate_power": self.learning_rate_power,
                "initial_accumulator_value": self.initial_accumulator_value,
                "l1_regularization_strength": self.l1_regularization_strength,
                "l2_regularization_strength": self.l2_regularization_strength,
                "l2_shrinkage_regularization_strength": self.l2_shrinkage_regularization_strength,  # noqa: E501
                "beta": self.beta,
            }
        )
        return config

