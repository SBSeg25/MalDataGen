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

    import tensorflow

    from Engine.optimizers.SGD import SGD

    from Engine.optimizers.Adam import Adam
    from Engine.optimizers.FTRL import FTRL

    from Engine.optimizers.Nadam import NADAM

    from Engine.optimizers.RSMProp import RMSProp
    from Engine.optimizers.AdaDelta import AdaDelta

except ImportError as error:
    print(error)
    sys.exit(-1)


class Optimizers:
    """
    A factory class for creating and configuring optimization algorithms.

    This class provides methods to create various optimizers with validated parameters.
    It stores default configurations for each optimizer type and provides static methods
    to create optimizer instances with custom parameters.


    Attributes:

        @adam_optimizer_beta_1 (float):
            Exponential decay rate for 1st moment estimates in Adam.
        @adam_optimizer_beta_2 (float):
            Exponential decay rate for 2nd moment estimates in Adam.
        @adam_optimizer_epsilon (float):
            Small constant for numerical stability in Adam.
        @adam_optimizer_amsgrad (bool):
            Whether to use AMSGrad variant of Adam.
        @adam_optimizer_learning_rate (float):
            Learning rate for Adam.

        @ada_delta_optimizer_rho (float):
            Decay rate for AdaDelta.
        @ada_delta_optimizer_epsilon (float):
            Small constant for numerical stability in AdaDelta.
        @ada_delta_optimizer_use_ema (bool):
            Whether to use exponential moving average in AdaDelta.
        @ada_delta_optimizer_ema_momentum (float):
            Momentum factor for EMA in AdaDelta.
        @ada_delta_optimizer_learning_rate (float):
            Learning rate for AdaDelta.

        @nadam_optimizer_beta_1 (float):
            Exponential decay rate for 1st moment estimates in NAdam.
        @nadam_optimizer_beta_2 (float):
            Exponential decay rate for 2nd moment estimates in NAdam.
        @nadam_optimizer_epsilon (float):
            Small constant for numerical stability in NAdam.
        @nadam_optimizer_use_ema (bool):
            Whether to use exponential moving average in NAdam.
        @nadam_optimizer_ema_momentum (float):
            Momentum factor for EMA in NAdam.
        @nadam_optimizer_learning_rate (float):
            Learning rate for NAdam.

        @rsmprop_optimizer_rho (float):
            Discounting factor for RMSProp.
        @rsmprop_optimizer_epsilon (float):
            Small constant for numerical stability in RMSProp.
        @rsmprop_optimizer_use_ema (bool):
            Whether to use exponential moving average in RMSProp.
        @rsmprop_optimizer_momentum (float):
            Momentum factor for RMSProp.
        @rsmprop_optimizer_ema_momentum (float):
            Momentum factor for EMA in RMSProp.
        @rsmprop_optimizer_learning_rate (float):
            Learning rate for RMSProp.

        @sgd_optimizer_use_ema (bool):
            Whether to use exponential moving average in SGD.
        @sgd_optimizer_momentum (float):
            Momentum factor for SGD.
        @sgd_optimizer_nesterov (bool):
            Whether to use Nesterov momentum in SGD.
        @sgd_optimizer_ema_momentum (float):
            Momentum factor for EMA in SGD.
        @sgd_optimizer_learning_rate (float):
            Learning rate for SGD.

        @ftrl_optimizer_beta (float):
            Beta parameter for FTRL.
        @ftrl_optimizer_use_ema (bool):
            Whether to use exponential moving average in FTRL.
        @ftrl_optimizer_ema_momentum (float):
            Momentum factor for EMA in FTRL.
        @ftrl_optimizer_learning_rate (float):
            Learning rate for FTRL.
        @ftrl_optimizer_learning_rate_power (float):
            Power for learning rate decay in FTRL.
        @ftrl_optimizer_initial_accumulator_value (float):
            Initial accumulator value for FTRL.
        @ftrl_optimizer_l1_regularization_strength (float):
            L1 regularization strength for FTRL.
        @ftrl_optimizer_l2_regularization_strength (float):
            L2 regularization strength for FTRL.
        @ftrl_optimizer_l2_shrinkage_regularization_strength (float):
            L2 shrinkage regularization strength for FTRL.
            
# This implementation is adapted from the original Keras source code,
# available at: https://github.com/keras-team/keras
# It has been modified for customization and integration into this specific context.

    """
    def __init__(self, arguments):

        self._adam_optimizer_beta_1 = arguments.adam_optimizer_beta_1
        self._adam_optimizer_beta_2 = arguments.adam_optimizer_beta_2
        self._adam_optimizer_epsilon = arguments.adam_optimizer_epsilon
        self._adam_optimizer_amsgrad = arguments.adam_optimizer_amsgrad
        self._adam_optimizer_learning_rate = arguments.adam_optimizer_learning_rate


        self._ada_delta_optimizer_rho = arguments.ada_delta_optimizer_rho
        self._ada_delta_optimizer_epsilon = arguments.ada_delta_optimizer_epsilon
        self._ada_delta_optimizer_use_ema = arguments.ada_delta_optimizer_use_ema
        self._ada_delta_optimizer_ema_momentum = arguments.ada_delta_optimizer_ema_momentum
        self._ada_delta_optimizer_learning_rate = arguments.ada_delta_optimizer_learning_rate

        self._nadam_optimizer_beta_1 = arguments.nadam_optimizer_beta_1
        self._nadam_optimizer_beta_2 = arguments.nadam_optimizer_beta_2
        self._nadam_optimizer_epsilon = arguments.nadam_optimizer_epsilon
        self._nadam_optimizer_use_ema = arguments.nadam_optimizer_use_ema
        self._nadam_optimizer_ema_momentum = arguments.nadam_optimizer_ema_momentum
        self._nadam_optimizer_learning_rate = arguments.nadam_optimizer_learning_rate

        self._rsmprop_optimizer_rho = arguments.rsmprop_optimizer_rho
        self._rsmprop_optimizer_epsilon = arguments.rsmprop_optimizer_epsilon
        self._rsmprop_optimizer_use_ema = arguments.rsmprop_optimizer_use_ema
        self._rsmprop_optimizer_momentum = arguments.rsmprop_optimizer_momentum
        self._rsmprop_optimizer_ema_momentum = arguments.rsmprop_optimizer_ema_momentum
        self._rsmprop_optimizer_learning_rate = arguments.rsmprop_optimizer_learning_rate

        self._sgd_optimizer_use_ema = arguments.sgd_optimizer_use_ema
        self._sgd_optimizer_momentum = arguments.sgd_optimizer_momentum
        self._sgd_optimizer_nesterov = arguments.sgd_optimizer_nesterov
        self._sgd_optimizer_ema_momentum = arguments.sgd_optimizer_ema_momentum
        self._sgd_optimizer_learning_rate = arguments.sgd_optimizer_learning_rate

        self._ftrl_optimizer_beta = arguments.ftrl_optimizer_beta
        self._ftrl_optimizer_use_ema = arguments.ftrl_optimizer_use_ema
        self._ftrl_optimizer_ema_momentum = arguments.ftrl_optimizer_ema_momentum
        self._ftrl_optimizer_learning_rate = arguments.ftrl_optimizer_learning_rate
        self._ftrl_optimizer_learning_rate_power = arguments.ftrl_optimizer_learning_rate_power
        self._ftrl_optimizer_initial_accumulator_value = arguments.ftrl_optimizer_initial_accumulator_value
        self._ftrl_optimizer_l1_regularization_strength = arguments.ftrl_optimizer_l1_regularization_strength
        self._ftrl_optimizer_l2_regularization_strength = arguments.ftrl_optimizer_l2_regularization_strength
        self._ftrl_optimizer_l2_shrinkage_regularization_strength = arguments.ftrl_optimizer_l2_shrinkage_regularization_strength



    def get_optimizer(self, optimizer_name):
        """
        Retrieves a configured optimizer instance based on the specified optimizer name.

        This method serves as a factory for different optimization algorithms, returning
        a fully configured optimizer instance with parameters initialized from the class's
        stored configuration. The method supports case-insensitive optimizer names and
        provides comprehensive error handling for unsupported optimizers.

        Args:
            optimizer_name (str):
                The name of the optimizer to instantiate. Supported values are:
                    - 'adadelta': AdaDelta optimizer
                    - 'adam': Adam optimizer
                    - 'ftrl': FTRL-Proximal optimizer
                    - 'nadam': Nesterov Adam optimizer
                    - 'rsmprop': RMSProp optimizer
                    - 'sgd': Stochastic Gradient Descent optimizer

        Returns:
            object: An instance of the requested optimizer class, configured with the parameters
                   stored in this optimizers factory instance.

        Raises:
            ValueError: If the specified optimizer_name is not supported. The error message
                      includes the list of available optimizers.

        Example:
            >>> optimizer_factory = optimizers(config_arguments)
            >>> adam_optimizer = optimizer_factory.get_optimizer('adam')
            >>> sgd_optimizer = optimizer_factory.get_optimizer('SGD')  # Case-insensitive

        Notes:
            - The method converts the optimizer_name to lowercase for case-insensitive matching
            - Each optimizer is configured with the parameters stored during the optimizers
              class initialization
            - The returned optimizer instances are ready for use in model training
            - The method provides both printed output and raised exception for unsupported
              optimizers to ensure visibility in both interactive and script usage

        Supported optimizers and Their Parameters:
            - AdaDelta:
                - learning_rate: self._ada_delta_optimizer_learning_rate
                - rho: self._ada_delta_optimizer_rho
                - epsilon: self._ada_delta_optimizer_epsilon
                - use_ema: self._ada_delta_optimizer_use_ema
                - ema_momentum: self._ada_delta_optimizer_ema_momentum

            - Adam:
                - learning_rate: self._adam_optimizer_learning_rate
                - beta_1: self._adam_optimizer_beta_1
                - beta_2: self._adam_optimizer_beta_2
                - epsilon: self._adam_optimizer_epsilon
                - amsgrad: self._adam_optimizer_amsgrad

            - FTRL:
                - learning_rate: self._ftrl_optimizer_learning_rate
                - learning_rate_power: self._ftrl_optimizer_learning_rate_power
                - initial_accumulator_value: self._ftrl_optimizer_initial_accumulator_value
                - l1_regularization_strength: self._ftrl_optimizer_l1_regularization_strength
                - l2_regularization_strength: self._ftrl_optimizer_l2_regularization_strength
                - l2_shrinkage_regularization_strength: self._ftrl_optimizer_l2_shrinkage_regularization_strength
                - beta: self._ftrl_optimizer_beta
                - use_ema: self._ftrl_optimizer_use_ema
                - ema_momentum: self._ftrl_optimizer_ema_momentum

            - NAdam:
                - learning_rate: self._nadam_optimizer_learning_rate
                - beta_1: self._nadam_optimizer_beta_1
                - beta_2: self._nadam_optimizer_beta_2
                - epsilon: self._nadam_optimizer_epsilon
                - use_ema: self._nadam_optimizer_use_ema
                - ema_momentum: self._nadam_optimizer_ema_momentum

            - RMSProp:
                - learning_rate: self._rsmprop_optimizer_learning_rate
                - rho: self._rsmprop_optimizer_rho
                - momentum: self._rsmprop_optimizer_momentum
                - epsilon: self._rsmprop_optimizer_epsilon
                - use_ema: self._rsmprop_optimizer_use_ema
                - ema_momentum: self._rsmprop_optimizer_ema_momentum

            - SGD:
                - learning_rate: self._sgd_optimizer_learning_rate
                - momentum: self._sgd_optimizer_momentum
                - nesterov: self._sgd_optimizer_nesterov
                - use_ema: self._sgd_optimizer_use_ema
                - ema_momentum: self._sgd_optimizer_ema_momentum
        """

        optimizers = {

            'adadelta': self.get_optimizer_ada_delta(learning_rate=self._ada_delta_optimizer_learning_rate,
                                                     rho=self._ada_delta_optimizer_rho,
                                                     epsilon=self._ada_delta_optimizer_epsilon,
                                                     use_ema=self._ada_delta_optimizer_use_ema,
                                                     ema_momentum=self._ada_delta_optimizer_ema_momentum),

            'adam': self.get_optimizer_adam(learning_rate=self._adam_optimizer_learning_rate,
                                            beta_1=self._adam_optimizer_beta_1,
                                            beta_2=self._adam_optimizer_beta_2,
                                            epsilon=self._adam_optimizer_epsilon,
                                            amsgrad=self._adam_optimizer_amsgrad),

            'ftrl': self.get_optimizer_ftrl(learning_rate=self._ftrl_optimizer_learning_rate,
                                            learning_rate_power=self._ftrl_optimizer_learning_rate_power,
                                            initial_accumulator_value=self._ftrl_optimizer_initial_accumulator_value,
                                            l1_regularization_strength=self._ftrl_optimizer_l1_regularization_strength,
                                            l2_regularization_strength=self._ftrl_optimizer_l2_regularization_strength,
                                            l2_shrinkage_regularization_strength=self._ftrl_optimizer_l2_shrinkage_regularization_strength,
                                            beta=self._ftrl_optimizer_beta,
                                            use_ema=self._ftrl_optimizer_use_ema,
                                            ema_momentum=self._ftrl_optimizer_ema_momentum),

            'nadam': self.get_optimizer_nadam(learning_rate=self._nadam_optimizer_learning_rate,
                                              beta_1=self._nadam_optimizer_beta_1,
                                              beta_2=self._nadam_optimizer_beta_2,
                                              epsilon=self._nadam_optimizer_epsilon,
                                              use_ema=self._nadam_optimizer_use_ema,
                                              ema_momentum=self._nadam_optimizer_ema_momentum),

            'rsmprop': self.get_optimizer_rsmprop(learning_rate=self._rsmprop_optimizer_learning_rate,
                                                  rho = self._rsmprop_optimizer_rho,
                                                  momentum = self._rsmprop_optimizer_momentum,
                                                  epsilon = self._rsmprop_optimizer_epsilon,
                                                  use_ema = self._rsmprop_optimizer_use_ema,
                                                  ema_momentum = self._rsmprop_optimizer_ema_momentum),

            'sgd': self.get_optimizer_sgd(learning_rate=self._sgd_optimizer_learning_rate,
                                          momentum=self._sgd_optimizer_momentum,
                                          nesterov=self._sgd_optimizer_nesterov,
                                          use_ema=self._sgd_optimizer_use_ema,
                                          ema_momentum=self._sgd_optimizer_ema_momentum)}

        # Convert the optimizer function name to lowercase to handle case insensitivity
        optimizer_lower = optimizer_name.lower()

        # Check if the optimizer is supported and return the corresponding layer
        if optimizer_lower in optimizers:
            return optimizers[optimizer_lower]

        else:
           # Raise an error if the optimizer function is unsupported
           print(f"Unsupported optimizer : '{optimizer_name}'. Please choose from: {list(optimizers.keys())}")
           raise ValueError(
               f"Unsupported optimizer: '{optimizer_name}'. Please choose from: {list(optimizers.keys())}")

    @staticmethod
    def get_optimizer_adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False):
        """
        Creates an Adam optimizer with parameter validation.

        Args:
            learning_rate: float > 0. Learning rate.
            beta_1: float in (0, 1). Exponential decay rate for 1st moment estimates.
            beta_2: float in (0, 1). Exponential decay rate for 2nd moment estimates.
            epsilon: float >= 0. Fuzz factor to avoid division by zero.
            amsgrad: boolean. Whether to apply AMSGrad variant.

        Returns:
            Adam optimizer instance

        Raises:
            ValueError: If any parameter is invalid
            TypeError: If any parameter has wrong type
        """

        try:

            if not isinstance(learning_rate, (float, int)) or learning_rate <= 0:
                raise ValueError(f"learning_rate must be positive float, got {learning_rate}")

            if not 0 < beta_1 < 1:
                raise ValueError(f"beta_1 must be between 0 and 1, got {beta_1}")

            if not 0 < beta_2 < 1:
                raise ValueError(f"beta_2 must be between 0 and 1, got {beta_2}")

            if not isinstance(epsilon, (float, int)) or epsilon < 0:
                raise ValueError(f"epsilon must be non-negative float, got {epsilon}")

            if not isinstance(amsgrad, bool):
                raise TypeError(f"amsgrad must be boolean, got {amsgrad}")

            return Adam(learning_rate, beta_1, beta_2, epsilon, amsgrad)

        except Exception as e:
            raise ValueError(f"Failed to create Adam optimizer: {str(e)}") from e


    @staticmethod
    def get_optimizer_ada_delta(learning_rate=0.001, rho=0.95, epsilon=1e-7, weight_decay=None, clipnorm=None,
                                 clipvalue=None, global_clipnorm=None, use_ema=False, ema_momentum=0.99,
                                 ema_overwrite_frequency=None, loss_scale_factor=None, gradient_accumulation_steps=None):

        """
        Creates an AdaDelta optimizer with parameter validation.

        Args:
            learning_rate: float > 0. Learning rate.
            rho: float in (0, 1). Decay rate.
            epsilon: float >= 0. Fuzz factor to avoid division by zero.
            weight_decay: None or float >= 0. Weight decay factor.
            clipnorm: None or float > 0. Gradient clipping by norm.
            clipvalue: None or float > 0. Gradient clipping by value.
            global_clipnorm: None or float > 0. Global gradient clipping by norm.
            use_ema: boolean. Whether to use exponential moving average.
            ema_momentum: float in (0, 1) if use_ema. EMA momentum factor.
            ema_overwrite_frequency: None or int > 0. EMA overwrite frequency.
            loss_scale_factor: None or float > 0. loss scaling factor.
            gradient_accumulation_steps: None or int > 0. Gradient accumulation steps.

        Returns:
            AdaDelta optimizer instance

        Raises:
            ValueError: If any parameter is invalid
            TypeError: If any parameter has wrong type
        """
        try:

            # Validate main parameters
            if not isinstance(learning_rate, (float, int)) or learning_rate <= 0:
                raise ValueError(f"learning_rate must be positive float, got {learning_rate}")

            if not 0 < rho < 1:
                raise ValueError(f"rho must be between 0 and 1, got {rho}")

            if not isinstance(epsilon, (float, int)) or epsilon < 0:
                raise ValueError(f"epsilon must be non-negative float, got {epsilon}")

            # Validate optional parameters
            if weight_decay is not None and (not isinstance(weight_decay, (float, int)) or weight_decay < 0):
                raise ValueError(f"weight_decay must be None or non-negative float, got {weight_decay}")

            if clipnorm is not None and (not isinstance(clipnorm, (float, int)) or clipnorm <= 0):
                raise ValueError(f"clipnorm must be None or positive float, got {clipnorm}")

            if clipvalue is not None and (not isinstance(clipvalue, (float, int)) or clipvalue <= 0):
                raise ValueError(f"clipvalue must be None or positive float, got {clipvalue}")

            if global_clipnorm is not None and (not isinstance(global_clipnorm, (float, int)) or global_clipnorm <= 0):
                raise ValueError(f"global_clipnorm must be None or positive float, got {global_clipnorm}")

            if not isinstance(use_ema, bool):
                raise TypeError(f"use_ema must be boolean, got {use_ema}")

            if use_ema and (not 0 < ema_momentum < 1):
                raise ValueError(f"ema_momentum must be between 0 and 1 when use_ema=True, got {ema_momentum}")

            if ema_overwrite_frequency is not None and (not isinstance(ema_overwrite_frequency, int)
                                                        or ema_overwrite_frequency <= 0):

                raise ValueError(f"ema_overwrite_frequency must be None or positive integer, got {
                ema_overwrite_frequency}")

            if loss_scale_factor is not None and (not isinstance(loss_scale_factor, (float, int))
                                                  or loss_scale_factor <= 0):

                raise ValueError(f"loss_scale_factor must be None or positive float, got {loss_scale_factor}")

            if gradient_accumulation_steps is not None and (not isinstance(gradient_accumulation_steps, int)
                                                            or gradient_accumulation_steps <= 0):

                raise ValueError(f"gradient_accumulation_steps must be None or positive integer,"
                                 f" got {gradient_accumulation_steps}")

            return AdaDelta(learning_rate, rho, epsilon, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema,
                            ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps)

        except Exception as e:
            raise ValueError(f"Failed to create AdaDelta optimizer: {str(e)}") from e

    def get_optimizer_ftrl(self, learning_rate=0.001, learning_rate_power=-0.5, initial_accumulator_value=0.1,
                           l1_regularization_strength=0.0, l2_regularization_strength=0.0,
                           l2_shrinkage_regularization_strength=0.0, beta=0.0, weight_decay=None, clipnorm=None,
                           clipvalue=None, global_clipnorm=None, use_ema=False, ema_momentum=0.99,
                           ema_overwrite_frequency=None, loss_scale_factor=None, gradient_accumulation_steps=None):
        """
        Creates an FTRL optimizer with parameter validation.

        Args:
            learning_rate: float > 0. Learning rate.
            learning_rate_power: float <= 0. Power for learning rate decay.
            initial_accumulator_value: float > 0. Initial accumulator value.
            l1_regularization_strength: float >= 0. L1 regularization factor.
            l2_regularization_strength: float >= 0. L2 regularization factor.
            l2_shrinkage_regularization_strength: float >= 0. L2 shrinkage factor.
            beta: float >= 0. Beta parameter.
            weight_decay: None or float >= 0. Weight decay factor.
            clipnorm/clipvalue/global_clipnorm: See AdaDelta
            use_ema/ema_momentum/etc: See AdaDelta

        Returns:
            FTRL optimizer instance

        Raises:
            ValueError: If any parameter is invalid
            TypeError: If any parameter has wrong type
        """

        try:

            # Validate main parameters
            if not isinstance(learning_rate, (float, int)) or learning_rate <= 0:
                raise ValueError(f"learning_rate must be positive float, got {learning_rate}")

            if not isinstance(learning_rate_power, (float, int)) or learning_rate_power > 0:
                raise ValueError(f"learning_rate_power must be <= 0, got {learning_rate_power}")

            if not isinstance(initial_accumulator_value, (float, int)) or initial_accumulator_value <= 0:
                raise ValueError(f"initial_accumulator_value must be positive float, got"
                                 f" {initial_accumulator_value}")

            if not isinstance(l1_regularization_strength, (float, int)) or l1_regularization_strength < 0:
                raise ValueError(f"l1_regularization_strength must be non-negative float, got"
                                 f" {l1_regularization_strength}")

            if not isinstance(l2_regularization_strength, (float, int)) or l2_regularization_strength < 0:
                raise ValueError(f"l2_regularization_strength must be non-negative float, got"
                                 f" {l2_regularization_strength}")

            if not isinstance(l2_shrinkage_regularization_strength,
                              (float, int)) or l2_shrinkage_regularization_strength < 0:
                raise ValueError(f"l2_shrinkage_regularization_strength must be non-negative float, got"
                                 f" {l2_shrinkage_regularization_strength}")

            if not isinstance(beta, (float, int)) or beta < 0:
                raise ValueError(f"beta must be non-negative float, got {beta}")

            # Validate common optional parameters (same as AdaDelta)
            self._validate_common_optimizer_args(weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema,
                                                 ema_momentum, ema_overwrite_frequency, loss_scale_factor,
                                                 gradient_accumulation_steps)

            return FTRL(learning_rate, learning_rate_power, initial_accumulator_value, l1_regularization_strength,
                        l2_regularization_strength, l2_shrinkage_regularization_strength, beta, weight_decay, clipnorm,
                        clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor,
                        gradient_accumulation_steps)

        except Exception as e:
            raise ValueError(f"Failed to create FTRL optimizer: {str(e)}") from e


    def get_optimizer_nadam(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, weight_decay=None,
                            clipnorm=None, clipvalue=None, global_clipnorm=None, use_ema=False, ema_momentum=0.99,
                            ema_overwrite_frequency=None, loss_scale_factor=None, gradient_accumulation_steps=None):

        """
        Creates a NAdam optimizer with parameter validation.

        Args:
            learning_rate: float > 0. Learning rate.
            beta_1: float in (0, 1). Exponential decay rate for 1st moment estimates.
            beta_2: float in (0, 1). Exponential decay rate for 2nd moment estimates.
            epsilon: float >= 0. Fuzz factor to avoid division by zero.
            weight_decay/clipnorm/etc: See AdaDelta

        Returns:
            NAdam optimizer instance

        Raises:
            ValueError: If any parameter is invalid
            TypeError: If any parameter has wrong type
        """
        try:

            # Validate main parameters (same as Adam)
            if not isinstance(learning_rate, (float, int)) or learning_rate <= 0:
                raise ValueError(f"learning_rate must be positive float, got {learning_rate}")

            if not 0 < beta_1 < 1:
                raise ValueError(f"beta_1 must be between 0 and 1, got {beta_1}")

            if not 0 < beta_2 < 1:
                raise ValueError(f"beta_2 must be between 0 and 1, got {beta_2}")

            if not isinstance(epsilon, (float, int)) or epsilon < 0:
                raise ValueError(f"epsilon must be non-negative float, got {epsilon}")

            # Validate common optional parameters (same as AdaDelta)
            self._validate_common_optimizer_args(weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema,
                                                 ema_momentum, ema_overwrite_frequency, loss_scale_factor,
                                                 gradient_accumulation_steps)

            return NADAM(learning_rate, beta_1, beta_2, epsilon, weight_decay, clipnorm, clipvalue, global_clipnorm,
                         use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps)

        except Exception as e:
            raise ValueError(f"Failed to create NAdam optimizer: {str(e)}") from e


    def get_optimizer_rsmprop(self, learning_rate=0.001, rho=0.95, momentum=0.0, epsilon=1e-7, centered=False,
                              weight_decay=None, clipnorm=None, clipvalue=None, global_clipnorm=None, use_ema=False,
                              ema_momentum=0.99, ema_overwrite_frequency=None, loss_scale_factor=None,
                              gradient_accumulation_steps=None):

        """
        Creates an RMSProp optimizer with parameter validation.

        Args:
            learning_rate: float > 0. Learning rate.
            rho: float in (0, 1). Discounting factor.
            momentum: float >= 0. Momentum factor.
            epsilon: float >= 0. Fuzz factor to avoid division by zero.
            centered: boolean. Whether to center gradients.
            weight_decay/clipnorm/etc: See AdaDelta

        Returns:
            RMSProp optimizer instance

        Raises:
            ValueError: If any parameter is invalid
            TypeError: If any parameter has wrong type
        """

        try:

            # Validate main parameters
            if not isinstance(learning_rate, (float, int)) or learning_rate <= 0:
                raise ValueError(f"learning_rate must be positive float, got {learning_rate}")

            if not 0 < rho < 1:
                raise ValueError(f"rho must be between 0 and 1, got {rho}")

            if not isinstance(momentum, (float, int)) or momentum < 0:
                raise ValueError(f"momentum must be non-negative float, got {momentum}")

            if not isinstance(epsilon, (float, int)) or epsilon < 0:
                raise ValueError(f"epsilon must be non-negative float, got {epsilon}")

            if not isinstance(centered, bool):
                raise TypeError(f"centered must be boolean, got {centered}")

            # Validate common optional parameters (same as AdaDelta)
            self._validate_common_optimizer_args(weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema,
                                                 ema_momentum, ema_overwrite_frequency, loss_scale_factor,
                                                 gradient_accumulation_steps)

            return RMSProp(learning_rate, rho, momentum, epsilon, centered, weight_decay, clipnorm, clipvalue,
                           global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor,
                           gradient_accumulation_steps)

        except Exception as e:
            raise ValueError(f"Failed to create RMSProp optimizer: {str(e)}") from e


    def get_optimizer_sgd(self, learning_rate=0.01, momentum=0.0, nesterov=False, weight_decay=None, clipnorm=None,
                          clipvalue=None, global_clipnorm=None, use_ema=False, ema_momentum=0.99,
                          ema_overwrite_frequency=None, loss_scale_factor=None, gradient_accumulation_steps=None):

        """
        Creates an SGD optimizer with parameter validation.

        Args:
            learning_rate: float > 0. Learning rate.
            momentum: float >= 0. Momentum factor.
            nesterov: boolean. Whether to use Nesterov momentum.
            weight_decay/clipnorm/etc: See AdaDelta

        Returns:
            SGD optimizer instance

        Raises:
            ValueError: If any parameter is invalid
            TypeError: If any parameter has wrong type

        """

        try:

            # Validate main parameters
            if not isinstance(learning_rate, (float, int)) or learning_rate <= 0:
                raise ValueError(f"learning_rate must be positive float, got {learning_rate}")

            if not isinstance(momentum, (float, int)) or momentum < 0:
                raise ValueError(f"momentum must be non-negative float, got {momentum}")

            if not isinstance(nesterov, bool):
                raise TypeError(f"nesterov must be boolean, got {nesterov}")

            # Validate common optional parameters (same as AdaDelta)
            self._validate_common_optimizer_args(weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema,
                                                 ema_momentum, ema_overwrite_frequency, loss_scale_factor,
                                                 gradient_accumulation_steps)

            return SGD(learning_rate, momentum, nesterov, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema,
                       ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps)

        except Exception as e:
            raise ValueError(f"Failed to create SGD optimizer: {str(e)}") from e

    @staticmethod
    def _validate_common_optimizer_args(weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum,
                                        ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps):

        """Validates parameters common to most optimizers."""
        if weight_decay is not None and (not isinstance(weight_decay, (float, int)) or weight_decay < 0):
            raise ValueError(f"weight_decay must be None or non-negative float, got {weight_decay}")

        if clipnorm is not None and (not isinstance(clipnorm, (float, int)) or clipnorm <= 0):
            raise ValueError(f"clipnorm must be None or positive float, got {clipnorm}")

        if clipvalue is not None and (not isinstance(clipvalue, (float, int)) or clipvalue <= 0):
            raise ValueError(f"clipvalue must be None or positive float, got {clipvalue}")

        if global_clipnorm is not None and (not isinstance(global_clipnorm, (float, int)) or global_clipnorm <= 0):
            raise ValueError(f"global_clipnorm must be None or positive float, got {global_clipnorm}")

        if not isinstance(use_ema, bool):
            raise TypeError(f"use_ema must be boolean, got {use_ema}")

        if use_ema and (not 0 < ema_momentum < 1):
            raise ValueError(f"ema_momentum must be between 0 and 1 when use_ema=True, got {ema_momentum}")

        if ema_overwrite_frequency is not None and (not isinstance(ema_overwrite_frequency,
                                                                   int) or ema_overwrite_frequency <= 0):
            raise ValueError(f"ema_overwrite_frequency must be None or positive integer, got {ema_overwrite_frequency}")

        if loss_scale_factor is not None and (not isinstance(loss_scale_factor,
                                                             (float, int)) or loss_scale_factor <= 0):
            raise ValueError(f"loss_scale_factor must be None or positive float, got {loss_scale_factor}")

        if gradient_accumulation_steps is not None and (not isinstance(gradient_accumulation_steps,
                                                                       int) or gradient_accumulation_steps <= 0):
            raise ValueError(f"gradient_accumulation_steps must be None or positive integer, got {gradient_accumulation_steps}")
