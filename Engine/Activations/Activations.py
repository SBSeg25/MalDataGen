#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{2}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/06'
__credits__ = ['Synthetic Ocean AI', 'Kayuã Oleques']

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
    from abc import ABC, abstractmethod
    from typing import Any, Callable

    # Detecta o framework a partir da variável de ambiente
    ML_FRAMEWORK = os.getenv('ML_FRAMEWORK', 'tensorflow').lower()

    # Importa condicionalmente os frameworks
    tf = None
    torch = None
    F = None

    if ML_FRAMEWORK == 'tensorflow':
        try:
            import tensorflow as tf
            # Importa as implementações específicas do TensorFlow
            from Engine.Activations.GLU import GLU
            from Engine.Activations.ELU import ELU
            from Engine.Activations.ReLU import ReLU
            from Engine.Activations.SELU import SELU
            from Engine.Activations.CeLU import CeLU
            from Engine.Activations.Tanh import Tanh
            from Engine.Activations.Swish import Swish
            from Engine.Activations.PReLU import PReLU
            from Engine.Activations.Linear import Linear
            from Engine.Activations.Sigmoid import Sigmoid
            from Engine.Activations.Softmax import Softmax
            from Engine.Activations.Softplus import Softplus
            from Engine.Activations.SoftSign import SoftSign
            from Engine.Activations.LeakyRelu import LeakyReLU
            from Engine.Activations.LogSigmoid import LogSigmoid
            from Engine.Activations.HardSigmoid import HardSigmoid
            from Engine.Activations.Exponential import Exponential
        except ImportError:
            raise ImportError("TensorFlow not found. Please install: pip install tensorflow")

    elif ML_FRAMEWORK == 'pytorch':
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("PyTorch not found. Please install: pip install torch")

    else:
        raise ValueError(f"Unsupported ML_FRAMEWORK: {ML_FRAMEWORK}. Use 'tensorflow' or 'pytorch'")

except ImportError as error:
    print(error)
    sys.exit(-1)


class BaseActivationImplementation(ABC):
    """
    Classe abstrata base para implementações específicas de framework.
    """

    @abstractmethod
    def add_activation(self, layer_or_tensor: Any, activation: str) -> Any:
        """
        Adiciona a função de ativação apropriada ao layer ou tensor.

        Args:
            layer_or_tensor: Layer do TensorFlow ou Tensor do PyTorch
            activation: Nome da função de ativação

        Returns:
            Layer ou Tensor com ativação aplicada
        """
        pass


if ML_FRAMEWORK == 'tensorflow':
    class TensorFlowActivationImpl(BaseActivationImplementation):
        """
        Implementação das ativações usando TensorFlow/Keras.
        """

        def __init__(self):
            # Dicionário mapeando nomes de ativação para suas funções correspondentes
            self.activations = {
                'leakyrelu': LeakyReLU(),
                'relu': ReLU(),
                'prelu': PReLU(),
                'sigmoid': Sigmoid(),
                'tanh': Tanh(),
                'elu': ELU(),
                'glu': GLU(),
                'logsigmoid': LogSigmoid(),
                'celu': CeLU(),
                'softsign': SoftSign(),
                'softmax': Softmax(),
                'swish': Swish(),
                'softplus': Softplus(),
                'hardsigmoid': HardSigmoid(),
                'selu': SELU(),
                'exponential': Exponential(),
                'linear': Linear(),
            }

        def add_activation(self, layer_or_tensor: Any, activation: str) -> Any:
            """
            Adiciona a função de ativação ao layer do TensorFlow.

            Args:
                layer_or_tensor: Layer do TensorFlow/Keras
                activation: Nome da função de ativação

            Returns:
                Layer com ativação aplicada
            """
            activation_lower = activation.lower()

            if activation_lower in self.activations:
                return self.activations[activation_lower](layer_or_tensor)
            else:
                print(f"Unsupported activation function: '{activation}'. "
                      f"Please choose from: {list(self.activations.keys())}")
                raise ValueError(
                    f"Unsupported activation function: '{activation}'. "
                    f"Please choose from: {list(self.activations.keys())}"
                )

if ML_FRAMEWORK == 'pytorch':
    class PyTorchActivationImpl(BaseActivationImplementation):
        """
        Implementação das ativações usando PyTorch.
        """

        def __init__(self):
            # Dicionário mapeando nomes de ativação para suas funções correspondentes
            self.activations = {
                'leakyrelu': lambda x: F.leaky_relu(x, negative_slope=0.2),
                'relu': F.relu,
                'prelu': lambda x: F.prelu(x, torch.tensor(0.25)),  # Default weight
                'sigmoid': torch.sigmoid,
                'tanh': torch.tanh,
                'elu': lambda x: F.elu(x, alpha=1.0),
                'glu': lambda x: F.glu(x, dim=-1),
                'logsigmoid': F.logsigmoid,
                'celu': lambda x: F.celu(x, alpha=1.0),
                'softsign': F.softsign,
                'softmax': lambda x: F.softmax(x, dim=-1),
                'swish': lambda x: x * torch.sigmoid(x),  # Swish/SiLU
                'softplus': F.softplus,
                'hardsigmoid': F.hardsigmoid,
                'selu': F.selu,
                'exponential': torch.exp,
                'linear': lambda x: x,
            }

        def add_activation(self, layer_or_tensor: Any, activation: str) -> Any:
            """
            Adiciona a função de ativação ao tensor do PyTorch.

            Args:
                layer_or_tensor: Tensor do PyTorch
                activation: Nome da função de ativação

            Returns:
                Tensor com ativação aplicada
            """
            activation_lower = activation.lower()

            if activation_lower in self.activations:
                return self.activations[activation_lower](layer_or_tensor)
            else:
                print(f"Unsupported activation function: '{activation}'. "
                      f"Please choose from: {list(self.activations.keys())}")
                raise ValueError(
                    f"Unsupported activation function: '{activation}'. "
                    f"Please choose from: {list(self.activations.keys())}"
                )


class Activations:
    """
    A utility class for managing the addition of various activation functions
    to neural network layers in a framework-agnostic way.

    This class provides methods to add activation layers to neural network models,
    supporting both TensorFlow/Keras and PyTorch frameworks. The framework is
    determined by the ML_FRAMEWORK environment variable.

    Environment Variable:
        ML_FRAMEWORK: Defines the framework to use ('tensorflow' or 'pytorch').
                     Default: 'tensorflow'

    Supported Activation Functions:
        - 'leakyrelu': LeakyReLU activation
        - 'relu': ReLU activation
        - 'prelu': PReLU activation
        - 'sigmoid': Sigmoid activation
        - 'tanh': Tanh activation
        - 'elu': ELU activation
        - 'glu': GLU activation
        - 'logsigmoid': LogSigmoid activation
        - 'celu': CeLU activation
        - 'softsign': SoftSign activation
        - 'softmax': Softmax activation
        - 'swish': Swish activation
        - 'softplus': Softplus activation
        - 'hardsigmoid': HardSigmoid activation
        - 'selu': SELU activation
        - 'exponential': Exponential activation
        - 'linear': Linear activation (identity)

    Example:
        >>> import os
        >>> os.environ['ML_FRAMEWORK'] = 'tensorflow'  # or 'pytorch'
        >>> activations = Activations()
        >>> # For TensorFlow
        >>> activated_layer = activations._add_activation_layer(layer, 'relu')
        >>> # For PyTorch
        >>> activated_tensor = activations._add_activation_layer(tensor, 'relu')
    """

    def __init__(self):
        """
        Initializes the Activations class.

        Selects the appropriate implementation based on the ML_FRAMEWORK
        environment variable.
        """
        self._framework = ML_FRAMEWORK

        if self._framework == 'tensorflow':
            self._implementation = TensorFlowActivationImpl()
        elif self._framework == 'pytorch':
            self._implementation = PyTorchActivationImpl()
        else:
            raise ValueError(f"Unsupported framework: {self._framework}")

    def _add_activation_layer(self, neural_model: Any, activation: str) -> Any:
        """
        Adds the specified activation function to a given neural network layer or tensor.

        This method is framework-agnostic and will apply the appropriate activation
        based on the ML_FRAMEWORK environment variable.

        Args:
            neural_model: The neural network layer (TensorFlow) or tensor (PyTorch)
                         to which the activation function will be applied.
            activation (str): The name of the activation function to add.

        Returns:
            The layer or tensor with the specified activation function applied.

        Raises:
            ValueError: If the specified activation function is not supported.

        Example:
            >>> activations = Activations()
            >>> activated = activations._add_activation_layer(layer, 'relu')
        """
        return self._implementation.add_activation(neural_model, activation)

    @staticmethod
    def _add_activation_layer_static(neural_model: Any, activation: str) -> Any:
        """
        Static method for backward compatibility with the original implementation.

        This method creates a temporary instance and calls the instance method.

        Args:
            neural_model: The neural network layer or tensor
            activation (str): The name of the activation function to add

        Returns:
            The layer or tensor with the specified activation function applied

        Raises:
            ValueError: If the specified activation function is not supported
        """
        instance = Activations()
        return instance._add_activation_layer(neural_model, activation)

    @property
    def framework(self) -> str:
        """
        Returns the framework being used.

        Returns:
            str: The name of the framework ('tensorflow' or 'pytorch')
        """
        return self._framework

    @property
    def supported_activations(self) -> list:
        """
        Returns a list of supported activation functions.

        Returns:
            list: List of supported activation function names
        """
        return list(self._implementation.activations.keys())