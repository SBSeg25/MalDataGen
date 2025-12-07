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

    import tensorflow

    from Engine.Activations.GLU import GLU
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import Dropout

    from tensorflow.keras.layers import Activation

    from tensorflow.keras.layers import DepthwiseConv1D

    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.layers import LayerNormalization

except ImportError as error:
    print(error)
    sys.exit(-1)

DEFAULT_NUMBER_FILTERS = 128
DEFAULT_SIZE_KERNEL = 3
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_CONVOLUTIONAL_PADDING = "same"


class ConvolutionalModule(Layer):
    """
    A convolutional module that applies a series of transformations to a 1D
    tensor, including point-wise convolutions, depth-wise convolutions,
    activations, batch normalization, and dropout. This module is designed
    to capture complex features in sequential data.

    The module consists of the following transformations:
        1. **Point-wise Convolution**: A 1x1 convolution (also known as point-wise
         convolution), used to adjust the number of filters.
        2. **GLU Activation**: Gated Linear Units (GLU) to introduce a non-linearity,
         improving the network's ability to model complex patterns.
        3. **Depth-wise Convolution**: A depth-wise separable convolution that applies
         a convolutional kernel to each input channel independently.
        4. **Batch Normalization**: Normalizes activations across the batch to improve
         training speed and stability.
        5. **Swish Activation**: A smooth non-linear activation function that can
         improve performance over ReLU.
        6. **Dropout**: Applied to prevent overfitting during training by randomly
         setting a fraction of the input units to zero.
        7. **Residual Connection**: The input is added to the output of the module,
         helping to mitigate the vanishing gradient problem.

    Args:
        @number_filters (int): The number of filters for the convolutional layers.
         Default is 64.
        @size_kernel (int): The size of the kernel for the depth-wise convolution.
         Default is 3.
        @dropout_decay (float): The dropout rate for the dropout layer.
         Default is 0.5.
        @convolutional_padding (str): Padding type for the convolutional layers
         ('same' or 'valid'). Default is 'same'.
        **kwargs: Additional keyword arguments passed to the base Layer class.

    Attributes:
        @convolutional_padding (str): Padding type for the convolutional layers.
        @layer_normalization (LayerNormalization): Layer normalization to stabilize
         the learning process.
        @first_point_wise_convolutional (Conv1D): Point-wise convolution (1x1 convolution)
         applied to the input tensor.
        @glu_activation (GLU): Gated Linear Unit activation applied after the first
         point-wise convolution.
        @depth_wise_convolutional (DepthwiseConv1D): Depth-wise convolution applied
         to the tensor.
        @batch_normalization (BatchNormalization): Batch normalization applied to
         the activations.
        @swish_activation (Activation): Swish activation applied after batch normalization.
        @second_point_wise_convolutional (Conv1D): Another point-wise convolution applied
         to the output of the depth-wise convolution.
        @dropout (Dropout): Dropout layer applied to prevent overfitting.

    Example
    -------
        >>> # Create an instance of the convolutional module with custom parameters
        ...     conv_module = ConvolutionalModule(number_filters=128, size_kernel=3)
        ...     # Example input tensor with shape (batch_size, sequence_length, num_features)
        ...     input_tensor = tf.random.normal([32, 100, 64])  # Batch of 32, sequence length of 100, and 64 features
        ...     # Apply the convolutional module to the input tensor
        ...     output_tensor = conv_module(input_tensor)
        ...     # Print the output shape
        ...     print("Output tensor shape:", output_tensor.shape)

    Output:
    -------
    Output tensor shape: (32, 100, 1)
    """

    def __init__(self,
                 number_filters: int = DEFAULT_NUMBER_FILTERS,
                 size_kernel: int = DEFAULT_SIZE_KERNEL,
                 dropout_decay: float = DEFAULT_DROPOUT_RATE,
                 convolutional_padding: str = DEFAULT_CONVOLUTIONAL_PADDING,
                 **kwargs):
        """
        Initializes the ConvolutionalModule with the specified parameters.

        Parameters
        ----------
            number_filters : int Number of filters for the convolutional layers.
            size_kernel : int Size of the kernel for the depth-wise convolutional layer.
            dropout_decay : float Dropout rate for the dropout layer.
            convolutional_padding : str Padding type for the convolutional layers.
            **kwargs : Additional keyword arguments.
        """
        # Calling the parent class (Layer) constructor to initialize the base layer
        super(ConvolutionalModule, self).__init__(**kwargs)

        # Defining the layer variables with provided or default values
        # Padding type for convolutions
        self.convolutional_padding = convolutional_padding

        # Layer normalization to stabilize the learning process
        self.layer_normalization = LayerNormalization()

        # First point-wise convolution (1x1), increasing the number of filters
        self.first_point_wise_convolutional = Conv1D(number_filters * 2, kernel_size=1)

        # GLU (Gated Linear Unit) activation
        self.glu_activation = GLU()

        # Depth-wise convolution, which applies a filter to each input channel individually
        self.depth_wise_convolutional = DepthwiseConv1D(kernel_size=size_kernel, padding=self.convolutional_padding)

        # Batch normalization after the depth-wise convolution
        self.batch_normalization = BatchNormalization()

        # Swish activation function
        self.swish_activation = Activation(tensorflow.nn.swish)

        # Second point-wise convolution (1x1) to project the results to the desired output
        self.second_point_wise_convolutional = Conv1D(1, kernel_size=1)

        # Dropout layer to prevent overfitting
        self.dropout = Dropout(dropout_decay)

    def call(self, neural_network_flow: tensorflow.Tensor) -> tensorflow.Tensor:
        """
        Applies the convolutional transformations to the input tensor.

        Parameters
        ----------
            neural_network_flow : tf.Tensor
                Input tensor to be processed by the convolutional module.

        Returns
        -------
            tf.Tensor
                Output tensor after applying the convolutional transformations.
        """
        # Storing the original input for applying the residual connection later
        residual_flow = neural_network_flow

        # Normalizing the input before the convolutions to stabilize the training process
        neural_network_flow = self.layer_normalization(neural_network_flow)

        # Applying the first point-wise convolution (1x1)
        neural_network_flow = self.first_point_wise_convolutional(neural_network_flow)

        # Applying the GLU activation (Gated Linear Unit)
        neural_network_flow = self.glu_activation(neural_network_flow)

        # Applying the depth-wise convolution
        neural_network_flow = self.depth_wise_convolutional(neural_network_flow)

        # Normalizing the output of the depth-wise convolution
        neural_network_flow = self.batch_normalization(neural_network_flow)

        # Applying the Swish activation
        neural_network_flow = self.swish_activation(neural_network_flow)

        # Applying the second point-wise convolution (1x1)
        neural_network_flow = self.second_point_wise_convolutional(neural_network_flow)

        # Applying the dropout layer to prevent overfitting
        neural_network_flow = self.dropout(neural_network_flow)

        # Adding the original input (residual connection) back to the final output
        return neural_network_flow + residual_flow  # Returning the output with the residual connection
