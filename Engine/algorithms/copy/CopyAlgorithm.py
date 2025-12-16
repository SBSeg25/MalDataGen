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

    import os
    import sys
    import numpy  

    import logging
    import tensorflow

    from pathlib import Path

    from tensorflow.keras.models import Model

    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import model_from_json

    from tensorflow.keras.losses import BinaryCrossentropy


except ImportError as error:
    logging.error(error)
    sys.exit(-1)


class CopyAlgorithm(Model):
    """
    copy is a naive machine learning model designed to generate synthetic data samples
    for specific classes based on provided real samples. This simple approach is primarily used
    for testing and comparison purposes, serving as a baseline method in experiments.

    This class extends the `Model` class from TensorFlow's Keras API, allowing integration into deep learning pipelines.

    Example:
        >>> python3
        ...     import numpy
        ...
        ...     # Define real samples and labels
        ...     x_real_samples = numpy.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        ...     y_real_samples = numpy.array([0, 1, 0, 1])
        ...
        ...     # Define the number of samples per class
        ...     number_samples_per_class = {
        ...     "classes": {0: 2, 1: 2},
        ...     "number_classes": 2
        ...     }
        ...
        ...     # Create the model instance
        ...     model = copy()
        ...
        ...     # Generate synthetic samples
        ...     generated_samples = model.get_samples(number_samples_per_class, x_real_samples, y_real_samples)
        ...     print(generated_samples)
        >>>
        ```

    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the copy model by calling the superclass constructor.

        Args:
            *args: Variable length argument list to be passed to the parent Model class.
            **kwargs: Arbitrary keyword arguments to be passed to the parent Model class.
        """
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_samples(number_samples_per_class, x_real_samples, y_real_samples):
        """
        Generates synthetic data samples for specified classes using real sample distributions.

        This naive approach selects samples from the provided dataset to create synthetic examples,
        ensuring that each class is represented by the specified number of samples. It does not apply
        any complex transformations or modifications to the data, making it a simple method for testing
        and comparison purposes.

        If no samples exist for a given class, a warning message is displayed.

        Args:
            number_samples_per_class (dict): A dictionary specifying the number of samples
                to generate for each class. Expected structure:
                {
                    "classes": {class_label: number_of_samples, ...},
                    "number_classes": total_number_of_classes
                }
            x_real_samples (numpy.ndarray): The array of real feature samples.
            y_real_samples (numpy.ndarray): The corresponding labels for feature samples.

        Returns:
            dict: A dictionary containing generated synthetic samples for each class.
                  Structure:
                  {
                      class_label: numpy.ndarray of shape (number_instances, feature_dim)
                  }
        """
        generated_data = {}

        for label_class, number_instances in number_samples_per_class["classes"].items():
            # Retrieve indices of samples belonging to the current class
            class_indices = numpy.where(y_real_samples == label_class)[0]

            if len(class_indices) == 0:
                print(f"Warning: No samples found for class {label_class}.")
                generated_data[label_class] = numpy.array([])
                continue

            # Randomly select samples from the available real samples for the class
            selected_indices = numpy.random.choice(class_indices, size=number_instances, replace=True)
            generated_samples = x_real_samples[selected_indices]

            # Store the generated samples in the output dictionary
            generated_data[label_class] = generated_samples

        return generated_data


     