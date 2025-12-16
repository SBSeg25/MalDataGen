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
    import pandas 

    import logging
    import tensorflow

    from pathlib import Path

    from tensorflow.keras.models import Model

    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import model_from_json

    from tensorflow.keras.losses import BinaryCrossentropy

    from sdv.single_table import CTGANSynthesizer
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.single_table import TVAESynthesizer

    from sdv.metadata import Metadata
    from sdv.sampling import Condition



except ImportError as error:
    logging.error(error)
    sys.exit(-1)


class SDVInterfaceAlgorithm:
    """
    SDVInterfaceAlgorithm

    Interface layer for integrating SDV (Synthetic Data Vault) models within the SynDataGen pipeline.

    This class encapsulates SDV's synthesizers (Gaussian Copula, CTGAN, and TVAE) and provides
    a standardized interface for training and sampling synthetic data conditioned on class labels.

    Designed to be used as a backend algorithm within the SynDataGen system, this class simplifies
    interaction with SDV’s metadata handling, conditional sampling, and model selection logic.

    Features:
    ---------
    - Automated metadata inference from real datasets
    - support for conditional generation per class
    - Naive handling of categorical/textual columns (temporary replacement)
    - Structured output for downstream ML tasks or augmentation
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor for SDVInterfaceAlgorithm.

        Acts as a lightweight interface component. Stores the internal synthesizer
        instance after initialization via `training_model`.

        Args:
            *args: Optional arguments for compatibility and extensibility.
            **kwargs: Optional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self._synthesizer = None 

        

    def training_model(self, x_real_samples, y_real_samples, columns, algorithm):
        """
        Fits the selected SDV model to real input data.

        Automatically appends the class column to the input feature matrix and detects
        the required metadata for SDV operation. This method prepares the internal state
        of the synthesizer for future sample generation.

        Parameters:
        -----------
        x_real_samples (array-like): Feature matrix of real samples.
        y_real_samples (array-like): Corresponding target class labels.
        columns (list): Column names for the feature matrix and target label.
                        The last element must correspond to the 'class' column.
        algorithm (str): SDV algorithm to use: one of ['copula', 'ctgan', 'tvae'].

        Raises:
        -------
        ValueError: If an unsupported algorithm is specified.
        """

        # Assemble a DataFrame with feature and class columns
        real_data = pandas.DataFrame(x_real_samples, columns=columns[:-1]) #last column is 'class', which will be added below
        
        real_data['class'] = y_real_samples

        # Generate SDV metadata from the DataFrame
        metadata = Metadata.detect_from_dataframe(data=real_data, table_name='table')

        # Select and instantiate appropriate SDV synthesizer
        if algorithm == "copula":
            self._synthesizer = GaussianCopulaSynthesizer(metadata)

        elif algorithm == "ctgan":
            self._synthesizer = CTGANSynthesizer(metadata)

        elif algorithm == "tvae":
            self._synthesizer = TVAESynthesizer(metadata)

        else:
            logging.error(f"algorithm not found! {algorithm}")

        # Fit the synthesizer to the training data
        self._synthesizer.fit(real_data)

    def get_samples(self, number_samples_per_class):
        """
        Generates synthetic feature samples for each target class with proper type handling.
        
        Parameters:
        -----------
        number_samples_per_class (dict): 
            {
                "classes": {
                    class_label (int/str): num_samples (int),
                    ...
                },
                "number_classes": int
            }

        Returns:
        --------
        dict: {class_label: np.ndarray of samples}
        """
        generated_data = {}
        conditions = []

        for label_class, number_instances in number_samples_per_class["classes"].items():

            synthetic_data = self._synthesizer.sample(num_rows=number_instances*5, )
            conditions.append(
                Condition(
                num_rows=int(number_instances),
                column_values={'class': label_class}
                )
            )
            
        sdv_data = self._synthesizer.sample_from_conditions(conditions=conditions, output_file_path=None)
           # Separate features (X) and target (y)
        sdv_x_samples = sdv_data.iloc[:, :-1].values  # Convert to numpy array

        #substitute any string to 1 
        sdv_x_samples =  numpy.where(
            numpy.vectorize(lambda x: isinstance(x, str))(sdv_x_samples),
            1,
            sdv_x_samples
        )
        
        sdv_y_samples = sdv_data.iloc[:, -1].values   # Convert to numpy array
        
        logging.info(f"\nFeature samples shape: {sdv_x_samples.shape}")
        logging.info(f"Target samples shape: {sdv_y_samples.shape}")
        logging.info("\nFirst 5 feature samples:")
        logging.info(sdv_x_samples[:5])
        logging.info("\nFirst 5 target samples:")
        logging.info(sdv_y_samples[:5])
        
        generated_data = {}
        
        for label_class, number_instances in number_samples_per_class["classes"].items():
            logging.info(f"\nProcessing class {label_class} (requested samples: {number_instances})")
            
            try:
                # Get indices of samples belonging to current class
                class_indices = numpy.where(sdv_y_samples == label_class)[0]
                
                if len(class_indices) == 0:
                    raise ValueError(f"No samples found for class {label_class}")
                    
                logging.info(f"Available samples for class: {len(class_indices)}")
                
                # Randomly select with replacement
                selected_indices = numpy.random.choice(
                    class_indices,
                    size=number_instances,
                    replace=True  # Allows oversampling if needed
                )
                
                ## Store selected samples
                class_samples = sdv_x_samples[selected_indices]
                generated_data[label_class] = class_samples
                logging.info(f"Successfully generated {len(class_samples)} samples")
                
            except Exception as e:
                logging.error(f"Error processing class {label_class}: {str(e)}")
                continue

        # Final log summary
        logging.info("\nGeneration complete")
        logging.info(f"Generated {sum(len(v) for v in generated_data.values())} total samples")
        
        return generated_data
