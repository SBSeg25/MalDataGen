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
    import time

    import numpy
    import pandas
    import logging

    from sklearn.utils import shuffle

    from Engine.data_io.CSVLoader import autoload

    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import train_test_split

except ImportError as error:
    print(error)

    sys.exit(-1)

def _save_data_to_csv(directory_output_data, data, labels, filename_prefix, fold):
        """Helper function to save data and labels to CSV."""
        # Concatenate data and labels
        data_with_labels = numpy.column_stack((data, labels))
        
        # Create column names
        columns = [f"f{i}" for i in range(data.shape[1])] + ["label"]
        
        # Create and save DataFrame
        data_frame = pandas.DataFrame(data_with_labels, columns=columns)
        csv_filename = f"{directory_output_data}/{filename_prefix}_fold_{fold + 1}.csv"
        data_frame.to_csv(csv_filename, index=False)

def StratifiedData(function):

    

    """
    A decorator that performs stratified K-Fold cross-validation on the provided dataset.
    This method shuffles the dataset, splits it into `k` stratified folds, and saves
    the corresponding training and validation datasets as CSV files. It also logs detailed
    information about the process and each fold.

    Args:
        function (Callable): The function to be wrapped. After completing the stratified K-Fold
                              cross-validation process, this function will be called with the
                              provided arguments.

    Returns:
        Callable: The wrapped function with stratified K-Fold processing applied.
    """

    @autoload  # Custom decorator, used for auto loading data
    def wrapper(self, *args, **kwargs):

        # Track the start time of the entire stratified K-fold process
        start_time = time.time()

        logging.info(
            "StratifiedKFold initialization started. Number of splits: %d, Shuffle: True, Random state: 42.",
            self.arguments.number_k_folds)

        try:
            # Shuffle the data before performing stratified splitting
            shuffled_data, shuffled_labels = shuffle(self._data_loaded, self._data_loaded_labels, random_state=42)

            # Initialize the StratifiedKFold instance with the specified number of folds
            stratified_instance = StratifiedKFold(n_splits=self.arguments.number_k_folds,
                                                  shuffle=True,
                                                  random_state=42)  # RM TODO: Add parameter for random seed

            logging.info("Data loaded for stratification. Total samples: %d", len(self._data_loaded))

            # Loop through each fold and process the stratified splits
            for fold, (train_index, val_index) in enumerate(
                    stratified_instance.split(shuffled_data, shuffled_labels)):
                # Track the start time of processing a specific fold
                fold_start_time = time.time()

                logging.info("Processing fold %d of %d.", fold + 1, self.arguments.number_k_folds)

                
                
                logging.info("Training   indices (size %d): %s", len(train_index), train_index)
                logging.info("Evaluating indices (size %d): %s", len(val_index), val_index)

                 

                # # Concatenate the data and labels for the shuffled dataset
                # data_with_labels = numpy.column_stack((shuffled_data, shuffled_labels))

                # # Create a DataFrame for saving the shuffled dataset as CSV or XLS
                # columns = [f"feature_{i}" for i in range(shuffled_data.shape[1])] + ["label"]
                # data_frame = pandas.DataFrame(data_with_labels, columns=columns)

                # # Save the shuffled data to CSV
                # csv_filename = "{}/data_shuffled_dataset.csv".format(self.directory_output_data)
                # data_frame.to_csv(csv_filename, index=False)

                # Dentro do seu loop for fold:
                # Save training data
                _save_data_to_csv(
                    self.directory_output_data,
                    self._data_loaded[train_index],
                    self._data_loaded_labels[train_index],
                    "data_training",
                    fold
                )
 

                # Save validation data
                _save_data_to_csv(
                    self.directory_output_data,
                    self._data_loaded[val_index],
                    self._data_loaded_labels[val_index],
                    "data_evaluation",
                    fold
                ) 

                # Shuffle the training and evaluation data
                training_shuffled_data, training_shuffled_labels = shuffle(self._data_loaded[train_index],
                                                                           self._data_loaded_labels[train_index],
                                                                           random_state=42)
                
                

                evaluation_shuffled_data, evaluation_shuffled_labels = shuffle(self._data_loaded[val_index],
                                                                               self._data_loaded_labels[val_index],
                                                                               random_state=42)

                # Store the training and evaluation data for later use
                self.list_folds.append({
                    'x_training_real': training_shuffled_data,
                    'y_training_real': training_shuffled_labels,

                    'x_evaluation_real': evaluation_shuffled_data,
                    'y_evaluation_real': evaluation_shuffled_labels,
                    
                    'x_training_synthetic': None,
                    'y_training_synthetic': None,

                    'x_evaluation_synthetic': None,
                    'y_evaluation_synthetic': None
                })

                # Track the end time for this fold
                fold_end_time = time.time()

                logging.info("Fold %d processed successfully in %.2f seconds.", fold + 1,
                             fold_end_time - fold_start_time)

            # Track the end time for the entire K-Fold process
            end_time = time.time()

            logging.info("All %d folds successfully created in %.2f seconds.", self.arguments.number_k_folds,
                         end_time - start_time)

            # Call the original function passed as a decorator argument
            return function(self, *args, **kwargs)

        except Exception as e:
            logging.error("An error occurred during StratifiedKFold processing: %s", str(e))
            raise  # Re-raise the exception after logging the error

    return wrapper
