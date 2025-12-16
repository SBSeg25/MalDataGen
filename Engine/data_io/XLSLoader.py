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
    import numpy
    import pandas
    import logging

    from sklearn.preprocessing import MinMaxScaler

except ImportError as error:
    print(error)
    sys.exit(-1)


class XLSDataProcessor:
    """
    A class to process and manage XLS data files, including loading, saving, normalization, and handling
    of dataset attributes such as labels and features. This class provides functionalities for managing
    dataset preprocessing steps such as cleaning, filtering, and handling missing data, as well as scaling
    features for machine learning tasks.

    Attributes:
        @_data_load_label_column (str):
            The name of the column containing the labels in the input dataset.
        @_data_load_max_samples (int):
            The maximum number of samples to load from the XLS file. If set to -1, all samples are loaded.
        @_data_load_max_columns (int):
            The maximum number of columns to load. If set to -1, all columns are loaded.
        @_data_load_start_column (int):
            The index of the starting column to load from the dataset.
        @_data_load_end_column (int):
            The index of the ending column to load from the dataset.
        @_data_load_path_file_input (str):
            The file path of the input XLS file.
        @_data_load_path_file_output (str):
            The file path where the processed XLS data will be saved.
        @_data_load_exclude_columns (list):
            List of column names to exclude from the dataset when loading.
        @_number_samples_per_class (dict):
            A dictionary mapping class labels to the number of samples for each class.
        @_data_loaded (numpy.ndarray):
            The processed data (features) as a NumPy array.
        @_data_loaded_labels (numpy.ndarray):
            The labels extracted from the dataset as a NumPy array.
        @_data_loaded_header (list):
            The header of the processed dataset (features + label column).
        @_data_original_header (list):
            The original column headers from the input XLS file.
        @_data_scaler (MinMaxScaler):
            A MinMaxScaler used for normalizing the feature data between 0 and 1.
        @_scaler_params (tuple):
            Stores the parameters of the scaler, such as the minimum and maximum values used for normalization.
        @list_folds (list):
            A list of dataset folds used for cross-validation during training.

    Example:
        >>> arguments = arguments(
        ...     data_load_label_column='Label',
        ...     data_load_max_samples=1000,
        ...     data_load_max_columns=50,
        ...     data_load_start_column=0,
        ...     data_load_end_column=49,
        ...     data_load_path_file_input='data/input.xlsx',
        ...     data_load_path_file_output='data/output.xlsx',
        ...     data_load_exclude_columns=['ID', 'Timestamp'],
        ...     number_samples_per_class={'ClassA': 500, 'ClassB': 500}
        ... )
        >>> processor = XLSDataProcessor(arguments)
    """

    def __init__(self, arguments):
        """
        Initializes the XLSDataProcessor with user-defined arguments. These arguments are used to configure
        the dataset loading, processing, and saving behaviors.

        Args:
            arguments: An object that contains various parameters to configure the data processing.
                These parameters include paths for input and output files, label columns,
                columns to exclude, maximum sample sizes, and other options for dataset processing.
        """
        self._data_load_label_column = arguments.data_load_label_column
        self._data_load_max_samples = arguments.data_load_max_samples
        self._data_load_max_columns = arguments.data_load_max_columns
        self._data_load_start_column = arguments.data_load_start_column
        self._data_load_end_column = arguments.data_load_end_column
        self._data_load_path_file_input = arguments.data_load_path_file_input
        self._data_load_path_file_output = arguments.data_load_path_file_output
        self._data_load_exclude_columns = arguments.data_load_exclude_columns
        self._number_samples_per_class = arguments.number_samples_per_class
        self._data_loaded = None
        self._data_loaded_labels = None
        self._data_loaded_header = None
        self._data_original_header = None
        self._data_scaler = MinMaxScaler(feature_range=(0, 1))
        self._scaler_params = None
        self.list_folds = []

    def load_xls(self):
        """
        Loads an XLS file from a specified path, processes the data by cleaning and filtering,
        and stores the structured dataset for further use.

        Steps:
        1. Reads the XLS file from the defined path.
        2. Handles missing files and errors gracefully.
        3. Replaces infinite values with NaN and drops missing values.
        4. Extracts and optionally limits the number of samples.
        5. Selects specific columns based on user-defined parameters.
        6. Extracts labels and normalizes data.

        Raises:
            FileNotFoundError: If the file is not found at the given path.
            ValueError: If the loaded file is empty.
            exception: If any other error occurs while loading the file.
        """
        logging.info(f"Attempting to load XLS from path: {self._data_load_path_file_input}")

        try:
            # Read the XLS file
            data_file = pandas.read_excel(self._data_load_path_file_input)
            logging.info(f"XLS loaded successfully with shape: {data_file.shape}")

        except FileNotFoundError:
            logging.error(f"XLS file not found at path: {self._data_load_path_file_input}")
            raise

        except Exception as e:
            logging.error(f"Error loading XLS: {str(e)}")
            raise

        # Check if the file is empty
        if data_file.empty:
            logging.error("The XLS file is empty.")
            raise ValueError("The XLS file is empty.")

        logging.info("XLS file is not empty, proceeding with processing.")

        # Handle infinite values and drop missing values
        data_file.replace([numpy.inf, -numpy.inf], numpy.nan, inplace=True)
        data_file.dropna(inplace=True)

        # Store original column headers
        self._data_original_header = data_file.columns.tolist()

        # Limit the number of samples if specified
        if self._data_load_max_samples and self._data_load_max_samples != -1:
            data_file = data_file.head(self._data_load_max_samples)

        # Select a subset of columns based on start and end column names
        if self._data_load_start_column and self._data_load_end_column:
            start_idx = data_file.columns.get_loc(self._data_load_start_column)
            end_idx = data_file.columns.get_loc(self._data_load_end_column) + 1
            data_file = data_file.iloc[:, start_idx:end_idx]

        # Exclude specified columns if defined
        if self._data_load_exclude_columns and self._data_load_exclude_columns != -1:
            data_file = data_file.drop(columns=self._data_load_exclude_columns)

        # Define label column if not specified
        if self._data_load_label_column == -1:
            self._data_load_label_column = data_file.columns[-1]

        # Extract labels and convert to numpy array
        self._data_loaded_labels = numpy.array(data_file[[self._data_load_label_column]].values, dtype=numpy.float32)

        # Remove label column from the data
        data_file = data_file.drop(columns=[self._data_load_label_column])

        # Limit the number of features (columns) if specified
        if self._data_load_max_columns != -1 and data_file.shape[1] > self._data_load_max_columns:
            data_file = data_file.iloc[:, :self._data_load_max_columns]

        # Convert data to a numpy array and store header information
        self._data_loaded = data_file.values.astype(numpy.float32)
        self._data_loaded_header = data_file.columns.tolist() + [self._data_load_label_column]

        # Normalize the data
        self._normalize_data()

    def save_xls(self, generated_data, fold_number, directory_name, generator_name):
        """
        Saves generated data into an XLS file with structured formatting.

        Steps:
        1. Organizes generated samples and labels into a structured DataFrame.
        2. Applies inverse scaling if required.
        3. Saves the DataFrame as an XLS file in the specified directory.

        Args:
            generated_data (dict): Dictionary containing label classes as keys and lists of generated samples as values.
            fold_number (int): The fold number for the output file name.
            directory_name (str): The directory path where the file will be saved.
            generator_name (str): Identifier for the generator used in the filename.

        Raises:
            exception: If an error occurs while creating the DataFrame or saving the file.
        """
        logging.info(f"Starting to save XLS for fold number {fold_number}.")
        labels, data = [], []

        # Organize generated data into a structured list
        for label_class, generated_samples in generated_data.items():
            labels.extend([label_class] * len(generated_samples))
            data.extend(generated_samples)

        try:
            # Create DataFrame with structured data
            data_file_output = pandas.DataFrame(data, columns=self._data_loaded_header[:-1])
            data_file_output[self._data_loaded_header[-1]] = labels
        except Exception as e:
            logging.error(f"Error creating DataFrame: {str(e)}")
            raise

        # Apply inverse scaling if scaler parameters exist
        if self._scaler_params:
            data_min, data_max = self._scaler_params
            data_file_output.iloc[:, :-1] = self._data_scaler.inverse_transform(data_file_output.iloc[:, :-1])

        try:
            # Construct output file path
            output_file_path = f"{directory_name}/DataOutput_K_fold_{fold_number}_{generator_name}.xlsx"
            # Save DataFrame as an XLS file
            data_file_output.to_excel(output_file_path, index=False)
            logging.info(f"Data successfully saved to {output_file_path}.")
        except Exception as e:
            logging.error(f"Error saving data to XLS: {str(e)}")
            raise

    def _normalize_data(self):
        """
        Normalizes the data between 0 and 1 using MinMaxScaler.
        """
        self._data_loaded = numpy.array(self._data_scaler.fit_transform(self._data_loaded), dtype=numpy.float32)

        # Store the scaler parameters for later use

        self._scaler_params = (self._data_scaler.data_min_, self._data_scaler.data_max_)

        # Logging to inform that the data has been normalized
        logging.info("Data normalized between 0 and 1.")


    def get_number_columns(self):

        return self._data_loaded.shape[-1]

    def get_number_classes(self):
        """
        Returns the number of unique classes in the labels.
        """
        number_classes = numpy.unique(self._data_loaded_labels).shape[0]

        # Logging to inform the number of classes
        logging.info(f"Number of classes: {number_classes}")

        return number_classes

    # Getter and Setter for label_column
    @property
    def data_load_label_column(self):
        return self._data_load_label_column

    @data_load_label_column.setter
    def data_load_label_column(self, value):
        self._data_load_label_column = value

    # Getter and Setter for max_samples
    @property
    def data_load_max_samples(self):
        return self._data_load_max_samples

    @data_load_max_samples.setter
    def data_load_max_samples(self, value):
        self._data_load_max_samples = value

    # Getter and Setter for max_columns
    @property
    def data_load_max_columns(self):
        return self._data_load_max_columns

    @data_load_max_columns.setter
    def data_load_max_columns(self, value):
        self._data_load_max_columns = value

    # Getter and Setter for start_column
    @property
    def data_load_start_column(self):
        return self._data_load_start_column

    @data_load_start_column.setter
    def data_load_start_column(self, value):
        self._data_load_start_column = value

    # Getter and Setter for end_column
    @property
    def data_load_end_column(self):
        return self._data_load_end_column

    @data_load_end_column.setter
    def data_load_end_column(self, value):
        self._data_load_end_column = value

    # Getter and Setter for path_file_input
    @property
    def data_load_path_file_input(self):
        return self._data_load_path_file_input

    @data_load_path_file_input.setter
    def data_load_path_file_input(self, value):
        self._data_load_path_file_input = value

    # Getter and Setter for path_file_output
    @property
    def data_load_path_file_output(self):
        return self._data_load_path_file_output

    @data_load_path_file_output.setter
    def data_load_path_file_output(self, value):
        self._data_load_path_file_output = value

    # Getter and Setter for exclude_columns
    @property
    def data_load_exclude_columns(self):
        return self._data_load_exclude_columns

    @data_load_exclude_columns.setter
    def data_load_exclude_columns(self, value):
        self._data_load_exclude_columns = value


def autoload(function):
    """
    Decorator to create an instance of the metrics class
    before executing the wrapped function.

    Parameters:
        function (callable): The function to be wrapped.

    Returns:
        callable: The wrapped function that initializes metrics.
    """
    def wrapper(self, *args, **kwargs):
        # Create an instance of metrics, passing the arguments from the instance
        XLSDataProcessor.__init__(self, self.arguments)
        self.load_xls()
        # Call the wrapped function with the metrics instance and other arguments
        return function(self, *args, **kwargs)

    return wrapper


def autosave(function):

    def wrapper(self, *args, **kwargs):
        self.save_xls(self.data_generated, self.fold_number, self.directory_output_data, self.generator_name)
        return function(self, *args, **kwargs)

    return wrapper