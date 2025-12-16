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

class CSVDataProcessor:
    """
    CSVDataProcessor - A comprehensive class for processing CSV data with advanced data handling capabilities.

    This class provides a robust interface for loading, filtering, normalizing, and saving tabular data from CSV files.
    It supports sophisticated data processing pipelines including column selection, sample limitation, class balancing,
    data normalization, and missing value handling. The processor maintains data integrity throughout all operations.

    Attributes:
    -----------
    data_load_label_column : str
        The name of the column containing class labels. This column will be treated specially during processing.
    data_load_max_samples : int
        Maximum number of samples to load (-1 for unlimited). Useful for working with large datasets.
    data_load_max_columns : int
        Maximum number of columns to retain (-1 for all columns). Enables dimensionality control.
    data_load_start_column : int
        Starting column index for column selection (0-based). Used with data_load_end_column for range selection.
    data_load_end_column : int
        Ending column index for column selection (inclusive). Defines the upper bound of column range.
    data_load_path_file_input : str
        Path to the input CSV file. Supports both relative and absolute paths.
    data_load_path_file_output : str
        Destination path for processed data output. Directory should be writable.
    data_load_exclude_columns : list[str]
        List of column names to explicitly exclude from processing.
    number_samples_per_class : dict[int, int]
        Dictionary specifying desired sample counts per class for balanced loading.
    data_loaded : pd.DataFrame or None
        The main DataFrame containing processed data. Initialized after load_csv().
    data_loaded_labels : pd.Series or None
        Extracted label series from the label column. Maintains original label order.
    data_loaded_header : list[str] or None
        Final column headers after all filtering operations.
    data_original_header : list[str] or None
        Original column headers before any processing. Preserved for reference.
    data_scaler : sklearn.preprocessing.MinMaxScaler
        Scaler instance for normalization. Configured during normalization.
    scaler_params : tuple or None
        (min, max) values used for normalization. Enables reproducible scaling.
    list_folds : list
        Contains dataset splits for cross-validation. Populated during fold creation.

    Methods:
    --------
    load_csv()
        Loads and processes the CSV file according to configuration parameters.
    normalize_data()
        Applies min-max normalization to scale features to specified range.
    save_processed_data()
        Writes the processed data to the output path in CSV format.
    get_number_columns()
        Returns the count of remaining columns after processing.
    get_number_samples()
        Returns the total number of samples loaded.

    Example Usage:
    --------------
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class arguments:
    ...     data_load_label_column: str = 'class'
    ...     data_load_max_samples: int = 1000
    ...     data_load_max_columns: int = 10
    ...     data_load_start_column: int = 1
    ...     data_load_end_column: int = 5
    ...     data_load_path_file_input: str = 'data/input.csv'
    ...     data_load_path_file_output: str = 'data/output.csv'
    ...     data_load_exclude_columns: list = None
    ...     number_samples_per_class: dict = None

    >>> config = arguments(
    ...     data_load_exclude_columns=['id', 'timestamp'],
    ...     number_samples_per_class={0: 300, 1: 300, 2: 300}
    ... )

    >>> processor = CSVDataProcessor(config)
    >>> processor.load_csv()
    >>> processor.normalize_data()
    >>> print(f"Processed {processor.get_number_samples()} samples with {processor.get_number_columns()} features")
    >>> processor.save_processed_data()

    Notes:
    ------
    - All column indices are 0-based unless otherwise specified
    - Missing values are handled by automatic removal during loading
    - Infinite values will cause processing errors and must be handled externally
    - The class maintains original data integrity; all operations return new objects
    """

    def __init__(self, arguments):
        """
        Initializes the CSVDataProcessor with user-defined arguments.

        Args:
            arguments: An object containing dataset parameters, including file paths, label column,
                       excluded columns, and normalization settings.
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
        self._scaler_params = None  # To store scaler parameters
        self.list_folds = []

    def load_csv(self):
        """Load and process CSV data according to configuration parameters.

        Performs the following operations:
        1. Loads CSV file with error handling
        2. Cleans data (inf values and missing data)
        3. Applies column and row filters
        4. Separates labels from features
        5. Normalizes the data

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: For empty data or invalid column specifications
            exception: For other loading/processing errors
        """
        try:
            data_file = self._load_csv_file()
            self._validate_data_not_empty(data_file)
            data_file = self._clean_data(data_file)

            self._store_original_header(data_file)
            data_file = self._apply_sample_limits(data_file)
            data_file = self._filter_columns(data_file)
            data_file = self._process_label_column(data_file)
            data_file = self._apply_column_limits(data_file)

            self._store_final_data(data_file)
            self._normalize_data()

        except Exception as e:
            logging.error(f"Failed to load CSV: {str(e)}")
            raise

    def _load_csv_file(self):
        """Load CSV file with proper error handling."""
        logging.info(f"Attempting to load CSV from path: {self._data_load_path_file_input}")

        try:
            data_file = pandas.read_csv(self._data_load_path_file_input)
            logging.info(f"CSV loaded successfully with shape: {data_file.shape}")
            return data_file

        except FileNotFoundError:
            logging.error(f"CSV file not found at path: {self._data_load_path_file_input}")
            raise

        except Exception as e:
            logging.error(f"Error loading CSV: {str(e)}")
            raise

    @staticmethod
    def _validate_data_not_empty(data_file):
        """Validate that the loaded data is not empty."""

        if data_file.empty:
            logging.error("The CSV file is empty.")
            raise ValueError("The CSV file is empty.")

        logging.info("CSV file is not empty, proceeding with processing.")

    @staticmethod
    def _clean_data(data_file):
        """Clean data by handling infinite values and missing data."""

        logging.info("Replacing infinite values and removing missing data.")
        data_file.replace([numpy.inf, -numpy.inf], numpy.nan, inplace=True)
        data_file.dropna(inplace=True)
        logging.info(f"Data after cleaning has shape: {data_file.shape}")

        return data_file

    def _store_original_header(self, data_file):

        """Store the original column headers."""
        self._data_original_header = data_file.columns.tolist()
        logging.info("Original CSV header stored.")

    def _apply_sample_limits(self, data_file):
        """Limit number of samples if specified."""

        if self._data_load_max_samples and self._data_load_max_samples != -1:
            logging.info(f"Limiting the data to {self._data_load_max_samples} samples.")
            data_file = data_file.head(self._data_load_max_samples)
            logging.info(f"Data now has {data_file.shape[0]} samples after limiting.")

        return data_file

    def _filter_columns(self, data_file):

        """Apply column filtering based on configuration."""
        data_file = self._filter_column_range(data_file)
        data_file = self._exclude_columns(data_file)

        return data_file

    def _filter_column_range(self, data_file):
        """Filter columns between start and end columns if specified."""

        if self._data_load_start_column and self._data_load_end_column:
            logging.info(f"Filtering columns between {self._data_load_start_column} and {self._data_load_end_column}.")

            if (self._data_load_start_column not in data_file.columns or
                    self._data_load_end_column not in data_file.columns):
                msg = f"Columns {self._data_load_start_column} or {self._data_load_end_column} do not exist in CSV."
                logging.error(msg)
                raise ValueError(msg)

            start_idx = data_file.columns.get_loc(self._data_load_start_column)
            end_idx = data_file.columns.get_loc(self._data_load_end_column) + 1
            data_file = data_file.iloc[:, start_idx:end_idx]
            logging.info(f"Columns filtered successfully, new data shape: {data_file.shape}")

        return data_file

    def _exclude_columns(self, data_file):

        """Exclude specified columns if requested."""
        if self._data_load_exclude_columns and self._data_load_exclude_columns != -1:
            logging.info(f"Dropping specified columns: {self._data_load_exclude_columns}")
            data_file = data_file.drop(columns=self._data_load_exclude_columns)
            logging.info(f"Columns dropped successfully, new data shape: {data_file.shape}")

        return data_file

    def _process_label_column(self, data_file):

        """Process label column configuration and separate labels."""
        if self._data_load_label_column == -1:
            self._data_load_label_column = data_file.columns[-1]
            logging.info(f"Label column automatically set to: {self._data_load_label_column}")

        logging.info(f"Separating the label column: {self._data_load_label_column}")
        self._data_loaded_labels = numpy.array(data_file[[self._data_load_label_column]].values, dtype=numpy.float32)
        data_file = data_file.drop(columns=[self._data_load_label_column])
        logging.info(f"Labels separated, remaining data shape: {data_file.shape}")

        return data_file

    def _apply_column_limits(self, data_file):
        """Limit number of columns if specified."""

        if self._data_load_max_columns != -1 and self._data_load_max_columns:
            logging.info(f"Limiting the number of columns to {self._data_load_max_columns}.")

            if data_file.shape[1] > self._data_load_max_columns:
                data_file = data_file.iloc[:, :self._data_load_max_columns]
                logging.info(f"Number of columns limited, new data shape: {data_file.shape}")

        return data_file

    def _store_final_data(self, data_file):
        """Store the final processed data and headers."""

        self._data_loaded = data_file.values.astype(numpy.float32)
        self._data_loaded_header = data_file.columns.tolist() + [self._data_load_label_column]
        logging.info(f"Data loaded and header updated. Final data shape: {self._data_loaded.shape}")

    def get_number_columns(self):

        return self._data_loaded.shape[-1]

    def get_data(self):
        return self._data_loaded

    def get_features_by_label(self, label):
        """
        Returns the feature matrix containing only samples with the specified label.

        Args:
            label: The target label value to filter samples by.

        Returns:
            numpy.ndarray: A 2D array containing only the features of samples with the specified label.

        Raises:
            ValueError: If the data hasn't been loaded yet or if the specified label doesn't exist.
        """
        if self._data_loaded is None or self._data_loaded_labels is None:
            logging.error("Data has not been loaded yet.")
            raise ValueError("Data has not been loaded yet. Call load_csv() first.")

        # Find indices where labels match the target label
        label_indices = numpy.where(self._data_loaded_labels == label)[0]

        if len(label_indices) == 0:
            logging.warning(f"No samples found with label {label}.")
            return numpy.array([])  # Return empty array if no samples found

        # Return only the rows corresponding to the label
        return self._data_loaded[label_indices]

    def _normalize_data(self):
        """
        Normalizes the data between 0 and 1 using MinMaxScaler.
        """
        self._data_loaded = numpy.array(self._data_scaler.fit_transform(self._data_loaded), dtype=numpy.float32)

        # Store the scaler parameters for later use
        self._scaler_params = (self._data_scaler.data_min_, self._data_scaler.data_max_)

        # Logging to inform that the data has been normalized
        logging.info("Data normalized between 0 and 1.")

    def save_csv(self, generated_data, fold_number, directory_name, generator_name):
        """
        Saves the processed data to a new CSV file, applying inverse normalization if needed.
        """
        logging.info(f"Starting to save CSV for fold number {fold_number}.")
        labels, data = [], []

        # Logging the size of the generated data
        total_samples = sum(len(samples) for samples in generated_data.values())
        logging.info(f"Total number of samples to be saved: {total_samples}")

        for label_class, generated_samples in generated_data.items():

            logging.info(f"Processing {len(generated_samples)} samples for label class {label_class}.")
            labels.extend([label_class] * len(generated_samples))
            data.extend(generated_samples)

        logging.info("Combining labels and data into a DataFrame.")
        # Create a DataFrame with the normalized data and add the labels
        try:
            data_file_output = pandas.DataFrame(data, columns=self._data_loaded_header[:-1])
            data_file_output[self._data_loaded_header[-1]] = labels
            logging.info(f"DataFrame created successfully with {data_file_output.shape[0]} rows and {data_file_output.shape[1]} columns.")

        except Exception as e:
            logging.error(f"Error creating DataFrame: {str(e)}")
            raise

        # Revert normalization if scaler_params are available
        if self._scaler_params:

            logging.info("Reverting normalization using stored scaler parameters.")

            try:
                data_min, data_max = self._scaler_params
                # Applying inverse transform to revert normalization
                data_file_output.iloc[:, :-1] = self._data_scaler.inverse_transform(data_file_output.iloc[:, :-1])
                logging.info("Normalization successfully reverted.")

            except Exception as e:
                logging.error(f"Error during normalization reversion: {str(e)}")
                raise

        else:
            logging.info("No normalization reversion needed as scaler_params are not provided.")

        # Save the DataFrame to a CSV file
        try:
            output_file_path = f"{directory_name}/DataOutput_K_fold_{fold_number}_{generator_name}.txt"
            data_file_output.to_csv(output_file_path, index=False)
            logging.info(f"Data successfully saved to {output_file_path}.")

        except Exception as e:
            logging.error(f"Error saving data to CSV: {str(e)}")
            raise

        # Logging the completion of the save process
        logging.info(f"CSV save operation for fold {fold_number} completed.")

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
        CSVDataProcessor.__init__(self, self.arguments)
        self.load_csv()
        # Call the wrapped function with the metrics instance and other arguments
        return function(self, *args, **kwargs)

    return wrapper


def autosave(function):

    def wrapper(self, *args, **kwargs):
        self.save_csv(self.data_generated, self.fold_number, self.directory_output_data, self.generator_name)
        return function(self, *args, **kwargs)

    return wrapper
