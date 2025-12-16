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

    import shutil
    import logging

    from datetime import datetime

except ImportError as error:
    print(error)
    sys.exit(-1)


class DirectoryManager:
    """
    A class responsible for managing directory structures for various purposes,
    such as storing models, logs, generated data, evaluation results, and monitoring files.

    This class provides methods for:
    - Creating a base directory and subdirectories.
    - Retrieving paths of specific subdirectories (Logs, Monitor, ModelsSaved, DataGenerated, EvaluationResults).
    - Compressing the current directory into a ZIP file and cleaning up the original directory.

    Attributes:
        @base_directory (str): The base directory where all subdirectories will be stored (default is "Results").
        @current_subdir (str): The path to the current subdirectory, which is timestamped when created.

    Methods:
        @__init__(): Initializes the DirectoryManager, setting the base directory and preparing the subdirectory.
        @_create_directories(base_directory=None): Creates the base directory and necessary subdirectories.
        @get_logs_path(): Returns the path of the Logs subdirectory.
        @get_monitor_path(): Returns the path of the Monitor subdirectory.
        @get_models_saved_path(): Returns the path of the ModelsSaved subdirectory.
        @get_data_generated_path(): Returns the path of the DataGenerated subdirectory.
        @get_evaluation_results_path(): Returns the path of the EvaluationResults subdirectory.
        @zip_and_cleanup(): Compresses the current directory into a ZIP file and removes the original directory.

    Example:
        >>>     dir_manager = DirectoryManager()
        ...     dir_manager._create_directories()
        ...     logs_path = dir_manager.get_logs_path()
        >>>     dir_manager.zip_and_cleanup()
    """

    def __init__(self):
        """
        Initializes the DirectoryManager instance.

        The constructor sets the base directory to "Results" and prepares for subdirectory creation.
        The current_subdir attribute is set to None initially, and will be assigned a timestamped value
        when directories are created.

        Attributes:
            @base_directory (str): Default base directory "Results".
            @current_subdir (str): Placeholder for the current subdirectory path.
        """
        self.base_directory = "Results"
        self.current_subdir = None

    def _create_directories(self, base_directory=None):
        """
        Creates the main and subdirectories under the specified base directory.

        If no base directory is provided, a timestamped subdirectory will be created inside the "Results" directory.

        Args:
            base_directory (str, optional): The base directory to use. If None, it defaults to "Results".

        This method ensures the creation of the following subdirectories:
            - ModelsSaved
            - Logs
            - DataGenerated
            - EvaluationResults
            - Monitor

        Logs the creation process, including any errors encountered during directory creation.

        Raises:
            OSError: If there is an error during directory creation.
        """
        if base_directory is None:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.current_subdir = os.path.join(self.base_directory, current_time)
        else:
            self.base_directory = base_directory
            self.current_subdir = self.base_directory

        try:
            # Ensure the base directory exists
            if not os.path.exists(self.base_directory):
                os.makedirs(self.base_directory)
                logging.info(f"Created base directory: {self.base_directory}")
            else:
                logging.info(f"Base directory already exists: {self.base_directory}")

            # Create the timestamped subdirectory
            if not os.path.exists(self.current_subdir):
                os.makedirs(self.current_subdir)

            # Define and create necessary subdirectories
            sub_directories = ['ModelsSaved', 'Logs', 'DataGenerated', 'EvaluationResults', 'Monitor']
            for subdir in sub_directories:
                os.makedirs(os.path.join(self.current_subdir, subdir))
                logging.info(f"Created subdirectory: {os.path.join(self.current_subdir, subdir)}")

        except OSError as e:
            logging.error(f"Failed to create directories: {e}")
            print(f"An error occurred while creating directories: {e}")

    def get_logs_path(self):
        """
        Retrieves the path to the "Logs" directory.

        This method returns the absolute path of the "Logs" subdirectory located under the current subdirectory.

        Returns:
            str: The path to the Logs directory, or None if an error occurs.

        Raises:
            TypeError: If the current subdirectory is not set.
        """
        try:
            path = os.path.join(self.current_subdir, 'Logs')
            logging.info(f"Retrieved Logs path: {path}")
            return path
        except TypeError as e:
            logging.error(f"Error retrieving Logs path: {e}")
            return None

    def get_monitor_path(self):
        """
        Retrieves the path to the "Monitor" directory.

        This method returns the absolute path of the "Monitor" subdirectory located under the current subdirectory.

        Returns:
            str: The path to the Monitor directory, or None if an error occurs.

        Raises:
            TypeError: If the current subdirectory is not set.
        """
        try:
            path = os.path.join(self.current_subdir, 'Monitor')
            logging.info(f"Retrieved Monitor path: {path}")
            return path
        except TypeError as e:
            logging.error(f"Error retrieving Monitor path: {e}")
            return None

    def get_models_saved_path(self):
        """
        Retrieves the path to the "ModelsSaved" directory.

        This method returns the absolute path of the "ModelsSaved" subdirectory located under the current subdirectory.

        Returns:
            str: The path to the ModelsSaved directory, or None if an error occurs.

        Raises:
            TypeError: If the current subdirectory is not set.
        """
        try:
            path = os.path.join(self.current_subdir, 'ModelsSaved')
            logging.info(f"Retrieved ModelsSaved path: {path}")
            return path
        except TypeError as e:
            logging.error(f"Error retrieving ModelsSaved path: {e}")
            return None

    def get_data_generated_path(self):
        """
        Retrieves the path to the "DataGenerated" directory.

        This method returns the absolute path of the "DataGenerated" subdirectory located under the current subdirectory.

        Returns:
            str: The path to the DataGenerated directory, or None if an error occurs.

        Raises:
            TypeError: If the current subdirectory is not set.
        """
        try:
            path = os.path.join(self.current_subdir, 'DataGenerated')
            logging.info(f"Retrieved DataGenerated path: {path}")
            return path

        except TypeError as e:
            logging.error(f"Error retrieving DataGenerated path: {e}")
            return None

    def get_evaluation_results_path(self):
        """
        Retrieves the path to the "EvaluationResults" directory.

        This method returns the absolute path of the "EvaluationResults" subdirectory located under the current subdirectory.

        Returns:
            str: The path to the EvaluationResults directory, or None if an error occurs.

        Raises:
            TypeError: If the current subdirectory is not set.
        """
        try:
            path = os.path.join(self.current_subdir, 'EvaluationResults')
            logging.info(f"Retrieved EvaluationResults path: {path}")
            return path

        except TypeError as e:
            logging.error(f"Error retrieving EvaluationResults path: {e}")
            return None

    def zip_and_cleanup(self):
        """
        Compresses the current subdirectory into a ZIP archive and removes the original directory.

        This method creates a ZIP archive of the current subdirectory, then deletes the original directory to clean up.
        The name of the ZIP file is based on the timestamped subdirectory name.

        Returns:
            None

        Raises:
            ValueError: If no directory is available to zip (i.e., the current subdirectory is None).
            OSError: If there is a file system error during the compression or cleanup process.
            exception: If an unexpected error occurs.
        """
        try:
            if self.current_subdir is None:
                raise ValueError("No directory to zip. Create the directories first.")

            # Create a zip file from the current subdirectory
            zip_filename = f"{self.current_subdir}.zip"
            shutil.make_archive(self.current_subdir, 'zip', self.current_subdir)
            logging.info(f"Directory zipped: {zip_filename}")

            # Remove the original directory after zipping
            shutil.rmtree(self.current_subdir)
            logging.info(f"Original directory removed: {self.current_subdir}")

            print(f"Directory '{self.current_subdir}' zipped and removed. Saved as '{zip_filename}'.")

        except ValueError as e:
            logging.error(f"Error: {e}")

        except OSError as e:
            logging.error(f"File system error during zip and cleanup: {e}")

        except Exception as e:
            logging.error(f"Unexpected error: {e}")

