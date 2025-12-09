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
    import logging

except ImportError as error:
    print(error)
    sys.exit(-1)

LOGGING_FILE_NAME = "logging.log"

class LoggerSetup:
    """
    A configurable logging system that sets up both file and console logging.

    This class provides:
    - Customizable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Different log formats based on verbosity level
    - File logging with rotation capabilities
    - Console output logging
    - Clean handler management

    Attributes:
        arguments (object): An object containing configuration arguments with:
            - output_dir (str): Base directory for log files
            - verbosity (int): Logging level (e.g., logging.DEBUG, logging.INFO)
    """
    def __init__(self, arguments):
        """
        Initialize the LoggerSetup with configuration arguments.

        Args:
            arguments (object): Configuration object containing:
                - output_dir: Base directory for output files
                - verbosity: Logging level constant from logging module
        """
        self.arguments = arguments

    def get_logs_path(self):
        """
        Determine and return the path to the logs directory.

        The path is constructed by combining the output directory from arguments
        with a 'Logs' subdirectory. Ensures proper path formatting with trailing slash.

        Returns:
            str: Full path to the logs directory with trailing slash
        """
        return "{}/Logs/".format(self.arguments.output_dir)

    def setup_logger(self):
        """
        Configure and initialize the root logger with file and console handlers.

        This method:
        1. Sets the global logging level based on provided verbosity
        2. Creates appropriate log formats (simpler for INFO+, detailed for DEBUG)
        3. Configures a file handler with log rotation
        4. Configures a console (stdout) handler
        5. Clears any existing handlers before adding new ones
        6. Applies both handlers to the root logger

        Note:
            Uses LOGGING_FILE_NAME constant for the log filename
            Creates log directory if it doesn't exist
        """
        # Get the root logger
        logger = logging.getLogger()

        # Set the global logging level from arguments
        logger.setLevel(self.arguments.verbosity)

        # Create format based on verbosity level
        logging_format = '%(asctime)s\t***\t%(message)s'
        if self.arguments.verbosity == logging.DEBUG:
            logging_format = '%(asctime)s\t***\t%(levelname)s {%(module)s} [%(funcName)s] %(message)s'

        # Create formatter with the selected format
        formatter = logging.Formatter(logging_format)

        # Configure file handler with rotation
        logging_filename = os.path.join(self.get_logs_path(), LOGGING_FILE_NAME)

        # Ensure log directory exists
        rotatingFileHandler = logging.FileHandler(filename=logging_filename)

        # Currently using basic FileHandler (commented RotatingFileHandler available)
        rotatingFileHandler.setLevel(self.arguments.verbosity)

        # Alternative with rotation (uncomment to use):
        # rotatingFileHandler = logging.RotatingFileHandler(
        #     filename=logging_filename,
        #     maxBytes=100000,
        #     backupCount=5
        # )
        rotatingFileHandler.setFormatter(formatter)

        # Configure console (stdout) handler
        streamHandler = logging.StreamHandler()

        streamHandler.setLevel(self.arguments.verbosity)
        streamHandler.setFormatter(formatter)

        # Clear existing handlers to avoid duplicate messages
        if logger.hasHandlers():
            logger.handlers.clear()

        # Add both handlers to the logger
        logger.addHandler(rotatingFileHandler)
        logger.addHandler(streamHandler)