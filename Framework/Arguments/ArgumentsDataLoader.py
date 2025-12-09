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

DEFAULT_DATA_LOAD_LABEL_COLUMN = -1
DEFAULT_DATA_LOAD_MAX_SAMPLES = -1
DEFAULT_DATA_LOAD_MAX_COLUMNS = -1
DEFAULT_DATA_LOAD_START_COLUMN = 0
DEFAULT_DATA_LOAD_END_COLUMN = -1
DEFAULT_DATA_LOAD_PATH_FILE_INPUT = 'Datasets/binaries/kronodroid_emulador-balanced.csv'
DEFAULT_DATA_LOAD_PATH_FILE_OUTPUT = 'OutputDir'
DEFAULT_DATA_LOAD_EXCLUDE_COLUMNS = -1

def add_argument_data_load(parser):

    parser.add_argument('-i', '--data_load_path_file_input', type=str, default=DEFAULT_DATA_LOAD_PATH_FILE_INPUT,
                       help='Path to the input CSV file.')


    parser.add_argument('--data_load_label_column', type=int, default=DEFAULT_DATA_LOAD_LABEL_COLUMN,
                        help='Index of the column to be used as the label.')

    parser.add_argument('--data_load_max_samples', type=int, default=DEFAULT_DATA_LOAD_MAX_SAMPLES,
                        help='Maximum number of samples to be loaded.')

    parser.add_argument('--data_load_max_columns', type=int, default=DEFAULT_DATA_LOAD_MAX_COLUMNS,
                        help='Maximum number of columns to be considered.')

    parser.add_argument('--data_load_start_column', type=int, default=DEFAULT_DATA_LOAD_START_COLUMN,
                        help='Index of the first column to be loaded.')

    parser.add_argument('--data_load_end_column', type=int, default=DEFAULT_DATA_LOAD_END_COLUMN,
                        help='Index of the last column to be loaded.')

    parser.add_argument('--data_load_path_file_output', type=str, default=DEFAULT_DATA_LOAD_PATH_FILE_OUTPUT,
                        help='Path to the output CSV file.')

    parser.add_argument('--data_load_exclude_columns', type=int, default=DEFAULT_DATA_LOAD_EXCLUDE_COLUMNS,
                        help='Columns to exclude from processing.')

    return parser
