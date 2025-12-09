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


# try:
import sys
import datetime
import argparse
import logging


# except ImportError as error:
#
#     print(error)
#     print()
#     print("1. (optional) Setup a virtual environment: ")
#     print("  python3 - m venv ~/Python3env/DroidAugmentor ")
#     print("  source ~/Python3env/DroidAugmentor/bin/activate ")
#     print()
#     print("2. Install requirements:")
#     print("  pip3 install --upgrade pip")
#     print("  pip3 install -r requirements.txt ")
#     print()
#     sys.exit(-1)

from Engine.Classifiers.Classifiers import Classifiers
DEFAULT_VERBOSITY = logging.INFO
TIME_FORMAT = '%Y-%m-%d,%H:%M:%S'
DEFAULT_DATA_TYPE = "float32"

DEFAULT_NUMBER_EPOCHS_CONDITIONAL_GAN = 100
DEFAULT_NUMBER_STRATIFICATION_FOLD = 5

DEFAULT_SAVE_MODELS = False
DEFAULT_SAVE_DATA = False
DEFAULT_OUTPUT_PATH_CONFUSION_MATRIX = "confusion_matrix"
DEFAULT_OUTPUT_PATH_TRAINING_CURVE = "training_curve"
DEFAULT_CLASSIFIER_LIST = ["RandomForest", "KNN", "DecisionTree"]
DEFAULT_EVALUATION_METHOD = ["TrAs", "TsAr"]

DEFAULT_VERBOSE_LIST = {logging.INFO: 2, logging.DEBUG: 1, logging.WARNING: 2,
                        logging.FATAL: 0, logging.ERROR: 0}

LOGGING_FILE_NAME = "logging.log"
 

def parse_number_samples(samples_str):

    samples = samples_str.split(',')
    parsed_samples = {}
    #max_class_id = 0

    for sample in samples:
        class_id, num_samples = sample.split(':')
        #max_class_id = max(max_class_id, int(class_id))
        parsed_samples[int(class_id)] = int(num_samples)

    return {"classes": parsed_samples,
            "number_classes": len(parsed_samples)}

def add_argument_framework():

    parser = argparse.ArgumentParser(description='SynDataGen Data Generator')

    
    parser.add_argument('--number_samples_per_class', type=parse_number_samples, default="1:256,2:256",
                        help="Class and number of samples in the format class1:num1,class2:num2,...")

    parser.add_argument('-c', '--classifier', type=str, default=DEFAULT_CLASSIFIER_LIST, nargs="+",
                        choices=Classifiers.dictionary_classifiers_name,
                        help="Classifier (or list of classifiers separated by empty space) default: {} availabe: {}.".format(
                            DEFAULT_CLASSIFIER_LIST, Classifiers.dictionary_classifiers_name))

    parser.add_argument('--evaluation', type=str, default=DEFAULT_EVALUATION_METHOD, nargs="+",
                        help="List Evaluation Methods ['TrAs', 'TsAr'}")

    parser.add_argument('-o', '--output_dir', type=str,
                        default=f'Results/out_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}',
                        help='Directory for saving output files.')

    parser.add_argument('--number_k_folds', type=int,
                        default=DEFAULT_NUMBER_STRATIFICATION_FOLD,
                        help='Number of folds for cross-validation.')

    parser.add_argument('--use_gpu', action='store_true',
                        default=False,
                        help='Enable GPU processing if available.')

    parser.add_argument("--verbosity", type=int,
                        help='Verbosity (Default {})'.format(DEFAULT_VERBOSITY),
                        default=DEFAULT_VERBOSITY)

    parser.add_argument("--save_models", type=bool,
                        help='Save trained models (Default {})'.format(DEFAULT_SAVE_MODELS),
                        default=DEFAULT_SAVE_MODELS)
    
    parser.add_argument("--save_data", type=bool,
                        help='Save generated data (Default {})'.format(DEFAULT_SAVE_DATA),
                        default=DEFAULT_SAVE_DATA)

    parser.add_argument("--path_confusion_matrix", type=str,
                        help='Output directory for confusion matrices',
                        default=DEFAULT_OUTPUT_PATH_CONFUSION_MATRIX)

    parser.add_argument("--path_curve_loss", type=str,
                        help='Output directory for training curve plots',
                        default=DEFAULT_OUTPUT_PATH_TRAINING_CURVE)

    parser.add_argument('--model_type',
                        choices=['smote',
                                 'random',
                                 'adversarial',
                                 'latent_diffusion',
                                 'denoising_diffusion',
                                 'wasserstein',
                                 'wasserstein_gp',
                                 'variational',
                                 'autoencoder',
                                 'quantized',
                                 'diffusion_kernel',
                                 'copy',
                                 'copula',
                                 'ctgan',
                                 'tvae'],
                        default='adversarial', help='Select the model type')

    return parser
