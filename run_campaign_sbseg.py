#!/usr/bin/env python3
 

# MIT License
#
# Copyright (c) 2025 MalDataGen
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


"""
Main script for running training and evaluation campaigns
of generative and supervised classification models, focused on
digital security and malware analysis applications.

Supported campaigns include:
- Generative Architectures: Autoencoders, Variational Autoencoders,
  WassersteinGP GANs, LatentDiffusion Architectures, CTGAN, TVAE, Copula GANs.
- Classifiers: Random Forest, SVM, KNN, etc.

This script sets up the campaign parameters, composes the execution commands,
and interfaces with the training (`main.py`) and visualization (`plots.py`) modules.
"""

try:
    import os
    import sys
    import shlex

    import logging
    import argparse
    import datetime
    import itertools
    import subprocess

    from pathlib import Path

    from logging.handlers import RotatingFileHandler

except ImportError as error:
    print(error)
    print()
    print("1. (optional) Setup a virtual environment: ")
    print("  python3 -m venv ~/Python3venv/SyntheticOceanAI ")
    print("  source ~/Python3venv/SyntheticOceanAI/bin/activate ")
    print()
    print("2. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)




DEFAULT_VERBOSITY_LEVEL = logging.INFO  # Logging verbosity level for the experiment
NUM_EPOCHS = 40  # Default number of training epochs (overridden per campaign)
TIME_FORMAT = '%Y-%m-%d_%H:%M:%S'  # Format for timestamps in logs
SAMPLES = '0:2000,1:2000'  # Default number of samples per class in string format (class:count)
#DEFAULT_CAMPAIGN = "wasserstein wasserstein_gp variational autoencoder adversarial latent_diffusion denoising_diffusion ctgan tvae copula".split()  # Active campaign types
DEFAULT_CAMPAIGN = "adversarial".split()  # Active campaign types


# Default classifiers to evaluate models with (space-separated string in list)
DEFAULT_CLASSIFIER = ['RandomForest SupportVectorMachine KNN DecisionTree NaiveBayes GradientBoosting StochasticGradientDescent']
DEFAULT_K_FOLDS = 5 # Default number of K-folds for cross-validation
PATH_LOG = 'logs'  # Directory to store logs
PATH_DATASETS = 'datasets'  # Directory where datasets are located
PATHS = [PATH_LOG]  # List of directories to create/check

arguments = None  # Placeholder for parsed arguments
COMMAND = "python3 main.py  "  # Default execution command
COMMAND_PLOT = "python3 plots.py  "  # Default plotting command
SAVE_DATA = 'True'  # Flag to indicate if synthetic/generated data should be saved



# ========================
# DATASETS PATH LIST
# ========================

datasets = [
    'Datasets/SBSeg_2025/reduced_balanced_androcrawl.csv',

]



# ========================
# CAMPAIGN DEFINITIONS
# ========================

campaigns_available = {}






campaigns_available['teste'] = {
    'classifier': DEFAULT_CLASSIFIER,
    'model_type': ['adversarial'],
    'number_samples_per_class': ['0:100,1:100'],
    'number_k_folds': [2],
    'adversarial_number_epochs': [NUM_EPOCHS],
    'save_data': [SAVE_DATA],
    'adversarial_batch_size': [256],
    'adversarial_dense_layer_sizes_d': [512],
    'adversarial_dense_layer_sizes_g': ['2048'],
    'adversarial_dropout_decay_rate_d': [0.4],
    'adversarial_dropout_decay_rate_g': [0.2],
    'adversarial_initializer_deviation': [0.5],
    'adversarial_initializer_mean': [0],
    'adversarial_latent_dimension': [128],
    'adversarial_latent_stander_deviation': [1],
    'adversarial_training_algorithm': ['Adam'],
    'adversarial_activation_function': ['LeakyReLU'],
    'adversarial_latent_mean_distribution': [0.0],
    'adversarial_loss_generator': ['binary_crossentropy'],
    'adversarial_loss_discriminator': ['binary_crossentropy'],
    'adversarial_smoothing_rate': [0.15],
}

# === Adversarial Campaign ===
campaigns_available['adversarial'] = {
    'classifier': DEFAULT_CLASSIFIER,
    'model_type': ['adversarial'],
    'number_samples_per_class': [SAMPLES],
    'number_k_folds': [DEFAULT_K_FOLDS],
    'adversarial_number_epochs': [NUM_EPOCHS],
    'save_data': [SAVE_DATA],
    'adversarial_batch_size': [64],
    'adversarial_dense_layer_sizes_g': ['256'],
    'adversarial_dense_layer_sizes_d': ['64'],
    'adversarial_dropout_decay_rate_g': [0.2],
    'adversarial_dropout_decay_rate_d': [0.4],
    'adversarial_initializer_deviation': [0.5],
    'adversarial_initializer_mean': [0],
    'adversarial_latent_dimension': [32],
    'adversarial_latent_mean_distribution': [0.5],
    'adversarial_latent_stander_deviation': [0.125],
    'adversarial_training_algorithm': ['Adam'],
    'adversarial_activation_function': ['leakyrelu'],
    'adversarial_loss_generator': ['binary_crossentropy'],
    'adversarial_loss_discriminator': ['binary_crossentropy'],
    'adversarial_smoothing_rate': [0.15]
}
# === Adversarial Campaign ===
campaigns_available['adversarial_demo'] = {
    'classifier': DEFAULT_CLASSIFIER,
    'model_type': ['adversarial'],
    'number_samples_per_class': [SAMPLES],
    'number_k_folds': [DEFAULT_K_FOLDS],
    'adversarial_number_epochs': [1],
    'save_data': [SAVE_DATA],
    'adversarial_batch_size': [64],
    'adversarial_dense_layer_sizes_g': ['256'],
    'adversarial_dense_layer_sizes_d': ['64'],
    'adversarial_dropout_decay_rate_g': [0.2],
    'adversarial_dropout_decay_rate_d': [0.4],
    'adversarial_initializer_deviation': [0.5],
    'adversarial_initializer_mean': [0],
    'adversarial_latent_dimension': [32],
    'adversarial_latent_mean_distribution': [0.5],
    'adversarial_latent_stander_deviation': [0.125],
    'adversarial_training_algorithm': ['Adam'],
    'adversarial_activation_function': ['leakyrelu'],
    'adversarial_loss_generator': ['binary_crossentropy'],
    'adversarial_loss_discriminator': ['binary_crossentropy'],
    'adversarial_smoothing_rate': [0.15]
}

# === autoencoder Campaign ===
campaigns_available['autoencoder'] = {
    'classifier': DEFAULT_CLASSIFIER,
    'model_type': ['autoencoder'],
    'number_samples_per_class': [SAMPLES],
    'number_k_folds': [DEFAULT_K_FOLDS],
    'save_data': [SAVE_DATA],
    'autoencoder_number_epochs': [NUM_EPOCHS],
    'autoencoder_latent_dimension': [128],
    'autoencoder_activation_function': ['elu'],
    'autoencoder_dropout_decay_rate_encoder': [0.10],
    'autoencoder_dropout_decay_rate_decoder': [0.10],
    'autoencoder_dense_layer_sizes_encoder': ['128 128'],
    'autoencoder_dense_layer_sizes_decoder': ['128 128'],
    'autoencoder_batch_size': [256],
    'autoencoder_number_classes': [2],
    'autoencoder_loss_function': ["binary_crossentropy"],
    'autoencoder_momentum': [0.8],
    'autoencoder_last_activation_layer': ["sigmoid"],
    'autoencoder_initializer_mean': [0.0],
    'autoencoder_initializer_deviation': [0.125],
    'autoencoder_latent_mean_distribution': [0.5],
    'autoencoder_latent_stander_deviation': [0.500],
}
# === Variational autoencoder Campaign ===
campaigns_available['variational_demo'] = {
    'classifier': DEFAULT_CLASSIFIER,
    'model_type': ['variational'],
    'number_samples_per_class': [SAMPLES],
    'number_k_folds': [DEFAULT_K_FOLDS],
    'save_data': [SAVE_DATA],
    'variational_autoencoder_number_epochs': [1],
    'variational_autoencoder_latent_dimension': [68],
    'variational_autoencoder_training_algorithm': ['Adam'],
    'variational_autoencoder_activation_function': ['relu'],
    'variational_autoencoder_dropout_decay_rate_encoder': [0.2],
    'variational_autoencoder_dropout_decay_rate_decoder': [0.2],
    'variational_autoencoder_dense_layer_sizes_encoder': ['128 64'],
    'variational_autoencoder_dense_layer_sizes_decoder': ['64 128'],
    'variational_autoencoder_batch_size': [32],
    'variational_autoencoder_number_classes': [2],
    'variational_autoencoder_loss_function': ['binary_crossentropy'],
    'variational_autoencoder_momentum': [0.8],
    'variational_autoencoder_last_activation_layer': ['sigmoid'],
    'variational_autoencoder_initializer_mean': [0.0],
    'variational_autoencoder_initializer_deviation': [0.125],
    'variational_autoencoder_mean_distribution': [0.5],
    'variational_autoencoder_stander_deviation': [0.125],
}



# === Variational autoencoder Campaign ===
campaigns_available['variational'] = {
    'classifier': DEFAULT_CLASSIFIER,
    'model_type': ['variational'],
    'number_samples_per_class': [SAMPLES],
    'number_k_folds': [DEFAULT_K_FOLDS],
    'save_data': [SAVE_DATA],
    'variational_autoencoder_number_epochs': [NUM_EPOCHS],
    'variational_autoencoder_latent_dimension': [68],
    'variational_autoencoder_training_algorithm': ['Adam'],
    'variational_autoencoder_activation_function': ['relu'],
    'variational_autoencoder_dropout_decay_rate_encoder': [0.2],
    'variational_autoencoder_dropout_decay_rate_decoder': [0.2],
    'variational_autoencoder_dense_layer_sizes_encoder': ['128 64'],
    'variational_autoencoder_dense_layer_sizes_decoder': ['64 128'],
    'variational_autoencoder_batch_size': [32],
    'variational_autoencoder_number_classes': [2],
    'variational_autoencoder_loss_function': ['binary_crossentropy'],
    'variational_autoencoder_momentum': [0.8],
    'variational_autoencoder_last_activation_layer': ['sigmoid'],
    'variational_autoencoder_initializer_mean': [0.0],
    'variational_autoencoder_initializer_deviation': [0.125],
    'variational_autoencoder_mean_distribution': [0.5],
    'variational_autoencoder_stander_deviation': [0.125],
}


# === Quantized VAE Campaign ===
campaigns_available['quantized'] = {
    'classifier': DEFAULT_CLASSIFIER,
    'model_type': ['quantized'],
    'number_samples_per_class': [SAMPLES],
    'number_k_folds': [DEFAULT_K_FOLDS],
    'save_data': [SAVE_DATA],
    'quantized_vae_number_epochs': [NUM_EPOCHS],
}



# === Wasserstein GAN Campaign ===
campaigns_available['wasserstein'] = {
    'classifier': DEFAULT_CLASSIFIER,
    'model_type': ['wasserstein'],
    'number_samples_per_class': [SAMPLES],
    'number_k_folds': [DEFAULT_K_FOLDS],
    'wasserstein_number_epochs': [NUM_EPOCHS],
    'save_data': [SAVE_DATA],
    'wasserstein_latent_dimension': [32],
    'wasserstein_training_algorithm': ['Adam'],
    'wasserstein_activation_function': ['elu'],
    'wasserstein_dropout_decay_rate_g': [0.0],
    'wasserstein_dropout_decay_rate_d': [0.0],
    'wasserstein_dense_layer_sizes_generator': [256],
    'wasserstein_dense_layer_sizes_discriminator': [64],
    'wasserstein_batch_size': [64],
    'wasserstein_number_classes': [2],
    'wasserstein_momentum': [0.8],
    'wasserstein_last_activation_layer': ['linear'],
    'wasserstein_initializer_mean': [0.0],
    'wasserstein_initializer_deviation': [0.125],
    'wasserstein_optimizer_generator_learning_rate': [0.001],
    'wasserstein_optimizer_discriminator_learning_rate': [0.001],
    'wasserstein_optimizer_generator_beta': [0.5],
    'wasserstein_optimizer_discriminator_beta': [0.5],
    'wasserstein_discriminator_steps': [3],
    'wasserstein_smoothing_rate': [0.10],
    'wasserstein_latent_mean_distribution': [0.5],
    'wasserstein_latent_stander_deviation': [0.125],
}


# === Wasserstein GAN GP Campaign ===
campaigns_available['wasserstein_gp'] = {
    'classifier': DEFAULT_CLASSIFIER,
    'model_type': ['wasserstein_gp'],
    'number_samples_per_class': [SAMPLES],
    'number_k_folds': [DEFAULT_K_FOLDS],
    'wasserstein_gp_number_epochs': [NUM_EPOCHS],
    'save_data': [SAVE_DATA],
    'wasserstein_gp_latent_dimension': [32],
    'wasserstein_gp_training_algorithm': ['Adam'],
    'wasserstein_gp_activation_function': ['leakyrelu'],
    'wasserstein_gp_dropout_decay_rate_g': [0.0],
    'wasserstein_gp_dropout_decay_rate_d': [0.0],
    'wasserstein_gp_dense_layer_sizes_generator': [256],
    'wasserstein_gp_dense_layer_sizes_discriminator': [64],
    'wasserstein_gp_batch_size': [64],
    'wasserstein_gp_number_classes': [2],
    'wasserstein_gp_momentum': [0.8],
    'wasserstein_gp_last_activation_layer': ['sigmoid'],
    'wasserstein_gp_initializer_mean': [0.0],
    'wasserstein_gp_initializer_deviation': [0.125],
    'wasserstein_gp_optimizer_generator_learning_rate': [0.001],
    'wasserstein_gp_optimizer_discriminator_learning_rate': [0.001],
    'wasserstein_gp_optimizer_generator_beta': [0.5],
    'wasserstein_gp_optimizer_discriminator_beta': [0.5],
    'wasserstein_gp_discriminator_steps': [3],
    'wasserstein_gp_smoothing_rate': [0.10],
    'wasserstein_gp_latent_mean_distribution': [0.5],
    'wasserstein_gp_latent_stander_deviation': [0.125],
    'wasserstein_gp_gradient_penalty': [10.0],
}


# === Latent Diffusion Model Campaign ===
campaigns_available['latent_diffusion'] = {
    'classifier': DEFAULT_CLASSIFIER,
    'model_type': ['latent_diffusion'],
    'number_samples_per_class': [SAMPLES],
    'number_k_folds': [DEFAULT_K_FOLDS],
    'save_data': [SAVE_DATA],
    'latent_diffusion_unet_epochs': [NUM_EPOCHS],
    'latent_diffusion_autoencoder_epochs': [NUM_EPOCHS],
    'latent_diffusion_unet_last_layer_activation': ['tanh'],
    'latent_diffusion_latent_dimension': [32],
    'latent_diffusion_unet_num_embedding_channels': [1],
    'latent_diffusion_unet_channels_per_level': ['1 2 4'],
    'latent_diffusion_unet_batch_size': [256],
    'latent_diffusion_unet_attention_mode': ['False True True'],
    'latent_diffusion_unet_num_residual_blocks': [2],
    'latent_diffusion_unet_group_normalization': [1],
    'latent_diffusion_unet_intermediary_activation': ['elu'],
    'latent_diffusion_unet_intermediary_activation_alpha': [0.05],
    'latent_diffusion_gaussian_beta_start': [1e-4],
    'latent_diffusion_gaussian_beta_end': [0.02],
    'latent_diffusion_gaussian_time_steps': [300],
    'latent_diffusion_gaussian_clip_min': [-1.0],
    'latent_diffusion_gaussian_clip_max': [1.0],
    'latent_diffusion_autoencoder_loss': ['mse'],
    'latent_diffusion_autoencoder_encoder_filters': ['64 32'],
    'latent_diffusion_autoencoder_decoder_filters': ['32 64'],
    'latent_diffusion_autoencoder_last_layer_activation': ['sigmoid'],
    'latent_diffusion_autoencoder_latent_dimension': [32],
    'latent_diffusion_autoencoder_batch_size_create_embedding': [128],
    'latent_diffusion_autoencoder_batch_size_training': [128],
    'latent_diffusion_autoencoder_intermediary_activation_function': ['elu'],
    'latent_diffusion_autoencoder_intermediary_activation_alpha': [0.05],
    'latent_diffusion_autoencoder_activation_output_encoder': ['sigmoid'],
    'latent_diffusion_margin': [0.5],
    'latent_diffusion_ema': [0.999],
    'latent_diffusion_time_steps': [300],
}


# === Denoising Diffusion Model Campaign ===
campaigns_available['denoising_diffusion'] = {
    'classifier': DEFAULT_CLASSIFIER,
    'model_type': ['denoising_diffusion'],
    'number_samples_per_class': [SAMPLES],
    'number_k_folds': [DEFAULT_K_FOLDS],
    'save_data': [SAVE_DATA],
    'denoising_diffusion_unet_epochs': [NUM_EPOCHS],
    'denoising_diffusion_unet_last_layer_activation': ['tanh'],
    'denoising_diffusion_unet_num_embedding_channels': [1],
    'denoising_diffusion_unet_channels_per_level': ['1 2 4'],
    'denoising_diffusion_unet_batch_size': [128],
    'denoising_diffusion_unet_attention_mode': ['False True True'],
    'denoising_diffusion_unet_num_residual_blocks': [2],
    'denoising_diffusion_unet_group_normalization': [1],
    'denoising_diffusion_unet_intermediary_activation': ['ELU'],
    'denoising_diffusion_unet_intermediary_activation_alpha': [0.05],
    'denoising_diffusion_gaussian_beta_start': [1e-4],
    'denoising_diffusion_gaussian_beta_end': [0.02],
    'denoising_diffusion_gaussian_time_steps': [300],
    'denoising_diffusion_gaussian_clip_min': [-1.0],
    'denoising_diffusion_gaussian_clip_max': [1.0],
    'denoising_diffusion_margin': [0.5],
    'denoising_diffusion_ema': [0.999],
    'denoising_diffusion_time_steps': [300]
}

# === Copula-based Synthetic Generation ===
campaigns_available['copula'] = {
    'classifier': DEFAULT_CLASSIFIER,
    'model_type': [ 'copula'],
    'number_samples_per_class': [SAMPLES],
    'number_k_folds': [DEFAULT_K_FOLDS],
    'save_data': [SAVE_DATA],
}

# === TVAE Model Campaign ===
campaigns_available['tvae'] = {
    'classifier': DEFAULT_CLASSIFIER,
    'model_type': ['tvae'],
    'number_samples_per_class': [SAMPLES],
    'number_k_folds': [DEFAULT_K_FOLDS],
    'save_data': [SAVE_DATA],
}

# === CTGAN Model Campaign ===
campaigns_available['ctgan'] = {
    'classifier': DEFAULT_CLASSIFIER,
    'model_type': ['ctgan'],
    'number_samples_per_class': [SAMPLES],
    'number_k_folds': [DEFAULT_K_FOLDS],
    'save_data': [SAVE_DATA],
}


# ========================
# UTILITY FUNCTIONS
# ========================
def print_all_settings(arguments):
    """Logs the full configuration of the current experiment based on parsed arguments."""
    logging.info("Campaign Command:\n\t{0}\n".format(" ".join([x for x in sys.argv])))
    logging.info("Campaign Settings:")
    lengths = [len(x) for x in vars(arguments).keys()]
    max_length = max(lengths)

    for key_item, values in sorted(vars(arguments).items()):
        message = "\t"
        message += key_item.ljust(max_length, " ")
        message += " : {}".format(values)
        logging.info(message)

    logging.info("")


def convert_flot_to_int(value):
    """
    Converts float values to integers by multiplying by 100.
    Used for normalizing certain hyperparameters or ID hashing.
    """
    if isinstance(value, float):
        value = int(value * 100)

    return value


# Custom argparse type representing a bounded int
# source: https://stackoverflow.com/questions/14117415/in-python-using-argparse-allow-only-positive-integers
class IntRange:
    """
    Custom argparse type to enforce an integer within a bounded range [imin, imax].

    Parameters
    ----------
    imin : int, optional
        Minimum allowed value (inclusive).
    imax : int, optional
        Maximum allowed value (inclusive).

    Raises
    ------
    argparse.ArgumentTypeError
        If the input cannot be parsed as int, or is outside the specified range.
    """
    def __init__(self, imin=None, imax=None):

        self.imin = imin
        self.imax = imax

    def __call__(self, arg):

        try:
            value = int(arg)

        except ValueError:
            raise self.exception()

        if (self.imin is not None and value < self.imin) or (self.imax is not None and value > self.imax):
            raise self.exception()

        return value

    def exception(self):

        if self.imin is not None and self.imax is not None:
            return argparse.ArgumentTypeError(f"Must be an integer in the range [{self.imin}, {self.imax}]")

        elif self.imin is not None:
            return argparse.ArgumentTypeError(f"Must be an integer >= {self.imin}")

        elif self.imax is not None:
            return argparse.ArgumentTypeError(f"Must be an integer <= {self.imax}")

        else:
            return argparse.ArgumentTypeError("Must be an integer")


def run_cmd(cmd, shell=False):
    """
    Executes a shell/system command with logging, respecting dry-run mode.

    Parameters
    ----------
    cmd : str
        The shell command to execute.
    shell : bool
        Whether to execute the command within a shell environment.

    Notes
    -----
    - Logs both the raw command and the parsed array (if applicable).
    - Supports dry-run mode via `args.dryrun`.
    """
    logging.info("Command line  : {}".format(cmd))
    cmd_array = shlex.split(cmd)
    logging.debug("Command array: {}".format(cmd_array))
    if not arguments.dryrun:
        subprocess.run(cmd_array, check=True, shell=shell)


class Campaign():
    """
    A data structure for storing campaign configuration metadata.

    Parameters
    ----------
    datasets : list
        Datasets involved in this campaign.
    training_algorithm : str
        The name or identifier of the training algorithm.
    dense_layer_sizes_g : list
        Hidden layers for the generator (if applicable).
    dense_layer_sizes_d : list
        Hidden layers for the discriminator (if applicable).
    """
    def __init__(self, datasets, training_algorithm, dense_layer_sizes_g, dense_layer_sizes_d):
        self.datasets = datasets
        self.training_algorithm = training_algorithm
        self.dense_layer_sizes_g = dense_layer_sizes_g
        self.dense_layer_sizes_d = dense_layer_sizes_d


def check_files(files, error=False):
    """
    Verifies that input files exist on the filesystem.

    Parameters
    ----------
    files : str or list
        File path(s) to check.
    error : bool
        If True, aborts execution if any file is missing.

    Returns
    -------
    bool
        True if all files are present, False otherwise.

    Notes
    -----
    Used for defensive checks prior to running expensive operations.
    """
    internal_files = files
    if isinstance(files, str):
        internal_files = [files]

    for f in internal_files:
        if not os.path.isfile(f):
            if error:
                logging.info("ERROR: file not found! {}".format(f))
                sys.exit(1)
            else:
                logging.info("File not found! {}".format(f))
                return False
        else:
            logging.info(("File found: {}".format(f)))

    return True


def main():
    """
    Main orchestration logic for launching model evaluation campaigns.

    Features
    --------
        - Parses CLI arguments for user configuration.
        - Dynamically selects datasets and campaigns.
        - Iterates over all combinations of hyperparameters.
        - Calls training and plotting scripts.
        - Logs time and performance statistics for each run.

    CLI Arguments
    -------------
        --campaign / -c : list of campaign identifiers
        --dryrun / -d   : disables actual execution, useful for debugging
        --pipenv / -p   : wraps commands using `pipenv run`
        --verbosity / -v: sets logging level (INFO, DEBUG, etc.)
        --dataset / -a  : selects a specific dataset by index

    Notes
    -----
        - Automatically creates output directories.
        - Uses RotatingFileHandler to manage large log files.
        - Handles Variational autoencoder (VAE) special cases (decoder = reversed encoder).
        - Plotting is skipped for baseline models like 'copula', 'ctgan', etc.

    Expected Globals
    ----------------
        - DEFAULT_CAMPAIGN
        - datasets
        - campaigns_available
        - PATHS
        - COMMAND, COMMAND_PLOT
        - TIME_FORMAT
    """

    parser = argparse.ArgumentParser(description='Torrent Trace Correct - Machine Learning')

    help_msg = "Campaign  (default={})".format([x for x in campaigns_available.keys()], )
    parser.add_argument("--campaign", "-c", help=help_msg, default=DEFAULT_CAMPAIGN, type=str, nargs='+')

    help_msg = "dry run mode (show commands but don´t run them) (default={})".format(False)
    parser.add_argument("--dryrun", "-d", help=help_msg, action='store_true')

    help_msg = "use pipenv (default={})".format(False)
    parser.add_argument("--pipenv", "-p", help=help_msg, action='store_true')

    help_msg = "verbosity logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
    parser.add_argument("--verbosity", "-v", help=help_msg, default=DEFAULT_VERBOSITY_LEVEL, type=int)

    help_msg = "dataset (0-6)"
    parser.add_argument("--dataset", "-a", help=help_msg, default=-1, type=int)

    global arguments
    arguments = parser.parse_args()

    print("Creating the structure of directories...")

    for p in PATHS:
        Path(p).mkdir(parents=True, exist_ok=True)

    print("done.")
    print("")

    campaigns_chosen = []
    print("args campaign", arguments.campaign)

    if arguments.campaign is None:
        campaigns_chosen = campaigns_available.keys()

    elif arguments.campaign == ['sf']:
        campaigns_chosen = ['variational_demo','adversarial_demo']
         
    elif arguments.campaign== ['sf2']:
        campaigns_chosen = ['copula','tvae','ctgan']


    else:

        campaigns_list = arguments.campaign

        if ',' in arguments.campaign:
            campaigns_list = arguments.campaign.split(',')

        for c in campaigns_list:

            if c in campaigns_available.keys():
                campaigns_chosen.append(c)

            else:
                logging.error("ERROR: Campaign '{}' not found".format(c))
                sys.exit(-1)
    
    output_dir = 'outputs/out_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    if "teste" in campaigns_chosen:
        output_dir = 'outputs/teste'

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging_filename = '{}/evaluation_campaigns.log'.format(output_dir)

    logging_format = '%(asctime)s\t---\t%(message)s'

    if arguments.verbosity == logging.DEBUG:
        logging_format = '%(asctime)s\t---\t%(levelname)s {%(module)s} [%(funcName)s] %(message)s'

    command = COMMAND
    command_plot = COMMAND_PLOT
    if arguments.pipenv:
        command = "pipenv run {}".format(command)
        command_plot = "pipenv run {}".format(command_plot)

    # formatter = logging.Formatter(logging_format, datefmt=TIME_FORMAT, level=args.verbosity)
    logging.basicConfig(format=logging_format, level=arguments.verbosity)

    # Add file rotating handler, with level DEBUG
    rotatingFileHandler = RotatingFileHandler(filename=logging_filename, maxBytes=100000, backupCount=5)
    rotatingFileHandler.setLevel(arguments.verbosity)
    rotatingFileHandler.setFormatter(logging.Formatter(logging_format))
    logging.getLogger().addHandler(rotatingFileHandler)

    print_all_settings(arguments)
    datasets_chosen = datasets

    if arguments.dataset > -1:
        datasets_chosen = [datasets_chosen[arguments.dataset]]

    logging.info(f"datasets : {datasets_chosen}")

    time_start_campaign = datetime.datetime.now()
    logging.info("\n\n\n")
    logging.info("##########################################")
    logging.info(" EVALUATION ")
    logging.info("##########################################")
    time_start_evaluation = datetime.datetime.now()

    results_grouping=[]
    count_dataset = 1

    for d in datasets_chosen:

        time_start_dataset = datetime.datetime.now()
        logging.info("\tDataset {} {}/{} ".format(d, count_dataset, len(datasets_chosen)))
        count_dataset +=1 
        dataset_name = d.split("/")[-1].split(".")[0]
        count_campaign = 1

        for c in campaigns_chosen:

            logging.info("\t\tCampaign {} {}/{} ".format(c, count_campaign, len(campaigns_chosen)))
            count_campaign += 1

            campaign = campaigns_available[c]
            params, values = zip(*campaign.items())
            combinations_dicts = [dict(zip(params, v)) for v in itertools.product(*values)]
             
            campaign_dir = os.path.join(output_dir, dataset_name, c)
            count_combination = 1

            for combination in combinations_dicts:
                plot_title = ""

                logging.info("\t\t\tcombination {}/{} ".format(count_combination, len(combinations_dicts)))
                logging.info("\t\t\t{}".format(combination))
                
                cmd = command
                cmd += f" --data_load_path_file_input {d}"
                cmd += " --verbosity {}".format(arguments.verbosity)
                output_dir_run = "{}".format(os.path.join(campaign_dir, "combination_{}".format(count_combination)))
                cmd += " --output_dir {}".format(output_dir_run)
                count_combination += 1

                for param in combination.keys():
                    cmd += " --{} {}".format(param, combination[param])

                    if param == 'variational_autoencoder_dense_layer_sizes_encoder':
                        layers = combination[param].split()
                        print(layers)
                        layers = layers[::-1]
                        print(layers)
                        layers = " ".join(layers)

                        cmd += " --variational_autoencoder_dense_layer_sizes_decoder {}".format(layers)

                plot_title += "{} ".format(combination["model_type"])
                plot_title += "{}".format(dataset_name)

                time_start_experiment = datetime.datetime.now()
                logging.info("\t\t\t\t\tBegin Experiment: {}".format(time_start_experiment.strftime(TIME_FORMAT)))
                run_cmd(cmd)
                time_end_experiment = datetime.datetime.now()
                duration = time_end_experiment - time_start_experiment
                logging.info("\t\t\t\t\tEnd                : {}".format(time_end_experiment.strftime(TIME_FORMAT)))
                logging.info("\t\t\t\t\tExperiment duration: {}".format(duration))

                time_start_experiment = datetime.datetime.now()
                logging.info("\t\t\t\t\tBegin Plot: {}".format(time_start_experiment.strftime(TIME_FORMAT)))
                cmd = command_plot
                if list(campaigns_chosen)[-1] == c:
                    # Create list of non-empty strings
                    results_grouping.append("".join([output_dir_run,"/EvaluationResults/Results.json"]))
                    results_str = [
                        str(item) for item in results_grouping  
                        if str(item).strip()  # This filters out empty/whitespace-only strings
                    ]
                    cmd += " --results {}".format(",".join(results_str))
                    cmd += " --f_plot  "
                    #results_grouping.append(output_dir_run)
                else: 
                    cmd += " --results {}".format("".join([output_dir_run,"/EvaluationResults/Results.json"]))
                    results_grouping.append("".join([output_dir_run,"/EvaluationResults/Results.json"]))
                cmd += " --title {}".format(plot_title)

                cmd += " --folds {}".format(combination['number_k_folds'])

                cmd += " --dataset {}".format(d)

                cmd += " --output_dir {}".format(output_dir_run)
                cmd += " --model {}".format(combination["model_type"])
                if combination["model_type"] not in ["copy", "copula", "ctgan", 'tvae']:

                    training = ["{}/Monitor/monitor_model_{}_fold.json".format(output_dir_run, x)
                                for x in range(combination['number_k_folds'])]

                    cmd += " --training {}".format(" ".join(map(str, training)))

                run_cmd(cmd)

                time_end_experiment = datetime.datetime.now()
                duration = time_end_experiment - time_start_experiment
                logging.info("\t\t\t\t\tEnd Plot                : {}".format(time_end_experiment.strftime(TIME_FORMAT)))
                logging.info("\t\t\t\t\tPlot duration: {}".format(duration))

            time_end_campaign = datetime.datetime.now()
            logging.info("\t\t Campaign duration: {}".format(time_end_campaign - time_start_campaign))

        time_end_dataset = datetime.datetime.now()
        logging.info("\t Dataset duration: {}".format(time_end_dataset - time_start_dataset))

    time_end_evaluation = datetime.datetime.now()
    logging.info("Evaluation duration: {}".format(time_end_evaluation - time_start_evaluation))


if __name__ == '__main__':
    sys.exit(main())

 
