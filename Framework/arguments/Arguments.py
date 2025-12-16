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


try:
    import os
    import sys

    import logging

    from logging import FileHandler

    from Framework.arguments.LoggerSetup import LoggerSetup

    from Engine.data_io.DirectoryManager import DirectoryManager
    from Framework.arguments.Models.ArgumentsSMOTE import add_argument_smote

    from Framework.arguments.ArgumentsFramework import add_argument_framework
    from Framework.arguments.classifiers.ArgumentsKNN import add_argument_knn
    from Framework.arguments.ArgumentsDataLoader import add_argument_data_load
    from Framework.arguments.ArgumentsEarlyStop import add_argument_early_stop
    from Framework.arguments.ArgumentsOptimizer import add_argument_optimizers

    from Framework.arguments.Models.ArgumentsAdversarial import add_argument_adversarial
    from Framework.arguments.Models.ArgumentsAutoencoder import add_argument_autoencoder

    from Framework.arguments.Models.ArgumentsRandomNoise import add_argument_random_noise

    from Framework.arguments.classifiers.ArgumentsKMeans import add_argument_k_means
    from Framework.arguments.Models.ArgumentsQuantizedVAE import add_argument_quantized_vae

    from Framework.arguments.Models.ArgumentsWassersteinGAN import add_argument_wasserstein_gan

    from Framework.arguments.Models.ArgumentsLatentDiffusion import add_argument_latent_diffusion

    from Framework.arguments.Models.ArgumentsWassersteinGANGP import add_argument_wasserstein_gan_gp

    from Framework.arguments.classifiers.ArgumentsPerceptron import add_argument_perceptron
    from Framework.arguments.classifiers.ArgumentsNaiveBayes import add_argument_naive_bayes

    from Framework.arguments.Models.ArgumentsDenoisingDiffusion import add_argument_denoising_diffusion
    from Framework.arguments.classifiers.ArgumentsDecisionTree import add_argument_decision_tree
    from Framework.arguments.classifiers.ArgumentsRandomForest import add_argument_random_forest

    from Framework.arguments.Models.ArgumentsVariationalAutoencoder import add_argument_variation_autoencoder
    from Framework.arguments.classifiers.ArgumentsGaussianProcess import add_argument_gaussian_process

    from Framework.arguments.classifiers.ArgumentsGradientBoosting import add_argument_gradient_boosting
    from Framework.arguments.classifiers.ArgumentsLinearRegression import add_argument_linear_regression

    from Framework.arguments.classifiers.ArgumentsSpectralClustering import add_argument_spectral_clustering
    from Framework.arguments.classifiers.ArgumentsSuportVectorMachine import add_argument_support_vector_machine

    from Framework.arguments.classifiers.ArgumentsStochasticGradientDescent import add_argument_stochastic_gradient_descent

    from Framework.arguments.classifiers.ArgumentsQuadraticDiscriminantAnalysis import add_argument_quadratic_discriminant_analysis

except ImportError as error:
    print(error)
    sys.exit(-1)


DEFAULT_VERBOSITY = logging.INFO
TIME_FORMAT = '%Y-%m-%d,%H:%M:%S'
DEFAULT_DATA_TYPE = "float32"
DEFAULT_VERBOSE_LIST = {logging.INFO: 2,
                        logging.DEBUG: 1,
                        logging.FATAL: 0,
                        logging.ERROR: 0,
                        logging.WARNING: 2}

LOGGING_FILE_NAME = "logging.log"


def arguments(function):
    """
    Decorator to initialize an instance of the arguments class
    before executing the wrapped function.

    Parameters:
        function (callable): The function to be wrapped.

    Returns:
        callable: The wrapped function that initializes arguments.
    """
    def wrapper(self, *args, **kwargs):
        # Initialize the arguments class for the instance
        Arguments.__init__(self)
        # Call the wrapped function with the provided arguments
        return function(self, *args, **kwargs)

    return wrapper

class Arguments(DirectoryManager):
    """
    Class to manage and parse command-line arguments for various machine
    learning models and settings. It initializes the arguments using
    multiple specific argument addition functions.

    Attributes:
        arguments (Namespace): Parsed command-line arguments.
    """

    def __init__(self):
        """
        Initializes the arguments class by adding various argument
        options for different machine learning models and settings.
        It also configures logging based on verbosity settings and
        prepares the output directory for storing logs.
        """

        # Initialize arguments from the framework
        super().__init__()

        self.arguments = add_argument_framework()

        # Add various model-specific argument parsers
        self.arguments = add_argument_adversarial(self.arguments)
        self.arguments = add_argument_smote(self.arguments)
        self.arguments = add_argument_optimizers(self.arguments)
        self.arguments = add_argument_early_stop(self.arguments)
        self.arguments = add_argument_data_load(self.arguments)
        self.arguments = add_argument_random_noise(self.arguments)
        self.arguments = add_argument_autoencoder(self.arguments)
        self.arguments = add_argument_latent_diffusion(self.arguments)
        self.arguments = add_argument_denoising_diffusion(self.arguments)
        self.arguments = add_argument_quantized_vae(self.arguments)
        self.arguments = add_argument_variation_autoencoder(self.arguments)
        self.arguments = add_argument_wasserstein_gan_gp(self.arguments)
        self.arguments = add_argument_wasserstein_gan(self.arguments)
        self.arguments = add_argument_decision_tree(self.arguments)
        self.arguments = add_argument_gaussian_process(self.arguments)
        self.arguments = add_argument_gradient_boosting(self.arguments)
        self.arguments = add_argument_k_means(self.arguments)
        self.arguments = add_argument_knn(self.arguments)
        self.arguments = add_argument_naive_bayes(self.arguments)
        self.arguments = add_argument_linear_regression(self.arguments)
        self.arguments = add_argument_spectral_clustering(self.arguments)
        self.arguments = add_argument_perceptron(self.arguments)
        self.arguments = add_argument_quadratic_discriminant_analysis(self.arguments)
        self.arguments = add_argument_random_forest(self.arguments)
        self.arguments = add_argument_stochastic_gradient_descent(self.arguments)
        self.arguments = add_argument_support_vector_machine(self.arguments)

        self.arguments = self.arguments.parse_args()
        self._create_directories(base_directory=self.arguments.output_dir)

        # view_splash_screen = View()
        # view_splash_screen.print_view()

        self.logger = LoggerSetup(self.arguments)
        self.logger.setup_logger()


    def show_all_settings(self):
        """
        Logs all settings and command-line arguments after parsing.
        Displays the command used to run the script along with the
        corresponding values for each argument.
        Skips arguments related to specific models and optimizers unless they match the selected ones.
        """
        args_dict_selected = {}
        # List of model-specific prefixes
        model_prefixes = {
            'adversarial': 'adversarial',
            'autoencoder': 'autoencoder',
            'variational': 'variational',
            'wasserstein': 'wasserstein',
            'wasserstein_gp': 'wasserstein_gp',
            'diffusion': 'diffusion',
            'quantized': 'quantized'
        }

        # List of optimizer prefixes (lowercase to match)
        optimizer_prefixes = {
            'Adam': 'adam_optimizer_',
            'SGD': 'sgd_optimizer_',
            'RMSprop': 'rsmprop_optimizer_',
            'Adagrad': 'adagrad_optimizer_',
            'Adamax': 'adamax_optimizer_',
            'Nadam': 'nadam_optimizer_',
            'Adadelta': 'ada_delta_optimizer_',
            'FTRL': 'ftrl_optimizer_'
        }

        # Get the current model type and optimizer
        current_model = getattr(self.arguments, 'model_type', None)
        current_optimizer = getattr(self.arguments, 'optimizer', None)

        # Log the command used to execute the script
        logging.info("Command:\n\t{0}\n".format(" ".join([x for x in sys.argv])))
        logging.info("Settings:")

        # Get all arguments and calculate max length for formatting
        args_dict = vars(self.arguments)
        lengths = [len(x) for x in args_dict.keys()]
        max_length = max(lengths) if lengths else 0

        # Log each argument and its value
        for key, value in sorted(args_dict.items()):
            # Skip if this is the optimizer parameter itself
            if key == 'optimizer':
                continue

            # Check if this argument belongs to a specific model
            skip_model = False
            for model, prefix in model_prefixes.items():
                if key.startswith(prefix) and current_model != model:
                    skip_model = True
                    break

            # Check if this argument belongs to a specific optimizer
            skip_optimizer = False
            if current_optimizer:
                for opt, prefix in optimizer_prefixes.items():
                    if key.startswith(prefix) and current_optimizer.lower() != opt.lower():
                        skip_optimizer = True
                        break

            if skip_model or skip_optimizer:
                continue

            settings_parser = "\t"  # Start with a tab for indentation

            # Left-justify the argument name for better readability
            settings_parser += key.ljust(max_length, " ")

            # Append the value of the argument
            settings_parser += " : {}".format(value)

            # Log the formatted argument and value
            logging.info(settings_parser)
            args_dict_selected[key] = value

        logging.info("")  # Log a newline for spacing
        self._dictionary_metrics = {}
        self._dictionary_metrics["arguments"] = args_dict_selected
