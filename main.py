#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Kayuã Oleques']

import os

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

os.environ['ML_FRAMEWORK'] = 'pytorch'
try:
    import sys
    import time
    import numpy
    import pandas 
    
    import logging

    from sklearn.utils import shuffle

    from Engine.Metrics.Metrics import Metrics

    from Engine.DataIO.CSVLoader import autosave
    from Engine.DataIO.CSVLoader import autoload

    from Engine.Arguments.Arguments import Arguments
    from Engine.Arguments.Arguments import arguments

    from Engine.Metrics.Metrics import import_metrics

    from Engine.Evaluation.Evaluation import Evaluation
    from sklearn.model_selection import StratifiedKFold

    from Engine.DataIO.CSVLoader import CSVDataProcessor

    from Engine.Classifiers.Classifiers import Classifiers

    from Engine.Models.GenerativeModels import import_models

    from Engine.Models.GenerativeModels import GenerativeModels

    from Engine.Evaluation.CrossValidation import StratifiedData
    from Engine.Classifiers.Classifiers import import_classifiers
    from Engine.Support.HardwareManager import HardwareManager

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



DEFAULT_VERBOSITY = logging.INFO
TIME_FORMAT = '%Y-%m-%d,%H:%M:%S'
DEFAULT_DATA_TYPE = "float32"



class SynDataGen(Arguments, CSVDataProcessor, Metrics, GenerativeModels, Classifiers, Evaluation):
    """
    SYNTHETIC DATA GENERATION AND EVALUATION FRAMEWORK
    =================================================

    The Synthetic Ocean AI library is designed for the generation of tabular data using generative models.
    It provides comprehensive GPU support, allowing for efficient processing, and can be used either as a
    framework or as a Python library, offering flexibility depending on the user's requirements.

    The library supports data ingestion from various formats, including CSV and XLS, enabling seamless
    integration with existing datasets. It features a wide array of pre-implemented generative algorithms
    and includes pre-trained models for immediate use, reducing the need for extensive training.

    One of the key strengths of Synthetic Ocean AI is its ability to provide fine-grained model and
    algorithm parameterization, giving users control over hyperparameters, training configurations,
    and other aspects of the generative process. The library also includes built-in support for evaluation
    metrics, allowing users to assess the quality of generated data. Additionally, it offers tools for
    graph generation, enabling visual analysis of model performance and data generation processes.

    A comprehensive pipeline for generating and evaluating synthetic data using various generative models.
    The class combines data processing, model training, generation, and evaluation capabilities.

    Key Features:
    -------------
        @ Supports multiple generative models (adversarial, autoencoder, variational, etc.)
        @ Built-in stratified k-fold cross validation
        @ Multiple evaluation strategies (TS-TR, TR-TS)
        @ Automated metrics calculation and reporting
        @ Model persistence and data export capabilities

    # Version: 1.0.1
    # Last Updated: 2025-5-28
    # Author: Synthetic Ocean AI - Team
    # License: MIT

    Purpose:
    --------
      Provides an end-to-end pipeline for:
        - Generating synthetic datasets using state-of-the-art generative models
        - Evaluating synthetic data quality through multiple validation strategies
        - Comparing model performance across different architectures
        - Producing publication-ready metrics and visualizations

    Architecture Overview:
    ---------------------

                                +--------+-------+      +--------+-------+      +--------+-------+
                                |  Activations   +------+     Layers     +------+    Specials    |
                                +-------+--------+      +-------+--------+      +--------+-------+
                                                                |
                                                                |
                                +-----------------+     +-------+--------+    +--------+-------+
                                |    Arguments    |     |     Models     |    |      Loss      |
                                +--------+--------+     +--------+-------+    +--------+-------+
                                        |                        |                     |
        +---------------+       +-------+--------+               |            +--------+-------+         +--------+-------+
        | DataProcessor +-------+   Generative   +---------------@------------+   Algorithms   +---------+   Optimizers   |
        +---------------+       |     Models     |                            +----------------+         +----------------+
                                +-------+--------+
                                        |
        +-------v--------+      +-------v--------+               +----------------+
        |     Plotter    +------+     Metrics    +---------------+   Classifiers  |
        +-------+--------+      +-------+--------+               +----------------+
                                        |
                                +-------v--------+
                                |   SynDataGen   |
                                +----------------+


    Model Catalog:
    --------------
        1. Adversarial (GAN) [model_type='adversarial']

            Implements an adversarial training algorithm, typically used in Generative Adversarial Networks (GANs).

            This class performs adversarial training by utilizing a generator and a discriminator,
            optimizing the generator to produce realistic data while training the discriminator to differentiate
            between real and fake data.

        2. Autoencoder [model_type='autoencoder']

            Implements a  AutoEncoder model for generating synthetic data.

            This class implements an Autoencoder model by inheriting from the VanillaEncoder and VanillaDecoder classes.
            It constructs an autoencoder architecture by combining both an encoder and a decoder with customizable
            hyperparameters. The autoencoder is typically used for tasks such as dimensionality reduction, feature learning,
            and denoising.

        3. Variational Autoencoder [model_type='variational']

            Implements a Variational AutoEncoder (VAE) model for generating synthetic data.

            The model includes an encoder and a decoder for encoding input data and reconstructing
            it from a learned latent space. During training, it computes both the reconstruction loss
            and the KL divergence loss. The trained decoder can be used to generate synthetic data.

        4. Vector Quantized Variational Autoencoder [model_type='quantized']

            Implements a Vector Quantized Variational Autoencoder (VQ-VAE) model for generating synthetic data.

            This class implements a VQ-VAE by combining an encoder, a quantized latent space with a codebook,
            and a decoder. The model learns discrete latent representations by mapping encoder outputs to
            the nearest codebook vectors during training. The decoder then reconstructs the input from
            these quantized latent embeddings.

        5. Wasserstein GAN [model_type='wasserstein']

            A Wasserstein Generative Adversarial Network (Wasserstein GAN) model.

           This class represents a Wasserstein GAN consisting of a generator and discriminator (critic) model.
           It implements the Wasserstein loss to train the discriminator and generator, promoting more
           stable training compared to traditional GANs.

        6. Wasserstein GP GAN [model_type='wasserstein_gp']

            A Wasserstein GP Generative Adversarial Network (WassersteinGP GAN) model.

            This class represents a WassersteinGP GAN consisting of a generator and discriminator model.
            It implements the WassersteinGP loss with gradient penalty to improve the training of the
            discriminator and generator.

        7. Latent Diffusion Models [model_type='latent_diffusion']

            Implements a diffusion process using UNet architectures for generating synthetic data.

            This model integrates an autoencoder and a diffusion network, enabling both data
            reconstruction and controlled generative modeling through Gaussian diffusion.

        8. Denoising Diffusion [model_type='denoising_diffusion']

            Implements a diffusion process using UNet architectures for generating synthetic data.

            This model integrates an autoencoder and a diffusion network, enabling both data
            reconstruction and controlled generative modeling through Gaussian diffusion.

        9. Copy/Paste [model_type='copy']

            Copy is a naive machine learning model designed to generate synthetic data samples
            for specific classes based on provided real samples. This simple approach is primarily used
            for testing and comparison purposes, serving as a baseline method in experiments.


    Data Flow:
    ---------
        1. Input Data → 2. Preprocessing → 3. Stratified Splitting
        ↓                                    ↓
        7. Results Collection ← 6. Evaluation ← 5. Generation ← 4. Model Training

    Evaluation Strategies:
    --------------------
        A. TS-TR (Train Synthetic - Assess Real)
            - Trains: On generated synthetic data
            - Tests: On held-out real validation data
            - Measures: Generalization capability

        B. TR-TS (Train Real - Assess Synthetic)
            - Trains: On real training data
            - Tests: On generated synthetic data
            - Measures: Generation quality

    Metrics Tracked:
    ---------------
        Primary Metrics:
        - Accuracy, Precision, Recall, F1, ROC-AUC, FalseNegativeRate, MSE, MAE, TrueNegativeRate

        Secondary Metrics:
        - EuclideanDistance, HellingerDistance, LogLikelihood, ManhattanDistance

    Example Workflows:
    -----------------
        1. Basic Usage:
            >>> gen = SynDataGen()
            >>> gen.run_experiments()

        2. Custom Configuration:
            >>> gen = SynDataGen()
            >>> gen.arguments.model_type = 'variational'
            >>> gen.arguments.number_k_folds = 5
            >>> gen.run_experiments()

        3. Research Pipeline:
            >>> for model in ['adversarial', 'variational', 'latent_diffusion']:
            ...     gen = SynDataGen()
            ...     gen.arguments.model_type = model
            ...     gen.run_experiments()
            >>>     gen.save_comparison_report()

    """
    @arguments
    def __init__(self):
        """
        CONSTRUCTOR
        ==========
        Initializes the synthetic data generation pipeline with default parameters.

        Detailed Initialization Sequence:
        -------------------------------
        1. Parent Class Initialization:
           - Arguments: Loads CLI/config file parameters
           - CSVDataProcessor: Initializes data loading pipelines
           - Metrics: Sets up metric tracking structures
           - GenerativeModels: Prepares model architectures
           - Classifiers: Loads evaluation classifiers

        2. Instance Variable Setup:
           - fold_number: Initialized to None, tracks current CV fold [0, n_folds-1]
           - data_generated: Dictionary structure:
               {
                   class_0: np.ndarray (n_samples, n_features),
                   class_1: np.ndarray (n_samples, n_features),
                   ...
               }
           - generator_name: String identifier matching model_type
           - directory_output_data: Path object with structure:
               ./output/
                   ├── models/
                   ├── data/
                   │   ├── fold_1/
                   │   ├── ...
                   └── metrics/

        3. Filesystem Preparation:
           - Creates required directory structure
           - Initializes log files with timestamp
           - Validates write permissions

        """

        super().__init__()

        self.fold_number = None
        self.data_generated = None
        self.generator_name = None
        self.directory_output_data = None
        self.directory_output_data = self.get_data_generated_path()
        self._manager = HardwareManager(use_gpu=self.arguments.use_gpu)
        self._manager.configure()

        self._sdv = None 

    @import_metrics
    @import_classifiers
    @StratifiedData
    def run_experiments(self):
        """
        Runs the experiment across multiple folds, using the stratified data splits.
        For each fold, the method trains a model, evaluates it on both synthetic and real data,
        and logs the results. The method also updates evaluation results and saves them in a JSON file.

        The method applies decorators for importing metrics, classifiers, and stratified data splits,
        and ensures that each experiment is logged and processed properly.

        Logs the progress and completion time for each fold and the total experiment runtime.

        This method involves the following steps:
            1. Stratified data splitting for training and evaluation.
            2. Model training and prediction for each fold.
            3. Evaluation using synthetic and real data.
            4. Saving the results to a JSON file.

        Args:
            :None
        """

        logging.info("Starting experiment runs across %d folds.", len(self.list_folds))

        # Start time for the entire experiment
        total_start_time = time.time()

        try:

            # Iterate over each fold in the stratified list of folds
            for fold, dictionary_data in enumerate(self.list_folds):

                # Start time for the current fold
                fold_start_time = time.time()
                logging.info("")

                # Log the fold number
                logging.info("Running experiment for fold %d.", fold + 1)

                # Log the size and shape of the training data (features and labels)
                logging.info("Fold %d training data shape: X=%s, Y=%s", fold + 1,
                              dictionary_data['x_training_real'].shape, 
                              dictionary_data['y_training_real'].shape)

                # Update the fold number in the class instance
                self.fold_number = fold
                logging.debug("\t\tFold number updated to %d in the class.", self.fold_number)

                # Get the path to monitor the experiment's progress
                monitor_path = self.get_monitor_path()
                # number_samples_per_class = self.arguments.number_samples_per_class
                # print("number_samples_per_class", number_samples_per_class)
                
            
                # Create the model and make predictions using the training data
                self.train_model(dictionary_data['x_training_real'], 
                                 dictionary_data['y_training_real'],
                                 monitor_path, fold)
                
                self.monitoring_start_generating()

                evaluation_synthetic = self.synthesize_data(
                                              dictionary_data['x_evaluation_real'], 
                                              dictionary_data['y_evaluation_real'],
                                              )
                
                self.monitoring_stop_generating(fold)
                logging.info("\t\tModel creation and prediction completed for fold %d.", fold + 1)

                # Log the start of the evaluation process
                logging.info("")
                logging.info("")
                logging.info(" starting evaluation for fold %d.", fold + 1)

                # Perform the evaluations using synthetic and real data
                self.evaluation_TR_TS(dictionary_data, evaluation_synthetic)  
                self.evaluation_TS_TR(dictionary_data, evaluation_synthetic)  
                
                #self.evaluation_TR_TR(dictionary_data)
                # self.calculate_sdv_metrics(dictionary_data, fold)

                # End of fold, log the time taken for the current fold
                fold_end_time = time.time()
                logging.info("Fold %d experiment completed in %.2f seconds.", fold + 1, fold_end_time - fold_start_time)
                logging.info("------\n\n")
                self.save_dictionary_to_json(self.get_evaluation_results_path()+"/Results.json")
                # sys.exit(0)

            # Update and log the mean and standard deviation of the evaluation results
            self.update_mean_std_fold()
            self.save_dictionary_to_json(self.get_evaluation_results_path()+"/Results.json")
            total_end_time = time.time()  # Separate folds for clarity in logs

            # Save the evaluation results to a JSON file
            logging.info("All experiments completed in %.2f seconds.", total_end_time - total_start_time)

        except Exception as e:
            logging.error("An error occurred during experiment execution: %s", str(e))
            raise

    @import_models
    def train_model(self, x_real_samples, y_real_samples, monitor_path, k_fold):
        """
        This method is responsible for creating a model based on the specified model type, training it on real data,
        generating synthetic data using the trained model, and optionally saving the model and generated data.

        It supports various model types including adversarial, autoencoder, variational, WassersteinGP, diffusion,
        and copy-paste algorithms. After training and generating data, it logs the completion of each step and saves the models
        and data if specified in the arguments.

        Args:
            x_real_samples (array): The real input samples (features) for training the model.
            y_real_samples (array): The real target labels corresponding to the input samples.
            monitor_path (str): Path to monitor the training process, such as for storing logs or checkpoints.
            k_fold (int): The fold number in a cross-validation setup, used to save models and data for each fold.

        Raises:
            Exception: If an error occurs during model creation, training, or data generation, an exception is raised.
        """

        logging.info("Starting model creation and prediction process.")
        logging.info("Number of real samples: %d", len(x_real_samples)) # Log the number of real samples

        try:
            # Train the model with the provided real samples and labels
            logging.info("Training model with %d samples and model type: %s", len(x_real_samples),
                         self.arguments.model_type)

            self.monitoring_start_training()
            
             
            if self.arguments.model_type in ["copula", "ctgan", "tvae"]:
                self.generator_name = self.arguments.model_type
                logging.info(f"Training SDV's model {self.generator_name} algorithm.")
                
                from Engine.Algorithms.ThirdParty.SDVInterfaceAlgorithm import SDVInterfaceAlgorithm
                
                self._sdv = SDVInterfaceAlgorithm()
                
                self._sdv.training_model( x_real_samples, y_real_samples, 
                                    self._data_original_header,
                                    self.arguments.model_type)
            else:
                self.training_model(self.arguments,
                                    self.get_number_columns(),
                                    x_real_samples,
                                    y_real_samples,
                                    monitor_path,
                                    k_fold)
            
            self.monitoring_stop_training(k_fold)

            logging.info("Model training completed.")

            if self.arguments.save_models:

                logging.info("Saving trained model.")

                if self.arguments.model_type == 'adversarial':
                    self._adversarial_algorithm.save_model(self.get_models_saved_path(), k_fold)

                elif self.arguments.model_type == 'autoencoder':
                    self._autoencoder_algorithm.save_model(self.get_models_saved_path(), k_fold)

                elif self.arguments.model_type == "variational":
                    self._latent_variational_algorithm.save_model(self.get_models_saved_path(), k_fold)

                elif self.arguments.model_type == "wasserstein":
                    self._wasserstein_algorithm.save_model(self.get_models_saved_path(), k_fold)

                elif self.arguments.model_type == "wasserstein_gp":
                    self._wasserstein_gp_algorithm.save_model(self.get_models_saved_path(), k_fold)

                elif self.arguments.model_type == "latent_diffusion":
                    self._latent_diffusion_algorithm.save_model(self.get_models_saved_path(), k_fold)

                elif self.arguments.model_type == "denoising_diffusion":
                    self._denoising_diffusion_algorithm.save_model(self.get_models_saved_path(), k_fold)

                elif self.arguments.model_type == "quantized":                 
                    self._quantized_vae_algorithm.save_model(self.get_models_saved_path(), k_fold)

                else:
                    # If an invalid model type is specified, log the error and exit the program
                    logging.error("Error during model selection")
                    exit(-1)
            
        except Exception as e:
            # If any error occurs during model creation, training, or data generation, log the error
            logging.error("Error during model creation or data generation: %s", str(e))

            raise  # Reraise the exception for further handling or termination
            

    def synthesize_data(self, x_real_samples, y_real_samples):

            # Generate synthetic data based on the specified model type
            # Depending on the selected model, we use the corresponding algorithm for data generation

            #dictionary_data['y_training_real']
            #labels = dictionary_data['y_evaluation_real']
            labels = y_real_samples
            labels = labels.astype(int)

            unique_classes, counts = numpy.unique(labels, return_counts=True)
            class_counts = dict(zip(unique_classes, counts))

            number_samples_per_class = {'classes': class_counts, 'number_classes': len(unique_classes)}
            logging.info("\t\tnumber_samples_per_class", number_samples_per_class)

            if self.arguments.model_type == 'adversarial':

                # Using adversarial model to generate synthetic data
                self.generator_name = 'adversarial'
                logging.info("Generating data using Adversarial algorithm.")
                self.data_generated = self._adversarial_algorithm.get_samples(number_samples_per_class)

            elif self.arguments.model_type == 'autoencoder':

                # Using autoencoder model to generate synthetic data
                self.generator_name = 'autoencoder'
                logging.info("Generating data using Autoencoder algorithm.")
                self.data_generated = self._autoencoder_algorithm.get_samples(number_samples_per_class)


            elif self.arguments.model_type == "variational":

                # Using variational model to generate synthetic data
                self.generator_name = 'variational'
                logging.info("Generating data using Variational algorithm.")
                self.data_generated = self._variational_algorithm.get_samples(number_samples_per_class)

            elif self.arguments.model_type == "wasserstein":

                # Using WassersteinGP model to generate synthetic data
                self.generator_name = 'wasserstein'
                logging.info("Generating data using Wasserstein algorithm.")
                self.data_generated = self._wasserstein_algorithm.get_samples(number_samples_per_class)

            elif self.arguments.model_type == "wasserstein_gp":

                # Using WassersteinGP model to generate synthetic data
                self.generator_name = 'wasserstein_gp'
                logging.info("Generating data using Wasserstein GP algorithm.")
                self.data_generated = self._wasserstein_gp_algorithm.get_samples(number_samples_per_class)

            elif self.arguments.model_type == "latent_diffusion":

                # Using diffusion model to generate synthetic data
                self.generator_name = 'latent_diffusion'
                logging.info("Generating data using LatentDiffusion algorithm.")
                self.data_generated = self._latent_diffusion_algorithm.get_samples(number_samples_per_class)

            elif self.arguments.model_type == "denoising_diffusion":

                # Using diffusion model to generate synthetic data
                self.generator_name = 'denoising_diffusion'
                logging.info("Generating data using Denoising Diffusion algorithm.")
                self.data_generated = self._denoising_diffusion_algorithm.get_samples(number_samples_per_class)


            elif self.arguments.model_type == "copy":
                # Using copy-paste model to generate synthetic data (by copying and pasting from real data)
                self.generator_name = 'copy'
                logging.info("Generating data using copy & paste algorithm.")
                
                self.data_generated = self._copy_algorithm.get_samples(number_samples_per_class,
                                                                       x_real_samples, y_real_samples)

            elif self.arguments.model_type == "quantized":
                # Using copy-paste model to generate synthetic data (by copying and pasting from real data)
                self.generator_name = 'quantized'
                logging.info("Generating data using Vector Quantized Variational Autoencoder algorithm.")

                self.data_generated = self._quantized_vae_algorithm.get_samples(number_samples_per_class)

            elif self.arguments.model_type == "random":

                # Using Random Noise Model
                self.generator_name = 'random'
                logging.info("Generating data using Random Noise  algorithm.")

                self.data_generated = self._random_noise_algorithm.get_samples(number_samples_per_class)

            elif self.arguments.model_type == "smote":

                # Using SMOTE model to generate synthetic data
                self.generator_name = 'smote'
                logging.info("Generating data using SMOTE algorithm.")
                self.data_generated = self._smote_algorithm.get_samples(number_samples_per_class)

            elif self.arguments.model_type in ["copula", "ctgan", "tvae"]:
                # Using copy-paste model to generate synthetic data (by copying and pasting from real data)
                self.generator_name = self.arguments.model_type
                logging.info(f"Generating data using SDV's {self.generator_name} algorithm.")
                
                self.data_generated = self._sdv.get_samples(number_samples_per_class)

            else:
                # If an invalid model type is specified, log the error and exit the program
                logging.error("Error during model selection")
                exit(-1)

            # Completion log for data generation process
            logging.info("Data generation completed successfully for model type: %s", self.arguments.model_type)

            # If specified, save the generated synthetic data
            if self.arguments.save_data:
                self.save_data_generated()
            
            return self.data_generated


    @autosave
    def save_data_generated(self):
        """
        Save the generated data to the specified location.
        """
        logging.info("Entered the save_data_generated method.")

        try:
            logging.info("Attempting to save generated data.")
            """
                @autosave
            
            """
            logging.info("Generated data saved successfully.")

        except Exception as e:
            logging.error(f"Error while saving generated data: {str(e)}")
            raise

if __name__ == "__main__":
    dataGeneration = SynDataGen()
    dataGeneration.show_all_settings()
    dataGeneration.run_experiments()

