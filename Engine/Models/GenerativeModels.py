#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

from Engine.Algorithms.Adversarial.AdversarialInstance import AdversarialInstance
from Engine.Algorithms.Autoencoder.AutoencoderInstance import AutoencoderInstance
from Engine.Algorithms.DenoisingDiffusion.DenoisingDiffusionInstance import DenoisingDiffusionInstance
from Engine.Algorithms.LatentDiffusion.LatentDiffusionInstance import LatentDiffusionInstance
from Engine.Algorithms.QuantizedVAE.QuantizedVAEInstance import QuantizedVAEInstance
from Engine.Models.Wasserstein.WassersteinInstance import WassersteinInstance
from Engine.Models.WassersteinGP.WassersteinGPInstance import WassersteinGPInstance

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

    import keras
    import numpy

    import logging
    import tensorflow

    from tensorflow.keras.optimizers import Adam

    from tensorflow.keras.utils import to_categorical

    from tensorflow.python.keras.losses import MeanSquaredError
    from Engine.Callbacks.CallbackEarlyStop import EarlyStopping

    from tensorflow.python.keras.losses import BinaryCrossentropy

    from Engine.Algorithms.Copy.CopyAlgorithm import CopyAlgorithm

    from Engine.Callbacks.CallbackModel import ModelMonitorCallback
    from Engine.Models.LatentDiffusion.DiffusionModelUnet import UNetModel

    from Engine.Algorithms.SMOTE.AlgorithmSMOTE import SMOTEAlgorithm

    from Engine.Callbacks.CallbackResources import ResourceMonitorCallback

    from Engine.Models.Adversarial.AdversarialModel import AdversarialModel
    from Engine.Models.Autoencoder.ModelAutoencoder import AutoencoderModel

    from Engine.Models.WassersteinGP.ModelWassersteinGPGAN import WassersteinGPModel
    from Engine.Models.QuantizedVAE.ModelQuantizedVAE import QuantizedVAEModel
    from Engine.Models.DenoisingDiffusion.DiffusionModelUnet import UNetDenoisingModel
    from Engine.Algorithms.LatentDiffusion.GaussianLatentDiffusion import GaussianDiffusion
    from Engine.Models.DiffusionKernel.DiffusionModelUnet import UNetModelKernel

    from Engine.Algorithms.Wasserstein.AlgorithmWassersteinGAN import WassersteinAlgorithm
    from Engine.Models.Wasserstein.ModelWassersteinGAN import WassersteinModel

    from Engine.Algorithms.RandomNoise.AlgorithmRandomNoise import RandomNoiseAlgorithm
    from Engine.Algorithms.Adversarial.AdversarialAlgorithm import AdversarialAlgorithm
    from Engine.Algorithms.Autoencoder.AutoencoderAlgorithm import AutoencoderAlgorithm

    from Engine.Algorithms.WassersteinGP.AlgorithmWassersteinGANGP import WassersteinGPAlgorithm
    from Engine.Algorithms.QuantizedVAE.AlgorithmQuantizedVAE import QuantizedVAEAlgorithm

    from Engine.Models.LatentDiffusion.VariationalAutoencoderModel import VariationalModelDiffusion

    from Engine.Models.VariationalAutoencoder.VariationalAutoencoderModel import VariationalModel
    from Engine.Algorithms.LatentDiffusion.AlgorithmLatentDiffusion import LatentDiffusionAlgorithm

    from Engine.Algorithms.LatentDiffusion.AlgorithmVAELatentDiffusion import VAELatentDiffusionAlgorithm
    from Engine.Algorithms.DenoisingDiffusion.AlgorithmDenoisingDiffusion import DenoisingDiffusionAlgorithm
    from Engine.Algorithms.VariationalAutoencoder.AlgorithmVariationalAutoencoder import VariationalAlgorithm



except ImportError as error:
    logging.error(error)
    sys.exit(-1)




class VariationalAutoencoderInstance:
    """
    A class that implements a Variational Autoencoder (VAE) for probabilistic generative modeling.
    This implementation combines an encoder-decoder architecture with variational inference to learn
    a compressed latent representation of input data while enabling efficient sampling and generation.

    Key Components:
    - Encoder network that maps inputs to a latent distribution
    - Decoder network that reconstructs inputs from latent samples
    - KL divergence regularization for latent space structure
    - Flexible architecture configuration via arguments
    - Complete training pipeline with monitoring

    Attributes:
        _variation_model: Contains the encoder and decoder networks
        _variational_algorithm: Manages the VAE training process

        # VAE Architecture Parameters
        _variational_autoencoder_latent_dimension: Dimensionality of latent space
        _variational_autoencoder_training_algorithm: Training methodology
        _variational_autoencoder_activation_function: Activation for hidden layers
        _variational_autoencoder_dropout_decay_rate_encoder: Dropout rate for encoder
        _variational_autoencoder_dropout_decay_rate_decoder: Dropout rate for decoder
        _variational_autoencoder_dense_layer_sizes_encoder: Layer sizes for encoder
        _variational_autoencoder_dense_layer_sizes_decoder: Layer sizes for decoder
        _variational_autoencoder_batch_size: Training batch size
        _variational_autoencoder_number_classes: Number of output classes
        _variational_autoencoder_loss_function: Composite loss (reconstruction + KL)
        _variational_autoencoder_momentum: Optimizer momentum parameter
        _variational_autoencoder_number_epochs: Training epochs
        _variational_autoencoder_last_activation_layer: Output layer activation
        _variational_autoencoder_initializer_mean: Weight init mean
        _variational_autoencoder_initializer_deviation: Weight init std dev

        # Latent Space Parameters
        _variational_autoencoder_mean_distribution: Distribution type for latent mean
        _variational_autoencoder_stander_deviation: Std dev for latent distribution

        # Model Persistence
        _variational_autoencoder_file_name_encoder: Encoder save filename
        _variational_autoencoder_file_name_decoder: Decoder save filename
        _variational_autoencoder_path_output_models: Model save directory
    """

    def __init__(self, arguments):
        """
        Initializes the VAE instance with configuration parameters.

        Args:
            arguments (Namespace): Configuration object containing:
                - Encoder/decoder architecture parameters
                - Training hyperparameters
                - Latent space configuration
                - Model persistence settings
        """
        self._variation_model = None
        self._variational_algorithm = None

        # ** Variational Autoencoder (VAE) Configuration Parameters **
        self._variational_autoencoder_latent_dimension = arguments.variational_autoencoder_latent_dimension
        self._variational_autoencoder_training_algorithm = arguments.variational_autoencoder_training_algorithm
        self._variational_autoencoder_activation_function = arguments.variational_autoencoder_activation_function
        self._variational_autoencoder_dropout_decay_rate_encoder = arguments.variational_autoencoder_dropout_decay_rate_encoder
        self._variational_autoencoder_dropout_decay_rate_decoder = arguments.variational_autoencoder_dropout_decay_rate_decoder
        self._variational_autoencoder_dense_layer_sizes_encoder = arguments.variational_autoencoder_dense_layer_sizes_encoder
        self._variational_autoencoder_dense_layer_sizes_decoder = arguments.variational_autoencoder_dense_layer_sizes_decoder
        self._variational_autoencoder_batch_size = arguments.variational_autoencoder_batch_size
        self._variational_autoencoder_number_classes = arguments.variational_autoencoder_number_classes
        self._variational_autoencoder_loss_function = arguments.variational_autoencoder_loss_function
        self._variational_autoencoder_momentum = arguments.variational_autoencoder_momentum
        self._variational_autoencoder_number_epochs = arguments.variational_autoencoder_number_epochs
        self._variational_autoencoder_last_activation_layer = arguments.variational_autoencoder_last_activation_layer

        # Latent Space Parameters
        self._variational_autoencoder_initializer_mean = arguments.variational_autoencoder_initializer_mean
        self._variational_autoencoder_initializer_deviation = arguments.variational_autoencoder_initializer_deviation
        self._variational_autoencoder_mean_distribution = arguments.variational_autoencoder_mean_distribution
        self._variational_autoencoder_stander_deviation = arguments.variational_autoencoder_stander_deviation

        # Model Persistence
        self._variational_autoencoder_file_name_encoder = arguments.variational_autoencoder_file_name_encoder
        self._variational_autoencoder_file_name_decoder = arguments.variational_autoencoder_file_name_decoder
        self._variational_autoencoder_path_output_models = arguments.variational_autoencoder_path_output_models


    def _get_variational_autoencoder(self, input_shape):
        """
        Initializes and sets up a Variational Autoencoder (VAE) model.

        This method creates an instance of a Variational Autoencoder (VAE) by configuring its encoder and decoder
        components. It uses a custom `VariationalModel` class to define and manage these components, and a `VariationalAlgorithm`
        to handle the training and operations of the VAE model. The VAE is designed for probabilistic inference and data generation.

        Args:
            input_shape (tuple):
                The shape of the input data, which is used to define the output shape of the model.

        Initializes:
            self._variation_model:
                An instance of the `VariationalModel` class that includes the encoder and decoder setup with
                configurations like latent dimension, activation functions, dropout rates, and neural network sizes.
            self._variational_algorithm:
                An instance of the `VariationalAlgorithm` class that handles the VAE's training process, loss function,
                and model parameters, including latent mean and standard deviation distributions.

        """

        # Variational Model setup for the VAE's encoder and decoder
        self._variation_model = VariationalModel(latent_dimension=self._variational_autoencoder_latent_dimension,
                                                 output_shape=input_shape,
                                                 activation_function=self._variational_autoencoder_activation_function,
                                                 initializer_mean=self._variational_autoencoder_initializer_mean,
                                                 initializer_deviation=self._variational_autoencoder_initializer_deviation,
                                                 dropout_decay_encoder=self._variational_autoencoder_dropout_decay_rate_encoder,
                                                 dropout_decay_decoder=self._variational_autoencoder_dropout_decay_rate_decoder,
                                                 last_layer_activation=self._variational_autoencoder_last_activation_layer,
                                                 number_neurons_encoder=self._variational_autoencoder_dense_layer_sizes_encoder,
                                                 number_neurons_decoder=self._variational_autoencoder_dense_layer_sizes_decoder,
                                                 dataset_type=numpy.float32, number_samples_per_class = self._number_samples_per_class)

        # Variational Algorithm setup for training and model operations
        self._variational_algorithm = VariationalAlgorithm(encoder_model=self._variation_model.get_encoder(),
                                                           decoder_model=self._variation_model.get_decoder(),
                                                           loss_function=self._variational_autoencoder_loss_function,
                                                           latent_dimension=self._variational_autoencoder_latent_dimension,
                                                           decoder_latent_dimension = self._variational_autoencoder_latent_dimension,
                                                           latent_mean_distribution=self._variational_autoencoder_mean_distribution,
                                                           latent_stander_deviation=self._variational_autoencoder_stander_deviation,
                                                           file_name_encoder=self._variational_autoencoder_file_name_encoder,
                                                           file_name_decoder=self._variational_autoencoder_file_name_decoder,
                                                           models_saved_path=self._variational_autoencoder_path_output_models)


    def _training_variational_autoencoder_model(self, input_shape, arguments, x_real_samples, y_real_samples):
        """
        Executes the complete VAE training pipeline.

        The training process:
        1. Initializes encoder and decoder models
        2. Configures the composite loss (reconstruction + KL divergence)
        3. Sets up optimizer with specified parameters
        4. Trains using minibatch gradient descent
        5. Manages training callbacks and monitoring

        Args:
            input_shape (tuple): Input data dimensions
            arguments (Namespace): Training configuration parameters
            x_real_samples (ndarray): Training dataset samples
            y_real_samples (ndarray): Corresponding sample labels
        """
        # Initialize the variational autoencoder model
        self._get_variational_autoencoder(input_shape)

        # Print the model summaries for the encoder and decoder
        self._variation_model.get_encoder().summary()
        self._variation_model.get_decoder().summary()

        variational_optimizer = keras.optimizers.Adam()
        # Compile the variational autoencoder algorithm with the specified loss function
        self._variational_algorithm.compile(loss=self._variational_autoencoder_loss_function,
                                            optimizer=variational_optimizer)

        # callbacks_list = [self._callback_resources_monitor, self._callback_model_monitor]
        callbacks_list = [self._callback_model_monitor]

        if arguments.use_early_stop:
            callbacks_list.append(self._callback_early_stop)

        # Fit the variational autoencoder model
        self._variational_algorithm.fit((x_real_samples, to_categorical(y_real_samples,
                                           num_classes=self._number_samples_per_class["number_classes"])),
                                        x_real_samples, epochs=self._variational_autoencoder_number_epochs,
                                        batch_size=self._variational_autoencoder_batch_size,
                                        callbacks=callbacks_list)


    # Getter and setter for variational_autoencoder_latent_dimension
    @property
    def variational_autoencoder_latent_dimension(self):
        return self._variational_autoencoder_latent_dimension

    @variational_autoencoder_latent_dimension.setter
    def variational_autoencoder_latent_dimension(self, value):
        self._variational_autoencoder_latent_dimension = value

    # Getter and setter for variational_autoencoder_training_algorithm
    @property
    def variational_autoencoder_training_algorithm(self):
        return self._variational_autoencoder_training_algorithm

    @variational_autoencoder_training_algorithm.setter
    def variational_autoencoder_training_algorithm(self, value):
        self._variational_autoencoder_training_algorithm = value

    # Getter and setter for variational_autoencoder_activation_function
    @property
    def variational_autoencoder_activation_function(self):
        return self._variational_autoencoder_activation_function

    @variational_autoencoder_activation_function.setter
    def variational_autoencoder_activation_function(self, value):
        self._variational_autoencoder_activation_function = value

    # Getter and setter for variational_autoencoder_dropout_decay_rate_encoder
    @property
    def variational_autoencoder_dropout_decay_rate_encoder(self):
        return self._variational_autoencoder_dropout_decay_rate_encoder

    @variational_autoencoder_dropout_decay_rate_encoder.setter
    def variational_autoencoder_dropout_decay_rate_encoder(self, value):
        self._variational_autoencoder_dropout_decay_rate_encoder = value

    # Getter and setter for variational_autoencoder_dropout_decay_rate_decoder
    @property
    def variational_autoencoder_dropout_decay_rate_decoder(self):
        return self._variational_autoencoder_dropout_decay_rate_decoder

    @variational_autoencoder_dropout_decay_rate_decoder.setter
    def variational_autoencoder_dropout_decay_rate_decoder(self, value):
        self._variational_autoencoder_dropout_decay_rate_decoder = value

    # Getter and setter for variational_autoencoder_dense_layer_sizes_encoder
    @property
    def variational_autoencoder_dense_layer_sizes_encoder(self):
        return self._variational_autoencoder_dense_layer_sizes_encoder

    @variational_autoencoder_dense_layer_sizes_encoder.setter
    def variational_autoencoder_dense_layer_sizes_encoder(self, value):
        self._variational_autoencoder_dense_layer_sizes_encoder = value

    # Getter and setter for variational_autoencoder_dense_layer_sizes_decoder
    @property
    def variational_autoencoder_dense_layer_sizes_decoder(self):
        return self._variational_autoencoder_dense_layer_sizes_decoder

    @variational_autoencoder_dense_layer_sizes_decoder.setter
    def variational_autoencoder_dense_layer_sizes_decoder(self, value):
        self._variational_autoencoder_dense_layer_sizes_decoder = value

    # Getter and setter for variational_autoencoder_batch_size
    @property
    def variational_autoencoder_batch_size(self):
        return self._variational_autoencoder_batch_size

    @variational_autoencoder_batch_size.setter
    def variational_autoencoder_batch_size(self, value):
        self._variational_autoencoder_batch_size = value

    # Getter and setter for variational_autoencoder_number_classes
    @property
    def variational_autoencoder_number_classes(self):
        return self._variational_autoencoder_number_classes

    @variational_autoencoder_number_classes.setter
    def variational_autoencoder_number_classes(self, value):
        self._variational_autoencoder_number_classes = value

    # Getter and setter for variational_autoencoder_loss_function
    @property
    def variational_autoencoder_loss_function(self):
        return self._variational_autoencoder_loss_function

    @variational_autoencoder_loss_function.setter
    def variational_autoencoder_loss_function(self, value):
        self._variational_autoencoder_loss_function = value

    # Getter and setter for variational_autoencoder_momentum
    @property
    def variational_autoencoder_momentum(self):
        return self._variational_autoencoder_momentum

    @variational_autoencoder_momentum.setter
    def variational_autoencoder_momentum(self, value):
        self._variational_autoencoder_momentum = value

    # Getter and setter for variational_autoencoder_last_activation_layer
    @property
    def variational_autoencoder_last_activation_layer(self):
        return self._variational_autoencoder_last_activation_layer

    @variational_autoencoder_last_activation_layer.setter
    def variational_autoencoder_last_activation_layer(self, value):
        self._variational_autoencoder_last_activation_layer = value

    # Getter and setter for variational_autoencoder_initializer_mean
    @property
    def variational_autoencoder_initializer_mean(self):
        return self._variational_autoencoder_initializer_mean

    @variational_autoencoder_initializer_mean.setter
    def variational_autoencoder_initializer_mean(self, value):
        self._variational_autoencoder_initializer_mean = value

    # Getter and setter for variational_autoencoder_initializer_deviation
    @property
    def variational_autoencoder_initializer_deviation(self):
        return self._variational_autoencoder_initializer_deviation

    @variational_autoencoder_initializer_deviation.setter
    def variational_autoencoder_initializer_deviation(self, value):
        self._variational_autoencoder_initializer_deviation = value

    # Getter and setter for variational_autoencoder_mean_distribution
    @property
    def variational_autoencoder_mean_distribution(self):
        return self._variational_autoencoder_mean_distribution

    @variational_autoencoder_mean_distribution.setter
    def variational_autoencoder_mean_distribution(self, value):
        self._variational_autoencoder_mean_distribution = value

    # Getter and setter for variational_autoencoder_stander_deviation
    @property
    def variational_autoencoder_stander_deviation(self):
        return self._variational_autoencoder_stander_deviation

    @variational_autoencoder_stander_deviation.setter
    def variational_autoencoder_stander_deviation(self, value):
        self._variational_autoencoder_stander_deviation = value

    # Getter and setter for variational_autoencoder_file_name_encoder
    @property
    def variational_autoencoder_file_name_encoder(self):
        return self._variational_autoencoder_file_name_encoder

    @variational_autoencoder_file_name_encoder.setter
    def variational_autoencoder_file_name_encoder(self, value):
        self._variational_autoencoder_file_name_encoder = value

    # Getter and setter for variational_autoencoder_file_name_decoder
    @property
    def variational_autoencoder_file_name_decoder(self):
        return self._variational_autoencoder_file_name_decoder

    @variational_autoencoder_file_name_decoder.setter
    def variational_autoencoder_file_name_decoder(self, value):
        self._variational_autoencoder_file_name_decoder = value

    # Getter and setter for variational_autoencoder_path_output_models
    @property
    def variational_autoencoder_path_output_models(self):
        return self._variational_autoencoder_path_output_models

    @variational_autoencoder_path_output_models.setter
    def variational_autoencoder_path_output_models(self, value):
        self._variational_autoencoder_path_output_models = value




class SmoteInstance:
    """
    A class that implements the Synthetic Minority Over-sampling Technique (SMOTE) for handling
    class imbalance in datasets. SMOTE generates synthetic samples for minority classes by
    interpolating between existing instances, effectively balancing the class distribution.

    Key Components:
    - SMOTE algorithm implementation for synthetic sample generation
    - Configurable neighborhood size for interpolation
    - Flexible sampling strategy for target class distribution
    - Random state control for reproducibility

    Attributes:
        _smote_algorithm: The core SMOTE algorithm instance
        _smote_sampling_strategy: Target sampling strategy for class balancing
        _smote_random_state: Seed for random number generation
        _smote_k_neighbors: Number of nearest neighbors to consider for interpolation
    """
    def __init__(self, arguments):
        """
        Initializes the SMOTE instance with configuration parameters.

        Args:
            arguments (Namespace): Configuration object containing:
                - smote_sampling_strategy: Target class distribution strategy
                - smote_random_state: Random seed for reproducibility
                - smote_k_neighbors: Number of neighbors for synthetic sample generation
        """
        self._smote_algorithm = None

        # SMOTE Configuration Parameters
        self._smote_sampling_strategy = arguments.smote_sampling_strategy
        self._smote_random_state = arguments.smote_random_state
        self._smote_k_neighbors = arguments.smote_k_neighbors


    def _get_smote(self, input_shape):
        """
        Initializes and configures the SMOTE algorithm with the specified parameters.

        This method creates an instance of the SMOTEAlgorithm with the configured:
        - Sampling strategy for target class distribution
        - Random state for reproducible results
        - Number of nearest neighbors for synthetic sample generation

        Args:
            input_shape (tuple): The shape of the input data (unused in SMOTE but kept for interface consistency)

        Initializes:
            self._smote_algorithm (SMOTEAlgorithm): The configured SMOTE algorithm instance
        """
        self._smote_algorithm = SMOTEAlgorithm(sampling_strategy = self._smote_sampling_strategy,
                                               random_state = self._smote_random_state,
                                               k_neighbors = self._smote_k_neighbors)


    def _training_smote_model(self, input_shape, arguments, x_real_samples, y_real_samples):
        """
        Executes the SMOTE training process to generate synthetic samples.

        The training process:
        1. Initializes the SMOTE algorithm with configured parameters
        2. Fits the SMOTE model to the input data
        3. Generates synthetic samples for minority classes

        Args:
            input_shape (tuple): Input data dimensions (unused but kept for interface consistency)
            arguments (Namespace): Training configuration (unused in this implementation)
            x_real_samples (ndarray): Original feature vectors
            y_real_samples (ndarray): Corresponding class labels

        Note:
            The method converts labels to categorical format internally to handle multi-class scenarios.
        """
        # Initialize the autoencoder model
        self._get_smote(input_shape)

        # Fit the autoencoder model
        self._smote_algorithm.fit(x_real_samples,
                                  to_categorical(y_real_samples,
                                                 num_classes=self._number_samples_per_class["number_classes"]))

    @property
    def smote_sampling_strategy(self):
        """Get the SMOTE sampling strategy."""
        return self._smote_sampling_strategy

    @smote_sampling_strategy.setter
    def smote_sampling_strategy(self, value):
        """Set the SMOTE sampling strategy."""
        self._smote_sampling_strategy = value

    @property
    def smote_random_state(self):
        """Get the SMOTE random state."""
        return self._smote_random_state

    @smote_random_state.setter
    def smote_random_state(self, value):
        """Set the SMOTE random state."""
        self._smote_random_state = value

    @property
    def smote_k_neighbors(self):
        """Get the SMOTE k-neighbors value."""
        return self._smote_k_neighbors

    @smote_k_neighbors.setter
    def smote_k_neighbors(self, value):
        """Set the SMOTE k-neighbors value."""
        self._smote_k_neighbors = value





class GenerativeModels(AdversarialInstance,
                       AutoencoderInstance,
                       QuantizedVAEInstance,
                       LatentDiffusionInstance,
                       WassersteinInstance,
                       WassersteinGPInstance,
                       VariationalAutoencoderInstance,
                       SmoteInstance,
                       DenoisingDiffusionInstance):

    """
    A class to manage and facilitate the training and generation of various types of generative models,
    including Generative Adversarial Networks (GANs), Autoencoders (AEs), Variational Autoencoders (VAEs),
    LatentDiffusion Models, and WassersteinGP GANs (WGANs). This class provides an interface to configure, initialize,
    and manage the training processes for these models, as well as to generate synthetic data from them.

    It supports flexibility in architecture selection and offers detailed configuration options for each model.
    Additionally, it handles various model types, training parameters, and their specific settings to ensure
    a smooth and efficient workflow for deep learning practitioners working with generative models.

    The class enables users to choose and fine-tune the architecture, training procedures, and hyperparameters
    of these models, facilitating experiments with different generative approaches. Each model type is encapsulated
    with distinct algorithms and training strategies, enabling easy experimentation and comparison.

    Supported Models:
    ----------------
    - **Generative Adversarial Networks (GANs)**:
        A class of generative models that consists of a generator and a discriminator, trained in a competitive process.
        The generator creates synthetic data, and the discriminator distinguishes real from fake data.

    - **Autoencoders (AEs)**:
        A type of neural network used to learn efficient codings of input data. Autoencoders are often used
        for data compression and denoising.

    - **Variational Autoencoders (VAEs)**: A probabilistic variant of autoencoders, designed to model the
        data distribution more effectively by learning a latent space with continuous values. This is
        particularly useful for generating new data samples.

    - **LatentDiffusion Models**: A family of generative models that gradually transform noise into data through
        a sequence of steps. They have gained significant attention for image generation tasks.

    - **Wasserstein GAN (WGAN)**: A type of GAN that uses the Wasserstein distance for training,
        which provides more stable training and better convergence than traditional GANs.

    - **Wasserstein GP GANs (WGAN-GP)**: A type of GAN that uses the Wasserstein distance for training
        + Gradient Penalty, which provides more stable training and better convergence than traditional GANs.

    - **Vector Quantizer Variational Autoencoder (VQ-VAE)**: A type of variational autoencoder that incorporates
        vector quantization for discrete latent representations. Unlike traditional VAEs, VQ-VAE maps inputs
        to a fixed set of learned embeddings, improving the quality and interpretability of the latent space.

    The class allows you to experiment with these models using a unified API, where you can easily configure
    each model's architecture, initialize weights, and set training hyperparameters.

    Attributes:
    -----------
        @arguments (dict):
            A dictionary containing configuration parameters necessary for model initialization. This includes
            model-specific hyperparameters like learning rate, batch size, latent dimensions, etc.

        @_callback_model_monitor (object):
            A callback for monitoring model performance during training. This can be used for logging,
            visualization, and tracking metrics.

        @_callback_resources_monitor (object):
            A callback for tracking resource usage (e.g., memory, GPU utilization) during training to ensure
            efficient resource management.

        @_decoder_diffusion (object):
            Instance of the decoder for the diffusion model, responsible for generating data samples by
            reversing the diffusion process.

        @_encoder_diffusion (object):
            Instance of the encoder for the diffusion model, which converts input data into a latent
            representation during the forward diffusion process.

        @_diffusion_algorithm (object):
            The core diffusion model algorithm, which orchestrates the noise process and the reverse diffusion steps.

        @_adversarial_algorithm (object):
            The generative adversarial algorithm for GANs, which includes both the generator and discriminator,
            and their adversarial training.

        @_autoencoder_algorithm (object):
            The autoencoder model algorithm that defines the encoding and decoding processes, used for
            unsupervised learning and data reconstruction.

        @_variational_algorithm_diffusion (object):
            The variational autoencoder algorithm specifically designed for diffusion models, enabling
            the generation of high-quality samples.

        @_wasserstein_algorithm (object):
            The Wasserstein GAN algorithm, which incorporates the Wasserstein distance metric to improve the
            stability and convergence of GAN training.

        @_wasserstein_gp_algorithm (object):
            The WassersteinGP GAN algorithm, which incorporates the WassersteinGP distance metric to improve the
            stability and convergence of GAN training, also include Gradient Penalty.

        @_vector_quantizer_vae (object):
            The Vector Quantizer VAE algorithm, which uses discrete latent embeddings through vector quantization
            to improve reconstruction quality and enable better generative modeling.

        @_copy_algorithm (Copy):
            A helper class for duplicating model configurations or weights, useful for saving model checkpoints,
            or transferring learned weights between models.


    Methods:
    --------
    @__init__(arguments: dict)
        Initializes the class by setting up all model parameters, callbacks, and algorithms according to
        the provided configuration dictionary.

    @build_models()
        Builds the generative models based on the selected architecture. This method initializes each model’s
        components (e.g., encoder/decoder, generator/discriminator) and sets up the training pipeline.

    @train_models()
        Trains the selected models based on the provided training configurations. It manages the entire training
        process, including the optimization and loss calculation.

    @generate_samples(model_type: str, num_samples: int)
        Generates synthetic data samples using the specified model type. This method can be used for generating
        data, texts, or other data formats depending on the model.

    @save_models()
        Saves the trained models to the specified output directory. Models are saved with their current weights,
        training state, and hyperparameters, allowing easy restoration later.

    @load_models(model_directory: str)
        Loads a trained model from the given directory. This method is used to restore previously trained models
        for further evaluation or fine-tuning.

    Example:
    --------
    >>> Sample configuration dictionary
    ...     arguments = {
    ...     "learning_rate": 0.0002,
    ...     "batch_size": 64,
    ...     "latent_dim": 128,
    ...     "epochs": 100,
    ...     "model_type": "GAN",
    ...     }
    ...     # Create and train a GAN model
    ...     generative_model = GenerativeModels(arguments)
    ...     generative_model.build_models()
    ...     generative_model.train_models()
    ...     # Generate synthetic images
    ...     synthetic_images = generative_model.generate_samples(model_type="GAN", num_samples=10)
    ...     # Save the trained model
    ...     generative_model.save_models()
    ...     # Evaluate the trained model
    >>>     evaluation_results = generative_model.evaluate_model(model_type="GAN", evaluation_data=test_data)
    """

    def __init__(self, arguments):
        """
        Initializes the GenerativeModels class with model configuration parameters.

        The constructor accepts a dictionary of configuration arguments that contain necessary parameters
        for initializing different types of generative models such as GANs, AEs, VAEs, and LatentDiffusion Models.
        It sets up placeholders for various components used in each model and algorithm, and it configures
        model-specific attributes based on the provided arguments.

        Args:
            arguments (dict): A dictionary containing configuration settings for model architectures, hyperparameters,
                              training options, and paths for saving model files. This dictionary is expected to
                              include keys for model parameters, batch sizes, epochs, learning rates, and other
                              settings necessary for training.
        """

        AdversarialInstance.__init__(self, arguments)
        AutoencoderInstance.__init__(self, arguments)
        QuantizedVAEInstance.__init__(self, arguments)
        LatentDiffusionInstance.__init__(self, arguments)
        WassersteinInstance.__init__(self, arguments)
        WassersteinGPInstance.__init__(self, arguments)
        VariationalAutoencoderInstance.__init__(self, arguments)
        SmoteInstance.__init__(self, arguments)
        DenoisingDiffusionInstance.__init__(self, arguments)

        self._callback_model_monitor = None
        self._callback_resources_monitor = None
        self._callback_early_stop = None

        self._random_noise_algorithm = None

        self._copy_algorithm = CopyAlgorithm()
        self.arguments = arguments

        self._random_noise_level = arguments.random_noise_level
        self._random_noise_type_noise = arguments.random_noise_type_noise

        self._number_samples_per_class = arguments.number_samples_per_class


    def _get_random_noise(self, input_shape):

        self._random_noise_algorithm = RandomNoiseAlgorithm(noise_level=self._random_noise_level,
                                                            noise_type=self._random_noise_type_noise)


    def training_model(self, arguments, input_shape, x_real_samples, y_real_samples, monitor_path, k_fold):
        """
        Trains a model based on the selected type: adversarial, diffusion, WassersteinGP, variational, or autoencoder.

        This method handles the training process by first initializing the model according to the `model_type`
        specified in `arguments`. Then, it compiles and fits the model using the provided samples. It also
        supports callback functions for monitoring resources during training.

        Parameters:
            - arguments (dict): Dictionary containing configuration options, including the
              model type (e.g., 'adversarial', 'autoencoder', etc.).
            - input_shape (tuple): Shape of the input data (e.g., (height, width, channels)).
            - x_real_samples (array): The real input samples used for training.
            - y_real_samples (array): The target labels corresponding to the real input samples.
            - monitor_path (str): Path to store the monitoring data for the callbacks.
            - k_fold (int): The k-fold cross-validation split number for monitoring.

        This function initializes and trains the model based on the specified model type.
        It supports the following model types:

            1. Adversarial (GAN)
            2. Autoencoder
            3. Variational Autoencoder
            4. Wasserstein + GP GAN
            5. Wasserstein GAN
            6. LatentDiffusion-based model
            7. LatentDiffusion-based model
            8. Denoising Diffusion Kernel-based model
            9. Smote model
            10. Random model
            11. Vector Quantized Variational Autoencoder

        The method also uses resource and model monitoring callbacks during training to track progress.

        """

        # Initialize resource and model monitoring callbacks
        self._callback_resources_monitor = ResourceMonitorCallback(monitor_path, k_fold)
        self._callback_model_monitor = ModelMonitorCallback(monitor_path, k_fold)
        self._callback_early_stop = EarlyStopping(arguments.early_stop_monitor,
                                                  arguments.early_stop_min_delta,
                                                  arguments.early_stop_patience,
                                                  arguments.early_stop_mode,
                                                  arguments.early_stop_baseline,
                                                  arguments.early_stop_restore_best_weights)

        # Adversarial model training
        if arguments.model_type == 'adversarial':
            self._training_adversarial_modelo(input_shape, arguments, x_real_samples, y_real_samples)

        # Autoencoder model training
        elif arguments.model_type == 'autoencoder':
            self._training_autoencoder_model(input_shape, arguments, x_real_samples, y_real_samples)

        # Autoencoder model training
        elif arguments.model_type == 'random':

            # Initialize the autoencoder model
            self._get_random_noise(input_shape)

            # Fit the autoencoder model
            self._random_noise_algorithm.fit(x_real_samples,
                                             to_categorical(y_real_samples,
                                                            num_classes=self._number_samples_per_class["number_classes"]))

        # Smote model training
        elif arguments.model_type == 'smote':

            # Initialize the autoencoder model
            self._get_smote(input_shape)

            # Fit the autoencoder model
            self._smote_algorithm.fit(
                x_real_samples, to_categorical(y_real_samples,
                                               num_classes=self._number_samples_per_class["number_classes"]))

        # Variational Autoencoder (VAE) model training
        elif arguments.model_type == 'variational':
            self._training_variational_autoencoder_model(input_shape, arguments, x_real_samples, y_real_samples)

        # WassersteinGP GAN model training
        elif arguments.model_type == 'wasserstein_gp':
            self._training_wasserstein_gp_model(input_shape, arguments, x_real_samples, y_real_samples)

        # WassersteinGP GAN model training
        elif arguments.model_type == 'wasserstein':
            self._training_wasserstein_model(input_shape, arguments, x_real_samples, y_real_samples)

        # LatentDiffusion model training
        elif arguments.model_type == 'latent_diffusion':
            self._training_latent_diffusion_model(input_shape, arguments, x_real_samples, y_real_samples)

        # LatentDiffusion model training
        elif arguments.model_type == 'denoising_diffusion':
            self._training_denoising_diffusion_model(input_shape, arguments, x_real_samples, y_real_samples)

        # LatentDiffusion Kernel model training
        elif arguments.model_type == 'diffusion_kernel':
            # Support for the 'diffusion_kernel' model type has not been implemented yet.
            # This section is reserved for initializing a generator based on diffusion kernels,
            # which may later leverage integral operators for non-parametric learning of diffusive dynamics.
            # Stay tuned — the diffusion hasn't spread here yet 😉
            pass

        # Vector Quantized Variational Autoencoder (VQ-VAE) model training
        elif arguments.model_type == 'quantized':
            self._training_quantized_VAE_model(input_shape, arguments, x_real_samples, y_real_samples)

        else:
            # If no valid model type is selected, do nothing
            pass


    def get_samples(self, number_samples_per_class):
        """
        Generate and retrieve samples from a trained model.

        This method generates synthetic samples using the trained model specified by the `model_type` argument.
        It supports multiple model types for generating samples, including adversarial, diffusion, WassersteinGP,
        variational, and autoencoder models. Depending on the selected model type, the corresponding algorithm
        is called to generate the samples.

        Args:
            number_samples_per_class (int): The number of samples to generate for each class.

        Supports the following model types:
            - 'adversarial': Uses the `AdversarialAlgorithm` to generate samples.
            - 'diffusion': Uses the `DiffusionAlgorithm` to generate samples.
            - 'wasserstein': Uses the `WassersteinAlgorithm` to generate samples.
            - 'variational': Uses the `VariationalAlgorithm` to generate samples.
            - 'autoencoder': Uses the `AutoencoderAlgorithm` to generate samples.

        Note:
            Additional models may be added in the future, such as a diffusion kernel model, but this is currently under
            implementation. The corresponding algorithm for that model is not yet available.

        """

        if self.arguments.model_type == 'adversarial':
            # Generate samples using the Adversarial algorithm
            self._adversarial_algorithm.get_samples(number_samples_per_class)
            pass

        elif self.arguments.model_type == 'latent_diffusion':
            # Generate samples using the LatentDiffusion algorithm
            self._latent_diffusion_algorithm.get_samples(number_samples_per_class)
            pass

        elif self.arguments.model_type == 'denoising_diffusion':
            # Generate samples using the LatentDiffusion algorithm
            self._denoising_diffusion_algorithm.get_samples(number_samples_per_class)
            pass

        elif self.arguments.model_type == 'wasserstein':
            # Generate samples using the WassersteinGP algorithm
            self._wasserstein_gp_algorithm.get_samples(number_samples_per_class)
            pass

        elif self.arguments.model_type == 'variational':
            # Generate samples using the Variational algorithm
            self._latent_variational_algorithm_diffusion.get_samples(number_samples_per_class)
            pass

        elif self.arguments.model_type == 'autoencoder':
            # Generate samples using the Autoencoder algorithm
            self._autoencoder_algorithm.get_samples(number_samples_per_class)
            pass

        elif self.arguments.model_type == 'random':
            # Generate samples using the Autoencoder algorithm
            self._random_noise_algorithm.get_samples(number_samples_per_class)
            pass

        elif self.arguments.model_type == 'quantized':
            # Generate samples using the Quantized Variational Autoencoder algorithm
            self._quantized_vae_algorithm.get_samples(number_samples_per_class)
            pass

        elif self.arguments.model_type == 'smote':
            # Generate samples using the Autoencoder algorithm
            self._smote_algorithm.get_samples(number_samples_per_class)
            pass


        # Algorithm in implementation process for future models
        # elif self.arguments.model_type == 'diffusion_kernel':
        #    self._diffusion_algorithm_kernel.get_samples(number_samples_per_class)
        #    pass

        else:
            # If the model type is not recognized, do nothing
            pass






def import_models(function):
    """
    Decorator to create an instance of the Metrics class
    before executing the wrapped function.

    Parameters:
        function (callable): The function to be wrapped.

    Returns:
        callable: The wrapped function that initializes Metrics.
    """
    def wrapper(self, *args, **kwargs):
        # Create an instance of Metrics, passing the arguments from the instance
        GenerativeModels.__init__(self, self.arguments)
        # Call the wrapped function with the metrics instance and other arguments
        return function(self, *args, **kwargs)

    return wrapper



