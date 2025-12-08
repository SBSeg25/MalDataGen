#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

from Framework.Instances.AdversarialInstance import AdversarialInstance
from Framework.Instances.AutoencoderInstanceTorch import AutoencoderInstance
from Engine.Algorithms.DenoisingDiffusion.DenoisingDiffusionInstance import DenoisingDiffusionInstance
from Engine.Algorithms.LatentDiffusion.LatentDiffusionInstance import LatentDiffusionInstance
from Framework.Instances.QuantizedVAEInstanceTorch import QuantizedVAEInstance
from Engine.Algorithms.RandomNoise.AlgorithmRandomNoise import RandomNoiseAlgorithm
from Engine.Callbacks.CallbackModel import ModelMonitorCallback
from Engine.Callbacks.CallbackResources import ResourceMonitorCallback
from Framework.Instances.SmoteInstance import SmoteInstance
from Framework.Instances.VariationalAutoencoderInstanceTorch import VariationalAutoencoderInstance
from Framework.Instances.WassersteinInstanceTorch import WassersteinInstance
from Framework.Instances.WassersteinGPInstanceTensorflow import WassersteinGPInstance

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

except ImportError as error:
    logging.error(error)
    sys.exit(-1)


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
            self._latent_variational_algorithm.get_samples(number_samples_per_class)
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



