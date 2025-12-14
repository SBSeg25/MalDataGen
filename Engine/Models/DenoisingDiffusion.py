#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/13'
__credits__ = ['Synthetic Ocean AI']

from Engine.Algorithms.denoising_diffusion.AlgorithmDenoisingDiffusion import AlgorithmDenoisingDiffusion
from Engine.Architectures.DenoisingDiffusion.DenoisingDiffusionUnetModel import DenoisingDiffusionUNetModel

try:
    import sys
    import os
    import logging
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from Engine.Algorithms.denoising_diffusion.GaussianDenoisingDiffusion import GaussianDenoisingDiffusion

except ImportError as error:
    logging.error(error)
    sys.exit(-1)

# Default values
DEFAULT_DIFFUSION_UNET_LAST_LAYER_ACTIVATION = 'linear'
DEFAULT_DIFFUSION_LATENT_DIMENSION = 32
DEFAULT_DIFFUSION_UNET_NUMBER_EMBEDDING_CHANNELS = 1
DEFAULT_DIFFUSION_UNET_CHANNELS_PER_LEVEL = [1, 2, 4]
DEFAULT_DIFFUSION_UNET_BATCH_SIZE = 128
DEFAULT_DIFFUSION_UNET_ATTENTION_MODE = [False, True, True]
DEFAULT_DIFFUSION_UNET_NUMBER_RESIDUAL_BLOCKS = 2
DEFAULT_DIFFUSION_UNET_GROUP_NORMALIZATION = 1
DEFAULT_DIFFUSION_UNET_INTERMEDIARY_ACTIVATION = 'swish'
DEFAULT_DIFFUSION_UNET_INTERMEDIARY_ACTIVATION_ALPHA = 0.05
DEFAULT_DIFFUSION_UNET_NUMBER_EPOCHS = 1000
DEFAULT_DIFFUSION_GAUSSIAN_BETA_START = 1e-4
DEFAULT_DIFFUSION_GAUSSIAN_BETA_END = 0.02
DEFAULT_DIFFUSION_GAUSSIAN_TIME_STEPS = 1000
DEFAULT_DIFFUSION_GAUSSIAN_CLIP_MIN = -1.0
DEFAULT_DIFFUSION_GAUSSIAN_CLIP_MAX = 1.0
DEFAULT_DIFFUSION_MARGIN = 0.5
DEFAULT_DIFFUSION_EMA = 0.999
DEFAULT_DIFFUSION_TIME_STEPS = 1000


class DenoisingDiffusion:
    """
    Framework-agnostic Denoising Diffusion Probabilistic Model (DDPM) implementation.

    NOW SUPPORTS MULTI-DIMENSIONAL DATA: Automatically handles 1D, 2D, 3D, and N-D inputs
    by flattening them during training and reshaping them during generation.

    This class contains BOTH TensorFlow and PyTorch implementations in a single class.
    The framework is selected based on the ML_FRAMEWORK environment variable.

    Supported frameworks:
        - 'Tensorflow': Uses TensorFlow/Keras implementation
        - 'pytorch': Uses PyTorch implementation

    Set the framework by: os.environ['ML_FRAMEWORK'] = 'pytorch' or 'Tensorflow'
    Default framework is TensorFlow if ML_FRAMEWORK is not set.

    Attributes:
        _original_input_shape (tuple): Stores the original shape of input data for reconstruction
    """

    def __init__(
            self,
            unet_last_layer_activation: str = DEFAULT_DIFFUSION_UNET_LAST_LAYER_ACTIVATION,
            latent_dimension: int = DEFAULT_DIFFUSION_LATENT_DIMENSION,
            unet_num_embedding_channels: int = DEFAULT_DIFFUSION_UNET_NUMBER_EMBEDDING_CHANNELS,
            unet_channels_per_level: list[int] = None,
            unet_batch_size: int = DEFAULT_DIFFUSION_UNET_BATCH_SIZE,
            unet_attention_mode: list[bool] = None,
            number_classes: int = 3,
            unet_num_residual_blocks: int = DEFAULT_DIFFUSION_UNET_NUMBER_RESIDUAL_BLOCKS,
            unet_group_normalization: int = DEFAULT_DIFFUSION_UNET_GROUP_NORMALIZATION,
            unet_intermediary_activation: str = DEFAULT_DIFFUSION_UNET_INTERMEDIARY_ACTIVATION,
            unet_intermediary_activation_alpha: float = DEFAULT_DIFFUSION_UNET_INTERMEDIARY_ACTIVATION_ALPHA,
            number_epochs: int = DEFAULT_DIFFUSION_UNET_NUMBER_EPOCHS,
            gaussian_beta_start: float = DEFAULT_DIFFUSION_GAUSSIAN_BETA_START,
            gaussian_beta_end: float = DEFAULT_DIFFUSION_GAUSSIAN_BETA_END,
            gaussian_time_steps: int = DEFAULT_DIFFUSION_GAUSSIAN_TIME_STEPS,
            gaussian_clip_min: float = DEFAULT_DIFFUSION_GAUSSIAN_CLIP_MIN,
            gaussian_clip_max: float = DEFAULT_DIFFUSION_GAUSSIAN_CLIP_MAX,
            margin: float = DEFAULT_DIFFUSION_MARGIN,
            ema: float = DEFAULT_DIFFUSION_EMA,
            time_steps: int = DEFAULT_DIFFUSION_TIME_STEPS,
            first_unet_model=None,
            second_unet_model=None,
            gaussian_diffusion_util=None,
            algorithm=None
    ) -> None:
        """
        Initializes the framework-agnostic denoising diffusion instance.

        Args:
            unet_last_layer_activation: Activation for last layer (default: 'linear')
            latent_dimension: Dimension of latent space (default: 64)
            unet_num_embedding_channels: Embedding channels (default: 1)
            unet_channels_per_level: Channels per U-Net level (default: [1, 2, 4])
            unet_batch_size: Batch size (default: 128)
            unet_attention_mode: Attention modes per level (default: [False, True, True])
            unet_num_residual_blocks: Residual blocks (default: 2)
            unet_group_normalization: Group norm groups (default: 1)
            unet_intermediary_activation: Intermediary activation (default: 'swish')
            unet_intermediary_activation_alpha: Activation alpha (default: 0.05)
            number_epochs: Training epochs (default: 1000)
            gaussian_beta_start: Beta start value (default: 1e-4)
            gaussian_beta_end: Beta end value (default: 0.02)
            gaussian_time_steps: Diffusion time steps (default: 1000)
            gaussian_clip_min: Noise clip minimum (default: -1.0)
            gaussian_clip_max: Noise clip maximum (default: 1.0)
            margin: Margin for diffusion (default: 0.5)
            ema: Exponential moving average (default: 0.999)
            time_steps: Time steps (default: 1000)
            first_unet_model: Optional pre-initialized first UNet (default: None)
            second_unet_model: Optional pre-initialized second UNet (default: None)
            gaussian_diffusion_util: Optional pre-initialized diffusion util (default: None)
            algorithm: Optional pre-initialized algorithm (default: None)
        """
        # Detect framework from environment variable
        self._framework = os.environ.get('ML_FRAMEWORK', 'Tensorflow').lower()

        if self._framework not in ['tensorflow', 'pytorch']:
            raise ValueError(
                f"Unsupported framework: {self._framework}. "
                f"Supported frameworks are 'Tensorflow' or 'pytorch'"
            )

        logging.info(f"Initializing DenoisingDiffusion with framework: {self._framework}")

        # Store pre-initialized instances if provided
        self._denoising_first_unet_model = first_unet_model
        self._denoising_second_unet_model = second_unet_model
        self._denoising_gaussian_diffusion_util = gaussian_diffusion_util
        self._denoising_diffusion_algorithm = algorithm

        # Internal instances
        self._denoising_first_instance_unet = None
        self._denoising_second_instance_unet = None

        # Configuration parameters
        self._denoising_diffusion_unet_last_layer_activation = unet_last_layer_activation
        self._denoising_diffusion_latent_dimension = latent_dimension
        self._denoising_diffusion_unet_num_embedding_channels = unet_num_embedding_channels

        # Handle mutable default values safely
        self._denoising_diffusion_unet_channels_per_level = (
            unet_channels_per_level
            if unet_channels_per_level is not None
            else DEFAULT_DIFFUSION_UNET_CHANNELS_PER_LEVEL.copy()
        )
        self._denoising_diffusion_unet_attention_mode = (
            unet_attention_mode
            if unet_attention_mode is not None
            else DEFAULT_DIFFUSION_UNET_ATTENTION_MODE.copy()
        )

        self._denoising_diffusion_unet_batch_size = unet_batch_size
        self._denoising_diffusion_unet_num_residual_blocks = unet_num_residual_blocks
        self._denoising_diffusion_unet_group_normalization = unet_group_normalization
        self._denoising_diffusion_unet_intermediary_activation = unet_intermediary_activation
        self._denoising_diffusion_unet_intermediary_activation_alpha = unet_intermediary_activation_alpha
        self._denoising_diffusion_unet_epochs = number_epochs

        # Diffusion Process Parameters
        self._denoising_diffusion_gaussian_beta_start = gaussian_beta_start
        self._denoising_diffusion_gaussian_beta_end = gaussian_beta_end
        self._denoising_diffusion_gaussian_time_steps = gaussian_time_steps
        self._denoising_diffusion_gaussian_clip_min = gaussian_clip_min
        self._denoising_diffusion_gaussian_clip_max = gaussian_clip_max
        self._denoising_diffusion_margin = margin
        self._denoising_diffusion_ema = ema
        self._denoising_diffusion_time_steps = time_steps

        # Framework-specific attributes
        if self._framework == 'pytorch':
            import torch
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Training history
        self._training_history = {
            'epoch': [],
            'loss': [],
            'avg_loss': []
        }

        # Flags to indicate if instances were provided
        self._has_external_first_unet = first_unet_model is not None
        self._has_external_second_unet = second_unet_model is not None
        self._has_external_diffusion_util = gaussian_diffusion_util is not None
        self._has_external_algorithm = algorithm is not None

        # Callback placeholders
        self._callback_model_monitor = None
        self._callback_early_stop = None
        self._callback_resources_monitor = None

        # Number of samples per class (set during training)
        self._number_samples_per_class = None

        # Storage for original input shape (for multi-dimensional data)
        self._original_input_shape = None

    def _get_denoising_diffusion_pytorch(self, input_shape: tuple[int, ...]) -> None:
        """
        PyTorch implementation: Initializes and configures the diffusion model using UNet architecture.
        """

        # Only create new UNet models if none were provided
        if not self._has_external_first_unet:
            self._denoising_first_instance_unet = DenoisingDiffusionUNetModel(
                output_shape=input_shape,
                embedding_channels=self._denoising_diffusion_unet_num_embedding_channels,
                list_neurons_per_level=self._denoising_diffusion_unet_channels_per_level,
                list_attentions=self._denoising_diffusion_unet_attention_mode,
                number_residual_blocks=self._denoising_diffusion_unet_num_residual_blocks,
                normalization_groups=self._denoising_diffusion_unet_group_normalization,
                intermediary_activation_function=self._denoising_diffusion_unet_intermediary_activation,
                intermediary_activation_alpha=self._denoising_diffusion_unet_intermediary_activation_alpha,
                last_layer_activation=self._denoising_diffusion_unet_last_layer_activation,
                number_samples_per_class=self._number_samples_per_class
            ).to(self._device)
            self._denoising_first_unet_model = self._denoising_first_instance_unet
        else:
            self._denoising_first_unet_model = self._denoising_first_unet_model.to(self._device)

        if not self._has_external_second_unet:
            self._denoising_second_instance_unet = DenoisingDiffusionUNetModel(
                output_shape=input_shape,
                embedding_channels=self._denoising_diffusion_unet_num_embedding_channels,
                list_neurons_per_level=self._denoising_diffusion_unet_channels_per_level,
                list_attentions=self._denoising_diffusion_unet_attention_mode,
                number_residual_blocks=self._denoising_diffusion_unet_num_residual_blocks,
                normalization_groups=self._denoising_diffusion_unet_group_normalization,
                intermediary_activation_function=self._denoising_diffusion_unet_intermediary_activation,
                intermediary_activation_alpha=self._denoising_diffusion_unet_intermediary_activation_alpha,
                last_layer_activation=self._denoising_diffusion_unet_last_layer_activation,
                number_samples_per_class=self._number_samples_per_class
            ).to(self._device)
            self._denoising_second_unet_model = self._denoising_second_instance_unet

            # Synchronize weights from first model if both were created
            if not self._has_external_first_unet:
                self._denoising_second_unet_model.load_state_dict(
                    self._denoising_first_unet_model.state_dict()
                )
        else:
            self._denoising_second_unet_model = self._denoising_second_unet_model.to(self._device)

        # Only create new GaussianDiffusion utility if none was provided
        if not self._has_external_diffusion_util:
            self._denoising_gaussian_diffusion_util = GaussianDenoisingDiffusion(
                beta_start=self._denoising_diffusion_gaussian_beta_start,
                beta_end=self._denoising_diffusion_gaussian_beta_end,
                time_steps=self._denoising_diffusion_gaussian_time_steps,
                clip_min=self._denoising_diffusion_gaussian_clip_min,
                clip_max=self._denoising_diffusion_gaussian_clip_max
            )

        # Only create new algorithm if none was provided
        if not self._has_external_algorithm:
            # Validate required components
            if self._denoising_first_unet_model is None:
                raise ValueError("First UNet model is required but was not provided.")
            if self._denoising_second_unet_model is None:
                raise ValueError("Second UNet model is required but was not provided.")
            if self._denoising_gaussian_diffusion_util is None:
                raise ValueError("GaussianDiffusion utility is required but was not provided.")

            optimizer_diffusion = torch.optim.Adam(
                self._denoising_first_unet_model.parameters(),
                lr=0.0001
            )
            optimizer_autoencoder = torch.optim.Adam(
                self._denoising_first_unet_model.parameters(),
                lr=0.0001
            )

            self._denoising_diffusion_algorithm = AlgorithmDenoisingDiffusion(
                output_shape=input_shape,
                first_unet_model=self._denoising_first_unet_model,
                second_unet_model=self._denoising_second_unet_model,
                gdf_util=self._denoising_gaussian_diffusion_util,
                optimizer_autoencoder=optimizer_autoencoder,
                optimizer_diffusion=optimizer_diffusion,
                time_steps=self._denoising_diffusion_gaussian_time_steps,
                ema=self._denoising_diffusion_ema,
                margin=self._denoising_diffusion_margin
            ).to(self._device)
        else:
            # Ensure external algorithm is properly assigned
            if self._denoising_diffusion_algorithm is None:
                raise ValueError(
                    "External algorithm was marked as provided but is None. "
                    "Please provide a valid algorithm instance."
                )
            # Update algorithm parameters
            if hasattr(self._denoising_diffusion_algorithm, 'time_steps'):
                self._denoising_diffusion_algorithm.time_steps = self._denoising_diffusion_gaussian_time_steps
            if hasattr(self._denoising_diffusion_algorithm, 'ema'):
                self._denoising_diffusion_algorithm.ema = self._denoising_diffusion_ema
            if hasattr(self._denoising_diffusion_algorithm, 'margin'):
                self._denoising_diffusion_algorithm.margin = self._denoising_diffusion_margin
            self._denoising_diffusion_algorithm = self._denoising_diffusion_algorithm.to(self._device)

    def _get_denoising_diffusion_tensorflow(self, input_shape: tuple[int, ...]) -> None:
        """
        TensorFlow implementation: Initializes and configures the diffusion model using UNet architecture.
        """
        # Skip if models already provided externally
        if (self._has_external_first_unet and
                self._has_external_second_unet and
                self._has_external_diffusion_util):
            logging.info("Using externally provided TensorFlow models")
            return

        try:
            from Engine.Architectures.DenoisingDiffusion.DenoisingDiffusionUnetModel import \
                DenoisingDiffusionUNetModel
        except ImportError as e:
            try:
                from Engine.Architectures.DenoisingDiffusion.Tensorflow.DenoisingDiffusionUnetModel import \
                    DenoisingDiffusionUNetModel
            except ImportError as e2:
                raise ImportError(
                    f"Failed to import DenoisingDiffusionUNetModel for TensorFlow. "
                    f"First attempt: {e}. Second attempt: {e2}"
                ) from e2

        # Initialize the first instance of UNet
        self._denoising_first_instance_unet = DenoisingDiffusionUNetModel(
            output_shape=input_shape,
            embedding_channels=self._denoising_diffusion_unet_num_embedding_channels,
            list_neurons_per_level=self._denoising_diffusion_unet_channels_per_level,
            list_attentions=self._denoising_diffusion_unet_attention_mode,
            number_residual_blocks=self._denoising_diffusion_unet_num_residual_blocks,
            normalization_groups=self._denoising_diffusion_unet_group_normalization,
            intermediary_activation_function=self._denoising_diffusion_unet_intermediary_activation,
            intermediary_activation_alpha=self._denoising_diffusion_unet_intermediary_activation_alpha,
            last_layer_activation=self._denoising_diffusion_unet_last_layer_activation,
            number_samples_per_class=self._number_samples_per_class
        )

        # Initialize the second instance of UNet
        self._denoising_second_instance_unet = DenoisingDiffusionUNetModel(
            output_shape=input_shape,
            embedding_channels=self._denoising_diffusion_unet_num_embedding_channels,
            list_neurons_per_level=self._denoising_diffusion_unet_channels_per_level,
            list_attentions=self._denoising_diffusion_unet_attention_mode,
            number_residual_blocks=self._denoising_diffusion_unet_num_residual_blocks,
            normalization_groups=self._denoising_diffusion_unet_group_normalization,
            intermediary_activation_function=self._denoising_diffusion_unet_intermediary_activation,
            intermediary_activation_alpha=self._denoising_diffusion_unet_intermediary_activation_alpha,
            last_layer_activation=self._denoising_diffusion_unet_last_layer_activation,
            number_samples_per_class=self._number_samples_per_class
        )

        # Build the models
        self._denoising_first_unet_model = self._denoising_first_instance_unet.build_model()
        self._denoising_second_unet_model = self._denoising_second_instance_unet.build_model()

        # Synchronize weights
        self._denoising_second_unet_model.set_weights(self._denoising_first_unet_model.get_weights())

        # Initialize GaussianDiffusion utility
        self._denoising_gaussian_diffusion_util = GaussianDenoisingDiffusion(
            beta_start=self._denoising_diffusion_gaussian_beta_start,
            beta_end=self._denoising_diffusion_gaussian_beta_end,
            time_steps=self._denoising_diffusion_gaussian_time_steps,
            clip_min=self._denoising_diffusion_gaussian_clip_min,
            clip_max=self._denoising_diffusion_gaussian_clip_max
        )

    def _training_denoising_diffusion_model_pytorch(
            self,
            input_shape: tuple[int, ...],
            x_real_samples: np.ndarray,
            y_real_samples: np.ndarray
    ) -> None:
        """
        PyTorch training implementation (simplified - delegates to AlgorithmDenoisingDiffusionTorch).
        """

        # Initialize the diffusion model
        self._get_denoising_diffusion_pytorch(input_shape)

        # CRITICAL VALIDATION: Ensure algorithm was created
        if self._denoising_diffusion_algorithm is None:
            raise RuntimeError(
                "Failed to initialize denoising diffusion algorithm. "
                "This may be due to missing models or incorrect configuration. "
                "Please check that all required components are properly initialized."
            )

        # Extract training parameters
        batch_size = self._denoising_diffusion_unet_batch_size
        epochs = self._denoising_diffusion_unet_epochs

        # Extract callbacks
        callback_model_monitor =  self._callback_model_monitor
        callback_early_stop = self._callback_early_stop
        use_early_stop = False

        # Delegate training to the algorithm's fit method
        history = self._denoising_diffusion_algorithm.fit(
            x_real_samples=x_real_samples,
            y_real_samples=y_real_samples,
            batch_size=batch_size,
            epochs=epochs,
            callback_model_monitor=callback_model_monitor,
            callback_early_stop=callback_early_stop,
            use_early_stop=use_early_stop
        )

        # Update local training history
        self._training_history = history


    def _training_denoising_diffusion_model_tensorflow(
            self,
            input_shape: tuple[int, ...],
            x_real_samples: np.ndarray,
            y_real_samples: np.ndarray
    ) -> None:
        """
        TensorFlow training implementation.
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.utils import to_categorical
            from tensorflow.python.keras.losses import MeanSquaredError
        except ImportError as e:
            raise ImportError(
                f"Failed to import TensorFlow dependencies: {e}. "
                "Make sure TensorFlow is installed."
            ) from e

        try:
            from Engine.Algorithms.denoising_diffusion.AlgorithmDenoisingDiffusion import \
                AlgorithmDenoisingDiffusion
        except ImportError:
            try:
                from Engine.Algorithms.denoising_diffusion.Tensorflow.AlgorithmDenoisingDiffusion import \
                    AlgorithmDenoisingDiffusion
            except ImportError as e:
                raise ImportError(
                    f"Failed to import AlgorithmDenoisingDiffusion: {e}. "
                    "Check your import paths."
                ) from e

        # Initialize the diffusion model
        self._get_denoising_diffusion_tensorflow(input_shape)

        # Setup callbacks
        callbacks_list = [self._callback_model_monitor] if self._callback_model_monitor else []

        if self._callback_early_stop:
            callbacks_list.append(self._callback_early_stop)

        # Initialize the diffusion algorithm
        self._denoising_diffusion_algorithm = AlgorithmDenoisingDiffusion(
            output_shape=input_shape,
            first_unet_model=self._denoising_first_unet_model,
            second_unet_model=self._denoising_second_unet_model,
            gdf_util=self._denoising_gaussian_diffusion_util,
            optimizer_autoencoder=Adam(learning_rate=0.0001),
            optimizer_diffusion=Adam(learning_rate=0.0001),
            time_steps=self._denoising_diffusion_gaussian_time_steps,
            ema=self._denoising_diffusion_ema,
            margin=self._denoising_diffusion_margin
        )

        # Compile the model
        self._denoising_diffusion_algorithm.compile(
            loss=MeanSquaredError(),
            optimizer=Adam(learning_rate=0.0001)
        )

        # Prepare data
        x_real_samples = np.array(x_real_samples)
        x_real_samples = tf.expand_dims(x_real_samples, axis=-1)

        # Train the model
        history = self._denoising_diffusion_algorithm.fit(
            x_real_samples,
            to_categorical(y_real_samples, num_classes=self._number_samples_per_class["number_classes"]),
            epochs=self._denoising_diffusion_unet_epochs,
            batch_size=self._denoising_diffusion_unet_batch_size,
            callbacks=callbacks_list
        )

        # Store training history
        if history and hasattr(history, 'history'):
            for epoch in range(len(history.history.get('loss', []))):
                self._training_history['epoch'].append(epoch + 1)
                self._training_history['loss'].append(history.history['loss'][epoch])
                self._training_history['avg_loss'].append(history.history['loss'][epoch])

    def fit_model(
            self,
            input_shape: tuple,
            x_real_samples: np.ndarray,
            y_real_samples: np.ndarray
    ) -> None:
        """
        Train the denoising diffusion model using the appropriate framework with automatic data flattening.

        NOW SUPPORTS MULTI-DIMENSIONAL DATA:
        - Automatically flattens N-D input data (1D, 2D, 3D, etc.)
        - Stores original shape for reconstruction during generation
        - Routes to framework-specific implementation

        Args:
            input_shape: Shape of input data samples (e.g., (784,) for 1D, (28, 28, 1) for 2D, (16, 16, 16) for 3D)
            arguments: Training arguments/configuration
            x_real_samples: Training samples (can be N-dimensional)
            y_real_samples: Training labels (1D array of class indices)
        """
        # Store original input shape for later reconstruction
        self._original_input_shape = input_shape

        # Calculate total flattened dimension
        flattened_dim = int(np.prod(input_shape))

        # CRITICAL: Set up number_samples_per_class from training labels
        # This must be done before building the models
        num_classes = int(y_real_samples.max()) + 1
        self._number_samples_per_class = {
            "number_classes": num_classes
        }


        # Flatten the input data if it has more than 2 dimensions
        # (batch_size, ...) -> (batch_size, flattened_features)
        if len(x_real_samples.shape) > 2:
            x_real_samples_flat = x_real_samples.reshape(x_real_samples.shape[0], -1)
        else:
            x_real_samples_flat = x_real_samples


        # Route to appropriate training method with flattened dimension
        if self._framework == 'pytorch':
            self._training_denoising_diffusion_model_pytorch(
                flattened_dim, x_real_samples_flat, y_real_samples
            )
        elif self._framework == 'tensorflow':
            self._training_denoising_diffusion_model_tensorflow(
                flattened_dim, x_real_samples_flat, y_real_samples
            )

    def train(
            self,
            input_shape: tuple,
            x_real_samples: np.ndarray,
            y_real_samples: np.ndarray
    ) -> None:
        """
        Public method to train the denoising diffusion model.
        This is an alias to fit_model for convenience.

        Args:
            input_shape: Shape of input data
            arguments: Training arguments/configuration
            x_real_samples: Training samples
            y_real_samples: Training labels
        """
        self.fit_model(
            input_shape, x_real_samples, y_real_samples
        )

    def get_samples(self, number_samples_per_class):
        """
        Generate samples using the trained model and reshape them to original input shape.

        NOW SUPPORTS MULTI-DIMENSIONAL OUTPUT:
        - Automatically reshapes generated samples to original input dimensions
        - Works with 1D, 2D, 3D, and N-D data

        Args:
            number_samples_per_class: Dictionary specifying number of samples per class
                Format 1: {"number_classes": N, "classes": {0: n0, 1: n1, ...}}
                Format 2 (simplified): {"number_classes": N, 0: n0, 1: n1, ...}

        Returns:
            np.ndarray: Generated samples reshaped to original input dimensions
                - Shape: (total_samples, *original_input_shape)
                - Example: For 3D input (16, 16, 16) with 100 total samples -> (100, 16, 16, 16)

        Raises:
            RuntimeError: If algorithm is not initialized
        """
        if self._denoising_diffusion_algorithm is None:
            raise RuntimeError(
                "Cannot generate samples: algorithm is not initialized. "
                "Please train the model first using fit_model() or train()."
            )

        # Generate flattened samples from algorithm
        generated_data = self._denoising_diffusion_algorithm.get_samples(number_samples_per_class)

        # Check if we have stored original input shape
        if not hasattr(self, '_original_input_shape') or self._original_input_shape is None:
            # If no original shape stored, return flattened data
            # Convert dict to array if needed
            if isinstance(generated_data, dict):
                all_samples = []
                for label in sorted(generated_data.keys()):
                    all_samples.append(generated_data[label])
                return np.concatenate(all_samples, axis=0)
            return generated_data

        # Reshape generated samples to original input shape
        reshaped_samples = []

        if isinstance(generated_data, dict):
            # If returned as dict, reshape each class
            for label in sorted(generated_data.keys()):
                samples = generated_data[label]
                # Reshape from (n_samples, flattened_features) to (n_samples, *original_shape)
                reshaped = samples.reshape(-1, *self._original_input_shape)
                reshaped_samples.append(reshaped)

            # Concatenate all classes
            return np.concatenate(reshaped_samples, axis=0)
        else:
            # If returned as array, reshape directly
            return generated_data.reshape(-1, *self._original_input_shape)

    def get_training_history(self) -> dict:
        """
        Returns the training history.

        Returns:
            dict: Dictionary containing epoch, loss, and avg_loss lists
        """
        return self._training_history

    # Properties
    @property
    def framework(self) -> str:
        """Get the current framework being used."""
        return self._framework

    @property
    def original_input_shape(self) -> tuple:
        """Get the original input shape (before flattening)."""
        return self._original_input_shape

    @property
    def denoising_first_unet_model(self):
        """Get the first UNet model instance."""
        return self._denoising_first_unet_model

    @property
    def denoising_second_unet_model(self):
        """Get the second UNet model instance."""
        return self._denoising_second_unet_model

    @property
    def denoising_gaussian_diffusion_util(self):
        """Get the Gaussian diffusion utility instance."""
        return self._denoising_gaussian_diffusion_util

    @property
    def denoising_diffusion_algorithm(self):
        """Get the diffusion algorithm instance."""
        return self._denoising_diffusion_algorithm

    # Property getters and setters for all configuration parameters
    @property
    def denoising_diffusion_unet_last_layer_activation(self) -> str:
        return self._denoising_diffusion_unet_last_layer_activation

    @denoising_diffusion_unet_last_layer_activation.setter
    def denoising_diffusion_unet_last_layer_activation(self, value: str) -> None:
        self._denoising_diffusion_unet_last_layer_activation = value

    @property
    def denoising_diffusion_latent_dimension(self) -> int:
        return self._denoising_diffusion_latent_dimension

    @denoising_diffusion_latent_dimension.setter
    def denoising_diffusion_latent_dimension(self, value: int) -> None:
        self._denoising_diffusion_latent_dimension = value

    @property
    def denoising_diffusion_unet_num_embedding_channels(self) -> int:
        return self._denoising_diffusion_unet_num_embedding_channels

    @denoising_diffusion_unet_num_embedding_channels.setter
    def denoising_diffusion_unet_num_embedding_channels(self, value: int) -> None:
        self._denoising_diffusion_unet_num_embedding_channels = value

    @property
    def denoising_diffusion_unet_channels_per_level(self) -> list[int]:
        return self._denoising_diffusion_unet_channels_per_level

    @denoising_diffusion_unet_channels_per_level.setter
    def denoising_diffusion_unet_channels_per_level(self, value: list[int]) -> None:
        self._denoising_diffusion_unet_channels_per_level = value

    @property
    def denoising_diffusion_unet_batch_size(self) -> int:
        return self._denoising_diffusion_unet_batch_size

    @denoising_diffusion_unet_batch_size.setter
    def denoising_diffusion_unet_batch_size(self, value: int) -> None:
        self._denoising_diffusion_unet_batch_size = value

    @property
    def denoising_diffusion_unet_attention_mode(self) -> list[bool]:
        return self._denoising_diffusion_unet_attention_mode

    @denoising_diffusion_unet_attention_mode.setter
    def denoising_diffusion_unet_attention_mode(self, value: list[bool]) -> None:
        self._denoising_diffusion_unet_attention_mode = value

    @property
    def denoising_diffusion_unet_num_residual_blocks(self) -> int:
        return self._denoising_diffusion_unet_num_residual_blocks

    @denoising_diffusion_unet_num_residual_blocks.setter
    def denoising_diffusion_unet_num_residual_blocks(self, value: int) -> None:
        self._denoising_diffusion_unet_num_residual_blocks = value

    @property
    def denoising_diffusion_unet_group_normalization(self) -> int:
        return self._denoising_diffusion_unet_group_normalization

    @denoising_diffusion_unet_group_normalization.setter
    def denoising_diffusion_unet_group_normalization(self, value: int) -> None:
        self._denoising_diffusion_unet_group_normalization = value

    @property
    def denoising_diffusion_unet_intermediary_activation(self) -> str:
        return self._denoising_diffusion_unet_intermediary_activation

    @denoising_diffusion_unet_intermediary_activation.setter
    def denoising_diffusion_unet_intermediary_activation(self, value: str) -> None:
        self._denoising_diffusion_unet_intermediary_activation = value

    @property
    def denoising_diffusion_unet_intermediary_activation_alpha(self) -> float:
        return self._denoising_diffusion_unet_intermediary_activation_alpha

    @denoising_diffusion_unet_intermediary_activation_alpha.setter
    def denoising_diffusion_unet_intermediary_activation_alpha(self, value: float) -> None:
        self._denoising_diffusion_unet_intermediary_activation_alpha = value

    @property
    def denoising_diffusion_unet_epochs(self) -> int:
        return self._denoising_diffusion_unet_epochs

    @denoising_diffusion_unet_epochs.setter
    def denoising_diffusion_unet_epochs(self, value: int) -> None:
        self._denoising_diffusion_unet_epochs = value

    @property
    def denoising_diffusion_gaussian_beta_start(self) -> float:
        return self._denoising_diffusion_gaussian_beta_start

    @denoising_diffusion_gaussian_beta_start.setter
    def denoising_diffusion_gaussian_beta_start(self, value: float) -> None:
        self._denoising_diffusion_gaussian_beta_start = value

    @property
    def denoising_diffusion_gaussian_beta_end(self) -> float:
        return self._denoising_diffusion_gaussian_beta_end

    @denoising_diffusion_gaussian_beta_end.setter
    def denoising_diffusion_gaussian_beta_end(self, value: float) -> None:
        self._denoising_diffusion_gaussian_beta_end = value

    @property
    def denoising_diffusion_gaussian_time_steps(self) -> int:
        return self._denoising_diffusion_gaussian_time_steps

    @denoising_diffusion_gaussian_time_steps.setter
    def denoising_diffusion_gaussian_time_steps(self, value: int) -> None:
        self._denoising_diffusion_gaussian_time_steps = value

    @property
    def denoising_diffusion_gaussian_clip_min(self) -> float:
        return self._denoising_diffusion_gaussian_clip_min

    @denoising_diffusion_gaussian_clip_min.setter
    def denoising_diffusion_gaussian_clip_min(self, value: float) -> None:
        self._denoising_diffusion_gaussian_clip_min = value

    @property
    def denoising_diffusion_gaussian_clip_max(self) -> float:
        return self._denoising_diffusion_gaussian_clip_max

    @denoising_diffusion_gaussian_clip_max.setter
    def denoising_diffusion_gaussian_clip_max(self, value: float) -> None:
        self._denoising_diffusion_gaussian_clip_max = value

    @property
    def denoising_diffusion_margin(self) -> float:
        return self._denoising_diffusion_margin

    @denoising_diffusion_margin.setter
    def denoising_diffusion_margin(self, value: float) -> None:
        self._denoising_diffusion_margin = value

    @property
    def denoising_diffusion_ema(self) -> float:
        return self._denoising_diffusion_ema

    @denoising_diffusion_ema.setter
    def denoising_diffusion_ema(self, value: float) -> None:
        self._denoising_diffusion_ema = value

    @property
    def denoising_diffusion_time_steps(self) -> int:
        return self._denoising_diffusion_time_steps

    @denoising_diffusion_time_steps.setter
    def denoising_diffusion_time_steps(self, value: int) -> None:
        self._denoising_diffusion_time_steps = value

    def __repr__(self) -> str:
        return (
            f"DenoisingDiffusion(framework='{self._framework}', "
            f"epochs={self._denoising_diffusion_unet_epochs}, "
            f"batch_size={self._denoising_diffusion_unet_batch_size})"
        )