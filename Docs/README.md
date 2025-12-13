# MalDataGen - Usage Examples

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Example Workflows](#example-workflows)
   - [Denoising Probabilistic Diffusion](#denoising-probabilistic-diffusion)
   - [Conditional GAN](#conditional-gan)
   - [Wasserstein GAN-GP](#wasserstein-gan-gp)
   - [Conditional Autoencoder](#conditional-autoencoder)
   - [Variational Autoencoder](#variational-autoencoder)
3. [Common Parameters](#common-parameters)

## Architecture Overview

The Synthetic Ocean AI library provides several generative architectures:

| Architecture               | Key Characteristics                          | Typical Use Cases                  |
|----------------------------|---------------------------------------------|------------------------------------|
| Denoising Probabilistic Diffusion | Iterative denoising process, high-quality outputs | High-fidelity data generation |
| Conditional GAN (CGAN)     | Label-guided generation                     | Conditional data augmentation  |
| Wasserstein GAN-GP         | Stable training with gradient penalty       | Robust generation tasks        |
| Conditional Autoencoder    | Deterministic reconstruction               | Data compression, denoising    |
| Variational Autoencoder    | Probabilistic latent space                 | Diverse sample generation      |

## Example Workflows

## Probabilistic Diffusion parameters
    
    Arguments(Variational Autoencoder):

        --diffusion_unet_last_layer_activation              Activation for the last layer of U-Net.
        --diffusion_latent_dimension                        Dimension of the latent space.
        --diffusion_unet_num_embedding_channels             Number of embedding channels for U-Net.
        --diffusion_unet_channels_per_level                 List of channels per level in U-Net.
        --diffusion_unet_batch_size                         Batch size for U-Net training.
        --diffusion_unet_attention_mode                     Attention mode for U-Net.
        --diffusion_unet_num_residual_blocks                Number of residual blocks in U-Net.
        --diffusion_unet_group_normalization                Group normalization value for U-Net.
        --diffusion_unet_intermediary_activation            Intermediary activation for U-Net.
        --diffusion_unet_intermediary_activation_alpha      Alpha value for intermediary activation function in U-Net.
        --diffusion_unet_epochs                             Number of epochs for U-Net training.
        --diffusion_gaussian_beta_start                     Starting value of beta for Gaussian diffusion.
        --diffusion_gaussian_beta_end                       Ending value of beta for Gaussian diffusion.
        --diffusion_gaussian_time_steps                     Number of time steps for Gaussian diffusion.
        --diffusion_gaussian_clip_min                       Minimum clipping value for Gaussian noise.
        --diffusion_gaussian_clip_max                       Maximum clipping value for Gaussian noise.
        --diffusion_autoencoder_loss                        Loss function for Autoencoder.
        --diffusion_autoencoder_encoder_filters             List of filters for Autoencoder encoder.
        --diffusion_autoencoder_decoder_filters             List of filters for Autoencoder decoder.
        --diffusion_autoencoder_last_layer_activation       Activation function for the last layer of Autoencoder.
        --diffusion_autoencoder_latent_dimension            Dimension of the latent in Autoencoder.
        --diffusion_autoencoder_batch_size_create_embedding Batch size for creating embeddings in Autoencoder.
        --diffusion_autoencoder_batch_size_training         Batch size for training Autoencoder.
        --diffusion_autoencoder_epochs                      Number of epochs for Autoencoder training.
        --diffusion_autoencoder_intermediary_activation_function Intermediary activation function for Autoencoder.
        --diffusion_autoencoder_intermediary_activation_alpha Alpha value for intermediary activation function in Autoencoder.
        --diffusion_autoencoder_activation_output_encoder   Activation function for the output of the encoder in Autoencoder.
        --diffusion_margin                                  Margin for diffusion process.
        --diffusion_ema                                     Exponential moving average for diffusion.
        --diffusion_time_steps                              Number of time steps for diffusion.
        --diffusion_autoencoder_initializer_mean 
        --diffusion_autoencoder_initializer_deviation
        --diffusion_autoencoder_dropout_decay_rate_encoder
        --diffusion_autoencoder_dropout_decay_rate_decoder
        --diffusion_autoencoder_file_name_encoder 
        --diffusion_autoencoder_file_name_decoder
        --diffusion_autoencoder_path_output_models
        --diffusion_autoencoder_mean_distribution
        --diffusion_autoencoder_stander_deviation

## Probabilistic Diffusion Library Mode:

```python3
import numpy
import tensorflow

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from SynDataGen.Engine.Models.Diffusion.DiffusionModelUnet import UNetModel

from SynDataGen.Engine.Algorithms.Diffusion.AlgorithmDiffusion import DiffusionModel
from SynDataGen.Engine.Algorithms.Diffusion.GaussianDiffusion import GaussianDiffusion

from SynDataGen.Engine.Models.Diffusion.VariationalAutoencoderModel import VariationalModelDiffusion

from SynDataGen.Engine.Algorithms.Diffusion.AlgorithmVariationalAutoencoderDiffusion import

VariationalAlgorithmDiffusion

number_samples_per_class = {
    "classes": {1: 100, 2: 200, 3: 150},
    "number_classes": 3
}
input_shape = (1200,)

# Initialize the first instance of UNet for the diffusion model
first_instance_unet = UNetModel(
    embedding_dimension=128,
    embedding_channels=1,
    list_neurons_per_level=[1, 2, 4],
    list_attentions=[False, True, True],
    number_residual_blocks=2,
    normalization_groups=1,
    intermediary_activation_function='swish',
    intermediary_activation_alpha=0.05,
    last_layer_activation='linear',
    number_samples_per_class=number_samples_per_class
)

# Initialize the second instance of UNet with the same configuration
second_instance_unet = UNetModel(
    embedding_dimension=128,
    embedding_channels=1,
    list_neurons_per_level=[1, 2, 4],
    list_attentions=[False, False, True, True],
    number_residual_blocks=2,
    normalization_groups=1,
    intermediary_activation_function='swish',
    intermediary_activation_alpha=0.05,
    last_layer_activation='linear',
    number_samples_per_class=number_samples_per_class
)

# Build the models for both UNet instances
first_unet_model = first_instance_unet.build_model()
second_unet_model = second_instance_unet.build_model()

# Synchronize the weights of the second UNet model with the first one
second_unet_model.set_weights(first_unet_model.get_weights())

# Initialize the GaussianDiffusion utility for the diffusion process
gaussian_diffusion_util = GaussianDiffusion(
    beta_start=1e-4,
    beta_end=0.02,
    time_steps=1000,
    clip_min=-1.0,
    clip_max=1.0
)

# Initialize the VariationalModelDiffusion for embedding learning and reconstructor
variation_model_diffusion = VariationalModelDiffusion(
    latent_dimension=128,
    output_shape=input_shape,
    activation_function='swish',
    initializer_mean=0.0,
    initializer_deviation=0.02,
    dropout_decay_encoder=0.2,
    dropout_decay_decoder=0.4,
    last_layer_activation='sigmoid',
    number_neurons_encoder=[128, 64],
    number_neurons_decoder=[64, 128],
    dataset_type=numpy.float32,
    number_samples_per_class=number_samples_per_class
)

# Initialize the VariationalAlgorithmDiffusion for the training and diffusion process
variational_algorithm_diffusion = VariationalAlgorithmDiffusion(
    encoder_model=variation_model_diffusion.get_encoder(),
    decoder_model=variation_model_diffusion.get_decoder(),
    loss_function='mse',
    latent_dimension=128,
    decoder_latent_dimension=128,
    latent_mean_distribution=0.0,
    latent_stander_deviation=0.5,
    file_name_encoder="encoder_model",
    file_name_decoder="decoder_model",
    models_saved_path="models_saved/"
)
encoder_diffusion = variation_model_diffusion.get_encoder()
decoder_diffusion = variation_model_diffusion.get_decoder()

diffusion_algorithm = DiffusionModel(first_unet_model=first_unet_model,
                                     second_unet_model=second_unet_model,
                                     encoder_model_image=encoder_diffusion,
                                     decoder_model_image=decoder_diffusion,
                                     gdf_util=gaussian_diffusion_util,
                                     optimizer_autoencoder=Adam(learning_rate=0.0002),
                                     optimizer_diffusion=Adam(learning_rate=0.0002),
                                     time_steps=1000,
                                     ema=0.9999,
                                     margin=0.001,
                                     embedding_dimension=128)

diffusion_algorithm.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.002))

# Prepare the data embedding and train the diffusion model
data_embedding = variational_algorithm_diffusion.create_embedding(
    [x_real_samples, to_categorical(y_real_samples, num_classes=number_samples_per_class["number_classes"])])

data_embedding = numpy.array(data_embedding)
data_embedding = tensorflow.expand_dims(data_embedding, axis=-1)

diffusion_algorithm.fit(
    data_embedding, to_categorical(y_real_samples, num_classes=number_samples_per_class["number_classes"]),
    epochs=1000, batch_size=32, verbose=2)

Samples = variational_algorithm_diffusion.get_samples(number_samples_per_class)

```

## 🧪 Generating data using a Conditional GAN (CGAN) model
## Conditional GAN parameters
    
    Arguments(Conditional GAN):

        --adversarial_number_epochs                         Number of epochs (training iterations).
        --adversarial_batch_size                            Number of epochs (training iterations).
        --adversarial_initializer_mean                      Mean value of the Gaussian initializer distribution.
        --adversarial_initializer_deviation                 Standard deviation of the Gaussian initializer distribution.
        --adversarial_latent_dimension                      Latent space dimension for cGAN training.
        --adversarial_training_algorithm                    Training algorithm for cGAN.
        --adversarial_activation_function                   Activation function for the cGAN.
        --adversarial_dropout_decay_rate_g                  Dropout decay rate for the cGAN generator.
        --adversarial_dropout_decay_rate_d                  Dropout decay rate for the cGAN discriminator.
        --adversarial_dense_layer_sizes_g                   Sizes of dense layers in the generator.
        --adversarial_dense_layer_sizes_d                   Sizes of dense layers in the discriminator.
        --adversarial_latent_mean_distribution              Mean of the random noise input distribution.
        --adversarial_latent_stander_deviation              Standard deviation of the latent space distribution.
        --adversarial_loss_generator                        Loss function for the generator.
        --adversarial_loss_discriminator                    Loss function for the discriminator.
        --adversarial_smoothing_rate                        Label smoothing rate for the adversarial training.
        --adversarial_file_name_discriminator               File name to save the trained discriminator model.
        --adversarial_file_name_generator                   File name to save the trained generator model.
        --adversarial_path_output_models                    Path to save the trained models.
        --adversarial_last_layer_activation                 Last layer activation.

## Conditional GAN Library Mode:

```python3
import numpy
import tensorflow

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from SynDataGen.Engine.Models.Adversarial.AdversarialModel import AdversarialModel
from SynDataGen.Engine.Algorithms.Adversarial.AdversarialAlgorithm import AdversarialAlgorithm

number_samples_per_class = {
    "classes": {1: 100, 2: 200, 3: 150},
    "number_classes": 3
}
input_shape = (1200,)

# adversarial Model setup for Generator and Discriminator
adversarial_model = AdversarialModel(latent_dimension=128,
                                     output_shape=input_shape,
                                     activation_function="LeakyReLU",
                                     initializer_mean=0.0,
                                     initializer_deviation=0.5,
                                     dropout_decay_rate_g=0.2,
                                     dropout_decay_rate_d=0.4,
                                     last_layer_activation="Sigmoid",
                                     dense_layer_sizes_g=[128],
                                     dense_layer_sizes_d=[128],
                                     dataset_type=numpy.float32,
                                     number_samples_per_class=number_samples_per_class)

# adversarial Algorithm setup for training and model operations
adversarial_algorithm = AdversarialAlgorithm(generator_model=adversarial_model.get_generator(),
                                             discriminator_model=adversarial_model.get_discriminator(),
                                             latent_dimension=128,
                                             loss_generator='binary_crossentropy',
                                             loss_discriminator='binary_crossentropy',
                                             file_name_discriminator="discriminator_model",
                                             file_name_generator="generator_model",
                                             models_saved_path="models_saved/",
                                             latent_mean_distribution=0.0,
                                             latent_stander_deviation=1.0,
                                             smoothing_rate=0.15)

# Print the model summaries for the generator and discriminator
adversarial_model.get_generator().summary()
adversarial_model.get_discriminator().summary()

# Set up optimizers for the generator and discriminator
generator_optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

# Compile the adversarial algorithm with binary cross-entropy loss
adversarial_algorithm.compile(
    generator_optimizer, discriminator_optimizer, BinaryCrossentropy(), BinaryCrossentropy())

# Fit the model with real samples and the corresponding labels
adversarial_algorithm.fit(
    x_real_samples, to_categorical(y_real_samples, num_classes=number_samples_per_class["number_classes"]),
    epochs=1000, batch_size=32)

Samples = adversarial_algorithm.get_samples(number_samples_per_class)

```

## 🧪 Generating data using a Conditional Wasserstein GAN with Gradient Penalty (WGAN-GP) model

## Wasserstein GAN-GP  parameters
    
    Arguments(Wasserstein GAN):

        --wasserstein_latent_dimension                       Latent space dimension for the Wasserstein GAN.
        --wasserstein_training_algorithm                     Training algorithm for the Wasserstein GAN.
        --wasserstein_activation_function                    Activation function for the Wasserstein GAN.
        --wasserstein_dropout_decay_rate_g                   Dropout decay rate for the generator.
        --wasserstein_dropout_decay_rate_d                   Dropout decay rate for the discriminator.
        --wasserstein_dense_layer_sizes_generator            Dense layer sizes for the generator.
        --wasserstein_dense_layer_sizes_discriminator        Dense layer sizes for the discriminator.
        --wasserstein_batch_size                             Batch size for the Wasserstein GAN.
        --wasserstein_number_epochs                          Number epochs for the Wasserstein GAN.
        --wasserstein_number_classes                         Number of classes for the Wasserstein GAN.
        --wasserstein_loss_function                          Loss function for the Wasserstein GAN.
        --wasserstein_momentum                               Momentum for the training algorithm.
        --wasserstein_last_activation_layer                  Activation function for the last layer.
        --wasserstein_initializer_mean                       Mean value of the Gaussian initializer distribution.
        --wasserstein_initializer_deviation                  Standard deviation of the Gaussian initializer distribution.
        --wasserstein_optimizer_generator_learning_rate      Learning rate for the generator optimizer.
        --wasserstein_optimizer_discriminator_learning_rate  Learning rate for the discriminator optimizer.
        --wasserstein_optimizer_generator_beta               Beta value for the generator optimizer.
        --wasserstein_optimizer_discriminator_beta           Beta value for the discriminator optimizer.
        --wasserstein_discriminator_steps                    Number of steps to update the discriminator per generator update.
        --wasserstein_smoothing_rate                         Smoothing rate for the Wasserstein GAN.
        --wasserstein_latent_mean_distribution               Mean of the random latent space distribution.
        --wasserstein_latent_stander_deviation               Standard deviation of the random latent space distribution.
        --wasserstein_gradient_penalty                       Gradient penalty value for the Wasserstein GAN.
        --wasserstein_file_name_discriminator                File name to save the discriminator model.
        --wasserstein_file_name_generator                    File name to save the generator model.
        --wasserstein_path_output_models                     Path to save the models.

## Wasserstein GAN-GP Library Mode:

```python3
import numpy
import tensorflow

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from SynDataGen.Engine.Models.Wasserstein.ModelWassersteinGAN import WassersteinModel
from SynDataGen.Engine.Algorithms.Wasserstein.AlgorithmWassersteinGan import WassersteinAlgorithm

number_samples_per_class = {
    "classes": {1: 100, 2: 200, 3: 150},
    "number_classes": 3
}
input_shape = (1200,)

# WassersteinGP Model setup for training and model operations
wasserstein_model = WassersteinModel(
    latent_dimension=128,
    output_shape=input_shape,
    activation_function="LeakyReLU",
    initializer_mean=0.0,
    initializer_deviation=0.02,
    dropout_decay_rate_g=0.2,
    dropout_decay_rate_d=0.4,
    last_layer_activation="sigmoid",
    dense_layer_sizes_g=[128],
    dense_layer_sizes_d=[128],
    dataset_type=numpy.float32,
    number_samples_per_class=number_samples_per_class
)

# WassersteinGP Algorithm setup for training and model operations
wasserstein_algorithm = WassersteinAlgorithm(
    generator_model=wasserstein_model.get_generator(),
    discriminator_model=wasserstein_model.get_discriminator(),
    latent_dimension=128,
    generator_loss_fn="binary_crossentropy",
    discriminator_loss_fn="binary_crossentropy",
    file_name_discriminator="discriminator_model",
    file_name_generator="generator_model",
    models_saved_path="models_saved/",
    latent_mean_distribution=0.0,
    latent_stander_deviation=1.0,
    smoothing_rate=0.15,
    gradient_penalty_weight=10.0,
    discriminator_steps=3
)

# Print the model summaries for the generator and discriminator
wasserstein_model.get_generator().summary()
wasserstein_model.get_discriminator().summary()

# Set up optimizers for the generator and discriminator
generator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)


# Define the custom loss functions for the discriminator and generator
def discriminator_loss(real_img, fake_img):
    return tensorflow.reduce_mean(fake_img) - tensorflow.reduce_mean(real_img)


def generator_loss(fake_img):
    return -tensorflow.reduce_mean(fake_img)


# Compile the WassersteinGP GAN algorithm
wasserstein_algorithm.compile(generator_optimizer,
                              discriminator_optimizer,
                              generator_loss,
                              discriminator_loss)

# Fit the WassersteinGP GAN model
wasserstein_algorithm.fit(x_real_samples,
                          to_categorical(y_real_samples, num_classes=number_samples_per_class["number_classes"]),
                          epochs=1000, batch_size=32)

Samples = wasserstein_algorithm.get_samples(number_samples_per_class)

```

## 🧪 Generating data using a Autoencoder Conditional (CAE) model

## Conditional Autoencoder parameters
    
    Arguments(Conditional Autoencoder):

        --autoencoder_latent_dimension                      Latent space dimension for the Autoencoder.
        --autoencoder_training_algorithm                    Training algorithm for the Autoencoder.
        --autoencoder_activation_function                   Activation function for the Autoencoder.
        --autoencoder_dropout_decay_rate_encoder            Dropout decay rate for the encoder.
        --autoencoder_dropout_decay_rate_decoder            Dropout decay rate for the decoder.
        --autoencoder_dense_layer_sizes_encoder             Dense layer sizes for the encoder.
        --autoencoder_dense_layer_sizes_decoder             Dense layer sizes for the decoder.
        --autoencoder_batch_size                            Batch size for the Autoencoder.
        --autoencoder_number_classes                        Number of classes for the Autoencoder.
        --autoencoder_number_epochs                         Number of classes for the Autoencoder.
        --autoencoder_loss_function                         Loss function for the Autoencoder.
        --autoencoder_momentum                              Momentum for the training algorithm.
        --autoencoder_last_activation_layer                 Activation function for the last layer.
        --autoencoder_initializer_mean                      Mean value of the Gaussian initializer distribution.
        --autoencoder_initializer_deviation                 Standard deviation of the Gaussian initializer distribution.
        --autoencoder_latent_mean_distribution              Mean of the random noise input distribution.
        --autoencoder_latent_stander_deviation              Standard deviation of the random noise input.
        --autoencoder_file_name_encoder                     File name to save the encoder model.
        --autoencoder_file_name_decoder                     File name to save the decoder model.
        --autoencoder_path_output_models                    Path to save the models.

## Conditional Autoencoder (CAE) Library Mode:

```python3
import numpy

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from SynDataGen.Engine.Models.Autoencoder.ModelAutoencoder import AutoencoderModel
from SynDataGen.Engine.Algorithms.Autoencoder.AutoencoderAlgorithm import AutoencoderAlgorithm

number_samples_per_class = {
    "classes": {1: 100, 2: 200, 3: 150},
    "number_classes": 3
}
input_shape = (1200,)

# autoencoder Model setup for Encoder and Decoder
autoencoder_model = AutoencoderModel(latent_dimension=64,
                                     output_shape=input_shape,
                                     activation_function="LeakyReLU",
                                     initializer_mean=0.0,
                                     initializer_deviation=0.50,
                                     dropout_decay_encoder=0.2,
                                     dropout_decay_decoder=0.2,
                                     last_layer_activation="sigmoid",
                                     number_neurons_encoder=[256, 128],
                                     number_neurons_decoder=[128, 256],
                                     dataset_type=numpy.float32,
                                     number_samples_per_class=2)

# autoencoder Algorithm setup for training and model operations
autoencoder_algorithm = AutoencoderAlgorithm(encoder_model=autoencoder_model.get_encoder(input_shape),
                                             decoder_model=autoencoder_model.get_decoder(input_shape),
                                             loss_function="binary_crossentropy",
                                             file_name_encoder="encoder_model",
                                             file_name_decoder="decoder_model",
                                             models_saved_path="models_saved/",
                                             latent_mean_distribution=0.5,
                                             latent_stander_deviation=0.5,
                                             latent_dimension=64)

# Print the model summaries for the encoder and decoder
autoencoder_model.get_encoder(input_shape).summary()
autoencoder_model.get_decoder(input_shape).summary()

# Compile the autoencoder algorithm with the specified loss function
autoencoder_algorithm.compile(loss='mse')

# Fit the autoencoder model
autoencoder_algorithm.fit((x_real_samples,
                           to_categorical(y_real_samples,
                                          num_classes=number_samples_per_class["number_classes"])),
                          x_real_samples, epochs=1000, batch_size=32)

Samples = autoencoder_algorithm.get_samples(number_samples_per_class)

```

## 🧪 Generating data using Variational Autoencoder Conditional (VAE) model

## Variational Autoencoder parameters
    
    Arguments(Variational Autoencoder):

        --variational_autoencoder_latent_dimension          Latent space dimension for the Variational Autoencoder
        --variational_autoencoder_training_algorithm        Training algorithm for the Variational Autoencoder.
        --variational_autoencoder_activation_function       Intermediate activation function of the Variational Autoencoder.
        --variational_autoencoder_dropout_decay_rate_encoder Dropout decay rate for the encoder of the Variational Autoencoder
        --variational_autoencoder_dropout_decay_rate_decoder Dropout decay rate for the discriminator of the Variational Autoencoder
        --variational_autoencoder_dense_layer_sizes_encoder Sizes of dense layers in the encoder
        --variational_autoencoder_dense_layer_sizes_decoder Sizes of dense layers in the decoder
        --variational_autoencoder_number_epochs             Number of classes for the Autoencoder.
        --variational_autoencoder_batch_size                Batch size for the Variational Autoencoder.
        --variational_autoencoder_number_classes            Number of classes for the Variational Autoencoder.
        --variational_autoencoder_loss_function             Loss function for the Variational Autoencoder.
        --variational_autoencoder_momentum                  Momentum for the training algorithm.
        --variational_autoencoder_last_activation_layer     Activation function of the last layer.
        --variational_autoencoder_initializer_mean          Mean value of the Gaussian initializer distribution.
        --variational_autoencoder_initializer_deviation     Standard deviation of the Gaussian initializer distribution.
        --variational_autoencoder_mean_distribution         Mean of the random noise distribution input
        --variational_autoencoder_stander_deviation         Standard deviation of the random noise input
        --variational_autoencoder_file_name_encoder         File name to save the encoder model.
        --variational_autoencoder_file_name_decoder         File name to save the decoder model.
        --variational_autoencoder_path_output_models        Path to save the models.

## Variational Autoencoder (VAE) Library Mode:

```python3
import numpy

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from SynDataGen.Engine.Models.VariationalAutoencoder.VariationalAutoencoderModel import VariationalModel
from SynDataGen.Engine.Algorithms.VariationalAutoencoder.AlgorithmVariationalAutoencoder import VariationalAlgorithm

number_samples_per_class = {
    "classes": {1: 100, 2: 200, 3: 150},
    "number_classes": 3
}
input_shape = (1200,)

# Variational Model setup for the VAE's encoder and decoder
variation_model = VariationalModel(latent_dimension=128,
                                   output_shape=input_shape,
                                   activation_function="LeakyReLU",
                                   initializer_mean=0.0,
                                   initializer_deviation=0.02,
                                   dropout_decay_encoder=0.2,
                                   dropout_decay_decoder=0.4,
                                   last_layer_activation="sigmoid",
                                   number_neurons_encoder=[128],
                                   number_neurons_decoder=[128],
                                   dataset_type=numpy.float32,
                                   number_samples_per_class=2)

# Variational Algorithm setup for training and model operations
variational_algorithm = VariationalAlgorithm(encoder_model=variation_model.get_encoder(),
                                             decoder_model=variation_model.get_decoder(),
                                             loss_function="binary_crossentropy",
                                             latent_dimension=64,
                                             decoder_latent_dimension=128,
                                             latent_mean_distribution=0.0,
                                             latent_stander_deviation=0.5,
                                             file_name_encoder="encoder_model",
                                             file_name_decoder="decoder_model",
                                             models_saved_path="models_saved/")

# Print the model summaries for the encoder and decoder
variation_model.get_encoder().summary()
variation_model.get_decoder().summary()

# Compile the variational autoencoder algorithm with the specified loss function
variational_algorithm.compile()

# Fit the variational autoencoder model
variational_algorithm.fit((x_real_samples,
                           to_categorical(y_real_samples,
                                          num_classes=number_samples_per_class["number_classes"])),
                          epochs=1000, batch_size=32, )

Samples = variational_algorithm.get_samples(number_samples_per_class)

```

## Data Load parameters

### Load CSV
    
    Arguments(Load CSV File)
        
        -i, --data_load_path_file_input                     Path to the input CSV file.
        --data_load_label_column                            Index of the column to be used as the label.
        --data_load_max_samples                             Maximum number of samples to be loaded.
        --data_load_max_columns                             Maximum number of columns to be considered.
        --data_load_start_column                            Index of the first column to be loaded.
        --data_load_end_column                              Index of the last column to be loaded.
        --data_load_path_file_output                        Path to the output CSV file.
        --data_load_exclude_columns                         Columns to exclude from processing.

## Classifier parameters

### Support Vector Machine
    Arguments(Support Vector Machine):
        
        --support_vector_machine_regularization             Regularization parameter for SVM.
        --support_vector_machine_kernel                     Kernel type for SVM.
        --support_vector_machine_kernel_degree              Degree for polynomial kernel function (SVM).
        --support_vector_machine_gamma                      Kernel coefficient for SVM.

### Stochastic Gradient Descent
    Arguments(Stochastic Gradient Descent):

        --stochastic_gradient_descent_loss                  Loss function to be used by the SGD algorithm.
        --stochastic_gradient_descent_penalty               Penalty term for regularization.
        --stochastic_gradient_descent_alpha                 Constant that multiplies the regularization term.
        --stochastic_gradient_descent_max_iterations        Maximum number of iterations for the SGD algorithm.
        --stochastic_gradient_descent_tolerance             Tolerance for stopping criteria.

### Random Forest
    Arguments(Random Forest):

        --random_forest_number_estimators                   Number of trees in the Random Forest.
        --random_forest_max_depth                           Maximum depth of the tree.
        --random_forest_max_leaf_nodes                      Maximum number of leaf nodes in Random Forest.

### Quadratic Discriminant Analysis   
    Arguments(Quadratic Discriminant Analysis):

        --quadratic_discriminant_analysis_priors            Prior probabilities of the classes in QDA.
        --quadratic_discriminant_analysis_regularization    Regularization parameter in QDA.
        --quadratic_discriminant_analysis_threshold         Threshold value for QDA.

### Multilayer Perceptron 
    Arguments(Multilayer Perceptron):
    
        --perceptron_training_algorithm                     Training algorithm for Perceptron.
        --perceptron_training_loss                          Loss function for Perceptron.
        --perceptron_layers_settings                        Layer settings for Perceptron.
        --perceptron_dropout_decay_rate                     Dropout decay rate in Perceptron.
        --perceptron_training_metric                        Evaluation metrics for Perceptron.
        --perceptron_layer_activation                       Activation function for layers in Perceptron.
        --perceptron_last_layer_activation                  Activation function for last layer in Perceptron.
        --perceptron_number_epochs                          Number of epochs for Perceptron.

### Spectral Cluster
    Arguments(Spectral Cluster):

        --spectral_number_clusters                          Number of clusters to form for spectral clustering.
        --spectral_eigen_solver                             The eigenvalue decomposition method to use.
        --spectral_affinity                                 How to construct the affinity matrix.
        --spectral_assign_labels                            The strategy to use for assigning labels in the embedding space.
        --spectral_random_state                             Seed for the random number generator.

### Linear Regression
    Arguments(Linear Regression):

        --linear_regression_fit_intercept                   Whether to calculate the intercept for the model.
        --linear_regression_normalize                       This parameter is ignored when `fit_intercept=False`.
        --linear_regression_copy_X                          If True, X will be copied; else, it may be overwritten.
        --linear_regression_number_jobs                     The number of jobs to use for the computation.

### Naive Bayes
    Arguments(Naive Bayes):

        --naive_bayes_priors                                Prior probabilities of the classes.
        --naive_bayes_variation_smoothing                   Portion of the largest variance to be added to variances for stability.
    
### K-Neighbors
    Arguments(K-Neighbors):

        --knn_number_neighbors                              Number of neighbors to use by default for KNN.
        --knn_weights                                       Weight function used in prediction for KNN.
        --knn_algorithm                                     Algorithm used to compute nearest neighbors.
        --knn_leaf_size                                     Leaf size passed to BallTree or KDTree in KNN.
        --knn_metric                                        The distance metric to use for KNN.

### K-Means
    Arguments(K-Means):

        --k_means_number_clusters                           Number of clusters to form.
        --k_means_init                                      Method for initialization of centroids.
        --k_means_max_iterations                            Maximum number of iterations for the K-Means algorithm.
        --k_means_tolerance                                 Convergence tolerance for the K-Means algorithm.
        --k_means_random_state                              Seed for the random number generator.

### Gradient Boosting
    Arguments(Gradient Boosting):
        
        --gradient_boosting_loss                            Loss function to be used by the gradient boosting model.
        --gradient_boosting_learning_rate                   Learning rate for the gradient boosting model.
        --gradient_boosting_number_estimators               Number of estimators (trees) in the gradient boosting model.
        --gradient_boosting_subsample                       Fraction of samples used for fitting the individual base learners.
        --gradient_boosting_criterion                       Criterion to measure the quality of a split.

### Gaussian Process
    Arguments(Gaussian Process):

        --gaussian_process_kernel                           Kernel to use in Gaussian Process.
        --gaussian_process_max_iterations                   Maximum number of iterations for optimizer.
        --gaussian_process_optimizer                        Optimizer to use in Gaussian Process.

### Decision Tree
    Arguments(Decision Tree):

        --decision_tree_criterion                           Function to measure the quality of a split.
        --decision_tree_max_depth                           Maximum depth of the decision tree.
        --decision_tree_max_features                        Number of features to consider when looking for the best split.
        --decision_tree_max_leaf_nodes                      Grow the tree with max leaf nodes.
