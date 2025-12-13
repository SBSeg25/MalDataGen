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

    import json
    import numpy

    import tensorflow
    from abc import ABC

    from tensorflow.keras import Model

    from tensorflow.keras.metrics import Mean

    from tensorflow.keras.utils import to_categorical

    from tensorflow.keras.losses import BinaryCrossentropy

except ImportError as error:
    print(error)
    sys.exit(-1)


class QuantizedVAEAlgorithmTensorflow(Model):
    """
    Implements a Vector Quantized Variational autoencoder (VQ-VAE) for discrete latent
    representation learning and generation. This model combines an encoder, decoder,
    and vector quantization layer to learn compressed representations of input data.

    The algorithm supports training with reconstruction loss and commitment loss for
    the quantization layer, enabling stable training of discrete latent variables.

    Attributes:
        @train_variance (float):
            Variance of the training data used to normalize the reconstruction loss.
        @latent_dimension (int):
            Dimensionality of the latent space before quantization.
        @number_embeddings (int):
            Number of embeddings in the vector quantization codebook.
        @encoder (Model):
            Encoder network that maps input data to latent representations.
        @decoder (Model):
            Decoder network that reconstructs data from quantized latent codes.
        @quantized_vae_model (Model):
            Complete VQ-VAE model combining encoder, quantization, and decoder.
        @file_name_encoder (str):
            Filename for saving the encoder weights.
        @file_name_decoder (str):
            Filename for saving the decoder weights.
        @models_saved_path (str):
            Directory path for saving model weights.
        @total_loss_tracker (Mean):
            Metric tracker for total training loss (reconstruction + VQ losses).
        @reconstruction_loss_tracker (Mean):
            Metric tracker for reconstruction loss component.
        @vq_loss_tracker (Mean):
            Metric tracker for vector quantization loss components.

    References:
        - van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). "Neural Discrete
        Representation Learning." Advances in Neural Information Processing Systems (NeurIPS).
        Available at: https://arxiv.org/abs/1711.00937

    Example:
        >>> vae = QuantizedVAEAlgorithmTensorflow(
        ...     encoder_model=encoder,
        ...     decoder_model=decoder,
        ...     quantized_vae_model=vq_vae,
        ...     train_variance=0.1,
        ...     latent_dimension=64,
        ...     number_embeddings=512,
        ...     file_name_encoder="encoder_weights.h5",
        ...     file_name_decoder="decoder_weights.h5",
        ...     models_saved_path="./saved_models/"
        ... )
        >>> vae.compile(optimizer=Tensorflow.keras.optimizers.Adam())
        >>> vae.fit(train_dataset, epochs=10)
        >>> generated_samples = vae.get_samples({
        ...     "classes": {0: 5, 1: 5},
        ...     "number_classes": 2
        ... })
    """

    def __init__(self,
                 encoder_model,
                 decoder_model,
                 quantized_vae_model,
                 train_variance,
                 latent_dimension,
                 number_embeddings,
                 file_name_encoder,
                 file_name_decoder,
                 models_saved_path,
                 **kwargs):

        """
        Initializes the QuantizedVAEAlgorithm with encoder, decoder, and VQ-VAE components.

        Args:
            @encoder_model (Model):
                Encoder network that compresses input data to latent space.
            @decoder_model (Model):
                Decoder network that reconstructs data from quantized latent codes.
            @quantized_vae_model (Model):
                Complete VQ-VAE model including quantization layer.
            @train_variance (float):
                Data variance used to scale reconstruction loss.
            @latent_dimension (int):
                Dimensionality of latent space before quantization.
            @number_embeddings (int):
                Size of quantization codebook (number of discrete latent codes).
            @file_name_encoder (str):
                Filename for saving encoder weights.
            @file_name_decoder (str):
                Filename for saving decoder weights.
            @models_saved_path (str):
                Directory path for model weight storage.
            @**kwargs:
                Additional keyword arguments passed to parent class.

        Raises:
            ValueError:
                If latent_dimension <= 0.
                If number_embeddings <= 0.
                If train_variance <= 0.
        """

        super().__init__(**kwargs)

        self._train_variance = train_variance
        self._latent_dimension = latent_dimension
        self._number_embeddings = number_embeddings

        self._encoder = encoder_model
        self._decoder = decoder_model
        self._quantized_vae_model = quantized_vae_model

        self._file_name_encoder = file_name_encoder
        self._file_name_decoder = file_name_decoder
        self._models_saved_path = models_saved_path
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.vq_loss_tracker = Mean(name="vq_loss")

    @property
    def metrics(self):
        """
        Returns the list of metrics tracked during training.

        Returns:
            list: List of metric trackers [total_loss, reconstruction_loss, vq_loss].
        """
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.vq_loss_tracker]


    @tensorflow.function
    def train_step(self, data):
        """
        Performs a single training step on a batch of data.

        The training step includes:
        1. Forward pass through the VQ-VAE
        2. Loss computation (reconstruction + VQ losses)
        3. Gradient computation and weight updates

        Args:
            data (tuple): Input data tuple containing (input_tensor, labels).

        Returns:
            dict: Dictionary with loss metrics for the current step.
        """
        x, y = data

        output_tensor, _ = x

        with tensorflow.GradientTape() as tape:

            # Forward pass through VQ-VAE
            reconstructions = self._quantized_vae_model(x)

            # Compute reconstruction loss (MSE normalized by data variance)
            reconstruction_loss = (tensorflow.reduce_mean((output_tensor - reconstructions) ** 2) / self._train_variance)

            # Total loss is reconstruction loss plus VQ losses (commitment + codebook)
            vae_model_loss = reconstruction_loss + sum(self._quantized_vae_model.losses)

        # Compute and apply gradients
        gradient_flow = tape.gradient(vae_model_loss, self._quantized_vae_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient_flow, self._quantized_vae_model.trainable_variables))

        # Update metric trackers
        self.total_loss_tracker.update_state(vae_model_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self._quantized_vae_model.losses))

        # Log results.
        return {"loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "vqvae_loss": self.vq_loss_tracker.result(),}

    def get_samples(self, number_samples_per_class):
        """
        Generates samples from the latent space using the decoder.

        Samples are generated by:
        1. Randomly selecting indices from the codebook
        2. Gathering corresponding latent vectors
        3. Decoding these vectors conditioned on class labels

        Args:
            number_samples_per_class (dict):
                Dictionary specifying samples to generate per class with structure:
                {
                    "classes": {class_label: num_samples, ...},
                    "number_classes": total_num_classes
                }

        Returns:
            dict: Generated samples keyed by class label.
        """

        generated_data = {}

        # Get codebook embeddings from the quantization layer
        codebook = self._quantized_vae_model.get_layer("vector_quantizer").embeddings

        number_embeddings = self._number_embeddings

        for label_class, number_instances in number_samples_per_class["classes"].items():

            # Create one-hot encoded labels for the samples
            label_samples_generated = to_categorical([label_class] * number_instances,
                                                     num_classes=number_samples_per_class["number_classes"])

            sampled_indices = numpy.random.choice(number_embeddings, size=number_instances)

            # Get corresponding latent vectors
            quantized_vectors = tensorflow.gather(codebook, sampled_indices)  # (number_instances, latent_dim)
            quantized_vectors = tensorflow.convert_to_tensor(quantized_vectors)

            # Decode the latent vectors
            generated_samples = self._decoder.predict([quantized_vectors, label_samples_generated], verbose=0)

            # Round to nearest integer (for discrete data like images)
            generated_samples = numpy.rint(generated_samples)

            generated_data[label_class] = generated_samples

        return generated_data


    def save_model(self, directory, file_name):
        """
        Save the encoder and decoder models in both JSON and H5 formats.

        Args:
            directory (str): Directory where models will be saved.
            file_name (str): Base file name for saving models.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Construct file names for encoder and decoder models
        encoder_file_name = os.path.join(directory, f"fold_{file_name}_encoder")
        decoder_file_name = os.path.join(directory, f"fold_{file_name}_decoder")

        # Save encoder model
        self._save_model_to_json(self._encoder, f"{encoder_file_name}.json")
        self._encoder.save_weights(f"{encoder_file_name}.weights.h5")

        # Save decoder model
        self._save_model_to_json(self._decoder, f"{decoder_file_name}.json")
        self._decoder.save_weights(f"{decoder_file_name}.weights.h5")


    @staticmethod
    def _save_model_to_json(model, file_path):
        """
        Save model architecture to a JSON file.

        Args:
            model (Model): Model to save.
            file_path (str): Path to the JSON file.
        """
        with open(file_path, "w") as json_file:
            json.dump(model.to_json(), json_file)


    def load_models(self, directory, file_name):
        """
        Load the encoder and decoder models from a directory.

        Args:
            directory (str): Directory where models are stored.
            file_name (str): Base file name for loading models.
        """

        # Construct file names for encoder and decoder models
        encoder_file_name = "{}_encoder".format(file_name)
        decoder_file_name = "{}_decoder".format(file_name)

        # Load the encoder and decoder models from the specified directory
        self._encoder = self._save_neural_network_model(encoder_file_name, directory)
        self._decoder = self._save_neural_network_model(decoder_file_name, directory)
