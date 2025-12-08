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

try:
    import os
    import sys
    import json
    import numpy
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

except ImportError as error:
    print(error)
    sys.exit(-1)


class VariationalAutoencoderAlgorithmTorch(nn.Module):
    """
    Implements a Variational AutoEncoder (VAE) model for generating synthetic data.
    """

    def __init__(self,
                 encoder_model,
                 decoder_model,
                 loss_function,
                 latent_dimension,
                 decoder_latent_dimension,
                 latent_mean_distribution,
                 latent_stander_deviation,
                 file_name_encoder,
                 file_name_decoder,
                 models_saved_path):
        """
        Initializes the VariationalAlgorithm model.
        """
        # Call parent __init__ first
        super(VariationalAutoencoderAlgorithmTorch, self).__init__()

        # Direct assignment to register as submodules
        self._encoder = encoder_model
        self._decoder = decoder_model

        # Loss function and metrics for tracking losses
        self._loss_function = loss_function
        self._total_loss_tracker = 0.0
        self._reconstruction_loss_tracker = 0.0
        self._kl_loss_tracker = 0.0
        self._latent_mean_distribution = latent_mean_distribution
        self._latent_stander_deviation = latent_stander_deviation
        self._latent_dimension = latent_dimension
        self._decoder_latent_dimension = decoder_latent_dimension

        # File names for saving models
        self._file_name_encoder = file_name_encoder
        self._file_name_decoder = file_name_decoder

        # Path for saving models
        self._models_saved_path = models_saved_path

        # Optimizer will be configured later
        self.optimizer = None

    def train_step(self, batch):
        """
        Perform a training step for the Variational AutoEncoder (VAE).
        """
        # Simplificar: assumir que batch é (x, y, labels) nessa ordem
        if len(batch) == 3:
            batch_x, batch_y, batch_y_labels = batch
        else:
            batch_x, batch_y = batch
            batch_y_labels = None
        # Move to device
        device = next(self.parameters()).device
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        if batch_y_labels is not None:
            batch_y_labels = batch_y_labels.to(device)

        # Zero gradients
        self.optimizer.zero_grad()

        try:
            # CORREÇÃO: Passar os argumentos corretamente para o encoder
            # O encoder espera (x, label) como dois argumentos separados, não uma tupla
            if batch_y_labels is not None:
                # Chamar encoder com dois argumentos separados
                encoder_output = self._encoder(batch_x, batch_y_labels)
            else:
                encoder_output = self._encoder(batch_x)

            # O encoder retorna uma tupla: (z_mean, z_log_var, z, label)
            if isinstance(encoder_output, tuple) and len(encoder_output) >= 3:
                z_mean, z_log_var, latent, label_output = encoder_output[:4]
            else:
                # Se for apenas um tensor, usar como latent
                latent = encoder_output
                label_output = batch_y_labels if batch_y_labels is not None else None
                z_mean = z_log_var = None

            # Se não temos label_output, criar um dummy
            if label_output is None:
                batch_size = latent.shape[0]
                label_output = torch.zeros((batch_size, 2)).to(device)

            # Decoder - agora usando latent e label_output
            reconstruction_data = self._decoder(latent, label_output)

        except Exception as e:
            print(f"ERROR in forward pass: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Calcular reconstruction loss
        reconstruction_loss = F.binary_cross_entropy(reconstruction_data, batch_y, reduction='mean')

        # Calcular KL divergence se temos z_mean e z_log_var
        if z_mean is not None and z_log_var is not None:
            kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
            kl_loss = kl_loss / batch_x.size(0)  # Normalizar pelo batch size
        else:
            # Para autoencoder simples, KL loss = 0
            kl_loss = torch.tensor(0.0).to(device)

        # Total loss
        total_loss = reconstruction_loss + kl_loss

        # Backward pass
        total_loss.backward()

        # Update weights
        self.optimizer.step()

        # Update loss metrics
        self._total_loss_tracker = total_loss.item()
        self._reconstruction_loss_tracker = reconstruction_loss.item()
        self._kl_loss_tracker = kl_loss.item()

        return {
            "loss": self._total_loss_tracker,
            "reconstruction_loss": self._reconstruction_loss_tracker,
            "kl_loss": self._kl_loss_tracker
        }
    def configure_optimizer(self,
                            learning_rate=0.001,
                            beta_1=0.9,
                            beta_2=0.999,
                            epsilon=1e-7,
                            amsgrad=False,
                            weight_decay=1e-5):
        """
        Configure the Adam optimizer with custom parameters.
        """
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            betas=(beta_1, beta_2),
            eps=epsilon,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )

    def get_decoder_trained(self):
        return self._decoder

    def get_encoder_trained(self):
        return self._encoder

    def create_embedding(self, data):
        """
        Generates latent space embeddings using the trained encoder.
        """
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            if isinstance(data, numpy.ndarray):
                data = torch.from_numpy(data).float().to(device)

            latent_mean, _, _, _ = self._encoder(data)

        return latent_mean.cpu().numpy()

    def get_samples(self, number_samples_per_class):
        """
        Generate synthetic samples for each specified class using the trained decoder.
        """

        device = next(self.parameters()).device
        generated_data = {}

        with torch.no_grad():
            for label_class, number_instances in number_samples_per_class["classes"].items():
                # Create one-hot encoded label array
                label_samples_generated = torch.zeros(number_instances, number_samples_per_class["number_classes"])
                label_samples_generated[:, label_class] = 1
                label_samples_generated = label_samples_generated.to(device)

                # Sample random latent vectors from a standard normal distribution
                latent_noise = torch.randn(number_instances, self._decoder_latent_dimension).to(device)

                # Use the decoder to generate samples
                generated_samples = self._decoder(latent_noise, label_samples_generated)

                # Round the generated samples to the nearest integer
                generated_samples = torch.round(generated_samples)

                # Store the generated samples
                generated_data[label_class] = generated_samples.cpu().numpy()

        return generated_data

    def generate_synthetic_data(self, number_samples_generate, labels, latent_dimension):
        """
        Generate synthetic data using the Variational AutoEncoder (VAE).
        """
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            # Generate random noise samples in the latent space
            random_noise_generate = torch.randn(
                number_samples_generate,
                latent_dimension,
                device=device
            ) * self._latent_stander_deviation + self._latent_mean_distribution

            # Create label vectors for the generated data
            label_list = torch.full((number_samples_generate, 1), labels, dtype=torch.float32, device=device)

            # Generate synthetic data by passing random noise and labels through the decoder
            synthetic_data = self._decoder(random_noise_generate, label_list)

        return synthetic_data

    @property
    def metrics(self):
        """
        Returns:
            dict: Dictionary of metrics tracked during training.
        """
        return {
            "loss": self._total_loss_tracker,
            "reconstruction_loss": self._reconstruction_loss_tracker,
            "kl_loss": self._kl_loss_tracker
        }

    def save_model(self, directory, file_name):
        """
        Save the encoder and decoder models.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Construct file names for encoder and decoder models
        encoder_file_name = os.path.join(directory, f"fold_{file_name}_encoder.pth")
        decoder_file_name = os.path.join(directory, f"fold_{file_name}_decoder.pth")

        # Save encoder model
        torch.save(self._encoder.state_dict(), encoder_file_name)

        # Save decoder model
        torch.save(self._decoder.state_dict(), decoder_file_name)

    def load_models(self, directory, file_name):
        """
        Load the encoder and decoder models from a directory.
        """
        device = next(self.parameters()).device

        # Construct file names for encoder and decoder models
        encoder_file_name = os.path.join(directory, f"fold_{file_name}_encoder.pth")
        decoder_file_name = os.path.join(directory, f"fold_{file_name}_decoder.pth")

        # Load the encoder and decoder models
        self._encoder.load_state_dict(torch.load(encoder_file_name, map_location=device))
        self._decoder.load_state_dict(torch.load(decoder_file_name, map_location=device))

    def compile(self, loss, optimizer):
        """
        Configure the model for training (PyTorch compatibility method).
        """
        self.optimizer = optimizer

    def fit(self, train_data, y_data, epochs, batch_size, callbacks=None):
        """
        Train the VAE model.
        """
        device = next(self.parameters()).device

        # Definir modo de treino sem causar recursão
        self.training = True

        # Unpack training data
        if isinstance(train_data, tuple):
            x_train, y_labels = train_data
        else:
            x_train = train_data
            y_labels = None

        # Convert to tensors
        if isinstance(x_train, numpy.ndarray):
            x_train = torch.from_numpy(x_train).float()
        if isinstance(y_data, numpy.ndarray):
            y_data = torch.from_numpy(y_data).float()
        if y_labels is not None and isinstance(y_labels, numpy.ndarray):
            y_labels = torch.from_numpy(y_labels).float()

        # Create dataset
        if y_labels is not None:
            dataset = TensorDataset(x_train, y_labels, y_data)
        else:
            dataset = TensorDataset(x_train, y_data)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        for epoch in range(epochs):
            epoch_losses = []

            for batch_idx, batch in enumerate(dataloader):
                if y_labels is not None:
                    batch_x, batch_y_labels, batch_y = batch
                else:
                    batch_x, batch_y = batch
                    batch_y_labels = None

                # Mover para dispositivo
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                if batch_y_labels is not None:
                    batch_y_labels = batch_y_labels.to(device)

                # Passar para train_step
                if batch_y_labels is not None:
                    loss_dict = self.train_step((batch_x, batch_y, batch_y_labels))
                else:
                    loss_dict = self.train_step((batch_x, batch_y))

                epoch_losses.append(loss_dict)

            # Calcular perdas médias para a época
            if epoch_losses:
                avg_loss = sum(d['loss'] for d in epoch_losses) / len(epoch_losses)
                avg_recon = sum(d['reconstruction_loss'] for d in epoch_losses) / len(epoch_losses)
                avg_kl = sum(d['kl_loss'] for d in epoch_losses) / len(epoch_losses)

                print(f"\nEpoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f} - "
                      f"reconstruction_loss: {avg_recon:.4f} - kl_loss: {avg_kl:.4f}")

                # Executar callbacks
                if callbacks:
                    for callback in callbacks:
                        if hasattr(callback, 'on_epoch_end'):
                            callback.on_epoch_end(epoch, {'loss': avg_loss})

    # REMOVED: The problematic property setters that interfere with module registration
    # Properties can still be used for read-only access if needed, but not for setting
    @property
    def decoder(self):
        return self._decoder

    @property
    def encoder(self):
        return self._encoder