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
                 latent_standard_deviation,  # CORRIGIDO: typo "stander" → "standard"
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
        self._latent_standard_deviation = latent_standard_deviation  # CORRIGIDO
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
        # Unpack batch based on its length
        if len(batch) == 3:
            batch_x, batch_y, batch_y_labels = batch
        elif len(batch) == 2:
            batch_x, batch_y = batch
            batch_y_labels = None
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")

        # Move to device
        device = next(self.parameters()).device
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        if batch_y_labels is not None:
            batch_y_labels = batch_y_labels.to(device)

        # IMPORTANTE: Em um VAE, queremos reconstruir batch_x
        # Se batch_y tem dimensão diferente de batch_x, então batch_y são os labels
        # e devemos usar batch_x como alvo de reconstrução
        if batch_x.shape != batch_y.shape:
            # batch_y são labels, usar batch_x como target
            reconstruction_target = batch_x
            if batch_y_labels is None:
                batch_y_labels = batch_y
        else:
            # batch_y é o target correto
            reconstruction_target = batch_y

        # Zero gradients
        self.optimizer.zero_grad()

        try:
            # Passar argumentos corretamente para o encoder
            if batch_y_labels is not None:
                encoder_output = self._encoder(batch_x, batch_y_labels)
            else:
                encoder_output = self._encoder(batch_x)

            # O encoder retorna: (z_mean, z_log_var, z, label)
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

            # Decoder
            reconstruction_data = self._decoder(latent, label_output)

        except Exception as e:
            print(f"ERROR in forward pass: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Calcular reconstruction loss usando o target correto
        reconstruction_loss = F.binary_cross_entropy(reconstruction_data, reconstruction_target, reduction='mean')

        # Calcular KL divergence se temos z_mean e z_log_var
        if z_mean is not None and z_log_var is not None:
            kl_loss = -0.5 * torch.mean(torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1))
        else:
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

    def train_step(self, batch):
        """
        Perform a training step for the Variational AutoEncoder (VAE).
        """
        # Unpack batch based on its length
        if len(batch) == 3:
            batch_x, batch_y, batch_y_labels = batch
        elif len(batch) == 2:
            batch_x, batch_y = batch
            batch_y_labels = None
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")

        # Move to device
        device = next(self.parameters()).device
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        if batch_y_labels is not None:
            batch_y_labels = batch_y_labels.to(device)

        # IMPORTANTE: Em um VAE, queremos reconstruir batch_x
        # Se batch_y tem dimensão diferente de batch_x, então batch_y são os labels
        # e devemos usar batch_x como alvo de reconstrução
        if batch_x.shape != batch_y.shape:
            # batch_y são labels, usar batch_x como target
            reconstruction_target = batch_x
            if batch_y_labels is None:
                batch_y_labels = batch_y
        else:
            # batch_y é o target correto
            reconstruction_target = batch_y

        # Zero gradients
        self.optimizer.zero_grad()

        try:
            # Passar argumentos corretamente para o encoder
            if batch_y_labels is not None:
                encoder_output = self._encoder(batch_x, batch_y_labels)
            else:
                encoder_output = self._encoder(batch_x)

            # O encoder retorna: (z_mean, z_log_var, z, label)
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

            # Decoder
            reconstruction_data = self._decoder(latent, label_output)

        except Exception as e:
            print(f"ERROR in forward pass: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Calcular reconstruction loss usando o target correto
        reconstruction_loss = F.binary_cross_entropy(reconstruction_data, reconstruction_target, reduction='mean')

        # Calcular KL divergence se temos z_mean e z_log_var
        if z_mean is not None and z_log_var is not None:
            kl_loss = -0.5 * torch.mean(torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1))
        else:
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

    def fit(self, x=None, y=None, batch_size=32, epochs=1, verbose=1,
            callbacks=None, validation_data=None, shuffle=True,
            initial_epoch=0, steps_per_epoch=None, validation_steps=None,
            validation_freq=1, optimizer=None, learning_rate=0.001, **kwargs):
        """
        Train the model with a simplified progress bar.

        Args:
            x: Input data (can be numpy array, tensor, tuple, or DataLoader).
            y: Target data or labels (can be numpy array, tensor, or tuple).
            batch_size: Number of samples per gradient update.
            epochs: Number of epochs to train.
            verbose: 0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks: List of callbacks to apply during training.
            validation_data: Data for validation.
            shuffle: Whether to shuffle data before each epoch.
            initial_epoch: Epoch at which to start training.
            steps_per_epoch: Number of steps per epoch.
            validation_steps: Number of validation steps.
            validation_freq: Validation frequency.
            optimizer: PyTorch optimizer (if None, uses already compiled optimizer).
            learning_rate: Learning rate for optimizer (only used if optimizer is None).

        Returns:
            A History object with training metrics.
        """
        device = next(self.parameters()).device

        # Set optimizer if provided
        if optimizer is not None:
            self.optimizer = optimizer
        elif self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Prepare the dataset
        if isinstance(x, DataLoader):
            train_dataloader = x
        else:
            # Handle different input formats
            if isinstance(x, tuple):
                # x is tuple: (data, target) or (data, target, labels)
                if len(x) == 3:
                    x_data, y_data, labels = x
                elif len(x) == 2:
                    x_data, y_data = x
                    labels = None
                else:
                    x_data = x[0]
                    y_data = x[0]  # Autoencoder: reconstruct input
                    labels = None
            else:
                x_data = x
                if y is None:
                    y_data = x  # Autoencoder: reconstruct input
                    labels = None
                elif isinstance(y, tuple):
                    if len(y) == 2:
                        y_data, labels = y
                    else:
                        y_data = y[0]
                        labels = None
                else:
                    # Se y tem dimensão diferente de x, y são os labels
                    if isinstance(y, numpy.ndarray):
                        y_np = y
                    elif torch.is_tensor(y):
                        y_np = y.cpu().numpy()
                    else:
                        y_np = numpy.array(y)

                    if isinstance(x, numpy.ndarray):
                        x_np = x
                    elif torch.is_tensor(x):
                        x_np = x.cpu().numpy()
                    else:
                        x_np = numpy.array(x)

                    # Verificar se dimensões são compatíveis
                    if y_np.shape[1:] != x_np.shape[1:]:
                        # y são labels, x é tanto input quanto target
                        y_data = x
                        labels = y
                    else:
                        y_data = y
                        labels = None

            # Convert to tensors
            if isinstance(x_data, numpy.ndarray):
                x_data = torch.from_numpy(x_data).float()
            if isinstance(y_data, numpy.ndarray):
                y_data = torch.from_numpy(y_data).float()
            if labels is not None and isinstance(labels, numpy.ndarray):
                labels = torch.from_numpy(labels).float()

            # Create TensorDataset
            if labels is not None:
                dataset = TensorDataset(x_data, y_data, labels)
            else:
                dataset = TensorDataset(x_data, y_data)

            train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        # Calculate steps per epoch
        if steps_per_epoch is None:
            steps_per_epoch = len(train_dataloader)

        # History to store metrics
        history = {'loss': [], 'reconstruction_loss': [], 'kl_loss': []}

        # Training loop
        for epoch in range(initial_epoch, epochs):
            self._total_loss_tracker = 0.0
            self._reconstruction_loss_tracker = 0.0
            self._kl_loss_tracker = 0.0

            epoch_losses = []
            epoch_recon_losses = []
            epoch_kl_losses = []

            if verbose == 1:
                print(f'\nEpoch {epoch + 1}/{epochs}')

            step = 0
            for batch_data in train_dataloader:
                step += 1

                # Perform training step
                metrics = self.train_step(batch_data)
                current_loss = float(metrics['loss'])
                current_recon_loss = float(metrics['reconstruction_loss'])
                current_kl_loss = float(metrics['kl_loss'])

                epoch_losses.append(current_loss)
                epoch_recon_losses.append(current_recon_loss)
                epoch_kl_losses.append(current_kl_loss)

                # Progress bar
                if verbose == 1:
                    progress = int(50 * step / steps_per_epoch)
                    bar = '=' * progress + '>' + '.' * (50 - progress - 1)
                    print(
                        f'\r[{bar}] {step}/{steps_per_epoch} - loss: {current_loss:.4f} - recon_loss: {current_recon_loss:.4f} - kl_loss: {current_kl_loss:.4f}',
                        end='', flush=True)

                if step >= steps_per_epoch:
                    break

            # Store epoch metrics
            epoch_loss = self._total_loss_tracker
            epoch_recon_loss = numpy.mean(epoch_recon_losses) if epoch_recon_losses else 0.0
            epoch_kl_loss = numpy.mean(epoch_kl_losses) if epoch_kl_losses else 0.0

            history['loss'].append(epoch_loss)
            history['reconstruction_loss'].append(epoch_recon_loss)
            history['kl_loss'].append(epoch_kl_loss)

            if verbose == 1:
                print(f' - loss: {epoch_loss:.4f} - recon_loss: {epoch_recon_loss:.4f} - kl_loss: {epoch_kl_loss:.4f}')
            elif verbose == 2:
                print(
                    f'Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.4f} - recon_loss: {epoch_recon_loss:.4f} - kl_loss: {epoch_kl_loss:.4f}')

            # Validation
            if validation_data is not None and (epoch + 1) % validation_freq == 0:
                val_loss = self._evaluate_validation(validation_data, validation_steps)
                if 'val_loss' not in history:
                    history['val_loss'] = []
                history['val_loss'].append(val_loss)

                if verbose >= 1:
                    print(f' - val_loss: {val_loss:.4f}')

            # Callbacks
            if callbacks is not None:
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, {
                            'loss': epoch_loss,
                            'reconstruction_loss': epoch_recon_loss,
                            'kl_loss': epoch_kl_loss
                        })

        # Return history
        class History:
            def __init__(self, history_dict):
                self.history = history_dict

        return History(history)
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

    def create_embedding(self, data, labels=None):
        """
        Generates latent space embeddings using the trained encoder.

        Args:
            data: Input data
            labels: Optional labels (one-hot encoded or indices)
        """
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            if isinstance(data, numpy.ndarray):
                data = torch.from_numpy(data).float().to(device)

            # CORRIGIDO: Aceitar labels opcionais
            if labels is not None:
                if isinstance(labels, numpy.ndarray):
                    labels = torch.from_numpy(labels).float().to(device)
                encoder_output = self._encoder(data, labels)
            else:
                encoder_output = self._encoder(data)

            # Extrair z_mean
            if isinstance(encoder_output, tuple) and len(encoder_output) >= 1:
                latent_mean = encoder_output[0]
            else:
                latent_mean = encoder_output

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
                # NOTA: Usando decoder_latent_dimension que pode ser diferente de latent_dimension
                latent_noise = torch.randn(number_instances, self._decoder_latent_dimension).to(device)

                # Use the decoder to generate samples
                generated_samples = self._decoder(latent_noise, label_samples_generated)

                # Store the generated samples
                generated_data[label_class] = generated_samples.cpu().numpy()

        return generated_data

    def generate_synthetic_data(self, number_samples_generate, label_class, num_classes, latent_dimension=None):
        """
        Generate synthetic data using the Variational AutoEncoder (VAE).

        Args:
            number_samples_generate: Number of samples to generate
            label_class: Class label (integer) to generate
            num_classes: Total number of classes
            latent_dimension: Dimension of latent space (uses self._latent_dimension if None)
        """
        self.eval()
        device = next(self.parameters()).device

        # CORRIGIDO: Usar latent_dimension correto
        if latent_dimension is None:
            latent_dimension = self._latent_dimension

        with torch.no_grad():
            # Generate random noise samples in the latent space
            random_noise_generate = torch.randn(
                number_samples_generate,
                latent_dimension,
                device=device
            ) * self._latent_standard_deviation + self._latent_mean_distribution

            # CORRIGIDO: Create one-hot encoded labels para compatibilidade com decoder
            label_list = torch.zeros(number_samples_generate, num_classes, device=device)
            label_list[:, label_class] = 1.0

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


    def _evaluate_validation(self, validation_data, validation_steps=None):
        """
        Evaluate the model on validation data.

        Args:
            validation_data: Validation dataset (DataLoader or tuple).
            validation_steps: Number of validation steps.

        Returns:
            Average validation loss.
        """
        self.eval()
        device = next(self.parameters()).device

        val_losses = []
        val_recon_losses = []
        val_kl_losses = []
        step = 0

        # Prepare validation dataset
        if isinstance(validation_data, DataLoader):
            val_dataloader = validation_data
        else:
            val_x, val_y = validation_data
            if isinstance(val_x, numpy.ndarray):
                val_x = torch.from_numpy(val_x).float()
            if isinstance(val_y, numpy.ndarray):
                val_y = torch.from_numpy(val_y).float()
            val_dataset = TensorDataset(val_x, val_y)
            val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        with torch.no_grad():
            for batch_data in val_dataloader:
                batch_x, batch_y = batch_data
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                # Forward pass through encoder and decoder
                encoder_output = self._encoder(batch_x)

                if isinstance(encoder_output, tuple) and len(encoder_output) >= 3:
                    latent_mean, latent_log_variation, latent, label = encoder_output[:4]
                else:
                    latent = encoder_output
                    latent_mean = latent_log_variation = None
                    label = torch.zeros((latent.shape[0], 2)).to(device)

                reconstruction_data = self._decoder(latent, label)

                # Calculate binary cross-entropy loss for reconstruction
                reconstruction_loss = F.binary_cross_entropy(reconstruction_data, batch_y, reduction='mean')

                # CORRIGIDO: Fórmula KL divergence correta
                if latent_mean is not None and latent_log_variation is not None:
                    # KL divergence: -0.5 * sum(1 + log(var) - mean^2 - var)
                    kl_loss = -0.5 * torch.sum(
                        1 + latent_log_variation - torch.square(latent_mean) - torch.exp(latent_log_variation),
                        dim=1
                    )
                    kl_divergence_loss = torch.mean(kl_loss)
                else:
                    kl_divergence_loss = torch.tensor(0.0).to(device)

                # Total loss
                total_loss = reconstruction_loss + kl_divergence_loss

                val_losses.append(float(total_loss))
                val_recon_losses.append(float(reconstruction_loss))
                val_kl_losses.append(float(kl_divergence_loss))

                step += 1
                if validation_steps is not None and step >= validation_steps:
                    break

        self.train()
        return numpy.mean(val_losses) if val_losses else 0.0

    @staticmethod
    def calculate_samples_per_class(y_labels):
        """
        Calculate the distribution of samples per class from labels.

        Args:
            y_labels (array-like): Labels array

        Returns:
            dict: Dictionary with 'classes' and 'number_classes' keys
        """
        # Convert to numpy if needed
        if torch.is_tensor(y_labels):
            y_labels = y_labels.cpu().numpy()

        # Handle one-hot encoded labels
        if len(y_labels.shape) == 2 and y_labels.shape[1] > 1:
            y_labels = numpy.argmax(y_labels, axis=1)

        # Count samples per class
        unique, counts = numpy.unique(y_labels, return_counts=True)

        return {
            "classes": dict(zip(unique.tolist(), counts.tolist())),
            "number_classes": len(unique)
        }

    # Properties for read-only access
    @property
    def decoder(self):
        return self._decoder

    @property
    def encoder(self):
        return self._encoder