#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/07'
__credits__ = ['Synthetic Ocean AI']

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
    import numpy
    import logging
    from pathlib import Path
    from typing import Dict, Optional, Tuple

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

except ImportError as error:
    logging.error(error)
    sys.exit(-1)



class AdversarialAlgorithmTorch(nn.Module):
    """
    Implements an adversarial training algorithm in PyTorch, typically used in Generative Adversarial Networks (GANs).

    This class performs adversarial training by utilizing a generator and a discriminator,
    optimizing the generator to produce realistic data while training the discriminator to differentiate
    between real and fake data.

    The concept of Generative Adversarial Networks was introduced by Ian Goodfellow and his collaborators.

    Attributes:
        @generator_model (nn.Module):
            The generator model.
        @discriminator_model (nn.Module):
            The discriminator model.
        @latent_dimension (int):
            Dimensionality of the latent space.
        @loss_generator (nn.Module):
            Loss function for the generator.
        @loss_discriminator (nn.Module):
            Loss function for the discriminator.
        @file_name_discriminator (str):
            Filename for saving the discriminator model.
        @file_name_generator (str):
            Filename for saving the generator model.
        @models_saved_path (str):
            Path where models will be saved.
        @latent_mean_distribution (float):
            Mean of the latent noise distribution.
        @latent_stander_deviation (float):
            Standard deviation of the latent noise distribution.
        @smoothing_rate (float):
            Smoothing rate applied to discriminator labels.

    Example:
        >>> generator_model = build_generator(latent_dimension=100)
        >>> discriminator_model = build_discriminator()
        >>> adversarial_algorithm = AdversarialAlgorithmTorch(
        ...     generator_model=generator_model,
        ...     discriminator_model=discriminator_model,
        ...     latent_dimension=100,
        ...     loss_generator=nn.BCELoss(),
        ...     loss_discriminator=nn.BCELoss(),
        ...     file_name_discriminator="discriminator",
        ...     file_name_generator="generator",
        ...     models_saved_path="./models/",
        ...     latent_mean_distribution=0.0,
        ...     latent_stander_deviation=1.0,
        ...     smoothing_rate=0.1
        ... )
        >>> adversarial_algorithm.compile(
        ...     optimizer_generator=torch.optim.Adam(generator.parameters(), lr=0.0002),
        ...     optimizer_discriminator=torch.optim.Adam(discriminator.parameters(), lr=0.0002)
        ... )
        >>> history = adversarial_algorithm.fit(train_dataset, epochs=100)
    """

    def __init__(self, generator_model: nn.Module,
                 discriminator_model: nn.Module,
                 latent_dimension: int,
                 loss_generator: nn.Module,
                 loss_discriminator: nn.Module,
                 file_name_discriminator: str,
                 file_name_generator: str,
                 models_saved_path: str,
                 latent_mean_distribution: float,
                 latent_stander_deviation: float,
                 smoothing_rate: float):
        """
        Initializes the adversarial algorithm with the specified generator, discriminator, and other configurations.

        Args:
            generator_model (nn.Module): The generator model.
            discriminator_model (nn.Module): The discriminator model.
            latent_dimension (int): Latent space dimension.
            loss_generator (nn.Module): Generator's loss function.
            loss_discriminator (nn.Module): Discriminator's loss function.
            file_name_discriminator (str): Filename for discriminator model.
            file_name_generator (str): Filename for generator model.
            models_saved_path (str): Path for saving models.
            latent_mean_distribution (float): Mean of the latent noise distribution.
            latent_stander_deviation (float): Standard deviation of the latent noise.
            smoothing_rate (float): Label smoothing rate.

        Raises:
            ValueError: If any validation fails.
        """
        super().__init__()

        if latent_dimension <= 0:
            raise ValueError("Latent dimension must be greater than 0.")

        if not isinstance(file_name_discriminator, str) or not file_name_discriminator:
            raise ValueError("Discriminator file name must be a non-empty string.")

        if not isinstance(file_name_generator, str) or not file_name_generator:
            raise ValueError("Generator file name must be a non-empty string.")

        if not isinstance(models_saved_path, str) or not models_saved_path:
            raise ValueError("Architectures saved path must be a non-empty string.")

        if not isinstance(latent_mean_distribution, (int, float)):
            raise TypeError("Latent mean distribution must be a number.")

        if not isinstance(latent_stander_deviation, (int, float)):
            raise TypeError("Latent standard deviation must be a number.")

        if latent_stander_deviation <= 0:
            raise ValueError("Latent standard deviation must be greater than 0.")

        if not (0.0 <= smoothing_rate <= 1.0):
            raise ValueError("Smoothing rate must be between 0 and 1.")

        self._generator = generator_model
        self._discriminator = discriminator_model
        self._latent_dimension = latent_dimension
        self._optimizer_generator = None
        self._optimizer_discriminator = None

        # Convert loss functions to PyTorch if needed
        self._loss_generator = self._convert_loss_to_pytorch(loss_generator)
        self._loss_discriminator = self._convert_loss_to_pytorch(loss_discriminator)

        self._smoothing_rate = smoothing_rate
        self._latent_mean_distribution = latent_mean_distribution
        self._latent_stander_deviation = latent_stander_deviation
        self._file_name_discriminator = file_name_discriminator
        self._file_name_generator = file_name_generator
        self._models_saved_path = models_saved_path

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._generator.to(self.device)
        self._discriminator.to(self.device)

    def _convert_loss_to_pytorch(self, loss):
        """
        Converts a loss function to PyTorch if it's from TensorFlow/Keras.

        Args:
            loss: Loss function (can be PyTorch nn.Module, TensorFlow loss, or string).

        Returns:
            nn.Module: PyTorch loss function.
        """
        # If it's already a PyTorch loss, return it
        if isinstance(loss, nn.Module):
            return loss

        # If it's a string, convert to PyTorch loss
        if isinstance(loss, str):
            loss_name = loss.lower()
            loss_map = {
                'binary_crossentropy': nn.BCELoss(),
                'bce': nn.BCELoss(),
                'mse': nn.MSELoss(),
                'mean_squared_error': nn.MSELoss(),
                'mae': nn.L1Loss(),
                'mean_absolute_error': nn.L1Loss(),
            }
            if loss_name in loss_map:
                return loss_map[loss_name]

        # If it's a TensorFlow/Keras loss or unknown type, default to BCELoss
        logging.warning(f"Non-PyTorch loss function detected: {type(loss)}. Using BCELoss() as default.")
        return nn.BCELoss()

    def compile(self, optimizer_generator, optimizer_discriminator,
                loss_generator=None, loss_discriminator=None):
        """
        Compiles the adversarial algorithm by setting optimizers and loss functions.

        Args:
            optimizer_generator: Optimizer for the generator (PyTorch optimizer or learning rate).
            optimizer_discriminator: Optimizer for the discriminator (PyTorch optimizer or learning rate).
            loss_generator (optional): Generator's loss function (overrides __init__ if provided).
            loss_discriminator (optional): Discriminator's loss function (overrides __init__ if provided).
        """
        # Handle optimizer_generator
        if isinstance(optimizer_generator, (int, float)):
            # If a learning rate is passed, create Adam optimizer
            self._optimizer_generator = torch.optim.Adam(
                self._generator.parameters(),
                lr=optimizer_generator
            )
        elif hasattr(optimizer_generator, 'zero_grad'):
            # It's a PyTorch optimizer
            self._optimizer_generator = optimizer_generator
        else:
            # It's likely a TensorFlow/Keras optimizer, create PyTorch Adam with default lr
            self._optimizer_generator = torch.optim.Adam(
                self._generator.parameters(),
                lr=0.0002,
                betas=(0.5, 0.999)
            )

        # Handle optimizer_discriminator
        if isinstance(optimizer_discriminator, (int, float)):
            # If a learning rate is passed, create Adam optimizer
            self._optimizer_discriminator = torch.optim.Adam(
                self._discriminator.parameters(),
                lr=optimizer_discriminator
            )
        elif hasattr(optimizer_discriminator, 'zero_grad'):
            # It's a PyTorch optimizer
            self._optimizer_discriminator = optimizer_discriminator
        else:
            # It's likely a TensorFlow/Keras optimizer, create PyTorch Adam with default lr
            self._optimizer_discriminator = torch.optim.Adam(
                self._discriminator.parameters(),
                lr=0.0002,
                betas=(0.5, 0.999)
            )

        # Handle loss functions - convert to PyTorch if needed
        if loss_generator is not None:
            self._loss_generator = self._convert_loss_to_pytorch(loss_generator)
        if loss_discriminator is not None:
            self._loss_discriminator = self._convert_loss_to_pytorch(loss_discriminator)

    def fit(self, x=None, y=None, batch_size=32, epochs=1, verbose=1,
            callbacks=None, validation_data=None, shuffle=True,
            initial_epoch=0, steps_per_epoch=None, validation_steps=None,
            validation_freq=1, optimizer=None, learning_rate=0.001, **kwargs):
        """
        Train the model with a simplified progress bar (PyTorch version).

        Args:
            x: Training data (can be DataLoader, tuple of arrays, or array).
            y: Target labels (optional, used when x is an array).
            batch_size (int): Batch size for training.
            epochs (int): Number of training epochs.
            verbose (int): Verbosity level (0=silent, 1=progress bar, 2=one line per epoch).
            callbacks: List of callbacks to be called during training.
            validation_data: Validation data (tuple of arrays or DataLoader).
            shuffle (bool): Whether to shuffle training data.
            initial_epoch (int): Epoch at which to start training.
            steps_per_epoch (int): Number of steps per epoch.
            validation_steps (int): Number of validation steps.
            validation_freq (int): Frequency of validation (in epochs).
            optimizer: Optional PyTorch optimizer (overrides learning_rate).
            learning_rate (float): Learning rate if optimizer not provided.
            **kwargs: Additional arguments.

        Returns:
            History: Training history object containing loss values.
        """

        # Set optimizer if provided
        if optimizer is not None:
            self._optimizer_generator = optimizer
            self._optimizer_discriminator = optimizer
        elif self._optimizer_generator is None or self._optimizer_discriminator is None:
            self._optimizer_generator = torch.optim.Adam(
                self._generator.parameters(),
                lr=learning_rate,
                betas=(0.5, 0.999)
            )
            self._optimizer_discriminator = torch.optim.Adam(
                self._discriminator.parameters(),
                lr=learning_rate,
                betas=(0.5, 0.999)
            )

        # Prepare the dataset
        if isinstance(x, DataLoader):
            train_dataset = x
        else:
            if y is None:
                y = x
            x_tensor = torch.FloatTensor(x) if not isinstance(x, torch.Tensor) else x
            y_tensor = torch.FloatTensor(y) if not isinstance(y, torch.Tensor) else y
            dataset = TensorDataset(x_tensor, y_tensor)
            train_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        # Calculate steps per epoch if not provided
        if steps_per_epoch is None:
            steps_per_epoch = len(train_dataset)

        # Prepare validation data if provided
        val_dataloader = None
        if validation_data is not None:
            if isinstance(validation_data, DataLoader):
                val_dataloader = validation_data
            elif isinstance(validation_data, tuple) and len(validation_data) == 2:
                val_x, val_y = validation_data
                val_x_tensor = torch.FloatTensor(val_x) if not isinstance(val_x, torch.Tensor) else val_x
                val_y_tensor = torch.FloatTensor(val_y) if not isinstance(val_y, torch.Tensor) else val_y
                val_dataset = TensorDataset(val_x_tensor, val_y_tensor)
                val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # History to store metrics
        history = {'loss': [], 'loss_d': [], 'loss_g': []}
        if validation_data is not None:
            history['val_loss'] = []

        # Training loop
        for epoch in range(initial_epoch, epochs):
            # Reset trackers at the start of each epoch
            epoch_loss_total = 0.0
            epoch_loss_d = 0.0
            epoch_loss_g = 0.0
            num_batches = 0

            if verbose == 1:
                print(f'\nEpoch {epoch + 1}/{epochs}')

            # Progress tracking
            step = 0
            for batch_data in train_dataset:
                step += 1

                # Perform training step
                metrics = self.train_step(batch_data)

                # Calculate total loss (average of discriminator and generator losses)
                current_loss = (metrics['loss_d'] + metrics['loss_g']) / 2.0
                current_loss_d = metrics['loss_d']
                current_loss_g = metrics['loss_g']

                # Accumulate losses
                epoch_loss_total += current_loss
                epoch_loss_d += current_loss_d
                epoch_loss_g += current_loss_g
                num_batches += 1

                # Simple progress bar
                if verbose == 1:
                    progress = int(50 * step / steps_per_epoch)
                    bar = '=' * progress + '>' + '.' * (50 - progress - 1)
                    print(
                        f'\r[{bar}] {step}/{steps_per_epoch} - loss: {current_loss:.4f} - loss_d: {current_loss_d:.4f} - loss_g: {current_loss_g:.4f}',
                        end='', flush=True)

                if step >= steps_per_epoch:
                    break

            # Calculate average losses for the epoch
            avg_loss = epoch_loss_total / num_batches if num_batches > 0 else 0.0
            avg_loss_d = epoch_loss_d / num_batches if num_batches > 0 else 0.0
            avg_loss_g = epoch_loss_g / num_batches if num_batches > 0 else 0.0

            # Store epoch losses
            history['loss'].append(avg_loss)
            history['loss_d'].append(avg_loss_d)
            history['loss_g'].append(avg_loss_g)

            if verbose == 1:
                print(f' - loss: {avg_loss:.4f} - loss_d: {avg_loss_d:.4f} - loss_g: {avg_loss_g:.4f}', end='')
            elif verbose == 2:
                print(
                    f'Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f} - loss_d: {avg_loss_d:.4f} - loss_g: {avg_loss_g:.4f}',
                    end='')

            # Validation
            if val_dataloader is not None and (epoch + 1) % validation_freq == 0:
                val_loss = self._evaluate_validation(val_dataloader, validation_steps)
                history['val_loss'].append(val_loss)

                if verbose >= 1:
                    print(f' - val_loss: {val_loss:.4f}')
            else:
                if verbose >= 1:
                    print()  # New line after metrics

            # Callbacks
            if callbacks is not None:
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, {
                            'loss': avg_loss,
                            'loss_d': avg_loss_d,
                            'loss_g': avg_loss_g
                        })

        # Return history object
        class History:
            def __init__(self, history_dict):
                self.history = history_dict

        return History(history)

    def _evaluate_validation(self, validation_data, validation_steps=None):
        """
        Evaluate the model on validation data.

        Args:
            validation_data: DataLoader with validation data.
            validation_steps (int): Number of validation steps.

        Returns:
            float: Average validation loss.
        """
        self._generator.eval()
        self._discriminator.eval()

        val_losses = []
        step = 0

        with torch.no_grad():
            for batch_data in validation_data:
                # Perform validation step (without updating weights)
                real_feature, real_samples_label = batch_data

                # Move to device
                real_feature = real_feature.to(self.device)
                real_samples_label = real_samples_label.to(self.device)

                # Get the current batch size
                batch_size = real_feature.shape[0]

                # Expand label dimension if needed
                if len(real_samples_label.shape) == 1:
                    real_samples_label = real_samples_label.unsqueeze(-1)

                # Sample random noise vectors
                latent_space = torch.randn(batch_size, self._latent_dimension, device=self.device)
                latent_space = latent_space * self._latent_stander_deviation + self._latent_mean_distribution

                # Generate synthetic features
                synthetic_feature = self._generator([latent_space, real_samples_label])

                # Get discriminator predictions
                label_predicted_real = self._discriminator([real_feature, real_samples_label])
                label_predicted_synthetic = self._discriminator([synthetic_feature, real_samples_label])

                # Concatenate predictions
                label_predicted_all_samples = torch.cat([label_predicted_real, label_predicted_synthetic], dim=0)

                # Create ground-truth labels
                tensor_labels_predicted = torch.cat([
                    torch.zeros_like(label_predicted_real),
                    torch.ones_like(label_predicted_synthetic)
                ], dim=0)

                # Compute discriminator loss
                loss_d = self._loss_discriminator(label_predicted_all_samples, tensor_labels_predicted)

                # Compute generator loss
                predicted_labels = self._discriminator([synthetic_feature, real_samples_label])
                loss_g = self._loss_generator(predicted_labels, torch.zeros_like(predicted_labels))

                # Total loss (average of both)
                total_loss = (loss_d.item() + loss_g.item()) / 2.0
                val_losses.append(total_loss)

                step += 1
                if validation_steps is not None and step >= validation_steps:
                    break

        return numpy.mean(val_losses) if val_losses else 0.0

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        Performs a single training step for both generator and discriminator.

        Args:
            batch (tuple): A tuple containing real features (input data) and their corresponding labels.

        Returns:
            dict: A dictionary containing the loss values for both generator (loss_g) and discriminator (loss_d).
        """
        # Unpack the batch into real features and real labels
        real_feature, real_samples_label = batch

        # Move to device
        real_feature = real_feature.to(self.device)
        real_samples_label = real_samples_label.to(self.device)

        # Get the current batch size
        batch_size = real_feature.shape[0]

        # Expand label dimension if needed (add channel dimension)
        if len(real_samples_label.shape) == 1:
            real_samples_label = real_samples_label.unsqueeze(-1)

        # ==================== TRAIN DISCRIMINATOR ====================
        self._discriminator.train()
        self._generator.eval()

        self._optimizer_discriminator.zero_grad()

        # Sample random noise vectors (latent space) for the generator input
        latent_space = torch.randn(batch_size, self._latent_dimension, device=self.device)
        latent_space = latent_space * self._latent_stander_deviation + self._latent_mean_distribution

        # Generate synthetic features
        with torch.no_grad():
            synthetic_feature = self._generator([latent_space, real_samples_label])

        # Get discriminator predictions on real and synthetic samples
        label_predicted_real = self._discriminator([real_feature, real_samples_label])
        label_predicted_synthetic = self._discriminator([synthetic_feature, real_samples_label])

        # Concatenate predictions
        label_predicted_all_samples = torch.cat([label_predicted_real, label_predicted_synthetic], dim=0)

        # Create ground-truth labels (real=0, fake=1)
        tensor_labels_predicted = torch.cat([
            torch.zeros_like(label_predicted_real),
            torch.ones_like(label_predicted_synthetic)
        ], dim=0)

        # Label smoothing
        smooth_tensor_real_data = 0.15 * torch.rand_like(label_predicted_real)
        smooth_tensor_synthetic_data = -0.15 * torch.rand_like(label_predicted_synthetic)

        tensor_labels_predicted += torch.cat([
            smooth_tensor_real_data,
            smooth_tensor_synthetic_data
        ], dim=0)

        # Compute discriminator loss
        loss_d = self._loss_discriminator(label_predicted_all_samples, tensor_labels_predicted)

        # Backward pass and optimization
        loss_d.backward()
        self._optimizer_discriminator.step()

        # ==================== TRAIN GENERATOR ====================
        self._generator.train()
        self._discriminator.eval()

        self._optimizer_generator.zero_grad()

        # Generate new synthetic samples
        latent_space = torch.randn(batch_size, self._latent_dimension, device=self.device)
        latent_space = latent_space * self._latent_stander_deviation + self._latent_mean_distribution

        synthetic_feature = self._generator([latent_space, real_samples_label])

        # Get discriminator predictions for synthetic samples
        predicted_labels = self._discriminator([synthetic_feature, real_samples_label])

        # Generator wants discriminator to classify synthetic data as real (label=0)
        loss_g = self._loss_generator(predicted_labels, torch.zeros_like(predicted_labels))

        # Backward pass and optimization
        loss_g.backward()
        self._optimizer_generator.step()

        # Return losses
        return {"loss_d": loss_d.item(), "loss_g": loss_g.item()}

    def get_samples(self, number_samples_per_class: Dict) -> Dict:
        """
        Generates synthetic data samples for each specified class using the trained generator.

        Args:
            number_samples_per_class (dict):
                A dictionary specifying the number of synthetic samples to generate per class.
                Expected structure:
                {
                    "classes": {class_label: number_of_samples, ...},
                    "number_classes": total_number_of_classes
                }

        Returns:
            dict: A dictionary where each key is a class label and the value is an array of generated samples.
        """
        self._generator.eval()
        generated_data = {}

        with torch.no_grad():
            for label_class, number_instances in number_samples_per_class["classes"].items():
                # Create one-hot encoded labels
                label_samples_generated = F.one_hot(
                    torch.tensor([label_class] * number_instances),
                    num_classes=number_samples_per_class["number_classes"]
                ).float().to(self.device)

                # Generate random noise vectors
                latent_noise = torch.normal(
                    mean=self._latent_mean_distribution,
                    std=self._latent_stander_deviation,
                    size=(number_instances, self._latent_dimension)
                ).to(self.device)

                # Generate synthetic samples
                generated_samples = self._generator([latent_noise, label_samples_generated])

                # Round to nearest integer
                generated_samples = torch.round(generated_samples)

                # Convert to numpy and store
                generated_data[label_class] = generated_samples.cpu().numpy()

        return generated_data

    def save_model(self, path_output: str, k_fold: int):
        """
        Save the generator and discriminator models.

        Args:
            path_output (str): Output directory path.
            k_fold (int): Fold number for cross-validation.
        """
        try:
            logging.info(f"Starting to save Adversarial Model for fold {k_fold}...")

            # Create directory
            path_directory = os.path.join(path_output, self._models_saved_path)
            Path(path_directory).mkdir(parents=True, exist_ok=True)
            logging.info(f"Created/verified directory at: {path_directory}")

            # Filenames
            discriminator_file_name = f"{self._file_name_discriminator}_{k_fold}"
            generator_file_name = f"{self._file_name_generator}_{k_fold}"

            # Fold directory
            path_model = os.path.join(path_directory, f"fold_{k_fold + 1}")
            Path(path_model).mkdir(parents=True, exist_ok=True)
            logging.info(f"Created/verified fold directory at: {path_model}")

            # Full paths
            discriminator_path = os.path.join(path_model, discriminator_file_name)
            generator_path = os.path.join(path_model, generator_file_name)

            # Save discriminator
            logging.info("Saving discriminator model...")
            torch.save({
                'model_state_dict': self._discriminator.state_dict(),
                'model_architecture': str(self._discriminator)
            }, discriminator_path + ".pth")
            logging.info(f"Discriminator model saved at: {discriminator_path}.pth")

            # Save generator
            logging.info("Saving generator model...")
            torch.save({
                'model_state_dict': self._generator.state_dict(),
                'model_architecture': str(self._generator)
            }, generator_path + ".pth")
            logging.info(f"Generator model saved at: {generator_path}.pth")

        except Exception as e:
            logging.error(f"An error occurred while saving the models: {e}")
            raise

    def load_models(self, path_output: str, k_fold: int):
        """
        Load the generator and discriminator models.

        Args:
            path_output (str): Output directory path.
            k_fold (int): Fold number for cross-validation.
        """
        try:
            logging.info(f"Loading Adversarial Model for fold {k_fold + 1}...")

            # Directory containing saved models
            path_directory = os.path.join(path_output, self._models_saved_path)

            # Filenames
            discriminator_file_name = f"{self._file_name_discriminator}_{k_fold + 1}"
            generator_file_name = f"{self._file_name_generator}_{k_fold + 1}"

            # Full paths
            discriminator_path = os.path.join(path_directory, discriminator_file_name)
            generator_path = os.path.join(path_directory, generator_file_name)

            # Load discriminator
            logging.info(f"Loading discriminator model from: {discriminator_path}.pth")
            discriminator_checkpoint = torch.load(discriminator_path + ".pth", map_location=self.device)
            self._discriminator.load_state_dict(discriminator_checkpoint['model_state_dict'])
            logging.info("Loaded discriminator weights")

            # Load generator
            logging.info(f"Loading generator model from: {generator_path}.pth")
            generator_checkpoint = torch.load(generator_path + ".pth", map_location=self.device)
            self._generator.load_state_dict(generator_checkpoint['model_state_dict'])
            logging.info("Loaded generator weights")

            # Set to evaluation mode
            self._generator.eval()
            self._discriminator.eval()

        except FileNotFoundError:
            logging.error("Model file not found. Please provide an existing and valid model.")
            raise
        except Exception as e:
            logging.error(f"An error occurred while loading the models: {e}")
            raise

    # Setter methods
    def set_generator(self, generator: nn.Module):
        self._generator = generator
        self._generator.to(self.device)

    def set_discriminator(self, discriminator: nn.Module):
        self._discriminator = discriminator
        self._discriminator.to(self.device)

    def set_latent_dimension(self, latent_dimension: int):
        if latent_dimension <= 0:
            raise ValueError("Latent dimension must be greater than 0.")
        self._latent_dimension = latent_dimension

    def set_optimizer_generator(self, optimizer_generator):
        self._optimizer_generator = optimizer_generator

    def set_optimizer_discriminator(self, optimizer_discriminator):
        self._optimizer_discriminator = optimizer_discriminator

    def set_loss_generator(self, loss_generator):
        """
        Sets the loss function for the generator.

        Args:
            loss_generator: Loss function (PyTorch nn.Module, TensorFlow loss, or string).
        """
        self._loss_generator = self._convert_loss_to_pytorch(loss_generator)

    def set_loss_discriminator(self, loss_discriminator):
        """
        Sets the loss function for the discriminator.

        Args:
            loss_discriminator: Loss function (PyTorch nn.Module, TensorFlow loss, or string).
        """
        self._loss_discriminator = self._convert_loss_to_pytorch(loss_discriminator)

    @property
    def generator(self):
        return self._generator

    @property
    def discriminator(self):
        return self._discriminator

    class History:
        """
        History object for Keras compatibility.
        Stores training history and provides access via .history attribute.
        """

        def __init__(self):
            self.history = {}

        def __getitem__(self, key):
            return self.history[key]

        def __setitem__(self, key, value):
            self.history[key] = value

        def keys(self):
            return self.history.keys()