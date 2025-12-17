#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com'
__version__ = '{1}.{0}.{2}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/12/16'
__credits__ = ['Kayuã Oleques']

# MIT License
#
# Copyright (c) 2025 Synthetic Ocean AI

try:
    import os
    import sys
    import json
    import time
    import numpy
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from typing import Any

except ImportError as error:
    print(error)
    sys.exit(-1)


class AlgorithmDenoisingDiffusionTorch(nn.Module):
    """
    Implements a diffusion process using UNet architectures with adaptive input handling.

    This model integrates an autoencoder and a diffusion network, enabling both data
    reconstruction and controlled generative modeling through Gaussian diffusion.
    It automatically adapts to any input shape: (x), (x, y), (x, y, z), etc.

    This class supports exponential moving average (EMA) updates for stable training,
    multiple training stages, and customizable hyperparameters to adapt to different tasks.

    References:
        - Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models."
        Advances in Neural Information Processing Systems (NeurIPS).
        Available at: https://arxiv.org/abs/2006.11239

    Example:
        >>> # Example with 1D time series data
        >>> diffusion_1d = AlgorithmDenoisingDiffusionTorch(
        ...     output_shape=100,
        ...     first_unet_model=primary_unet,
        ...     second_unet_model=ema_unet,
        ...     gdf_util=gaussian_diffusion,
        ...     optimizer_autoencoder=torch.optim.Adam(...),
        ...     optimizer_diffusion=torch.optim.Adam(...),
        ...     time_steps=1000,
        ...     ema=0.999,
        ...     margin=0.1,
        ...     input_shape=(100,)
        ... )
        >>> diffusion_1d.fit(data_1d, labels_1d, epochs=50)

        >>> # Example with 2D image data
        >>> diffusion_2d = AlgorithmDenoisingDiffusionTorch(
        ...     output_shape=784,
        ...     first_unet_model=unet_2d,
        ...     second_unet_model=ema_unet_2d,
        ...     gdf_util=gaussian_diffusion,
        ...     optimizer_autoencoder=torch.optim.Adam(...),
        ...     optimizer_diffusion=torch.optim.Adam(...),
        ...     time_steps=1000,
        ...     ema=0.999,
        ...     margin=0.1,
        ...     input_shape=(28, 28)
        ... )
        >>> diffusion_2d.fit(images, labels, epochs=100)
    """

    def __init__(self,
                 output_shape,
                 first_unet_model,
                 second_unet_model,
                 gdf_util,
                 optimizer_autoencoder,
                 optimizer_diffusion,
                 time_steps,
                 ema,
                 margin,
                 train_stage='all',
                 input_shape=None,
                 auto_adapt_shape=True):
        """
        Initializes the DiffusionModel with provided sub-models, optimizers, and hyperparameters.

        Args:
            output_shape: Expected output dimension for generated data
            first_unet_model: Primary UNet model for diffusion-based generation
            second_unet_model: Secondary UNet model for maintaining EMA-based weight updates
            gdf_util: Utility object responsible for Gaussian diffusion operations
            optimizer_autoencoder: Optimizer handling the training of the encoder-decoder network
            optimizer_diffusion: Optimizer applied to the diffusion process
            time_steps: Number of discrete time steps for the diffusion process
            ema: Exponential moving average decay factor
            margin: Margin value used in loss calculations or regularization
            train_stage: Current training stage ('all', 'diffusion', etc.)
            input_shape: Expected input shape (without batch dimension)
            auto_adapt_shape: Whether to automatically adapt to input data shape
        """
        super().__init__()

        self._ema = ema
        self._margin = margin
        self._gdf_util = gdf_util
        self._time_steps = time_steps
        self._train_stage = train_stage
        self._network = first_unet_model
        self._output_shape = output_shape
        self._original_shape = output_shape
        self._second_unet_model = second_unet_model
        self._optimizer_diffusion = optimizer_diffusion
        self._optimizer_autoencoder = optimizer_autoencoder
        self._loss_fn = nn.MSELoss()

        # Shape adaptation
        self._input_shape = input_shape
        self._auto_adapt_shape = auto_adapt_shape
        self._inferred_shape = None

        self._total_loss = 0.0
        self._total_loss_count = 0

        # Training history
        self._training_history = {
            'epoch': [],
            'loss': [],
            'avg_loss': []
        }

    @staticmethod
    def _infer_data_shape(data):
        """
        Infer the shape of input data, excluding the batch dimension.

        Args:
            data: Input data (tensor, array, or tuple/list).

        Returns:
            tuple: Shape of the data excluding batch dimension.
        """
        if isinstance(data, (tuple, list)):
            # If data is a tuple/list, infer from first element
            data = data[0]

        if isinstance(data, torch.Tensor):
            shape = tuple(data.shape[1:])
        elif isinstance(data, numpy.ndarray):
            shape = data.shape[1:] if len(data.shape) > 1 else data.shape
        else:
            # Try to convert to tensor and get shape
            try:
                tensor_data = torch.tensor(data)
                shape = tuple(tensor_data.shape[1:])
            except:
                raise ValueError(f"Cannot infer shape from data of type {type(data)}")

        return shape

    def _validate_and_adapt_shape(self, data):
        """
        Validate input data shape and adapt if necessary.

        Args:
            data: Input data.

        Returns:
            bool: True if shape is valid or successfully adapted.
        """
        current_shape = self._infer_data_shape(data)

        if self._inferred_shape is None:
            self._inferred_shape = current_shape
            if self._input_shape is not None and self._input_shape != current_shape:
                print(f"Warning: Specified input_shape {self._input_shape} differs from inferred shape {current_shape}")
                if self._auto_adapt_shape:
                    print(f"Auto-adapting to shape: {current_shape}")
                    self._input_shape = current_shape
            elif self._input_shape is None:
                self._input_shape = current_shape
                print(f"Inferred input shape: {current_shape}")
        else:
            if current_shape != self._inferred_shape:
                if self._auto_adapt_shape:
                    print(f"Warning: Input shape changed from {self._inferred_shape} to {current_shape}")
                    self._inferred_shape = current_shape
                else:
                    raise ValueError(
                        f"Input shape mismatch: expected {self._inferred_shape}, got {current_shape}. "
                        f"Set auto_adapt_shape=True to allow dynamic shape changes."
                    )

        return True

    @staticmethod
    def _prepare_batch(batch):
        """
        Prepare batch data, handling different input formats.

        Args:
            batch: Input batch (can be single tensor, tuple, or list).

        Returns:
            tuple: (batch_x, batch_y) formatted for diffusion training.
        """
        if isinstance(batch, (tuple, list)):
            if len(batch) == 1:
                # Single input - need labels for diffusion
                raise ValueError("Diffusion model requires both data and labels")
            elif len(batch) == 2:
                # (data, labels)
                batch_x = batch[0]
                batch_y = batch[1]
            else:
                # Multiple inputs
                batch_x = batch[0]
                batch_y = batch[1]
        else:
            # Single tensor - need labels
            raise ValueError("Diffusion model requires both data and labels")

        return batch_x, batch_y

    def get_input_shape(self):
        """
        Get the current input shape.

        Returns:
            tuple: Current input shape (excluding batch dimension).
        """
        return self._inferred_shape if self._inferred_shape is not None else self._input_shape

    @staticmethod
    def reshape_data(data, target_shape):
        """
        Reshape data to target shape if needed.

        Args:
            data: Input data.
            target_shape: Desired shape (excluding batch dimension).

        Returns:
            Reshaped data.
        """
        if isinstance(data, numpy.ndarray):
            if data.shape[1:] != target_shape:
                batch_size = data.shape[0]
                return data.reshape((batch_size,) + target_shape)
        elif isinstance(data, torch.Tensor):
            if tuple(data.shape[1:]) != target_shape:
                batch_size = data.shape[0]
                return data.reshape((batch_size,) + target_shape)
        return data

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
        if isinstance(y_labels, torch.Tensor):
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

    def set_stage_training(self, training_stage):
        """
        Sets the current training stage.

        Args:
            training_stage (str): New training stage ('all', 'diffusion', etc.).
        """
        self._train_stage = training_stage

    def train_step(self, data, labels):
        """
        Performs a single training step with automatic batch handling.

        Args:
            data: Input data tensor
            labels: Label tensor

        Returns:
            dict: A dictionary with the computed loss for diffusion.
        """
        loss_diffusion = self.train_diffusion_model(data, labels)
        self.sync_skip_projections()
        self.update_ema_weights()
        return {"Diffusion_loss": loss_diffusion.item() if loss_diffusion is not None else 0}

    def sync_skip_projections(self):
        """
        Synchronizes dynamically created skip projection layers from the first UNet
        to the second UNet (EMA model).
        """
        if hasattr(self._network, '_skip_projections') and hasattr(self._second_unet_model, '_skip_projections'):
            for key, projection in self._network._skip_projections.items():
                if key not in self._second_unet_model._skip_projections:
                    self._second_unet_model._skip_projections[key] = nn.Linear(
                        projection.in_features,
                        projection.out_features
                    ).to(projection.weight.device)
                    self._second_unet_model._skip_projections[key].load_state_dict(
                        projection.state_dict()
                    )

    def train_diffusion_model(self, data, ground_truth):
        """
        Performs a single training step for the diffusion model.

        Args:
            data (torch.Tensor): Input data embeddings
            ground_truth (torch.Tensor): Corresponding class labels

        Returns:
            torch.Tensor: The computed loss for this training step.
        """
        embedding_label = ground_truth
        embedding_data_expanded = data

        batch_size = data.shape[0]

        embedding_data_expanded = embedding_data_expanded.float()

        random_time_steps = torch.randint(0, self._time_steps, (batch_size,),
                                          dtype=torch.long, device=data.device)

        self._optimizer_diffusion.zero_grad()

        random_noise = torch.randn_like(embedding_data_expanded)

        embedding_with_noise = self._gdf_util.q_sample(embedding_data_expanded,
                                                       random_time_steps,
                                                       random_noise)

        predicted_noise = self._network(embedding_with_noise, random_time_steps, embedding_label)

        if len(predicted_noise.shape) > len(random_noise.shape):
            predicted_noise = predicted_noise.squeeze(-1)

        if predicted_noise.shape != random_noise.shape:
            raise RuntimeError(
                f"Shape mismatch: predicted_noise {predicted_noise.shape} "
                f"!= random_noise {random_noise.shape}. "
                f"Model output_shape: {self._network._output_shape}, "
                f"Input shape: {embedding_data_expanded.shape}"
            )

        loss_diffusion = self._loss_fn(random_noise, predicted_noise)

        loss_diffusion.backward()

        self._optimizer_diffusion.step()

        return loss_diffusion

    def update_ema_weights(self):
        """
        Updates the weights of the second UNet model using exponential moving average.
        """
        with torch.no_grad():
            first_params = dict(self._network.named_parameters())
            second_params = dict(self._second_unet_model.named_parameters())

            for name in first_params.keys():
                if name in second_params:
                    param = first_params[name]
                    ema_param = second_params[name]

                    if param.shape == ema_param.shape:
                        ema_param.data.mul_(self._ema).add_(param.data, alpha=1 - self._ema)
                    else:
                        print(f"Warning: Skipping EMA update for {name} due to shape mismatch: "
                              f"{param.shape} vs {ema_param.shape}")

    def fit(self,
            x=None,
            y=None,
            batch_size=32,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_data=None,
            validation_freq=1,
            shuffle=True,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            optimizer=None,
            learning_rate=0.001,
            # Legacy parameters for backward compatibility
            x_real_samples=None,
            y_real_samples=None,
            callback_model_monitor=None,
            callback_early_stop=None,
            use_early_stop=False,
            **kwargs):
        """
        Train the model with automatic shape adaptation.

        Args:
            x: Input data (any shape) or DataLoader.
            y: Target data (labels).
            batch_size: Number of samples per gradient update.
            epochs: Number of epochs to train.
            verbose: 0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks: List of callbacks to apply during training.
            validation_data: Data for validation.
            validation_freq: Validation frequency in epochs.
            shuffle: Whether to shuffle data before each epoch.
            initial_epoch: Epoch at which to start training.
            steps_per_epoch: Number of steps per epoch.
            validation_steps: Number of validation steps.
            optimizer: PyTorch optimizer (if None, uses already compiled optimizer).
            learning_rate: Learning rate for optimizer.

        Returns:
            A History object with training metrics.
        """

        # Handle legacy parameter names
        if x_real_samples is not None:
            x = x_real_samples
        if y_real_samples is not None:
            y = y_real_samples

        # Set optimizer if provided
        if optimizer is not None:
            self._optimizer_diffusion = optimizer
        elif not hasattr(self, '_optimizer_diffusion') or self._optimizer_diffusion is None:
            self._optimizer_diffusion = torch.optim.Adam(self._network.parameters(),
                                                         lr=learning_rate)

        device = next(self._network.parameters()).device

        # Prepare the dataset
        if isinstance(x, DataLoader):
            dataloader = x
            # Try to infer shape from dataset
            for batch in dataloader:
                self._validate_and_adapt_shape(batch)
                break
        else:
            # Validate and adapt shape
            self._validate_and_adapt_shape(x)

            # Handle different input formats
            if isinstance(x, tuple):
                # x is tuple: (data, labels)
                if len(x) == 2:
                    x_data, target_or_labels = x
                else:
                    x_data = x[0]
                    target_or_labels = None
            else:
                x_data = x
                target_or_labels = y

            if target_or_labels is None:
                raise ValueError("Labels (y) must be provided for diffusion training")

            # Convert to numpy arrays
            x_data = numpy.array(x_data)

            # Convert to tensors
            x_data = torch.from_numpy(x_data).float()

            # Add channel dimension if needed
            if len(x_data.shape) == 2:
                x_data = x_data.unsqueeze(-1)

            # Handle labels - convert to one-hot if needed
            if isinstance(target_or_labels, numpy.ndarray):
                target_or_labels = torch.from_numpy(target_or_labels)

            if len(target_or_labels.shape) == 1:
                # Convert to one-hot
                num_classes = int(target_or_labels.max()) + 1
                y_one_hot = torch.zeros(len(target_or_labels), num_classes)
                y_one_hot[torch.arange(len(target_or_labels)), target_or_labels.long()] = 1
                target_or_labels = y_one_hot

            # Create dataset
            dataset = TensorDataset(x_data, target_or_labels)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        # Calculate steps per epoch if not provided
        if steps_per_epoch is None:
            try:
                steps_per_epoch = len(dataloader)
            except:
                steps_per_epoch = 100  # Default fallback

        # Prepare validation data if provided
        val_dataloader = None
        if validation_data is not None:
            if isinstance(validation_data, DataLoader):
                val_dataloader = validation_data
            else:
                x_val, y_val = validation_data
                x_val = numpy.array(x_val)
                x_val = torch.from_numpy(x_val).float()

                if len(x_val.shape) == 2:
                    x_val = x_val.unsqueeze(-1)

                if isinstance(y_val, numpy.ndarray):
                    y_val = torch.from_numpy(y_val)

                if len(y_val.shape) == 1:
                    num_classes = int(y_val.max()) + 1
                    y_val_one_hot = torch.zeros(len(y_val), num_classes)
                    y_val_one_hot[torch.arange(len(y_val)), y_val.long()] = 1
                    y_val = y_val_one_hot

                val_dataset = TensorDataset(x_val, y_val)
                val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize callbacks
        if callbacks:
            for callback in callbacks:
                if not hasattr(callback, 'params') or callback.params is None:
                    callback.params = {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "steps": steps_per_epoch,
                        "samples": len(dataloader.dataset) if hasattr(dataloader, 'dataset') else 0,
                        "verbose": verbose,
                        "metrics": ["loss", "avg_loss"]
                    }

                if not hasattr(callback, 'data') or callback.data is None:
                    callback.data = {"start_time": time.time()}
                elif "start_time" not in callback.data:
                    callback.data["start_time"] = time.time()

                if hasattr(callback, 'on_train_begin'):
                    callback.on_train_begin()

        # History to store metrics
        history = {'loss': [], 'avg_loss': []}
        if validation_data is not None:
            history['val_loss'] = []

        self.train()
        best_loss = float('inf')

        for epoch in range(initial_epoch, epochs):
            epoch_loss = 0.0
            num_batches = 0

            if verbose == 1:
                print(f"\nEpoch {epoch + 1}/{epochs}")

            step = 0
            for batch_data, batch_labels in dataloader:
                step += 1
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)

                loss_dict = self.train_step(batch_data, batch_labels)

                current_loss = loss_dict["Diffusion_loss"]
                epoch_loss += current_loss
                num_batches += 1

                if verbose == 1:
                    progress = int(50 * step / steps_per_epoch)
                    bar = '=' * progress + '>' + '.' * (50 - progress - 1)
                    print(f'\r[{bar}] {step}/{steps_per_epoch} - loss: {current_loss:.4f}',
                          end='', flush=True)

                if step >= steps_per_epoch:
                    break

            avg_loss = epoch_loss / num_batches

            self._training_history['epoch'].append(epoch + 1)
            self._training_history['loss'].append(epoch_loss)
            self._training_history['avg_loss'].append(avg_loss)

            history['loss'].append(epoch_loss)
            history['avg_loss'].append(avg_loss)

            if verbose == 1:
                print(f' - avg_loss: {avg_loss:.4f}')
            elif verbose == 2:
                print(f'Epoch {epoch + 1}/{epochs} - avg_loss: {avg_loss:.4f}')

            if avg_loss < best_loss:
                best_loss = avg_loss

            # Validation
            if val_dataloader is not None and (epoch + 1) % validation_freq == 0:
                val_loss = self._evaluate_validation(val_dataloader, validation_steps, device)
                history['val_loss'].append(val_loss)

                if verbose >= 1:
                    print(f' - val_loss: {val_loss:.4f}')

            # Callbacks
            if callbacks is not None:
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, {'loss': avg_loss, 'avg_loss': avg_loss})

            # Handle legacy callbacks
            if callback_model_monitor:
                try:
                    if callable(callback_model_monitor):
                        callback_model_monitor(epoch, avg_loss)
                    elif hasattr(callback_model_monitor, 'on_epoch_end'):
                        callback_model_monitor.on_epoch_end(epoch, {'loss': avg_loss})
                except Exception as e:
                    if verbose >= 1:
                        print(f"Warning: Could not call model monitor callback: {e}")

            # Handle early stopping
            if use_early_stop and callback_early_stop:
                try:
                    should_stop = False
                    if callable(callback_early_stop):
                        should_stop = callback_early_stop(epoch, avg_loss)
                    elif hasattr(callback_early_stop, 'on_epoch_end'):
                        should_stop = callback_early_stop.on_epoch_end(epoch, {'loss': avg_loss})

                    if should_stop:
                        if verbose >= 1:
                            print("\n" + "=" * 80)
                            print("EARLY STOPPING TRIGGERED")
                            print("=" * 80 + "\n")
                        break
                except Exception as e:
                    if verbose >= 1:
                        print(f"Warning: Could not call early stop callback: {e}")

        # Call on_train_end
        if callbacks:
            for callback in callbacks:
                if hasattr(callback, 'on_train_end'):
                    callback.on_train_end()

        # Return history object
        class History:
            def __init__(self, history_dict):
                self.history = history_dict

        return History(history)

    def _evaluate_validation(self, val_dataloader, validation_steps=None, device=None):
        """
        Evaluate the model on validation data with automatic shape handling.

        Args:
            val_dataloader: Validation data loader
            validation_steps: Number of validation steps
            device: Device to run evaluation on

        Returns:
            Average validation loss
        """
        if device is None:
            device = next(self._network.parameters()).device

        self.eval()
        val_losses = []
        step = 0

        with torch.no_grad():
            for batch_data, batch_labels in val_dataloader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)

                loss_dict = self.train_step(batch_data, batch_labels)
                val_losses.append(loss_dict["Diffusion_loss"])

                step += 1
                if validation_steps is not None and step >= validation_steps:
                    break

        self.train()
        return numpy.mean(val_losses) if val_losses else 0.0

    def generate_data(self, labels, batch_size):
        """
        Generates synthetic data by reversing the diffusion process.

        Args:
            labels (torch.Tensor): Class labels used to condition the generated data.
            batch_size (int): Number of data samples to generate in a single batch.

        Returns:
            numpy.ndarray: Generated synthetic data samples.
        """
        device = next(self._network.parameters()).device

        model_output_shape = self._network._output_shape

        synthetic_data = torch.randn(labels.shape[0], model_output_shape, 1,
                                     dtype=torch.float32, device=device)

        labels_vector = labels.unsqueeze(-1) if len(labels.shape) == 2 else labels

        self._network.eval()
        with torch.no_grad():
            for time_step in reversed(range(0, self._time_steps)):
                array_time = torch.full((labels_vector.shape[0],), time_step,
                                        dtype=torch.long, device=device)

                predicted_noise = self._network(synthetic_data, array_time, labels_vector)

                synthetic_data = self._gdf_util.p_sample(
                    predicted_noise[0] if isinstance(predicted_noise, tuple) else predicted_noise,
                    synthetic_data,
                    array_time,
                    clip_denoised=True
                )

        self._network.train()

        generated_data = self._crop_tensor_to_original_size(
            synthetic_data.cpu().numpy(),
            self._original_shape
        )

        return generated_data

    @staticmethod
    def _crop_tensor_to_original_size(tensor: numpy.ndarray, original_size: int) -> numpy.ndarray:
        """
        Crops the input tensor along the second dimension to match the original size.

        Args:
            tensor (np.ndarray): A 3D NumPy array of shape (X, Y, Z)
            original_size (int): The desired size for the second dimension (Y)

        Returns:
            np.ndarray: A cropped 3D tensor with shape (X, original_size, Z)
        """
        if tensor.ndim != 3:
            raise ValueError(f"Expected 3D tensor (X, Y, Z), got shape: {tensor.shape}")

        current_size = tensor.shape[1]

        if current_size <= original_size:
            return tensor

        return tensor[:, :original_size, :]

    def _padding_input_tensor(self, input_tensor):
        """
        Pads the input tensor along the feature dimension to match the expected input shape.

        Args:
            input_tensor (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Padded tensor
        """
        input_tensor = input_tensor.float()

        if hasattr(self._network, 'module'):
            target_dimension = self._network.module._output_shape
        else:
            target_dimension = self._network._output_shape

        current_dimension = input_tensor.shape[-2]

        padding_needed = max(0, target_dimension - current_dimension)

        if padding_needed == 0:
            return input_tensor

        pad = (0, 0, 0, padding_needed)
        padded_tensor = F.pad(input_tensor, pad, mode='constant', value=0)

        return padded_tensor

    def get_samples(self, number_samples_per_class):
        """
        Generates synthetic data samples for each class.

        Args:
            number_samples_per_class (dict): Dictionary with 'classes' and 'number_classes' keys

        Returns:
            dict: Generated samples keyed by class label.
        """
        generated_data = {}
        device = next(self._network.parameters()).device

        for label_class, number_instances in number_samples_per_class["classes"].items():
            labels = torch.zeros(number_instances, number_samples_per_class["number_classes"],
                                 device=device)
            labels[:, label_class] = 1

            generated_samples = self.generate_data(labels, batch_size=64)

            generated_samples = numpy.rint(numpy.squeeze(generated_samples, axis=-1))

            generated_data[label_class] = generated_samples

        return generated_data

    def get_training_history(self):
        """
        Returns the training history.

        Returns:
            dict: Dictionary containing epoch, loss, and avg_loss lists
        """
        return self._training_history

    def save_model(self, directory, file_name):
        """
        Save the model weights with shape information.

        Args:
            directory (str): Directory where models will be saved
            file_name (str): Base file name for saving models
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        first_unet_path = os.path.join(directory, f"{file_name}_first_unet.pth")
        second_unet_path = os.path.join(directory, f"{file_name}_second_unet.pth")

        torch.save(self._network.state_dict(), first_unet_path)
        torch.save(self._second_unet_model.state_dict(), second_unet_path)

        # Save shape information
        shape_info = {
            'input_shape': self._input_shape,
            'inferred_shape': self._inferred_shape,
            'output_shape': self._output_shape,
            'original_shape': self._original_shape,
            'time_steps': self._time_steps,
            'ema': self._ema,
            'margin': self._margin
        }
        shape_file = os.path.join(directory, f"{file_name}_shape_info.json")
        with open(shape_file, 'w') as f:
            json.dump(shape_info, f)

        print(f"Models saved to {directory}")

    def load_models(self, directory, file_name):
        """
        Load the model weights with shape information.

        Args:
            directory (str): Directory where models are stored
            file_name (str): Base file name for loading models
        """
        first_unet_path = os.path.join(directory, f"{file_name}_first_unet.pth")
        second_unet_path = os.path.join(directory, f"{file_name}_second_unet.pth")

        self._network.load_state_dict(torch.load(first_unet_path))
        self._second_unet_model.load_state_dict(torch.load(second_unet_path))

        # Load shape information if available
        shape_file = os.path.join(directory, f"{file_name}_shape_info.json")
        if os.path.exists(shape_file):
            with open(shape_file, 'r') as f:
                shape_info = json.load(f)
                self._input_shape = tuple(shape_info.get('input_shape', ()))
                self._inferred_shape = tuple(shape_info.get('inferred_shape', ()))
                self._output_shape = shape_info.get('output_shape', self._output_shape)
                self._original_shape = shape_info.get('original_shape', self._original_shape)
                self._time_steps = shape_info.get('time_steps', self._time_steps)
                self._ema = shape_info.get('ema', self._ema)
                self._margin = shape_info.get('margin', self._margin)

        print(f"Models loaded from {directory}")

    # ========== PROPERTIES ==========

    @property
    def ema(self) -> Any:
        """Get the Exponential Moving Average (EMA) model."""
        return self._ema

    @ema.setter
    def ema(self, value: Any) -> None:
        """Set the Exponential Moving Average (EMA) model."""
        self._ema = value

    @property
    def margin(self) -> float:
        """Get the margin value."""
        return self._margin

    @margin.setter
    def margin(self, value: float) -> None:
        """Set the margin value."""
        if value <= 0:
            raise ValueError("Margin must be positive")
        self._margin = value

    @property
    def gdf_util(self) -> Any:
        """Get the Gradient Descent Filter utility."""
        return self._gdf_util

    @gdf_util.setter
    def gdf_util(self, value: Any) -> None:
        """Set the Gradient Descent Filter utility."""
        self._gdf_util = value

    @property
    def time_steps(self) -> int:
        """Get the number of diffusion time steps."""
        return self._time_steps

    @time_steps.setter
    def time_steps(self, value: int) -> None:
        """Set the number of diffusion time steps."""
        if value <= 0:
            raise ValueError("Time steps must be positive")
        self._time_steps = value

    @property
    def train_stage(self) -> str:
        """Get the current training stage."""
        return self._train_stage

    @train_stage.setter
    def train_stage(self, value: str) -> None:
        """Set the current training stage."""
        self._train_stage = value

    @property
    def network(self) -> Any:
        """Get the primary U-Net model."""
        return self._network

    @network.setter
    def network(self, value: Any) -> None:
        """Set the primary U-Net model."""
        self._network = value

    @property
    def second_unet_model(self) -> Any:
        """Get the secondary U-Net model."""
        return self._second_unet_model

    @second_unet_model.setter
    def second_unet_model(self, value: Any) -> None:
        """Set the secondary U-Net model."""
        self._second_unet_model = value

    @property
    def optimizer_diffusion(self) -> Any:
        """Get the diffusion model optimizer."""
        return self._optimizer_diffusion

    @optimizer_diffusion.setter
    def optimizer_diffusion(self, value: Any) -> None:
        """Set the diffusion model optimizer."""
        self._optimizer_diffusion = value

    @property
    def optimizer_autoencoder(self) -> Any:
        """Get the autoencoder optimizer."""
        return self._optimizer_autoencoder

    @optimizer_autoencoder.setter
    def optimizer_autoencoder(self, value: Any) -> None:
        """Set the autoencoder optimizer."""
        self._optimizer_autoencoder = value

    @property
    def input_shape(self):
        """Get the current input shape."""
        return self.get_input_shape()

    @property
    def output_shape(self):
        """Get the output shape."""
        return self._output_shape

    @output_shape.setter
    def output_shape(self, value):
        """Set the output shape."""
        self._output_shape = value