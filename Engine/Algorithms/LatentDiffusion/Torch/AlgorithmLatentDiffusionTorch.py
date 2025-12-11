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
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from typing import Any

except ImportError as error:
    print(error)
    sys.exit(-1)


class AlgorithmLatentDiffusionTorch(nn.Module):
    """
    Implements a diffusion process using UNet architectures for generating synthetic data.
    This model integrates an autoencoder and a diffusion network, enabling both data
    reconstruction and controlled generative modeling through Gaussian diffusion.

    This class supports exponential moving average (EMA) updates for stable training,
    multiple training stages, and customizable hyperparameters to adapt to different tasks.
    """

    def __init__(self, first_unet_model,
                 second_unet_model,
                 encoder_model_image,
                 decoder_model_image,
                 gdf_util,
                 optimizer_autoencoder,
                 optimizer_diffusion,
                 time_steps,
                 ema,
                 margin,
                 embedding_dimension,
                 train_stage='all'):
        """
        Initializes the DiffusionModel with provided sub-models, optimizers, and hyperparameters.
        """
        super(AlgorithmLatentDiffusionTorch, self).__init__()

        self._ema = ema
        self._margin = margin
        self._gdf_util = gdf_util
        self._time_steps = time_steps
        self._train_stage = train_stage
        self._network = first_unet_model
        self._second_unet_model = second_unet_model
        self._embedding_dimension = embedding_dimension
        self._encoder_model_data = encoder_model_image
        self._decoder_model_data = decoder_model_image
        self._optimizer_diffusion = optimizer_diffusion
        self._optimizer_autoencoder = optimizer_autoencoder
        self._loss_fn = nn.MSELoss()

    def set_stage_training(self, training_stage):
        """
        Sets the current training stage.

        Args:
            training_stage (str): New training stage ('all', 'diffusion', etc.).
        """
        self._train_stage = training_stage

    def train_step(self, data):
        """
        Performs a single training step.

        Args:
            data (tuple): A tuple containing input data and labels.

        Returns:
            dict: A dictionary with the computed loss for diffusion.
        """
        raw_data, label = data

        loss_encoder, loss_diffusion = None, None

        loss_diffusion = self.train_diffusion_model(raw_data, label)
        self.update_ema_weights()
        return {"Diffusion_loss": loss_diffusion.item() if loss_diffusion is not None else 0}

    def train_diffusion_model(self, data, ground_truth):
        """
        Performs a single training step for the diffusion model.

        This method applies the forward diffusion process (adding noise to the data),
        predicts the noise using the model, computes the loss, and updates the model weights.

        Args:
            data (torch.Tensor): Input data embeddings (e.g., image or text embeddings).
            ground_truth (torch.Tensor): Corresponding class labels or conditioning embeddings.

        Returns:
            torch.Tensor: The computed loss for this training step.
        """
        self._network.fit_model()

        # Labels (conditioning information) and input data embeddings
        embedding_label = ground_truth
        embedding_data_expanded = data

        # Batch size of the current data batch
        batch_size = data.shape[0]

        # Sample random time steps for each sample in the batch
        random_time_steps = torch.randint(0, self._time_steps, (batch_size,),
                                          dtype=torch.long, device=data.device)

        # Zero gradients
        self._optimizer_diffusion.zero_grad()

        # Sample random noise to add to the data
        random_noise = torch.randn_like(embedding_data_expanded)

        # Apply forward diffusion process (add noise based on the current time step t)
        embedding_with_noise = self._gdf_util.q_sample(embedding_data_expanded,
                                                       random_time_steps,
                                                       random_noise)

        # Predict noise using the diffusion model (network), conditioned on time and label
        predicted_noise = self._network(embedding_with_noise, random_time_steps, embedding_label)

        # Compute the loss by comparing the true noise with the predicted noise
        loss_diffusion = self._loss_fn(random_noise, predicted_noise)

        # Backpropagation
        loss_diffusion.backward()

        # Update weights
        self._optimizer_diffusion.step()

        # Return the computed diffusion loss for monitoring
        return loss_diffusion

    def update_ema_weights(self):
        """
        Updates the weights of the second UNet model using exponential moving average.
        Now properly handles dynamically created projection layers.

        FIXED VERSION: Handles dynamic decoder_projections dictionary properly.
        """
        with torch.no_grad():
            # First, ensure both models have the same dynamically created projection layers
            if hasattr(self._network, 'decoder_projections'):
                for key in self._network.decoder_projections.keys():
                    if key not in self._second_unet_model.decoder_projections:
                        # Copy the layer structure if it doesn't exist in EMA model
                        layer = self._network.decoder_projections[key]
                        self._second_unet_model.decoder_projections[key] = nn.Linear(
                            layer.in_features,
                            layer.out_features
                        ).to(layer.weight.device)
                        # Initialize with current weights
                        self._second_unet_model.decoder_projections[key].load_state_dict(
                            layer.state_dict()
                        )

            # Update all standard parameters
            for (name1, param1), (name2, param2) in zip(
                    self._network.named_parameters(),
                    self._second_unet_model.named_parameters()
            ):
                # Only update if shapes match
                if param1.shape == param2.shape:
                    param2.data.mul_(self._ema).add_(param1.data, alpha=1 - self._ema)

            # Update dynamically created projection layers
            for key in self._network.decoder_projections.keys():
                if key in self._second_unet_model.decoder_projections:
                    for param1, param2 in zip(
                            self._network.decoder_projections[key].parameters(),
                            self._second_unet_model.decoder_projections[key].parameters()
                    ):
                        param2.data.mul_(self._ema).add_(param1.data, alpha=1 - self._ema)

    def generate_data(self, labels, batch_size):
        """
        Generates synthetic data by reversing the diffusion process, starting from pure noise
        and iteratively denoising to create data samples conditioned on class labels.

        Args:
            labels (torch.Tensor): Class labels used to condition the generated data (e.g., class embeddings).
            batch_size (int): Number of data samples to generate in a single batch.

        Returns:
            numpy.ndarray: Generated synthetic data samples after reversing the diffusion process.
        """
        self._network.eval()
        device = next(self._network.parameters()).device

        # Ensure labels are on the correct device
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.float32, device=device)
        else:
            labels = labels.to(device)

        with torch.no_grad():
            # Start with random noise in the embedding space
            embedding_diffusion = torch.randn(
                labels.shape[0], self._embedding_dimension, 1,
                dtype=torch.float32, device=device
            )

            # Reshape labels to ensure they have the correct shape for conditioning
            labels_vector = labels.unsqueeze(-1)

            # Reverse the diffusion process by iterating over the time steps (from T to 0)
            for time_step in reversed(range(0, self._time_steps)):
                # Create an array with the current time step for each sample in the batch
                array_time = torch.full((labels_vector.shape[0],), time_step,
                                        dtype=torch.long, device=device)

                # Predict the noise at the current time step using the trained network
                predicted_noise = self._network(embedding_diffusion, array_time, labels_vector)

                # Apply the reverse diffusion step to remove noise from the embeddings
                embedding_diffusion = self._gdf_util.p_sample(predicted_noise, embedding_diffusion,
                                                              array_time, clip_denoised=True)

            # Use the decoder model to transform the denoised embeddings into real data samples
            self._decoder_model_data.eval()

            # Process in smaller batches if needed
            generated_samples = []
            decode_batch_size = max(1, batch_size // 4)

            for i in range(0, embedding_diffusion.shape[0], decode_batch_size):
                batch_embed = embedding_diffusion[i:i + decode_batch_size]
                batch_labels = labels_vector[i:i + decode_batch_size]

                generated_batch = self._decoder_model_data(batch_embed, batch_labels)
                generated_samples.append(generated_batch.cpu())

            generated_data = torch.cat(generated_samples, dim=0)

        # Return the generated data as numpy array
        return generated_data.numpy()

    @staticmethod
    def _crop_tensor_to_original_size(tensor: numpy.ndarray, original_size: int) -> numpy.ndarray:
        """
        Crops the input tensor along the second dimension (axis=1) to match the original size.
        """
        # Validate input dimensions
        if tensor.ndim != 3:
            raise ValueError(f"Expected 3D tensor (X, Y, Z), got shape: {tensor.shape}")

        current_size = tensor.shape[1]

        # No cropping needed
        if current_size <= original_size:
            return tensor

        # Slice the tensor along axis 1 (sequence length) to crop the excess at the end
        return tensor[:, :original_size, :]

    def _padding_input_tensor(self, input_tensor):
        """
        Pads the input tensor along the feature dimension to match the expected input shape
        required by the diffusion network.
        """
        # Ensure tensor is in float32 for consistency with model expectations
        input_tensor = input_tensor.float()

        # Get the target dimension from the network's expected input
        if hasattr(self._network, 'embedding_dimension'):
            target_dimension = self._network.embedding_dimension
        else:
            # Fallback to embedding dimension
            target_dimension = self._embedding_dimension

        # Get current dimension
        current_dimension = input_tensor.shape[-2]

        # Calculate padding needed
        padding_needed = max(0, target_dimension - current_dimension)

        # If no padding needed, return as-is
        if padding_needed == 0:
            return input_tensor

        # Apply padding: pad along the second-to-last dimension
        padded_tensor = F.pad(input_tensor, (0, 0, 0, padding_needed), mode='constant', value=0)

        return padded_tensor

    def get_samples(self, number_samples_per_class):
        """
        Generates synthetic data samples for each class, using the specified number of samples per class.

        Args:
            number_samples_per_class (dict): A dictionary where the "classes" key maps to a dictionary of class labels
                                             and their corresponding sample counts, and the "number_classes" key specifies
                                             the total number of classes.

        Returns:
            dict: A dictionary where keys are class labels and values are the generated samples for each class.
        """
        generated_data = {}
        device = next(self._network.parameters()).device

        # Iterate over each class and generate the specified number of samples
        for label_class, number_instances in number_samples_per_class["classes"].items():
            # Create one-hot encoded labels for the current class and number of instances
            label_samples_generated = F.one_hot(
                torch.tensor([label_class] * number_instances, dtype=torch.long),
                num_classes=number_samples_per_class["number_classes"]
            ).float().to(device)

            # Generate synthetic data using the diffusion model
            generated_samples = self.generate_data(label_samples_generated, batch_size=64)

            # Round the generated samples to ensure valid output format (e.g., pixel values)
            generated_samples = numpy.rint(generated_samples)

            # Store the generated samples for the current class
            generated_data[label_class] = generated_samples

        return generated_data

    def save_model(self, directory, file_name):
        """
        Save the encoder and decoder models.

        Args:
            directory (str): Directory where models will be saved.
            file_name (str): Base file name for saving models.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Construct file names for encoder and decoder models
        encoder_file_name = os.path.join(directory, f"{file_name}_encoder.pth")
        decoder_file_name = os.path.join(directory, f"{file_name}_decoder.pth")
        first_unet_file_name = os.path.join(directory, f"{file_name}_first_unet.pth")
        second_unet_file_name = os.path.join(directory, f"{file_name}_second_unet.pth")

        # Save encoder model
        torch.save(self._encoder_model_data.state_dict(), encoder_file_name)

        # Save decoder model
        torch.save(self._decoder_model_data.state_dict(), decoder_file_name)

        # Save first UNet model
        torch.save(self._network.state_dict(), first_unet_file_name)

        # Save second UNet model (EMA)
        torch.save(self._second_unet_model.state_dict(), second_unet_file_name)

        print(f"Models successfully saved to {directory}")

    def load_model(self, directory, file_name):
        """
        Load the encoder and decoder models.

        Args:
            directory (str): Directory where models are saved.
            file_name (str): Base file name for loading models.
        """
        # Construct file names
        encoder_file_name = os.path.join(directory, f"{file_name}_encoder.pth")
        decoder_file_name = os.path.join(directory, f"{file_name}_decoder.pth")
        first_unet_file_name = os.path.join(directory, f"{file_name}_first_unet.pth")
        second_unet_file_name = os.path.join(directory, f"{file_name}_second_unet.pth")

        # Load encoder model
        self._encoder_model_data.load_state_dict(torch.load(encoder_file_name))

        # Load decoder model
        self._decoder_model_data.load_state_dict(torch.load(decoder_file_name))

        # Load first UNet model
        self._network.load_state_dict(torch.load(first_unet_file_name))

        # Load second UNet model (EMA)
        self._second_unet_model.load_state_dict(torch.load(second_unet_file_name))

        print(f"Models successfully loaded from {directory}")

    @staticmethod
    def _save_model_to_json(model, file_path):
        """
        Save model architecture information to a JSON file.
        """
        try:
            model_info = {
                "model_class": model.__class__.__name__,
                "model_string": str(model)
            }

            with open(file_path, "w") as json_file:
                json.dump(model_info, json_file, indent=2)
            print(f"Model info successfully saved to {file_path}.")

        except Exception as e:
            error_message = f"Error occurred while saving model info: {str(e)}"
            with open(file_path, "w") as error_file:
                error_file.write(error_message)
            print(f"An error occurred and was saved to {file_path}: {error_message}")

    # Properties
    @property
    def ema(self) -> Any:
        """Get the Exponential Moving Average (EMA) decay rate."""
        return self._ema

    @ema.setter
    def ema(self, value: Any) -> None:
        """Set the Exponential Moving Average (EMA) decay rate."""
        self._ema = value

    @property
    def margin(self) -> float:
        """Get the margin value used in contrastive loss."""
        return self._margin

    @margin.setter
    def margin(self, value: float) -> None:
        """Set the margin value for contrastive loss."""
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
    def embedding_dimension(self) -> int:
        """Get the embedding dimension size."""
        return self._embedding_dimension

    @embedding_dimension.setter
    def embedding_dimension(self, value: int) -> None:
        """Set the embedding dimension size."""
        if value <= 0:
            raise ValueError("Embedding dimension must be positive")
        self._embedding_dimension = value

    @property
    def encoder_model_data(self) -> Any:
        """Get the image encoder model."""
        return self._encoder_model_data

    @encoder_model_data.setter
    def encoder_model_data(self, value: Any) -> None:
        """Set the image encoder model."""
        self._encoder_model_data = value

    @property
    def decoder_model_data(self) -> Any:
        """Get the image decoder model."""
        return self._decoder_model_data

    @decoder_model_data.setter
    def decoder_model_data(self, value: Any) -> None:
        """Set the image decoder model."""
        self._decoder_model_data = value

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