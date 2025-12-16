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
    import GPUtil
    import psutil
    import logging
    import platform
    import tensorflow
    from typing import List
    from typing import Optional
    from typing import Union

except ImportError as error:
    print(error)
    sys.exit(-1)

# Configure module-level logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class HardwareManager:
    """
    Hardware configuration and resource management utility for TensorFlow.

    This class provides a flexible interface to configure parallelism (CPU thread pools),
    control GPU visibility and memory allocation behavior, enforce CPU affinity, and inspect
    hardware availability for high-performance and reproducible TensorFlow workloads.

    Typical use cases include:
        - Restricting TensorFlow to specific CPU or GPU devices
        - Dynamically controlling memory growth for GPU devices
        - Enforcing deterministic thread allocation via intra/inter-op settings
        - Collecting system-level hardware diagnostics

    Example:
        >>> manager = HardwareManager(use_gpu=True, visible_devices=[0], enable_memory_growth=True)
        >>> manager.configure()

        >>> manager = HardwareManager(use_gpu=False, inter_op_threads=4, intra_op_threads=8, cpu_affinity=[0, 2, 4, 6])
        >>> manager.configure()

        >>> print(manager.get_system_info())
        >>> print(manager.get_gpu_info())
    """

    def __init__(
        self,
        use_gpu: bool = True,
        visible_devices: Optional[Union[List[int], str]] = None,
        enable_memory_growth: bool = True,
        log_device_placement: bool = False,
        inter_op_threads: Optional[int] = None,
        intra_op_threads: Optional[int] = None,
        cpu_affinity: Optional[List[int]] = None,
    ):
        """
        Initialize the hardware manager with the specified configuration.

        Args:
            use_gpu: Whether to enable GPU acceleration (default: True)
            visible_devices: List of GPU indices to make visible, 'all' for all GPUs, or None
            enable_memory_growth: Enable dynamic memory allocation for visible GPUs (default: True)
            log_device_placement: Enable TensorFlow device placement logging (default: False)
            inter_op_threads: Number of inter-operation parallelism threads
            intra_op_threads: Number of intra-operation parallelism threads
            cpu_affinity: Optional list of CPU core indices to which the process will be pinned
        """
        self.use_gpu = use_gpu
        self.visible_devices = visible_devices
        self.enable_memory_growth = enable_memory_growth
        self.log_device_placement = log_device_placement
        self.inter_op_threads = inter_op_threads
        self.intra_op_threads = intra_op_threads
        self.cpu_affinity = cpu_affinity

    def configure(self):
        """
        Apply the configured hardware settings.

        The configuration is applied in the following order:
        1. CPU threading parameters
        2. CPU core affinity (if specified)
        3. GPU visibility and memory growth behavior
        4. Optional TensorFlow device placement logging
        """
        self._configure_cpu_threads()
        self._configure_cpu_affinity()
        if not self.use_gpu:
            self._disable_all_gpus()
        else:
            self._enable_gpus()

        if self.log_device_placement:
            tensorflow.debugging.set_log_device_placement(True)

    def _configure_cpu_threads(self):
        """
        Configure TensorFlow CPU thread pools for intra-op and inter-op parallelism.
        """
        if self.intra_op_threads is not None:
            tensorflow.config.threading.set_intra_op_parallelism_threads(self.intra_op_threads)
            logger.info(f"Set intra-op threads to {self.intra_op_threads}")

        if self.inter_op_threads is not None:
            tensorflow.config.threading.set_inter_op_parallelism_threads(self.inter_op_threads)
            logger.info(f"Set inter-op threads to {self.inter_op_threads}")

    def _configure_cpu_affinity(self):
        """
        Apply CPU affinity settings to the current process using psutil.
        """
        if self.cpu_affinity:
            try:
                psutil_instance = psutil.Process(os.getpid())
                psutil_instance.cpu_affinity(self.cpu_affinity)
                logger.info(f"CPU affinity set to cores: {self.cpu_affinity}")
            except Exception as e:
                logger.warning(f"Failed to set CPU affinity: {e}")

    @staticmethod
    def _disable_all_gpus():
        """
        Disable all GPU devices by setting CUDA visibility to -1.
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            tensorflow.config.set_visible_devices([], 'GPU')
            logger.info("All GPUs disabled. Running on CPU.")
        except RuntimeError as e:
            logger.warning(f"Could not disable GPUs: {e}")

    def _enable_gpus(self):
        """
        Enable and configure GPU devices according to the provided settings.
        Applies memory growth if requested.
        """
        gpus = tensorflow.config.list_physical_devices('GPU')
        if not gpus:
            logger.warning("No GPUs found.")
            return

        if self.visible_devices == "all" or self.visible_devices is None:
            visible = gpus
        elif isinstance(self.visible_devices, list):
            try:
                visible = [gpus[i] for i in self.visible_devices]
            except IndexError as e:
                logger.error(f"Invalid GPU index: {e}")
                return
        else:
            logger.error("Invalid format for visible_devices")
            return

        try:
            tensorflow.config.set_visible_devices(visible, 'GPU')
            logger.info(f"Visible GPUs: {[gpu.name for gpu in visible]}")
        except RuntimeError as e:
            logger.warning(f"Runtime error setting visible GPUs: {e}")

        if self.enable_memory_growth:
            for gpu in visible:
                try:
                    tensorflow.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"Enabled memory growth on: {gpu.name}")
                except Exception as e:
                    logger.warning(f"Memory growth error on {gpu.name}: {e}")

    @staticmethod
    def get_gpu_info() -> List[dict]:
        """
        Retrieve detailed information about all available GPUs using GPUtil.

        Returns:
            A list of dictionaries, one per GPU, including:
                - id: GPU index
                - name: GPU model name
                - memory_total_MB: Total memory
                - memory_used_MB: Used memory
                - memory_free_MB: Free memory
                - load: Utilization percentage (0-1)
                - temperature_C: Current temperature in Celsius
        """
        gpus = GPUtil.getGPUs()
        return [{
            'id': gpu.id,
            'name': gpu.name,
            'memory_total_MB': gpu.memoryTotal,
            'memory_used_MB': gpu.memoryUsed,
            'memory_free_MB': gpu.memoryFree,
            'load': gpu.load,
            'temperature_C': gpu.temperature
        } for gpu in gpus]

    @staticmethod
    def get_system_info() -> dict:
        """
        Retrieve basic system information including CPU and memory.

        Returns:
            Dictionary with keys:
                - platform: OS name
                - platform_version: OS version
                - architecture: CPU architecture
                - cpu_count: Number of physical cores
                - cpu_logical_count: Number of logical cores
                - total_memory_GB: Total RAM
                - available_memory_GB: Available RAM
        """
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'cpu_count': psutil.cpu_count(logical=False),
            'cpu_logical_count': psutil.cpu_count(logical=True),
            'total_memory_GB': round(psutil.virtual_memory().total / (1024 ** 3), 2),
            'available_memory_GB': round(psutil.virtual_memory().available / (1024 ** 3), 2)
        }

    @staticmethod
    def list_available_gpus() -> List[str]:
        """
        List all physical GPU devices visible to TensorFlow.

        Returns:
            List of physical GPU device names.
        """
        return [gpu.name for gpu in tensorflow.config.list_physical_devices('GPU')]

    @staticmethod
    def list_logical_gpus() -> List[str]:
        """
        List all logical GPU devices available to TensorFlow.

        Returns:
            List of logical GPU device names.
        """
        return [gpu.name for gpu in tensorflow.config.list_logical_devices('GPU')]

    @staticmethod
    def is_gpu_available() -> bool:
        """
        Check if TensorFlow detects any available GPU.

        Returns:
            True if at least one GPU is available, otherwise False.
        """
        return len(tensorflow.config.list_physical_devices('GPU')) > 0

    # --- Setters ---

    def set_use_gpu(self, use_gpu: bool):
        """
        Set whether to enable GPU usage.

        Args:
            use_gpu: Boolean flag to enable or disable GPU.
        """
        self.use_gpu = use_gpu
        logger.info(f"use_gpu set to {self.use_gpu}")

    def set_visible_devices(self, devices: Union[List[int], str, None]):
        """
        Define which GPU devices should be made visible.

        Args:
            devices: List of GPU indices, 'all', or None.
        """
        self.visible_devices = devices
        logger.info(f"visible_devices set to {self.visible_devices}")

    def set_enable_memory_growth(self, enable: bool):
        """
        Enable or disable memory growth for GPUs.

        Args:
            enable: True to enable, False to disable.
        """
        self.enable_memory_growth = enable
        logger.info(f"enable_memory_growth set to {self.enable_memory_growth}")

    def set_log_device_placement(self, log: bool):
        """
        Enable or disable TensorFlow device placement logging.

        Args:
            log: True to enable logging, False to disable.
        """
        self.log_device_placement = log
        logger.info(f"log_device_placement set to {self.log_device_placement}")

    def set_inter_op_threads(self, num_threads: Optional[int]):
        """
        Set the number of inter-op parallelism threads.

        Args:
            num_threads: Integer number of threads or None.
        """
        self.inter_op_threads = num_threads
        logger.info(f"inter_op_threads set to {self.inter_op_threads}")

    def set_intra_op_threads(self, num_threads: Optional[int]):
        """
        Set the number of intra-op parallelism threads.

        Args:
            num_threads: Integer number of threads or None.
        """
        self.intra_op_threads = num_threads
        logger.info(f"intra_op_threads set to {self.intra_op_threads}")

    def set_cpu_affinity(self, core_indices: Optional[List[int]]):
        """
        Define CPU core affinity (process pinning).

        Args:
            core_indices: List of CPU core indices.
        """
        self.cpu_affinity = core_indices
        logger.info(f"cpu_affinity set to {self.cpu_affinity}")
