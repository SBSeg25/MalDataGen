#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
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
    import sys
    import json
    import numpy
    import psutil
    import logging
    import platform

    import subprocess
    import tensorflow

    from tensorflow.keras.callbacks import Callback

except ImportError as error:
    print(error)
    sys.exit(-1)


class ResourceMonitorCallback(Callback):
    """
    Callback for monitoring system resources during Keras model training.

    This callback monitors resources such as CPU, GPU, RAM, storage, temperature, and network
    and records them in a JSON file at user-defined intervals.

    Parameters:
        @json_file_path (str): Path to the JSON file where the data will be saved.
        @save_interval (int): Interval of epochs for saving the data. By default, it is 1 (every epoch).
    """

    def __init__(self, json_file_path, k_fold, save_interval=1):
        super(ResourceMonitorCallback, self).__init__()
        self.json_file_path = '{}/monitor_resource_{}_fold.json'.format(json_file_path, k_fold)
        self.save_interval = save_interval
        self.machine_info = self._get_machine_info()
        self.epoch_counter = 0

    def _get_machine_info(self):
        """
        Collects basic system information at the start of training.

        Returns:
            dict: Dictionary containing platform details, Python version, TensorFlow version,
                  CPU, memory, disk information, and placeholders for epoch statistics.
        """
        try:
            machine_info = {
                "platform": platform.platform(),
                "tensorflow_version": tensorflow.__version__,
                "numpy_version": numpy.__version__,
                "python_version": platform.python_version(),
                "cpu_info": self._get_cpu_info(),
                "memory_info": self._get_memory_info(),
                "disk_info": self._get_disk_info(),
                "epoch": [],
                "cpu_percent": {"mean": [], "per_core": []},
                "ram_usage": [],
                "swap_usage": [],
                "storage_available": [],
                "cpu_temperature": {"mean": [], "per_core": []},
                "gpu_temperature": [],
                "network_bandwidth": [],
                "network_transfer_rate": []
            }
            return machine_info

        except Exception as e:
            logging.error(f'Failed to collect machine info: {e}')
            return {}

    @staticmethod
    def _get_cpu_info():
        """
        Retrieves detailed CPU specifications.
        """
        try:
            cpu_info = subprocess.check_output(['lscpu']).decode()
            logging.info('CPU info collected.')
            return cpu_info

        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logging.error(f'Error collecting CPU info: {e}')
            return "Unknown CPU info"

    @staticmethod
    def _get_memory_info():
        """
        Retrieves memory statistics.
        """
        try:
            memory_info = subprocess.check_output(['free', '-h']).decode()
            logging.debug('Memory info collected.')
            return memory_info

        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logging.error(f'Error collecting memory info: {e}')
            return "Unknown memory info"

    @staticmethod
    def _get_disk_info():
        """
        Retrieves disk storage statistics.
        """
        try:
            disk_info = subprocess.check_output(['df', '-h']).decode()
            logging.debug('Disk info collected.')
            return disk_info

        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logging.error(f'Error collecting disk info: {e}')
            return "Unknown disk info"

    def on_train_begin(self, logs=None):
        """
        Saves the initial machine configuration when training begins.
        """
        try:

            with open(self.json_file_path, 'w') as json_file:
                json.dump(self.machine_info, json_file, indent=4)
                json_file.write('\n')

            logging.info('Training started and initial machine info saved.')

        except IOError as e:
            logging.error(f'Error writing to JSON file at training start: {e}')

    def on_epoch_end(self, epoch, logs=None):
        """
        Logs system resource usage at the end of each epoch.
        """
        self.epoch_counter += 1
        logging.info(f'Epoch {epoch} ended. Logs: {logs}')

        if self.epoch_counter % self.save_interval == 0:
            try:
                cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
                cpu_percent_mean = psutil.cpu_percent(interval=None)
                ram_usage = psutil.virtual_memory().percent
                swap_usage = psutil.swap_memory().percent
                storage_available = psutil.disk_usage('/').percent

                data = {
                    "epoch": epoch,
                    "cpu_percent": {"mean": cpu_percent_mean, "per_core": cpu_percent},
                    "ram_usage": ram_usage,
                    "swap_usage": swap_usage,
                    "storage_available": storage_available,
                }

                with open(self.json_file_path, 'r+') as json_file:
                    try:
                        data_list = json.load(json_file)
                    except json.JSONDecodeError as e:
                        logging.error(f'Error loading JSON file: {e}')
                        data_list = self.machine_info  # Initialize with machine info

                    for key, value in data.items():
                        if isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                data_list[key][subkey].append(subvalue)
                        else:
                            data_list[key].append(value)

                    json_file.seek(0)
                    json.dump(data_list, json_file, indent=4)
                    json_file.truncate()
                logging.info(f'Resource data for epoch {epoch} saved successfully.')

            except IOError as e:
                logging.error(f'Error writing resource data for epoch {epoch} to JSON: {e}')

    @staticmethod
    def _get_gpu_info():

        try:
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'])
            gpu_data = result.decode('utf-8').strip().split('\n')
            gpu_utilization = int(gpu_data[0].split(',')[0])
            gpu_memory_usage = int(gpu_data[0].split(',')[1])
            logging.debug('GPU info collected.')
            return gpu_utilization, gpu_memory_usage

        except (FileNotFoundError, subprocess.CalledProcessError) as e:

            logging.error(f'Error collecting GPU info: {e}')
            return None, None

    @staticmethod
    def _get_system_temperatures():

        try:

            cpu_temp_result = subprocess.check_output(['sensors'])
            cpu_temp_data = cpu_temp_result.decode('utf-8').split('\n')
            cpu_temps = [line.split(':')[-1].strip().split()[0] for line in cpu_temp_data if 'Core' in line]
            cpu_temperature = {
                "mean": numpy.mean([float(temp.replace('°C', '')) for temp in cpu_temps]),
                "per_core": [float(temp.replace('°C', '')) for temp in cpu_temps]
            }
            logging.debug('CPU temperature info collected.')

        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logging.error(f'Error collecting CPU temperature info: {e}')
            cpu_temperature = {"mean": None, "per_core": None}

        try:

            gpu_temp_result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'])
            gpu_temperature = int(gpu_temp_result.decode('utf-8').strip())
            logging.debug('GPU temperature info collected.')

        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logging.error(f'Error collecting GPU temperature info: {e}')
            gpu_temperature = None

        return cpu_temperature, gpu_temperature

    @staticmethod
    def _get_network_info():

        try:
            network_stats = psutil.net_io_counters()
            bandwidth = network_stats.bytes_sent + network_stats.bytes_recv
            transfer_rate = network_stats.bytes_sent / (1024 * 1024)  # In MB
            logging.debug('Network info collected.')
            return bandwidth, transfer_rate

        except Exception as e:
            logging.error(f'Error collecting network info: {e}')
            return None, None
