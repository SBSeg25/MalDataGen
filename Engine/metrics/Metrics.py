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
    import pandas as pd 
     
    import time 

    from Engine.metrics.binary.Recall import Recall

    from Engine.metrics.binary.F1_Score import F1Score
    from Engine.metrics.binary.Accuracy import Accuracy

    from Engine.metrics.binary.Precision import Precision

    from Engine.metrics.binary.Specificity import Specificity

    from Engine.metrics.binary.TruePositive import TruePositive
    from Engine.metrics.binary.TrueNegative import TrueNegative

    from Engine.metrics.binary.FalsePositive import FalsePositive
    from Engine.metrics.binary.FalseNegative import FalseNegative

    from Engine.metrics.binary.AreaUnderCurve import AreaUnderCurve

    from Engine.metrics.binary.MeanSquaredError import MeanSquareError
    from Engine.metrics.binary.TrueNegativeRate import TrueNegativeRate

    from Engine.metrics.binary.FalsePositiveRate import FalsePositiveRate
    from Engine.metrics.binary.MeanAbsoluteError import MeanAbsoluteError

    from Engine.metrics.distance.EuclideanDistance import EuclideanDistance
    from Engine.metrics.distance.HellingerDistance import HellingerDistance
    from Engine.metrics.distance.ManhattanDistance import ManhattanDistance

    from Engine.metrics.distance.HammingDistance import HammingDistance
    from Engine.metrics.distance.JaccardDistance import JaccardDistance
    from Engine.metrics.distance.PermutationTest import PermutationTest


except ImportError as error:
    print(error)
    sys.exit(-1)

def import_metrics(function):
    """
    Decorator to create an instance of the metrics class
    before executing the wrapped function.

    Parameters:
        function (callable): The function to be wrapped.

    Returns:
        callable: The wrapped function that initializes metrics.
    """
    def wrapper(self, *args, **kwargs):
        # Create an instance of metrics, passing the arguments from the instance
        Metrics.__init__(self, self.arguments)
        # Call the wrapped function with the metrics instance and other arguments
        return function(self, *args, **kwargs)

    return wrapper

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (numpy.float32, numpy.float64)):
            return float(obj)
        elif isinstance(obj, (numpy.int32, numpy.int64)):
            return int(obj)
        return super().default(obj)
    
class Metrics:
    """
    Class to calculate and manage various evaluation metrics for machine learning models.
    Initializes dictionaries to store binary metrics, distance metrics, and area under curve metrics.

    Attributes:
        _dictionary_binary_metrics (dict): Dictionary of binary evaluation metric instances.
        _dictionary_distance_metrics (dict): Dictionary of distance metric instances.
        _dictionary_area_under_curve (dict): Dictionary for area under curve metric instances.
        _classifier_list (list): List of classifiers used in the evaluation.
        _dictionary_metrics (dict): Dictionary to store metrics for different classifiers and evaluation types.
    """

    def __init__(self, arguments):
        """
        Initializes the metrics class by creating instances of various metrics
        and setting up the metrics dictionary based on provided arguments.

        Parameters:
            arguments (Namespace): Command-line arguments containing settings for the metrics.
        """
        # Initialize dictionaries for binary metrics with their corresponding instances
        self._dictionary_binary_metrics = {
            Accuracy.__name__: Accuracy(),
            Precision.__name__: Precision(),
            Recall.__name__: Recall(),
            F1Score.__name__: F1Score(),
            Specificity.__name__: Specificity(),
            FalsePositiveRate.__name__: FalsePositiveRate(),
            TrueNegativeRate.__name__: TrueNegativeRate(),
            MeanSquareError.__name__: MeanSquareError(),
            MeanAbsoluteError.__name__: MeanAbsoluteError(),
            TruePositive.__name__: TruePositive(),
            FalsePositive.__name__: FalsePositive(),
            TrueNegative.__name__: TrueNegative(),
            FalseNegative.__name__: FalseNegative(),
        }

        # Initialize dictionaries for distance metrics
        self._dictionary_distance_metrics = {
            EuclideanDistance.__name__ : EuclideanDistance(),
            HellingerDistance.__name__ : HellingerDistance(),
            ManhattanDistance.__name__ : ManhattanDistance(),
            HammingDistance.__name__ : HammingDistance(),
            JaccardDistance.__name__ : JaccardDistance(),
        }

        # Initialize dictionary for area under curve metrics
        self._dictionary_area_under_curve = {"AreaUnderCurve": AreaUnderCurve()}

        self._classifier_list = list()
        for c in arguments.classifier:
            if c in self.dictionary_classifiers_name:
                self._classifier_list.append(c)

        # Initialize the metrics dictionary based on the provided arguments
        self.__initialize_dictionary(arguments)
        self._process = psutil.Process()

    def __initialize_dictionary(self, arguments):
        """
        Initializes the metrics dictionary to store evaluation metrics for different classifiers
        and evaluation types, including cross-validation folds.

        Parameters:
            arguments (Namespace): Command-line arguments containing the number of folds.
        """

        self.list_classifier_metrics = self._dictionary_binary_metrics.keys()

        # # List of distribution metrics
        self.list_distance_metrics = self._dictionary_distance_metrics.keys()
        self.list_efficiency_metrics =  ["Process_CPU_%", "Process_Memory_MB", "System_CPU_%", "System_Memory_MB", "System_Memory_%", "Time_training_ms", "Time_generating_ms"]
        self.list_sdv_metrics = ["diagnostic", "quality"]

        # Initialize the main metrics dictionary for different evaluation types and classifiers
        self._dictionary_metrics = self._dictionary_metrics  | {
            "TS-TR": {
                classifier: {
                    **{
                        f'{fold}-Fold': {metric: 0 for metric in self.list_classifier_metrics}
                        for fold in range(1, arguments.number_k_folds + 1)
                    },
                    'Summary': {
                        metric: {'mean': 0, 'std': 0} for metric in self.list_classifier_metrics
                    }
                } for classifier in self._classifier_list
            },
            "TR-TS": {
                classifier: {
                    **{
                        f"{fold}-Fold": {metric: 0 for metric in self.list_classifier_metrics}
                        for fold in range(1, arguments.number_k_folds + 1)
                    },
                    "Summary": {
                        metric: {'mean': 0, 'std': 0} for metric in self.list_classifier_metrics
                    }
                } for classifier in self._classifier_list
            },
           "TR-TR": {
               classifier: {
                   **{
                       f'{fold}-Fold': {metric: 0 for metric in self.list_classifier_metrics}
                       for fold in range(1, arguments.number_k_folds + 1)
                   },
                   'Summary': {
                       metric: {'mean': 0, 'std': 0} for metric in self.list_classifier_metrics
                   }
               } for classifier in self._classifier_list
           },

            "DistanceMetrics": {
                methodology: {
                **{
                    f'{fold}-Fold': {metric: 0 for metric in self.list_distance_metrics}
                    for fold in range(1, arguments.number_k_folds + 1)
                },
                'Summary': {
                    metric: {'mean': 0, 'std': 0} for metric in self.list_distance_metrics
                }
                } for methodology in [ "R-S", "R-R"]
            },

            "EfficiencyMetrics": {
                **{
                    f'{fold}-Fold': {metric: 0 for metric in self.list_efficiency_metrics}
                    for fold in range(1, arguments.number_k_folds + 1)
                },
                'Summary': {
                    metric: {'mean': 0, 'std': 0} for metric in self.list_efficiency_metrics
                }
            },

            "SDVMetrics": {
                **{
                    f'{fold}-Fold': {metric: 0 for metric in self.list_sdv_metrics}
                    for fold in range(1, arguments.number_k_folds + 1)
                },
                'Summary': {
                    metric: {'mean': 0, 'std': 0} for metric in self.list_sdv_metrics
                }
            }

        }


    def monitoring_start_training(self):
        self._time_start_training = time.perf_counter_ns()
        self._process_cpu_start = self._process.cpu_percent(interval=None)
        self._process_mem_start = self._process.memory_info().rss / (1024 * 1024)  #   MB
        self._system_cpu_start = self._process.cpu_percent(interval=None)
        self._system_mem_start = psutil.virtual_memory().used / (1000**2) #MB
        self._system_mem_start_perc = psutil.virtual_memory().percent



    def monitoring_stop_training(self, fold):
        self._time_end_training = time.perf_counter_ns()
        duration_ns  = self._time_end_training - self._time_start_training
        self._dictionary_metrics["EfficiencyMetrics"][f'{fold+1}-Fold']['Time_training_ms'] = duration_ns / 1_000_000
    

    def monitoring_start_generating(self):
        self._time_start_generating = time.perf_counter_ns()

    def monitoring_stop_generating(self, fold):
        self._time_end_generating = time.perf_counter_ns()
        duration_ns = self._time_end_generating - self._time_start_generating
        #self._dictionary_metrics["EfficiencyMetrics"][f'{fold+1}-Fold']['Time_generating_secs'] = duration.total_seconds()

        # Uso de CPU (percentual médio durante a execução)
        cpu_usage = self._process.cpu_percent(interval=None) / psutil.cpu_count()
        
        # Uso de memória (diferença entre início e fim)
        process_mem_end = self._process.memory_info().rss / (1000**2)  # Em MB
        process_mem_usage = process_mem_end - self._process_mem_start

        self._dictionary_metrics["EfficiencyMetrics"][f'{fold+1}-Fold'].update({
        'Time_generating_ms':  duration_ns / 1_000_000,
        'Process_CPU_%': cpu_usage,
        'Process_Memory_MB': process_mem_usage,
        'System_CPU_%': psutil.cpu_percent(interval=None),
        'System_Memory_MB': psutil.virtual_memory().used / (1000**2),
        'System_Memory_%': psutil.virtual_memory().percent,
        })

    def save_dictionary_to_json(self, output_file_results):
        """
        Saves the metrics dictionary to a JSON file.

        Parameters:
            output_file_results (str): The file path where the JSON will be saved.
        """

        try:
            # Open the specified output file in write mode
            with open(output_file_results, 'w') as json_file:
                # Write the metrics dictionary to the JSON file
                json.dump(self._dictionary_metrics, json_file, indent=4, cls=NumpyEncoder)
                print(f"Dictionary successfully saved to {output_file_results}")

        except Exception as e:
            # Print an error message if saving fails
            print(f"Error saving the dictionary: {e}")

    def update_mean_std_fold(self):
        """Updates the mean and standard deviation of evaluation metrics across all folds.

        This method calculates and stores the mean and standard deviation for each metric
        across all cross-validation folds, for each methodology and classifier combination.
        The results are stored back in the metrics dictionary under the 'Mean-Fold' entry.

        The method processes two methodologies ("TS-TR" and "TR-TS") and all classifiers
        stored in the dictionary. For each combination, it:

            1. Identifies all fold entries (keys ending with "-Fold")

            2. Computes mean and std for each metric across folds

            3. Stores the results in the 'Mean-Fold' dictionary structure

        Note:
            - Expects self._dictionary_metrics to be properly initialized
            - The 'Mean-Fold' entry must exist for each classifier-methodology pair
            - Modifies the dictionary in-place by adding mean/std values
        """

        for methodology in ["TS-TR", "TR-TS", "TR-TR"]:
            for classifier in self._dictionary_classifiers_name:

                # Get the metrics data for current methodology and classifier
                data = self._dictionary_metrics[methodology][classifier]

                # Identify all fold keys (excluding "Mean-Fold")
                folds = [key for key in data if key.endswith("-Fold")]

                # Calculate mean and std for each metric across folds
                for metric in data["Summary"]:

                    # Collect all values for this metric across folds
                    values = [data[fold][metric] for fold in folds]

                    # Compute statistics
                    mean_value = numpy.mean(values)
                    std_value = numpy.std(values)

                    # Store results in Mean-Fold structure
                    self._dictionary_metrics[methodology][classifier]["Summary"][metric]["mean"] = mean_value
                    self._dictionary_metrics[methodology][classifier]["Summary"][metric]["std"] = std_value
        
         
        data = self._dictionary_metrics["DistanceMetrics"]
        folds = [key for key in data["R-S"] if key.endswith("-Fold")]
        for methodology in ["R-S", "R-R"]:
            for metric in data[methodology]["Summary"].keys():
                values = [data[methodology][fold][metric] for fold in folds]
                data[methodology]["Summary"][metric]["mean"] = numpy.mean(values)
                data[methodology]["Summary"][metric]["std"] = numpy.std(values)

        data = self._dictionary_metrics["EfficiencyMetrics"]
        folds = [key for key in data if key.endswith("-Fold")]
        for metric in data["Summary"].keys():
            values = [data[fold][metric] for fold in folds]
            data["Summary"][metric]["mean"] = numpy.mean(values)
            data["Summary"][metric]["std"] = numpy.std(values)

    def get_binary_metrics(self, real_labels, predict_labels, evaluation_type, classifier, fold):
        """
        Calculates binary metrics using real and predicted labels and updates the metrics dictionary.

        Parameters:
            real_labels (array-like): The true labels.
            predict_labels (array-like): The predicted labels from the model.
            evaluation_type (str): The evaluation type (e.g., "TS-TR").
            classifier (str): The name of the classifier being evaluated.
            fold (str): The current fold number in cross-validation.
        """
         
        logging.info(f"\t\t\t Binary metrics")
        for metric_name, instance in self._dictionary_binary_metrics.items():
            # Calculate the metric and update the dictionary with the result
            self._dictionary_metrics[evaluation_type][classifier][f"{fold}-Fold"][metric_name] = (
                instance.get_metric(real_labels, predict_labels)
            )

    
    def get_distance_metrics(self, x_evaluation_real, x_evaluation_synthetic, evaluation_type, fold):
        """
        Calculates distance metrics between two distributions and updates the metrics dictionary.

        Parameters:
            real_distribution (array-like): The true distribution.
            synthetic_distribution (array-like): The synthetic distribution generated by the model.
            fold (str): The current fold number in cross-validation.
        """

        logging.info(f"\t\t\t Distance metrics")
        for metric_name, instance in self._dictionary_distance_metrics.items():
            # Calculate the distance metric and update the dictionary with the result
            self._dictionary_metrics["DistanceMetrics"][evaluation_type][f"{fold}-Fold"][metric_name] = (  
                instance.get_metric(x_evaluation_real, x_evaluation_synthetic)
            )

    def get_AUC_metric(self, real_label, synthetic_label_probability, evaluation_type, classifier, fold):
        """
        Calculates the Area Under Curve (AUC) metric and updates the metrics dictionary.

        Parameters:
            real_label (array-like): The true labels.
            synthetic_label_probability (array-like): The predicted probabilities from the model.
            evaluation_type (str): The evaluation type (e.g., "TS-TR").
            classifier (str): The name of the classifier being evaluated.
            fold (str): The current fold number in cross-validation.
        """
        for metric_name, instance in self._dictionary_area_under_curve.items():
            # Calculate the AUC and update the dictionary with the result
            self._dictionary_metrics[evaluation_type][classifier][fold][metric_name] = (
                instance.get_metric(real_label, synthetic_label_probability)
            )

