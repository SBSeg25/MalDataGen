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
    import numpy
    import logging

except ImportError as error:
    print(error)
    sys.exit(-1)

class TrTr:

    def evaluation_TR_TR(self, dictionary_data):

        """
        Evaluates the performance of classifiers trained on synthetic data and evaluated on real data.
        This method trains classifiers using synthetic data (generated data) and evaluates them using
        real evaluation data. The binary classification metrics for each classifier are calculated and
        logged for each fold.

        Args:
            dictionary_data (dict): A dictionary containing real evaluation data. The key 'x_evaluation_real'
                                    holds the features for evaluation, and the key 'y_evaluation_real' holds the true labels.
        """
        # Logging the evaluation strategy
        logging.info(f"\tTR-TR: train on real, test on real")

        # Initialize empty lists for labels and data
        labels, data = [], []

        # # Logging the total number of generated synthetic samples to be processed
        # total_samples = sum(len(samples) for samples in self.data_generated.values())
        # logging.info(f"\t\tTR-RS: Total number of samples to be saved: {total_samples}")

        # Iterate through each class and its corresponding generated synthetic samples
        # for label_class, generated_samples in self.data_generated.items():
        logging.info(f"")
        logging.info(f"")
        logging.info(f"")
        logging.info(f"#################################################################################")
        logging.info(f"\tTR-TR: ")

            # Add the label (class) for each generated sample
            #labels.extend([label_class] * len(generated_samples))

            # Add generated samples to the data list
            #data.extend(generated_samples)

        # Train classifiers using the real training data and corresponding labels
        classifiers = self.get_trained_classifiers(dictionary_data['x_training_real'],
                                                    numpy.squeeze(dictionary_data['y_training_real'], axis=-1),
                                                    numpy.float32, self.get_number_columns())

        # Evaluate the classifiers on synthetic data for each classifier instancevaluation
        for classifier_name, classifier_instances in zip(self._dictionary_classifiers_name, classifiers):
            # Predict the labels using the trained classifier on the synthetic data
            label_predicted = classifier_instances.predict(dictionary_data['x_evaluation_real'])
            logging.info("")
            logging.info(f"\t\t\t TR-TR {classifier_name}")
            # Calculate and log the binary classification metrics (such as accuracy, precision, recall, etc.)
            self.get_binary_metrics(numpy.array(dictionary_data['y_evaluation_real']), numpy.array(label_predicted),
                                    "TR-TR", classifier_name, self.fold_number + 1)
            
        
        data_real = numpy.array(dictionary_data['x_training_real'])
        n_real, m_real = data_real.shape
        logging.info(f"x_real_eva size:{data_real.size} shape: {n_real}, {m_real}")

        data_real_b = numpy.array(dictionary_data['x_evaluation_real'])
        n_synt, m_synt = data_real_b.shape
        logging.info(f"x_real_val size:{data_real_b.size} shape: {n_synt}, {m_synt}")

        assert m_synt == m_real, \
            f"Synthetic data has {m_synt} columns != real columns {m_real}"

        if n_real > n_synt:
            data_real = data_real[:n_synt]
        elif n_real < n_synt:
            data_real_b = data_real_b[:n_real]
        else:
            pass

        self.get_distance_metrics(data_real, data_real_b, "R-R", self.fold_number + 1)