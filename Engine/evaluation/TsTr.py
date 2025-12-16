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

    from sklearn.utils import shuffle   
except ImportError as error:
    print(error)
    sys.exit(-1)

class TsTr:

    def evaluation_TS_TR(self, dictionary_data, synthetic_data):

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
        logging.info(f"")
        logging.info(f"")
        logging.info(f"")
        logging.info(f"#################################################################################")
        logging.info(f"\tTS-TR: train on synthetic, test on real")

        # Initialize empty lists for labels and data
        labels, data = [], []

        # Logging the total number of generated synthetic samples to be processed
        total_samples = sum(len(samples) for samples in self.data_generated.values())
        logging.info(f"\t\tTS-TR: Total number of samples to be saved: {total_samples}")

        # Iterate through each class and its corresponding generated synthetic samples
        for label_class, generated_samples in synthetic_data.items():
            
            logging.info(f"\t\tTS-TR: Processing {len(generated_samples)} samples for label class {label_class}.")

            # Add the label (class) for each generated sample
            labels.extend([label_class] * len(generated_samples))

            # Add generated samples to the data list
            data.extend(generated_samples)

        shuffled_data, shuffled_labels = shuffle(data, numpy.array(labels), random_state=42)
        # Train classifiers using the generated synthetic data and corresponding labels
        classifiers = self.get_trained_classifiers(shuffled_data, shuffled_labels, numpy.float32, self.get_number_columns())

        # Evaluate the classifiers on real data for each classifier instance
        for classifier_name, classifier_instances in zip(self._dictionary_classifiers_name, classifiers):
            # Predict the labels using the trained classifier on the real evaluation data
            label_predicted = classifier_instances.predict(dictionary_data['x_evaluation_real'])

            logging.info("")
            logging.info(f"\t\tTS-TR {classifier_name}")
            # Calculate and log the binary classification metrics (such as accuracy, precision, recall, etc.)
            self.get_binary_metrics(numpy.squeeze(dictionary_data['y_evaluation_real'], axis=-1)[:total_samples],
                                    numpy.array(label_predicted)[:total_samples],
                                    "TS-TR", classifier_name, self.fold_number + 1)
            
        
        data_real = numpy.array(dictionary_data['x_training_real'])
        n_real, m_real = data_real.shape
        logging.info(f"x_real_eva size:{data_real.size} shape: {n_real}, {m_real}")

        data_synthetic = numpy.array(data)
        n_synt, m_synt = data_synthetic.shape
        logging.info(f"x_synt size:{data_synthetic.size} shape: {n_synt}, {m_synt}")

        assert m_synt == m_real, \
            f"Synthetic data has {m_synt} columns != real columns {m_real}"

        if n_real > n_synt:
            data_real = data_real[:n_synt]
        elif n_real < n_synt:
            data_synthetic = data_synthetic[:n_real]
        else:
            pass

        self.get_distance_metrics(data_real, data_synthetic, "R-S", self.fold_number + 1)