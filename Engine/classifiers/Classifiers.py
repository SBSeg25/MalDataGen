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
    import logging

    from Engine.classifiers.algorithms.AdaBoost import AdaBoost

    from Engine.classifiers.algorithms.NaiveBayes import NaiveBayes
    from Engine.classifiers.algorithms.Perceptron import Perceptron

    from Engine.classifiers.algorithms.RandomForest import RandomForest
    from Engine.classifiers.algorithms.DecisionTree import DecisionTree

    from Engine.classifiers.algorithms.GaussianProcess import GaussianProcess

    from Engine.classifiers.algorithms.GradientBoosting import GradientBoosting
    from Engine.classifiers.algorithms.KMeansClustering import KMeansClustering

    from Engine.classifiers.algorithms.KNearestNeighbors import KNearestNeighbors

    from Engine.classifiers.algorithms.LinearRegression import LinearRegressionModel

    from Engine.classifiers.algorithms.SupportVectorMachine import SupportVectorMachine
    from Engine.classifiers.algorithms.SpectralClustering import SpectralClusteringModel

    from Engine.classifiers.algorithms.StochasticGradientDescent import StochasticGradientDescent
    from Engine.classifiers.algorithms.QuadraticDiscriminant import QuadranticDiscriminantAnalysis

except ImportError as error:
    print(error)
    sys.exit(-1)

def import_classifiers(function):

    def wrapper(self, *args, **kwargs):
        Classifiers.__init__(self, self.arguments)
        return function(self, *args, **kwargs)

    return wrapper


class Classifiers:
    dictionary_classifiers_name = ["RandomForest",
                                             "SupportVectorMachine",
                                             "KNN",
                                             #"GaussianPrecess",
                                             "DecisionTree",
                                             #"AdaBoost", #TODO corrigir
                                             "NaiveBayes",
                                             #"QuadraticDiscriminant",
                                             #"Perceptron",
                                             "GradientBoosting",
                                             #"KMeansClustering",
                                             #"LinearRegression",
                                             "StochasticGradientDescent",
                                             #"SpectralClusteringModel"
                                             ]
    
    
    
    def __init__(self, arguments):

        dictionary_classifiers = {"RandomForest": RandomForest(arguments),
                                        "SupportVectorMachine": SupportVectorMachine(arguments),
                                        "KNN": KNearestNeighbors(arguments),
                                        "GaussianPrecess": GaussianProcess(arguments),
                                        "DecisionTree": DecisionTree(arguments),
                                        # "AdaBoost": AdaBoost(arguments),
                                        "NaiveBayes": NaiveBayes(arguments),
                                        "QuadraticDiscriminant": QuadranticDiscriminantAnalysis(arguments),
                                        "Perceptron": Perceptron(arguments),
                                        "GradientBoosting": GradientBoosting(arguments),
                                        "KMeansClustering": KMeansClustering(arguments),
                                        "LinearRegression": LinearRegressionModel(arguments),
                                        "StochasticGradientDescent": StochasticGradientDescent(arguments),
                                        "SpectralClusteringModel": SpectralClusteringModel(arguments)}
        
        self._dictionary_classifiers_name = list()
        self._dictionary_classifiers = {}

        for c in arguments.classifier: 
            if c in self.dictionary_classifiers_name and c in  dictionary_classifiers.keys():
                self._dictionary_classifiers_name.append(c)
                self._dictionary_classifiers[c] = dictionary_classifiers[c]


    def get_trained_classifiers(self, x_samples_training, y_samples_training, dataset_type, input_dataset_shape):

        logging.info("")
        logging.info("Starting Training classifiers")
        list_instance_classifiers = []

        for classifier_algorithm in self._dictionary_classifiers_name:

            classifier_model = self._dictionary_classifiers[classifier_algorithm]
            list_instance_classifiers.append(classifier_model.get_model(x_samples_training,
                                                                        y_samples_training,
                                                                        dataset_type, input_dataset_shape))


        return list_instance_classifiers
