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

    from sklearn.exceptions import NotFittedError
    from sklearn.tree import DecisionTreeClassifier

except ImportError as error:
    print(error)
    sys.exit(-1)

class DecisionTree:
    """
    A Decision Tree classifier wrapper that encapsulates the configuration and training process.

    Attributes:
        _decision_tree_criterion (str): Criterion used to measure the quality of a split.
        _decision_tree_max_depth (int): The maximum depth of the tree.
        _decision_tree_max_feature (str): The number of features to consider when looking for the best split.
        _decision_tree_max_leaf (int): The maximum number of leaf nodes.

    Methods:
        get_model(x_samples_training, y_samples_training, dataset_type, input_dataset_shape):
            Trains a Decision Tree model using the provided training samples and labels.
    """

    def __init__(self, arguments):
        """
        Initializes the DecisionTree class with hyperparameters.

        Args:
            arguments: An object containing hyperparameters for the Decision Tree model.
        """
        self._decision_tree_criterion = arguments.decision_tree_criterion
        self._decision_tree_max_depth = arguments.decision_tree_max_depth
        self._decision_tree_max_feature = arguments.decision_tree_max_features
        self._decision_tree_max_leaf = arguments.decision_tree_max_leaf_nodes

        logging.debug(f"DecisionTree initialized with criterion={self._decision_tree_criterion}, "
                      f"max_depth={self._decision_tree_max_depth}, max_features={self._decision_tree_max_feature}, "
                      f"max_leaf_nodes={self._decision_tree_max_leaf}")

    def get_model(self, x_samples_training, y_samples_training, dataset_type, input_dataset_shape):
        """
        Trains a Decision Tree classifier using the provided training samples and labels.

        Args:
            x_samples_training (array-like): The training feature samples.
            y_samples_training (array-like): The training labels corresponding to the samples.
            dataset_type: The data type for the training samples (e.g., float32).
            input_dataset_shape: The shape of the input dataset (used for logging purposes).

        Returns:
            DecisionTreeClassifier: A trained Decision Tree classifier instance.

        Raises:
            ValueError: If training samples or labels are empty or do not match in shape.
            Exception: For any other issues encountered during model fitting.
        """
        logging.info("Starting training classifier: DECISION TREE")

        try:
            # Convert input samples to the specified data type
            logging.debug(f"Converting training samples to numpy arrays with type {dataset_type}.")
            x_samples_training = numpy.array(x_samples_training, dtype=dataset_type)
            y_samples_training = numpy.array(y_samples_training, dtype=dataset_type)

            logging.debug(f"Training data shape: {x_samples_training.shape}, Labels shape: {y_samples_training.shape}")

            # Validate training data
            if x_samples_training.size == 0 or y_samples_training.size == 0:
                raise ValueError("Training samples or labels are empty.")
            if x_samples_training.shape[0] != y_samples_training.shape[0]:
                raise ValueError("The number of samples in training data and labels do not match.")

            # Create and train the Decision Tree classifier
            instance_model_classifier = DecisionTreeClassifier(
                criterion=self._decision_tree_criterion,
                max_depth=self._decision_tree_max_depth,
                max_features=self._decision_tree_max_feature,
                max_leaf_nodes=self._decision_tree_max_leaf
            )

            logging.info("Fitting the Decision Tree model to the training data.")
            instance_model_classifier.fit(x_samples_training, y_samples_training)
            logging.info("Finished training DECISION TREE model.")
            return instance_model_classifier

        except ValueError as ve:
            logging.error(f"ValueError occurred: {ve}")
            raise  # Re-raise to propagate the exception
        except Exception as e:
            logging.error(f"An error occurred during model training: {e}")
            raise  # Re-raise to propagate the exception

    def set_decision_tree_criterion(self, decision_tree_criterion):
        """
        Sets the criterion used to measure the quality of a split.

        Args:
            decision_tree_criterion (str): New criterion (e.g., 'gini' or 'entropy').
        """
        logging.debug(f"Setting new Decision Tree criterion: criterion={decision_tree_criterion}")
        self._decision_tree_criterion = decision_tree_criterion

    def set_decision_tree_max_depth(self, decision_tree_max_depth):
        """
        Sets the maximum depth of the Decision Tree.

        Args:
            decision_tree_max_depth (int): New maximum depth.
        """
        logging.debug(f"Setting new Decision Tree max depth: max_depth={decision_tree_max_depth}")
        self._decision_tree_max_depth = decision_tree_max_depth

    def set_decision_tree_max_feature(self, decision_tree_max_feature):
        """
        Sets the maximum number of features to consider when looking for the best split.

        Args:
            decision_tree_max_feature (str): New max feature option (e.g., 'auto', 'sqrt').
        """
        logging.debug(f"Setting new Decision Tree max features: max_features={decision_tree_max_feature}")
        self._decision_tree_max_feature = decision_tree_max_feature

    def set_decision_tree_max_leaf(self, decision_tree_max_leaf):
        """
        Sets the maximum number of leaf nodes in the Decision Tree.

        Args:
            decision_tree_max_leaf (int): New maximum number of leaf nodes.
        """
        logging.debug(f"Setting new Decision Tree max leaf nodes: max_leaf_nodes={decision_tree_max_leaf}")
        self._decision_tree_max_leaf = decision_tree_max_leaf

    def predict(self, model, x_samples):
        """
        Makes predictions using the trained Decision Tree model.

        Args:
            model (DecisionTreeClassifier): The trained Decision Tree classifier.
            x_samples (array-like): The samples for which to make predictions.

        Returns:
            array: Predictions for the provided samples.

        Raises:
            NotFittedError: If the model has not been trained yet.
            ValueError: If the input samples are empty or do not match the model's expected input shape.
        """
        try:
            if not isinstance(model, DecisionTreeClassifier):
                raise NotFittedError("The model is not fitted. Please train the model first.")

            x_samples = numpy.array(x_samples, dtype=numpy.float32)

            if x_samples.size == 0:
                raise ValueError("Input samples for prediction cannot be empty.")

            predictions = model.predict(x_samples)
            logging.debug(f"Predictions made for samples: {predictions}")
            return predictions

        except ValueError as ve:
            logging.error(f"ValueError during prediction: {ve}")
            raise  # Re-raise to propagate the exception
        except NotFittedError as nfe:
            logging.error(f"Model not fitted error: {nfe}")
            raise  # Re-raise to propagate the exception
        except Exception as e:
            logging.error(f"An error occurred during prediction: {e}")
            raise  # Re-raise to propagate the exception