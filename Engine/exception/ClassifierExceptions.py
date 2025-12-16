#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2022/06/01'
__last_update__ = '2023/08/03'
__credits__ = ['unknown']


class BoltzmannActivationValidationError(Exception):
    """
    Custom exception for activation validation errors in the BoltzmannActivation class.
    """
    def __init__(self, name, message="Activation validation error."):
        self.name = name
        self.message = message
        super().__init__(self.message)


class BoltzmannInputValidationError(Exception):
    """
    Custom exception for activation validation errors in the BoltzmannActivation class.
    """
    def __init__(self, message="Input validation error."):
        self.message = message
        super().__init__(self.message)


class BoltzmannPredictionValidationError(Exception):
    """
    Custom exception for activation validation errors in the BoltzmannActivation class.
    """
    def __init__(self, message="Prediction validation error."):
        self.message = message
        super().__init__(self.message)


class BoltzmannPhaseValidationError(Exception):
    """
    Custom exception for phase validation errors in the BoltzmannPhase class.
    """
    def __init__(self, name, message="Phase validation error."):
        self.name = name
        self.message = message
        super().__init__(self.message)


class BoltzmannSampleValidationError(Exception):
    """
    Custom exception for sample validation errors in the BoltzmannSample class.
    """
    def __init__(self, message="Sample validation error."):
        self.message = message
        super().__init__(self.message)


class BootstrapAggregatingPredictionError(Exception):
    """
    Custom exception for errors during prediction in the BootstrapAggregating class.
    """
    def __init__(self, message="Error during prediction."):
        self.message = message
        super().__init__(self.message)


class BootstrapAggregatingTrainingError(Exception):
    """
    Custom exception for errors during training in the BootstrapAggregating class.
    """
    def __init__(self, message="Error during training."):
        self.message = message
        super().__init__(self.message)


class GaussianFilterError(Exception):
    """
    Custom exception for errors related to Gaussian filter in image processing.
    """
    def __init__(self, message="Error on define filter."):
        self.message = message
        super().__init__(self.message)


class GradientBoostingEmptyDataError(Exception):
    """
    Custom exception for errors related to empty datasets in Gradient Boosting.
    """
    def __init__(self, message="Error: Datasets is empty."):
        self.message = message
        super().__init__(self.message)


class GradientBoostingIncompatibleShapesError(Exception):
    """
    Custom exception for errors related to incompatible input shapes in Gradient Boosting.
    """
    def __init__(self, message="Error: Input dimension is not compatible"):
        self.message = message
        super().__init__(self.message)


class GradientBoostingPredictionError(Exception):
    """
    Custom exception for errors during prediction in Gradient Boosting.
    """
    def __init__(self, message="An error occurred during prediction"):
        self.message = message
        super().__init__(self.message)


class GradientBoostingTrainingError(Exception):
    """
    Custom exception for errors during training in Gradient Boosting.
    """
    def __init__(self, message="An error occurred during training"):
        self.message = message
        super().__init__(self.message)


class KMeansPredictionError(Exception):
    """
    Custom exception for errors during prediction in K-Means clustering.
    """
    def __init__(self, message="An error occurred during prediction"):
        self.message = message
        super().__init__(self.message)


class KMeansTrainingError(Exception):
    """
    Custom exception for errors during training in K-Means clustering.
    """
    def __init__(self, message="An error occurred during training"):
        self.message = message
        super().__init__(self.message)


class LogisticRegressionPredictionError(Exception):
    """
    Custom exception for errors during prediction in Logistic Regression.
    """
    def __init__(self, message="An error occurred during prediction"):
        self.message = message
        super().__init__(self.message)


class LogisticRegressionTrainingError(Exception):
    """
    Custom exception for errors during training in Logistic Regression.
    """
    def __init__(self, message="An error occurred during training"):
        self.message = message
        super().__init__(self.message)


class NaiveBayesPredictionError(Exception):
    """
    Custom exception for errors during prediction in Naive Bayes.
    """
    def __init__(self, message="An error occurred during prediction"):
        self.message = message
        super().__init__(self.message)


class NaiveBayesTrainingError(Exception):
    """
    Custom exception for errors during training in Naive Bayes.
    """
    def __init__(self, message="An error occurred during training"):
        self.message = message
        super().__init__(self.message)


class QuadraticDiscriminatorPredictionError(Exception):
    """
    Custom exception for errors during prediction in Quadratic Discriminator.
    """
    def __init__(self, message="An error occurred during prediction"):
        self.message = message
        super().__init__(self.message)


class QuadraticDiscriminatorTrainingError(Exception):
    """
    Custom exception for errors during training in Quadratic Discriminator.
    """
    def __init__(self, message="An error occurred during training"):
        self.message = message
        super().__init__(self.message)


class RandomForestPredictionError(Exception):
    """
    Custom exception for errors during prediction in Random Forest.
    """
    def __init__(self, message="An error occurred during prediction"):
        self.message = message
        super().__init__(self.message)


class RandomForestTrainingError(Exception):
    """
    Custom exception for errors during training in Random Forest.
    """
    def __init__(self, message="An error occurred during training"):
        self.message = message
        super().__init__(self.message)


class SupportVectorMachinePredictionError(Exception):
    """
    Custom exception for errors during prediction in support Vector Machine (SVM).
    """
    def __init__(self, message="An error occurred during prediction"):
        self.message = message
        super().__init__(self.message)


class SupportVectorMachineTrainingError(Exception):
    """
    Custom exception for errors during training in support Vector Machine (SVM).
    """
    def __init__(self, message="An error occurred during training"):
        self.message = message
        super().__init__(self.message)
