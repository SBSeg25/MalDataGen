class AccuracyError(Exception):
    def __init__(self, name, message="Unknown error."):
        self.name = name
        self.message = message
        super().__init__(self.message)
    pass
class AreaUnderCurveError(Exception):
    """Custom exception for AUC-related errors."""


class CohensKappaError(Exception):
    """Custom exception for Cohen's Kappa coefficient related errors."""


class EuclideanDistanceError(Exception):
    """Custom exception for Euclidean distance related errors."""


class F1ScoreError(Exception):
    def __init__(self, name, message="Unknown error."):
        self.name = name
        self.message = message
        super().__init__(self.message)

class FalseNegativeError(Exception):
    """Custom exception for False Negative related errors."""

class FalsePositiveError(Exception):
    """Custom exception for False Positive related errors."""

class FalsePositiveRateError(Exception):
    """Custom exception for False Positive Rate (FPR) related errors."""


class HellingerDistanceError(Exception):
    """Custom exception for Hellinger distance related errors."""


class JaccardError(Exception):
    """Custom exception for Jaccard similarity coefficient related errors."""


class JensenShannonDivergenceError(Exception):
    """Custom exception for Jensen-Shannon Divergence related errors."""


class KLDivergenceError(Exception):
    """Custom exception for Kullback-Leibler Divergence related errors."""


class LogLikelihoodError(Exception):
    """Custom exception for Log-Likelihood related errors."""


class MatthewsCorrelationCoefficientError(Exception):
    """Custom exception for Matthews Correlation Coefficient (MCC) related errors."""


class MAEError(Exception):
    """Custom exception for Mean Absolute Error (MAE) related errors."""


class MeanSquareEError(Exception):
    """Custom exception for Mean Squared Error (MSE) related errors."""


class MinkowskiDistanceError(Exception):
    """Custom exception for Minkowski distance related errors."""


class YoudenIndexError(Exception):
    """Custom exception for Youden's Index (J) related errors."""


class ManhattanDistanceError(Exception):
    """Custom exception for Manhattan distance related errors."""


class RecallError(Exception):
    def __init__(self, name, message="Unknown error."):
        self.name = name
        self.message = message
        super().__init__(self.message)

class WassersteinDistanceError(Exception):
    def __init__(self, name, message="Unknown error."):
        self.name = name
        self.message = message
        super().__init__(self.message)


class SpecificityError(Exception):
    def __init__(self, name, message="Unknown error."):
        self.name = name
        self.message = message
        super().__init__(self.message)

class TrueNegativeError(Exception):
    """Custom exception for True Negative related errors."""

class TrueNegativeRateError(Exception):
    """Custom exception for True Negative Rate (TNR) related errors."""

class TruePositiveError(Exception):
    """Custom exception for True Positive related errors."""

class PrecisionError(Exception):
    def __init__(self, name, message="Unknown error."):
        self.name = name
        self.message = message
        super().__init__(self.message)


class KolmogorovSmirnovError(Exception):
    pass


class HammingDistanceError(Exception):
    def __init__(self, name, message="Unknown error in Hamming distance calculation."):
        self.name = name
        self.message = message
        super().__init__(self.message)

class JaccardDistanceError(Exception):
    def __init__(self, name, message="Unknown error in Jaccard distance calculation."):
        self.name = name
        self.message = message
        super().__init__(self.message)

class PermutationTestError(Exception):
    def __init__(self, name, message="Unknown error in Permutation Test."):
        self.name = name
        self.message = message
        super().__init__(self.message)