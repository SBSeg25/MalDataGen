class InvalidConfusionMatrixError(Exception):
    def __init__(self, message="Invalid confusion matrix."):
        self.message = message
        super().__init__(self.message)
