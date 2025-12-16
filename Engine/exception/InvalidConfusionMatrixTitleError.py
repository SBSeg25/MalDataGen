class InvalidConfusionMatrixTitleError(Exception):
    def __init__(self, message="Invalid confusion matrix title."):
        self.message = message
        super().__init__(self.message)