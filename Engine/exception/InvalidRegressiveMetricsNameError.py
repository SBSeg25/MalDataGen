class InvalidRegressiveMetricsNameError(Exception):
    def __init__(self, message="Invalid regressive metrics names."):
        self.message = message
        super().__init__(self.message)