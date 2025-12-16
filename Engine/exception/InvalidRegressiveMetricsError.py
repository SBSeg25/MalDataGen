class InvalidRegressiveMetricsError(Exception):
    def __init__(self, message="Invalid regressive metrics."):
        self.message = message
        super().__init__(self.message)