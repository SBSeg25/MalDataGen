class InvalidComparativeMetricsError(Exception):
    def __init__(self, message="Invalid comparative metrics."):
        self.message = message
        super().__init__(self.message)