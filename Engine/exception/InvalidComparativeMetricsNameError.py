class InvalidComparativeMetricsNameError(Exception):
    def __init__(self, message="Invalid comparative metrics names."):
        self.message = message
        super().__init__(self.message)