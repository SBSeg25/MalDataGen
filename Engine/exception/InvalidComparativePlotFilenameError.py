class InvalidComparativePlotFilenameError(Exception):
    def __init__(self, message="Invalid comparative plot filename."):
        self.message = message
        super().__init__(self.message)