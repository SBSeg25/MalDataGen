class InvalidPlotFilenameError(Exception):
    def __init__(self, message="Invalid plot filename."):
        self.message = message
        super().__init__(self.message)