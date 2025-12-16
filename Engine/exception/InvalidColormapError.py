class InvalidColormapError(Exception):
    def __init__(self, message="Invalid colormap."):
        self.message = message
        super().__init__(self.message)