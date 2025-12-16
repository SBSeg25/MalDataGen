class InvalidFontWeightError(Exception):
    """exception raised for invalid font weight."""

    def __init__(self, message="Invalid font_weight. It must be a string."):
        super().__init__(message)
