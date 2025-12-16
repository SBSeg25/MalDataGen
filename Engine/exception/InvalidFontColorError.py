class InvalidFontColorError(Exception):
    """exception raised for invalid font color."""

    def __init__(self, message="Invalid font_color. It must be a string."):
        super().__init__(message)