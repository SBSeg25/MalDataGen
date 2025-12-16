class InvalidFontFamilyError(Exception):
    """exception raised for invalid font family."""

    def __init__(self, message="Invalid font_family. It must be a string."):
        super().__init__(message)
