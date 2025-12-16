class InvalidBackgroundColorError(Exception):
    """exception raised for invalid background color."""

    def __init__(self, message="Invalid background_color. It must be a string."):
        super().__init__(message)