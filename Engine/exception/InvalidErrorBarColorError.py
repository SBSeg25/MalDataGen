class InvalidErrorBarColorError(Exception):
    """exception raised for invalid error bar color."""

    def __init__(self, message="Invalid error_bar_color. It must be a string."):
        super().__init__(message)