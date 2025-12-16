class InvalidBarColorError(Exception):
    """exception raised for invalid bar color."""

    def __init__(self, message="Invalid bar_color. It must be a string."):
        super().__init__(message)
