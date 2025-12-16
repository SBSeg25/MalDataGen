class InvalidBorderWidthFigureError(Exception):
    """exception raised for invalid border width of the figure."""

    def __init__(self, message="Invalid border_width_figure. It must be an integer."):
        super().__init__(message)