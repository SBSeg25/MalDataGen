class InvalidLegendLabelsError(Exception):
    """exception raised for invalid legend labels."""

    def __init__(self, message="Invalid legend_labels. It must be a list of strings."):
        super().__init__(message)