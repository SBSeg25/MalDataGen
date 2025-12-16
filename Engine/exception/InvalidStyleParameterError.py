class InvalidStyleParameterError(Exception):
    """exception raised for invalid style-related parameters."""

    def __init__(self, parameter_name):
        message = f"Invalid value for style-related parameter: {parameter_name}."
        super().__init__(message)
