class PipelineError(RuntimeError):
    """Base error for pipeline startup and runtime failures."""


class CaptureAccessError(PipelineError):
    """Raised when screen capture cannot be initialized."""


class ModelInitializationError(PipelineError):
    """Raised when the detector cannot be created or warmed up."""
