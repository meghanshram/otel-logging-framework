__version__ = "1.0.2"

from .otel_logger import (
    configure_logger,
    log_info,
    log_error,
    log_debug,
    log_warning,
    otel_trace,
    otel_span,
    otel_log
)