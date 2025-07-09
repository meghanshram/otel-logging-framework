import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from otel_logger import (
    otel_trace,
    configure_logger,
    otel_log,
    log_info,
    log_error,
    otel_span,
)

# Initialize logger
logger = configure_logger(config_file="logger_config.yaml")

# Log a message
log_info("Test log message", attributes={"user_id": "test123"})

# Shutdown
logger.close()