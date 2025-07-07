from otel_logger import configure_logger, log_info

# Initialize logger
logger = configure_logger(config_file="logger_config.yaml")

# Log a message
log_info("Test log message", attributes={"user_id": "test123"})

# Shutdown
logger.close()