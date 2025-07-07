import logging

debug_logger = logging.getLogger("otel_debug")
debug_logger.setLevel(logging.DEBUG)
if not debug_logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    debug_logger.addHandler(console_handler)
