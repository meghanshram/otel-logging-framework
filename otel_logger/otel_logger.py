import json
import logging
import os
import time
import traceback
import uuid
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, Optional, List
import threading
import inspect
import yaml
from pathlib import Path
from .otel_types import LogBackend, LogBackendConfig, LogBackendInterface
from .handlers.debug import debug_logger

# Thread-local storage for trace context and correlation ID
_trace_context = threading.local()

class OTELFormatter(logging.Formatter):
    """Custom formatter that outputs logs in OpenTelemetry format with correlation_id"""

    def format(self, record: logging.LogRecord) -> str:
        trace_id = getattr(_trace_context, "trace_id", None)
        span_id = getattr(_trace_context, "span_id", None)
        correlation_id = getattr(_trace_context, "correlation_id", None)

        otel_log = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "severity_text": record.levelname,
            "severity_number": self._get_severity_number(record.levelno),
            "body": record.getMessage(),
            "resource": {
                "service.name": getattr(record, "service_name", "blueprint-framework"),
                "service.version": getattr(record, "service_version", "1.0.0"),
                "service.instance.id": getattr(record, "instance_id", str(uuid.uuid4())),
            },
            "attributes": {
                "logger.name": record.name,
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
                "filename": record.filename,
                "pathname": record.pathname,
                "process.pid": os.getpid(),
                "thread.id": record.thread,
                "thread.name": record.threadName,
            },
        }

        if trace_id:
            otel_log["trace_id"] = trace_id
        if span_id:
            otel_log["span_id"] = span_id
        if correlation_id:
            otel_log["attributes"]["correlation_id"] = correlation_id

        if hasattr(record, "otel_attributes"):
            otel_log["attributes"].update(record.otel_attributes)

        if record.exc_info:
            otel_log["attributes"]["exception.type"] = record.exc_info[0].__name__
            otel_log["attributes"]["exception.message"] = str(record.exc_info[1])
            otel_log["attributes"]["exception.stacktrace"] = self.formatException(record.exc_info)

        return json.dumps(otel_log, default=str)

    def _get_severity_number(self, level: int) -> int:
        mapping = {
            logging.DEBUG: 5,
            logging.INFO: 9,
            logging.WARNING: 13,
            logging.ERROR: 17,
            logging.CRITICAL: 21,
        }
        return mapping.get(level, 9)

class OTELLogger:
    """Centralized OTEL Logger with multiple backend support"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(OTELLogger, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        backends: Optional[List[LogBackendConfig]] = None,
        service_name: str = "blueprint-framework",
        service_version: str = "1.0.0",
        log_level: str = "INFO",
        enable_console: bool = True,
    ):
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self.service_name = service_name
        self.service_version = service_version
        self.instance_id = str(uuid.uuid4())
        self.backends = []

        if backends is None:
            backends = [LogBackendConfig(
                backend_type=LogBackend.FILESYSTEM,
                config={}
            )]

        self._initialize_backends(backends)

        self.logger = logging.getLogger("otel_logger")
        self.logger.setLevel(getattr(logging, log_level.upper()))

        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(OTELFormatter())
            self.logger.addHandler(console_handler)

        self.logger.propagate = False

        self.info(
            "OTEL Logger initialized",
            {
                "service_name": self.service_name,
                "service_version": self.service_version,
                "instance_id": self.instance_id,
                "backends": [b.backend_type.value for b in backends],
            },
        )

    def _initialize_backends(self, backend_configs: List[LogBackendConfig]) -> None:
        from .handlers.filesystem import FilesystemBackend
        from .handlers.elasticsearch import ElasticsearchBackend
        from .handlers.postgres import PostgresBackend

        backend_classes = {
            LogBackend.FILESYSTEM: FilesystemBackend,
            LogBackend.ELASTICSEARCH: ElasticsearchBackend,
            LogBackend.POSTGRES: PostgresBackend,
        }

        self.backends = []
        for config in backend_configs:
            try:
                debug_logger.debug("Attempting to initialize backend: %s with config: %s", 
                                 config.backend_type.value, config.config)
                backend_class = backend_classes[config.backend_type]
                backend = backend_class()
                backend.initialize(config.config)
                self.backends.append(backend)
                debug_logger.info("Successfully initialized %s backend", config.backend_type.value)
                print(f"Initialized {config.backend_type.value} backend")
            except Exception as e:
                debug_logger.error("Failed to initialize %s backend: %s", config.backend_type.value, e)
                print(f"Failed to initialize {config.backend_type.value} backend: {e}")
                raise

    def _log(
        self,
        level: str,
        message: str,
        attributes: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ):
        record = self.logger.makeRecord(
            name=self.logger.name,
            level=getattr(logging, level.upper()),
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=None,
            extra={
                "service_name": self.service_name,
                "service_version": self.service_version,
                "instance_id": self.instance_id
            },
        )

        record.service_name = self.service_name
        record.service_version = self.service_version
        record.instance_id = self.instance_id

        if attributes:
            record.otel_attributes = attributes

        if trace_id:
            _trace_context.trace_id = trace_id
        if span_id:
            _trace_context.span_id = span_id
        if correlation_id:
            _trace_context.correlation_id = correlation_id

        self.logger.handle(record)

        formatter = OTELFormatter()
        log_entry_str = formatter.format(record)
        log_entry = json.loads(log_entry_str)

        for backend in self.backends:
            try:
                backend.write_log(log_entry)
            except Exception as e:
                print(f"Backend write failed: {e}")

    def debug(self, message: str, attributes: Optional[Dict[str, Any]] = None, correlation_id: Optional[str] = None, **kwargs):
        self._log("DEBUG", message, attributes, correlation_id=correlation_id, **kwargs)

    def info(self, message: str, attributes: Optional[Dict[str, Any]] = None, correlation_id: Optional[str] = None, **kwargs):
        self._log("INFO", message, attributes, correlation_id=correlation_id, **kwargs)

    def warning(self, message: str, attributes: Optional[Dict[str, Any]] = None, correlation_id: Optional[str] = None, **kwargs):
        self._log("WARNING", message, attributes, correlation_id=correlation_id, **kwargs)

    def error(self, message: str, attributes: Optional[Dict[str, Any]] = None, correlation_id: Optional[str] = None, **kwargs):
        self._log("ERROR", message, attributes, correlation_id=correlation_id, **kwargs)

    def critical(self, message: str, attributes: Optional[Dict[str, Any]] = None, correlation_id: Optional[str] = None, **kwargs):
        self._log("CRITICAL", message, attributes, correlation_id=correlation_id, **kwargs)

    def close(self):
        for backend in self.backends:
            try:
                backend.close()
            except Exception as e:
                print(f"Error closing backend: {e}")

def read_logger_config(config_file: str = "logger_config.yaml") -> Dict[str, Any]:
    """Read logger configuration from a YAML file"""
    config_path = Path(config_file)
    
    if not config_path.exists():
        debug_logger.warning("Logger config file %s not found, using default Filesystem backend", config_file)
        return {
            "service_name": "blueprint-framework",
            "service_version": "1.0.0",
            "log_level": "INFO",
            "enable_console": True,
            "backends": [
                {
                    "backend_type": "filesystem",
                    "config": {
                        "log_file": f"./logs/blueprint-{datetime.now().strftime('%Y%m%d')}.jsonl",
                        "max_bytes": 10 * 1024 * 1024,
                        "backup_count": 5
                    }
                }
            ]
        }
    
    try:
        with config_path.open('r') as f:
            config = yaml.safe_load(f)
        debug_logger.info("Loaded logger configuration from %s", config_file)
        
        # Validate config structure
        if not config or "logger" not in config:
            debug_logger.error("Invalid logger config: 'logger' key missing")
            return {
                "service_name": "blueprint-framework",
                "service_version": "1.0.0",
                "log_level": "INFO",
                "enable_console": True,
                "backends": [
                    {
                        "backend_type": "filesystem",
                        "config": {
                            "log_file": f"./logs/blueprint-{datetime.now().strftime('%Y%m%d')}.jsonl",
                            "max_bytes": 10 * 1024 * 1024,
                            "backup_count": 5
                        }
                    }
                ]
            }

        logger_config = config["logger"]
        if not isinstance(logger_config.get("backends"), list):
            debug_logger.warning("Invalid or missing backends in config, using default Filesystem backend")
            logger_config["backends"] = [
                {
                    "backend_type": "filesystem",
                    "config": {
                        "log_file": f"./logs/blueprint-{datetime.now().strftime('%Y%m%d')}.jsonl",
                        "max_bytes": 10 * 1024 * 1024,
                        "backup_count": 5
                    }
                }
            ]
        
        return logger_config
    except Exception as e:
        debug_logger.error("Failed to load logger config from %s: %s", config_file, e)
        return {
            "service_name": "blueprint-framework",
            "service_version": "1.0.0",
            "log_level": "INFO",
            "enable_console": True,
            "backends": [
                {
                    "backend_type": "filesystem",
                    "config": {
                        "log_file": f"./logs/blueprint-{datetime.now().strftime('%Y%m%d')}.jsonl",
                        "max_bytes": 10 * 1024 * 1024,
                        "backup_count": 5
                    }
                }
            ]
        }

def create_otel_logger(
    config_file: Optional[str] = None,
    backend_type: str = None,
    backend_config: Optional[Dict[str, Any]] = None,
    service_name: str = None,
    service_version: str = None,
    log_level: str = None,
    enable_console: bool = None,
) -> OTELLogger:
    """Create OTELLogger from config file or parameters"""
    if config_file:
        config = read_logger_config(config_file)
    else:
        config = {
            "service_name": service_name or "blueprint-framework",
            "service_version": service_version or "1.0.0",
            "log_level": log_level or "INFO",
            "enable_console": enable_console if enable_console is not None else True,
            "backends": []
        }
        if backend_type:
            config["backends"].append({
                "backend_type": backend_type,
                "config": backend_config or {}
            })
        else:
            config["backends"].append({
                "backend_type": "filesystem",
                "config": {
                    "log_file": f"./logs/blueprint-{datetime.now().strftime('%Y%m%d')}.jsonl",
                    "max_bytes": 10 * 1024 * 1024,
                    "backup_count": 5
                }
            })
    
    backends = [
        LogBackendConfig(
            backend_type=LogBackend(backend["backend_type"]),
            config=backend["config"]
        )
        for backend in config["backends"]
    ]
    
    return OTELLogger(
        backends=backends,
        service_name=config["service_name"],
        service_version=config["service_version"],
        log_level=config["log_level"],
        enable_console=config["enable_console"],
    )

otel_logger = None

def configure_logger(**kwargs):
    """Configure the global OTEL logger"""
    global otel_logger
    otel_logger = create_otel_logger(**kwargs)
    return otel_logger

def otel_trace(
    operation_name: Optional[str] = None,
    service_name: Optional[str] = None,
    include_args: bool = True,
    include_result: bool = True,
    log_exceptions: bool = True,
    custom_attributes: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            trace_id = str(uuid.uuid4()).replace("-", "")
            span_id = str(uuid.uuid4()).replace("-", "")[:16]
            local_correlation_id = correlation_id or str(uuid.uuid4()).replace("-", "")

            _trace_context.trace_id = trace_id
            _trace_context.span_id = span_id
            _trace_context.correlation_id = local_correlation_id

            op_name = operation_name or f"{func.__module__}.{func.__qualname__}"

            attributes = {
                "operation.name": op_name,
                "function.name": func.__name__,
                "function.module": func.__module__,
                "function.file": inspect.getfile(func),
                "span.kind": "internal",
            }

            if custom_attributes:
                attributes.update(custom_attributes)

            if service_name:
                attributes["service.name"] = service_name

            if include_args and (args or kwargs):
                try:
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    safe_args = {}
                    for param_name, param_value in bound_args.arguments.items():
                        if any(sensitive in param_name.lower() for sensitive in ["password", "token", "secret", "key"]):
                            safe_args[param_name] = "[REDACTED]"
                        else:
                            safe_args[param_name] = str(param_value)[:200]
                    attributes["function.args"] = safe_args
                except Exception as e:
                    attributes["function.args_error"] = str(e)

            start_time = time.time()
            otel_logger.info(
                f"Starting operation: {op_name}",
                attributes={**attributes, "span.start_time": start_time},
                trace_id=trace_id,
                span_id=span_id,
                correlation_id=local_correlation_id,
            )

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                success_attributes = {
                    **attributes,
                    "span.duration_ms": round(duration * 1000, 2),
                    "span.status": "success",
                }

                if include_result and result is not None:
                    try:
                        result_str = str(result)
                        if len(result_str) > 500:
                            result_str = result_str[:500] + "..."
                        success_attributes["function.result"] = result_str
                    except Exception as e:
                        success_attributes["function.result_error"] = str(e)

                otel_logger.info(
                    f"Completed operation: {op_name}",
                    attributes=success_attributes,
                    trace_id=trace_id,
                    span_id=span_id,
                    correlation_id=local_correlation_id,
                )

                return result

            except Exception as e:
                duration = time.time() - start_time

                error_attributes = {
                    **attributes,
                    "span.duration_ms": round(duration * 1000, 2),
                    "span.status": "error",
                    "error.type": type(e).__name__,
                    "error.message": str(e),
                    "error.stack": traceback.format_exc(),
                }

                if log_exceptions:
                    otel_logger.error(
                        f"Operation failed: {op_name}",
                        attributes=error_attributes,
                        trace_id=trace_id,
                        span_id=span_id,
                        correlation_id=local_correlation_id,
                    )

                raise

            finally:
                if hasattr(_trace_context, "trace_id"):
                    delattr(_trace_context, "trace_id")
                if hasattr(_trace_context, "span_id"):
                    delattr(_trace_context, "span_id")
                if hasattr(_trace_context, "correlation_id"):
                    delattr(_trace_context, "correlation_id")

        return wrapper

    return decorator

def otel_log(
    level: str = "INFO",
    message: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"
            log_message = message or f"Executing {func_name}"
            local_correlation_id = correlation_id or str(uuid.uuid4()).replace("-", "")

            log_attributes = {
                "function.name": func.__name__,
                "function.module": func.__module__,
                **(attributes or {}),
            }

            getattr(otel_logger, level.lower())(
                log_message, log_attributes, correlation_id=local_correlation_id
            )

            try:
                result = func(*args, **kwargs)

                getattr(otel_logger, level.lower())(
                    f"Completed {func_name}",
                    {**log_attributes, "status": "success"},
                    correlation_id=local_correlation_id
                )

                return result

            except Exception as e:
                otel_logger.error(
                    f"Failed {func_name}: {str(e)}",
                    {**log_attributes, "status": "error", "error": str(e)},
                    correlation_id=local_correlation_id
                )
                raise

        return wrapper

    return decorator

def log_info(message: str, attributes: Optional[Dict[str, Any]] = None, correlation_id: Optional[str] = None, **kwargs):
    if attributes is None:
        attributes = kwargs
    else:
        attributes.update(kwargs)
    otel_logger.info(message, attributes, correlation_id=correlation_id)

def log_error(message: str, attributes: Optional[Dict[str, Any]] = None, correlation_id: Optional[str] = None, **kwargs):
    if attributes is None:
        attributes = kwargs
    else:
        attributes.update(kwargs)
    otel_logger.error(message, attributes, correlation_id=correlation_id)

def log_debug(message: str, attributes: Optional[Dict[str, Any]] = None, correlation_id: Optional[str] = None, **kwargs):
    if attributes is None:
        attributes = kwargs
    else:
        attributes.update(kwargs)
    otel_logger.debug(message, attributes, correlation_id=correlation_id)

def log_warning(message: str, attributes: Optional[Dict[str, Any]] = None, correlation_id: Optional[str] = None, **kwargs):
    if attributes is None:
        attributes = kwargs
    else:
        attributes.update(kwargs)
    otel_logger.warning(message, attributes, correlation_id=correlation_id)

class otel_span:
    def __init__(
        self, operation_name: str, attributes: Optional[Dict[str, Any]] = None, correlation_id: Optional[str] = None
    ):
        self.operation_name = operation_name
        self.attributes = attributes or {}
        self.correlation_id = correlation_id or str(uuid.uuid4()).replace("-", "")
        self.trace_id = None
        self.span_id = None
        self.start_time = None

    def __enter__(self):
        self.trace_id = str(uuid.uuid4()).replace("-", "")
        self.span_id = str(uuid.uuid4()).replace("-", "")[:16]
        self.start_time = time.time()

        _trace_context.trace_id = self.trace_id
        _trace_context.span_id = self.span_id
        _trace_context.correlation_id = self.correlation_id

        otel_logger.info(
            f"Starting span: {self.operation_name}",
            attributes={
                **self.attributes,
                "span.start_time": self.start_time,
                "operation.name": self.operation_name,
            },
            trace_id=self.trace_id,
            span_id=self.span_id,
            correlation_id=self.correlation_id,
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time

        if exc_type is None:
            otel_logger.info(
                f"Completed span: {self.operation_name}",
                attributes={
                    **self.attributes,
                    "span.duration_ms": round(duration * 1000, 2),
                    "span.status": "success",
                    "operation.name": self.operation_name,
                },
                trace_id=self.trace_id,
                span_id=self.span_id,
                correlation_id=self.correlation_id,
            )
        else:
            otel_logger.error(
                f"Span failed: {self.operation_name}",
                attributes={
                    **self.attributes,
                    "span.duration_ms": round(duration * 1000, 2),
                    "span.status": "error",
                    "operation.name": self.operation_name,
                    "error.type": exc_type.__name__,
                    "error.message": str(exc_val),
                    "error.stack": traceback.format_exc(),
                },
                trace_id=self.trace_id,
                span_id=self.span_id,
                correlation_id=self.correlation_id,
            )

        if hasattr(_trace_context, "trace_id"):
            delattr(_trace_context, "trace_id")
        if hasattr(_trace_context, "span_id"):
            delattr(_trace_context, "span_id")
        if hasattr(_trace_context, "correlation_id"):
            delattr(_trace_context, "correlation_id")

if __name__ == "__main__":
    # Test with config file
    print("=== Testing Logger with Config File ===")
    logger = configure_logger(config_file="logger_config.yaml")
    
    @otel_trace(
        operation_name="example.calculation", 
        include_args=True, 
        include_result=True,
        correlation_id="req-001"
    )
    def calculate_something(x: int, y: int, operation: str = "add") -> int:
        time.sleep(0.1)
        if operation == "add":
            return x + y
        elif operation == "multiply":
            return x * y
        else:
            raise ValueError(f"Unknown operation: {operation}")

    @otel_log(level="INFO", message="Processing data", correlation_id="req-002")
    def process_data(data: list) -> dict:
        return {"processed": len(data), "data": data}

    result1 = calculate_something(5, 3, "add")
    print(f"Result 1: {result1}")

    result2 = process_data([1, 2, 3, 4, 5])
    print(f"Result 2: {result2}")

    with otel_span("data.processing", {"batch_size": 100}, correlation_id="req-003"):
        time.sleep(0.05)
        log_info("Processing batch", batch_id="batch_001", size=100, correlation_id="req-003")

    try:
        calculate_something(1, 2, "unknown")
    except ValueError as e:
        print(f"Caught expected error: {e}")

    log_debug("Debug message", {"debug_info": "test"}, correlation_id="req-004")
    log_info("Info message", {"info_data": "test"}, correlation_id="req-005")
    log_warning("Warning message", {"warning_type": "test"}, correlation_id="req-006")
    log_error("Error message", {"error_code": "TEST001"}, correlation_id="req-007")

    print("Testing completed! Check your configured backends for logs.")
    
    logger.close()
