
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from ..otel_types import LogBackendInterface
from .debug import debug_logger
import queue
import threading
from logging.handlers import RotatingFileHandler
from filelock import FileLock
import logging

class FilesystemBackend(LogBackendInterface):
    """Asynchronous filesystem logging backend with log rotation and thread-safety"""
    
    def __init__(self):
        self.log_queue = queue.Queue()
        self.running = True
        self.worker = None
        self.log_file = None
        self.file_handler = None
        
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize filesystem backend with log rotation"""
        debug_logger.debug("Initializing Filesystem backend with config: %s", config)
        
        log_file = config.get('log_file')
        max_bytes = config.get('max_bytes', 10 * 1024 * 1024)  # 10MB default
        backup_count = config.get('backup_count', 5)
        
        if log_file is None:
            log_dir = Path.cwd() / "logs"
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"blueprint-{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.file_handler = RotatingFileHandler(
                self.log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            self.file_handler.setFormatter(logging.Formatter('%(message)s'))
            debug_logger.info("Filesystem backend initialized with log file: %s", self.log_file)
        except Exception as e:
            debug_logger.error("Failed to initialize file handler: %s", e)
            raise
        
        self.worker = threading.Thread(target=self._write_worker)
        self.worker.daemon = True
        self.worker.start()
    
    def write_log(self, log_entry: Dict[str, Any]) -> None:
        """Queue log entry for asynchronous writing"""
        debug_logger.debug("Queuing log entry: %s", log_entry)
        self.log_queue.put(log_entry)
    
    def _write_worker(self):
        """Process log queue and write to file with thread-safety"""
        while self.running:
            try:
                log_entry = self.log_queue.get(timeout=1)
                lock_file = f"{self.log_file}.lock"
                with FileLock(lock_file):
                    self.file_handler.emit(
                        logging.makeLogRecord({"msg": json.dumps(log_entry, default=str)})
                    )
                self.log_queue.task_done()
                debug_logger.debug("Wrote log entry to %s", self.log_file)
            except queue.Empty:
                continue
            except Exception as e:
                debug_logger.error("Failed to write log: %s", e)
    
    def close(self) -> None:
        """Close the filesystem backend"""
        debug_logger.debug("Closing Filesystem backend")
        self.running = False
        self.worker.join()
        if self.file_handler:
            try:
                self.file_handler.close()
                debug_logger.info("Filesystem backend closed successfully")
            except Exception as e:
                debug_logger.error("Error closing file handler: %s", e)
                raise



































# import json
# from datetime import datetime
# from pathlib import Path
# from typing import Any, Dict
# from otel_logger import LogBackendInterface
# from .debug import debug_logger
# import queue
# import threading
# from logging.handlers import RotatingFileHandler
# from filelock import FileLock
# import logging

# class FilesystemBackend(LogBackendInterface):
#     """Asynchronous filesystem logging backend with log rotation and thread-safety"""
    
#     def __init__(self):
#         self.log_queue = queue.Queue()
#         self.running = True
#         self.worker = None
#         self.log_file = None
#         self.file_handler = None
        
#     def initialize(self, config: Dict[str, Any]) -> None:
#         """Initialize filesystem backend with log rotation"""
#         debug_logger.debug("Initializing Filesystem backend with config: %s", config)
        
#         log_file = config.get('log_file')
#         max_bytes = config.get('max_bytes', 10 * 1024 * 1024)  # 10MB default
#         backup_count = config.get('backup_count', 5)
        
#         if log_file is None:
#             log_dir = Path.cwd() / "logs"
#             log_dir.mkdir(exist_ok=True)
#             log_file = log_dir / f"blueprint-{datetime.now().strftime('%Y%m%d')}.jsonl"
        
#         self.log_file = Path(log_file)
#         self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
#         try:
#             self.file_handler = RotatingFileHandler(
#                 self.log_file,
#                 maxBytes=max_bytes,
#                 backupCount=backup_count
#             )
#             self.file_handler.setFormatter(logging.Formatter('%(message)s'))
#             debug_logger.info("Filesystem backend initialized with log file: %s", self.log_file)
#         except Exception as e:
#             debug_logger.error("Failed to initialize file handler: %s", e)
#             raise
        
#         self.worker = threading.Thread(target=self._write_worker)
#         self.worker.daemon = True
#         self.worker.start()
    
#     def write_log(self, log_entry: Dict[str, Any]) -> None:
#         """Queue log entry for asynchronous writing"""
#         debug_logger.debug("Queuing log entry: %s", log_entry)
#         self.log_queue.put(log_entry)
    
#     def _write_worker(self):
#         """Process log queue and write to file with thread-safety"""
#         while self.running:
#             try:
#                 log_entry = self.log_queue.get(timeout=1)
#                 lock_file = f"{self.log_file}.lock"
#                 with FileLock(lock_file):
#                     self.file_handler.emit(
#                         logging.makeLogRecord({"msg": json.dumps(log_entry, default=str)})
#                     )
#                 self.log_queue.task_done()
#                 debug_logger.debug("Wrote log entry to %s", self.log_file)
#             except queue.Empty:
#                 continue
#             except Exception as e:
#                 debug_logger.error("Failed to write log: %s", e)
    
#     def close(self) -> None:
#         """Close the filesystem backend"""
#         debug_logger.debug("Closing Filesystem backend")
#         self.running = False
#         self.worker.join()
#         if self.file_handler:
#             try:
#                 self.file_handler.close()
#                 debug_logger.info("Filesystem backend closed successfully")
#             except Exception as e:
#                 debug_logger.error("Error closing file handler: %s", e)
#                 raise
