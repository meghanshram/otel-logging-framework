
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict
from dataclasses import dataclass

class LogBackend(Enum):
    """Supported logging backends"""
    FILESYSTEM = "filesystem"
    ELASTICSEARCH = "elasticsearch"
    POSTGRES = "postgres"

@dataclass
class LogBackendConfig:
    """Configuration for logging backends"""
    backend_type: LogBackend
    config: Dict[str, Any]

class LogBackendInterface(ABC):
    """Abstract interface for logging backends"""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def write_log(self, log_entry: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def close(self) -> None:
        pass
