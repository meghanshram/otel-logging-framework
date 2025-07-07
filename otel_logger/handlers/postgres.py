import json
from typing import Any, Dict
from ..otel_logger import LogBackendInterface
from .debug import debug_logger

class PostgresBackend(LogBackendInterface):
    """PostgreSQL logging backend"""
    
    def __init__(self):
        self.connection = None
        self.table_name = None
        
    def initialize(self, config: Dict[str, Any]) -> None:
        debug_logger.debug("Initializing PostgreSQL backend with config: %s", config)
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            debug_logger.error("psycopg2 package is required for PostgreSQL backend. Install with: pip install psycopg2-binary")
            raise ImportError("psycopg2 package is required for PostgreSQL backend. Install with: pip install psycopg2-binary")
        
        self.table_name = config.get('table_name', 'otel_logs')
        
        try:
            self.connection = psycopg2.connect(
                host=config.get('host', 'localhost'),
                port=config.get('port', 5432),
                database=config.get('database', 'logs'),
                user=config.get('user', 'postgres'),
                password=config.get('password', 'postgres')
            )
            debug_logger.info("Successfully connected to PostgreSQL")
            self._create_table()
            debug_logger.info("PostgreSQL table created successfully")
        except Exception as e:
            debug_logger.error("Failed to initialize PostgreSQL client: %s", e)
            raise
    
    def write_log(self, log_entry: Dict[str, Any]) -> None:
        debug_logger.debug("Writing log to PostgreSQL: %s", log_entry)
        query = f"""
        INSERT INTO {self.table_name} 
        (timestamp, severity_text, severity_number, body, trace_id, span_id, resource, attributes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, (
                    log_entry.get('timestamp'),
                    log_entry.get('severity_text'),
                    log_entry.get('severity_number'),
                    log_entry.get('body'),
                    log_entry.get('trace_id'),
                    log_entry.get('span_id'),
                    json.dumps(log_entry.get('resource', {})),
                    json.dumps(log_entry.get('attributes', {}))
                ))
                self.connection.commit()
                debug_logger.debug("Log written to PostgreSQL")
        except Exception as e:
            self.connection.rollback()
            debug_logger.error("Failed to write to PostgreSQL: %s", e)
            print(f"Failed to write to PostgreSQL: {e}")
            print(json.dumps(log_entry, default=str))
    
    def _create_table(self) -> None:
        debug_logger.debug("Creating PostgreSQL table")
        query = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            severity_text VARCHAR(20),
            severity_number INTEGER,
            body TEXT,
            trace_id VARCHAR(32),
            span_id VARCHAR(16),
            resource JSONB,
            attributes JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_timestamp ON {self.table_name}(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_trace_id ON {self.table_name}(trace_id);
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_severity ON {self.table_name}(severity_text);
        """
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                self.connection.commit()
                debug_logger.debug("PostgreSQL table and indices created")
        except Exception as e:
            self.connection.rollback()
            debug_logger.error("Failed to create PostgreSQL table: %s", e)
            raise
    
    def close(self) -> None:
        debug_logger.debug("Closing PostgreSQL connection")
        if self.connection:
            self.connection.close()
            debug_logger.info("PostgreSQL connection closed")
