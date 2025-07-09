
import json
from datetime import datetime
from typing import Any, Dict
from ..otel_logger import LogBackendInterface
from .debug import debug_logger
from elasticsearch import ConnectionError, TransportError, SSLError
import time

class ElasticsearchBackend(LogBackendInterface):
    """Direct Elasticsearch logging backend"""
    
    def __init__(self):
        self.es_client = None
        self.index_pattern = None
        
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize Elasticsearch backend"""
        debug_logger.debug("Initializing Elasticsearch backend with config: %s", config)
        
        try:
            from elasticsearch import Elasticsearch
            # Log client version
            import importlib.metadata
            try:
                version = importlib.metadata.version('elasticsearch')
                debug_logger.debug("Elasticsearch client module version: %s", version)
            except importlib.metadata.PackageNotFoundError:
                debug_logger.debug("Elasticsearch client module version: unknown")
        except ImportError as e:
            debug_logger.error("Failed to import elasticsearch package: %s", e)
            raise ImportError("elasticsearch package is required for Elasticsearch backend. Install with: pip install elasticsearch")
        
        hosts = config.get('hosts', ['localhost:9200'])
        self.index_pattern = config.get('index_pattern', 'blueprint-logs-%Y.%m.%d')
        
        # Log configuration details
        debug_logger.debug("Elasticsearch hosts: %s, index_pattern: %s", hosts, self.index_pattern)
        
        # Initialize Elasticsearch client with retries
        retries = 3
        for attempt in range(1, retries + 1):
            try:
                self.es_client = Elasticsearch(
                    hosts=hosts,
                    basic_auth=config.get('basic_auth'),
                    ca_certs=config.get('ca_certs'),
                    verify_certs=config.get('verify_certs', False),
                    request_timeout=config.get('timeout', 30)
                )
                # Test connection
                if not self.es_client.ping():
                    debug_logger.error("Elasticsearch ping failed on attempt %d", attempt)
                    try:
                        cluster_info = self.es_client.info()
                        debug_logger.error("Cluster info retrieved despite ping failure: %s", cluster_info)
                    except Exception as info_error:
                        debug_logger.error("Failed to retrieve cluster info: %s", info_error)
                    raise ConnectionError("Could not connect to Elasticsearch")
                debug_logger.info("Successfully connected to Elasticsearch")
                
                # Log cluster info
                cluster_info = self.es_client.info()
                debug_logger.debug("Elasticsearch cluster info: %s", cluster_info)
                
                # Create index template
                self._create_index_template()
                debug_logger.info("Elasticsearch index template created successfully")
                break
            except ConnectionError as ce:
                debug_logger.error("ConnectionError on attempt %d: %s", attempt, ce)
                if attempt == retries:
                    raise ConnectionError(f"Could not connect to Elasticsearch after {retries} attempts: {ce}")
                time.sleep(2)
            except SSLError as ssl_err:
                debug_logger.error("SSLError on attempt %d: %s", attempt, ssl_err)
                if attempt == retries:
                    raise ConnectionError(f"Could not connect to Elasticsearch after {retries} attempts due to SSL error: {ssl_err}")
                time.sleep(2)
            except TransportError as te:
                debug_logger.error("TransportError on attempt %d: %s", attempt, te)
                if attempt == retries:
                    raise ConnectionError(f"Could not connect to Elasticsearch after {retries} attempts: {te}")
                time.sleep(2)
            except Exception as e:
                debug_logger.error("Unexpected error: %s", e)
                raise
    
    def write_log(self, log_entry: Dict[str, Any]) -> None:
        """Write log entry directly to Elasticsearch with retries"""
        debug_logger.debug("Writing log to Elasticsearch: %s", log_entry)
        index_name = datetime.now().strftime(self.index_pattern)
        
        es_entry = {
            "@timestamp": log_entry.get("timestamp"),
            **log_entry
        }
        
        retries = 3
        for attempt in range(1, retries + 1):
            try:
                response = self.es_client.index(index=index_name, body=es_entry)
                debug_logger.debug("Log written to Elasticsearch, response: %s", response)
                break
            except (ConnectionError, TransportError, SSLError) as e:
                debug_logger.error("Failed to write to Elasticsearch on attempt %d: %s", attempt, e)
                if attempt == retries:
                    debug_logger.error("Failed to write to Elasticsearch after %d attempts: %s", retries, e)
                    print(f"Failed to write to Elasticsearch after {retries} attempts: {e}")
                    print(json.dumps(log_entry, default=str))
                time.sleep(2)  # Wait before retrying
            except Exception as e:
                debug_logger.error("Unexpected error writing to Elasticsearch: %s", e)
                print(f"Unexpected error writing to Elasticsearch: {e}")
                print(json.dumps(log_entry, default=str))
                break
    
    def _create_index_template(self) -> None:
        """Create Elasticsearch index template"""
        debug_logger.debug("Creating Elasticsearch index template")
        template = {
            "index_patterns": [self.index_pattern.replace("%Y.%m.%d", "*")],  # Use configured index_pattern
            "priority": 1,
            "template": {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "index.refresh_interval": "5s"
                },
                "mappings": {
                    "_source": {"enabled": True},
                    "properties": {
                        "@timestamp": {"type": "date"},
                        "timestamp": {"type": "date"},
                        "severity_text": {"type": "keyword"},
                        "severity_number": {"type": "integer"},
                        "body": {"type": "text"},
                        "trace_id": {"type": "keyword"},
                        "span_id": {"type": "keyword"},
                        "resource": {
                            "type": "object",
                            "properties": {
                                "service.name": {"type": "keyword"},
                                "service.version": {"type": "keyword"},
                                "service.instance.id": {"type": "keyword"}
                            }
                        },
                        "attributes": {"type": "object", "dynamic": True}
                    }
                }
            }
        }
        
        try:
            # Check if template exists to avoid overwriting
            if not self.es_client.indices.exists_index_template(name="my-app-logs-template"):
                self.es_client.indices.put_index_template(name="my-app-logs-template", body=template)
                debug_logger.info("Index template 'my-app-logs-template' created successfully")
            else:
                debug_logger.debug("Index template 'my-app-logs-template' already exists, skipping creation")
        except Exception as e:
            debug_logger.warning("Could not create index template: %s", e)
            print(f"Warning: Could not create index template: {e}")
    
    def close(self) -> None:
        """Close Elasticsearch connection"""
        debug_logger.debug("Closing Elasticsearch connection")
        if self.es_client:
            self.es_client.close()
            debug_logger.info("Elasticsearch connection closed")
