import json
import asyncio
from datetime import datetime
from typing import Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import threading
from contextlib import asynccontextmanager
from ..otel_logger import LogBackendInterface
from .debug import debug_logger
from elasticsearch import ConnectionError, TransportError, SSLError
import time

class AsyncElasticsearchBackend(LogBackendInterface):
    """Async Elasticsearch logging backend with production-ready features"""
    
    def __init__(self):
        self.es_client = None
        self.index_pattern = None
        self._executor = None
        self._shutdown_event = None
        self._health_check_task = None
        self._connection_lock = threading.RLock()
        self._is_healthy = False
        self._consecutive_failures = 0
        self._circuit_breaker_open = False
        self._last_failure_time = None
        self._config = None
        
        # Configuration defaults
        self._max_retries = 3
        self._retry_delay = 2.0
        self._health_check_interval = 30.0
        self._circuit_breaker_threshold = 5
        self._circuit_breaker_timeout = 60.0
        self._max_workers = 10
        
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize Elasticsearch backend asynchronously"""
        debug_logger.debug("Initializing async Elasticsearch backend with config: %s", config)
        
        self._config = config
        self._max_retries = config.get('max_retries', self._max_retries)
        self._retry_delay = config.get('retry_delay', self._retry_delay)
        self._health_check_interval = config.get('health_check_interval', self._health_check_interval)
        self._circuit_breaker_threshold = config.get('circuit_breaker_threshold', self._circuit_breaker_threshold)
        self._circuit_breaker_timeout = config.get('circuit_breaker_timeout', self._circuit_breaker_timeout)
        self._max_workers = config.get('max_workers', self._max_workers)
        
        # Initialize thread pool executor for blocking operations
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="elasticsearch-backend"
        )
        
        # Initialize shutdown event
        self._shutdown_event = asyncio.Event()
        
        try:
            await self._initialize_elasticsearch()
            
            # Start health check task
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            debug_logger.info("Async Elasticsearch backend initialized successfully")
            
        except Exception as e:
            await self._cleanup()
            raise
    
    async def _initialize_elasticsearch(self) -> None:
        """Initialize Elasticsearch client with async retry logic"""
        try:
            # Import in executor to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                self._executor, self._import_elasticsearch
            )
        except ImportError as e:
            debug_logger.error("Failed to import elasticsearch package: %s", e)
            raise ImportError("elasticsearch package is required for Elasticsearch backend. Install with: pip install elasticsearch")
        
        hosts = self._config.get('hosts', ['localhost:9200'])
        self.index_pattern = self._config.get('index_pattern', 'blueprint-logs-%Y.%m.%d')
        
        debug_logger.debug("Elasticsearch hosts: %s, index_pattern: %s", hosts, self.index_pattern)
        
        # Initialize Elasticsearch client with retries
        for attempt in range(1, self._max_retries + 1):
            try:
                await self._create_es_client(hosts)
                
                # Test connection
                if not await self._ping_elasticsearch():
                    debug_logger.error("Elasticsearch ping failed on attempt %d", attempt)
                    try:
                        cluster_info = await self._get_cluster_info()
                        debug_logger.error("Cluster info retrieved despite ping failure: %s", cluster_info)
                    except Exception as info_error:
                        debug_logger.error("Failed to retrieve cluster info: %s", info_error)
                    raise ConnectionError("Could not connect to Elasticsearch")
                
                debug_logger.info("Successfully connected to Elasticsearch")
                
                # Log cluster info
                cluster_info = await self._get_cluster_info()
                debug_logger.debug("Elasticsearch cluster info: %s", cluster_info)
                
                # Create index template
                await self._create_index_template()
                debug_logger.info("Elasticsearch index template created successfully")
                
                self._is_healthy = True
                self._consecutive_failures = 0
                self._circuit_breaker_open = False
                break
                
            except (ConnectionError, SSLError, TransportError) as e:
                debug_logger.error("Connection error on attempt %d: %s", attempt, e)
                self._handle_connection_failure()
                
                if attempt == self._max_retries:
                    raise ConnectionError(f"Could not connect to Elasticsearch after {self._max_retries} attempts: {e}")
                
                await asyncio.sleep(self._retry_delay)
                
            except Exception as e:
                debug_logger.error("Unexpected error: %s", e)
                self._handle_connection_failure()
                raise
    
    def _import_elasticsearch(self) -> None:
        """Import Elasticsearch in thread executor"""
        from elasticsearch import Elasticsearch
        import importlib.metadata
        
        try:
            version = importlib.metadata.version('elasticsearch')
            debug_logger.debug("Elasticsearch client module version: %s", version)
        except importlib.metadata.PackageNotFoundError:
            debug_logger.debug("Elasticsearch client module version: unknown")
    
    async def _create_es_client(self, hosts: list) -> None:
        """Create Elasticsearch client in executor"""
        def _create_client():
            from elasticsearch import Elasticsearch
            return Elasticsearch(
                hosts=hosts,
                basic_auth=self._config.get('basic_auth'),
                ca_certs=self._config.get('ca_certs'),
                verify_certs=self._config.get('verify_certs', False),
                request_timeout=self._config.get('timeout', 30),
                max_retries=0,  # We handle retries ourselves
                retry_on_timeout=False
            )
        
        with self._connection_lock:
            self.es_client = await asyncio.get_event_loop().run_in_executor(
                self._executor, _create_client
            )
    
    async def _ping_elasticsearch(self) -> bool:
        """Ping Elasticsearch asynchronously"""
        def _ping():
            return self.es_client.ping()
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, _ping
        )
    
    async def _get_cluster_info(self) -> Dict[str, Any]:
        """Get cluster info asynchronously"""
        def _get_info():
            return self.es_client.info()
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, _get_info
        )
    
    async def write_log(self, log_entry: Dict[str, Any]) -> None:
        """Write log entry asynchronously with circuit breaker"""
        if self._circuit_breaker_open:
            if self._should_attempt_reset():
                debug_logger.info("Attempting to reset circuit breaker")
                self._circuit_breaker_open = False
            else:
                debug_logger.warning("Circuit breaker is open, dropping log entry")
                return
        
        debug_logger.debug("Writing log to Elasticsearch: %s", log_entry)
        
        try:
            await self._write_log_with_retries(log_entry)
            self._handle_write_success()
        except Exception as e:
            self._handle_write_failure(e, log_entry)
    
    async def _write_log_with_retries(self, log_entry: Dict[str, Any]) -> None:
        """Write log with async retry logic"""
        index_name = datetime.now().strftime(self.index_pattern)
        
        es_entry = {
            "@timestamp": log_entry.get("timestamp"),
            **log_entry
        }
        
        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._index_document(index_name, es_entry)
                debug_logger.debug("Log written to Elasticsearch, response: %s", response)
                return
                
            except (ConnectionError, TransportError, SSLError) as e:
                debug_logger.error("Failed to write to Elasticsearch on attempt %d: %s", attempt, e)
                
                if attempt == self._max_retries:
                    debug_logger.error("Failed to write to Elasticsearch after %d attempts: %s", self._max_retries, e)
                    raise
                
                await asyncio.sleep(self._retry_delay)
                
            except Exception as e:
                debug_logger.error("Unexpected error writing to Elasticsearch: %s", e)
                raise
    
    async def _index_document(self, index_name: str, es_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Index document asynchronously"""
        def _index():
            return self.es_client.index(index=index_name, body=es_entry)
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, _index
        )
    
    async def _create_index_template(self) -> None:
        """Create Elasticsearch index template asynchronously"""
        debug_logger.debug("Creating Elasticsearch index template")
        
        template = {
            "index_patterns": [self.index_pattern.replace("%Y.%m.%d", "*")],
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
            template_exists = await self._check_template_exists("my-app-logs-template")
            if not template_exists:
                await self._put_index_template("my-app-logs-template", template)
                debug_logger.info("Index template 'my-app-logs-template' created successfully")
            else:
                debug_logger.debug("Index template 'my-app-logs-template' already exists, skipping creation")
        except Exception as e:
            debug_logger.warning("Could not create index template: %s", e)
            print(f"Warning: Could not create index template: {e}")
    
    async def _check_template_exists(self, template_name: str) -> bool:
        """Check if index template exists"""
        def _check():
            return self.es_client.indices.exists_index_template(name=template_name)
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, _check
        )
    
    async def _put_index_template(self, template_name: str, template: Dict[str, Any]) -> None:
        """Put index template"""
        def _put():
            return self.es_client.indices.put_index_template(name=template_name, body=template)
        
        await asyncio.get_event_loop().run_in_executor(
            self._executor, _put
        )
    
    async def _health_check_loop(self) -> None:
        """Background health check loop"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self._health_check_interval)
                
                if self._shutdown_event.is_set():
                    break
                
                is_healthy = await self._ping_elasticsearch()
                
                if is_healthy != self._is_healthy:
                    self._is_healthy = is_healthy
                    if is_healthy:
                        debug_logger.info("Elasticsearch connection restored")
                        self._consecutive_failures = 0
                    else:
                        debug_logger.warning("Elasticsearch health check failed")
                        self._handle_connection_failure()
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                debug_logger.error("Health check error: %s", e)
                self._handle_connection_failure()
    
    def _handle_connection_failure(self) -> None:
        """Handle connection failure for circuit breaker"""
        self._consecutive_failures += 1
        self._last_failure_time = time.time()
        
        if self._consecutive_failures >= self._circuit_breaker_threshold:
            self._circuit_breaker_open = True
            debug_logger.warning("Circuit breaker opened due to %d consecutive failures", self._consecutive_failures)
    
    def _handle_write_success(self) -> None:
        """Handle successful write"""
        if self._consecutive_failures > 0:
            self._consecutive_failures = 0
            debug_logger.info("Write successful, resetting failure count")
    
    def _handle_write_failure(self, error: Exception, log_entry: Dict[str, Any]) -> None:
        """Handle write failure"""
        self._handle_connection_failure()
        print(f"Failed to write to Elasticsearch: {error}")
        print(json.dumps(log_entry, default=str))
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self._last_failure_time is None:
            return True
        
        return (time.time() - self._last_failure_time) > self._circuit_breaker_timeout
    
    async def close(self) -> None:
        """Close Elasticsearch connection gracefully"""
        debug_logger.debug("Closing async Elasticsearch connection")
        
        # Signal shutdown
        if self._shutdown_event:
            self._shutdown_event.set()
        
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        await self._cleanup()
        debug_logger.info("Async Elasticsearch connection closed")
    
    async def _cleanup(self) -> None:
        """Cleanup resources"""
        # Close ES client
        if self.es_client:
            def _close_client():
                self.es_client.close()
            
            await asyncio.get_event_loop().run_in_executor(
                self._executor, _close_client
            )
        
        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=True)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        return {
            "healthy": self._is_healthy,
            "circuit_breaker_open": self._circuit_breaker_open,
            "consecutive_failures": self._consecutive_failures,
            "last_failure_time": self._last_failure_time
        }


# Fixed sync wrapper that handles event loop conflicts properly
class ElasticsearchBackend(LogBackendInterface):
    """Sync wrapper that properly handles event loop conflicts"""
    
    def __init__(self):
        self._async_backend = AsyncElasticsearchBackend()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="es-sync-wrapper")
        self._own_loop = None
        self._initialized = False
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Synchronous initialize that handles event loop conflicts"""
        debug_logger.debug("Initializing sync Elasticsearch wrapper")
        
        # Check if we're in an async context
        try:
            # Try to get running loop
            current_loop = asyncio.get_running_loop()
            debug_logger.debug("Running in async context, using executor")
            
            # We're in an async context, run initialization in a separate thread
            def run_async_init():
                # Create new event loop in thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    new_loop.run_until_complete(self._async_backend.initialize(config))
                    return True
                except Exception as e:
                    debug_logger.error("Failed to initialize in thread: %s", e)
                    raise
                finally:
                    new_loop.close()
            
            # Run in thread to avoid event loop conflict
            future = self._executor.submit(run_async_init)
            result = future.result(timeout=60)  # 60 second timeout
            self._initialized = True
            
        except RuntimeError:
            # No event loop running, we can create our own
            debug_logger.debug("No async context, creating own event loop")
            self._own_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._own_loop)
            try:
                self._own_loop.run_until_complete(self._async_backend.initialize(config))
                self._initialized = True
            except Exception as e:
                self._own_loop.close()
                self._own_loop = None
                raise
        
        debug_logger.info("Sync Elasticsearch wrapper initialized successfully")
    
    def write_log(self, log_entry: Dict[str, Any]) -> None:
        """Synchronous write_log that handles event loop conflicts"""
        if not self._initialized:
            debug_logger.warning("Backend not initialized, dropping log entry")
            return
        
        try:
            # Check if we're in an async context
            current_loop = asyncio.get_running_loop()
            
            # We're in an async context, run in executor
            def run_async_write():
                # Create new event loop in thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    # Create a new async backend instance for this thread
                    temp_backend = AsyncElasticsearchBackend()
                    temp_backend.es_client = self._async_backend.es_client
                    temp_backend.index_pattern = self._async_backend.index_pattern
                    temp_backend._config = self._async_backend._config
                    temp_backend._executor = self._async_backend._executor
                    
                    new_loop.run_until_complete(temp_backend.write_log(log_entry))
                except Exception as e:
                    debug_logger.error("Failed to write in thread: %s", e)
                    print(f"Failed to write to Elasticsearch: {e}")
                    print(json.dumps(log_entry, default=str))
                finally:
                    new_loop.close()
            
            # Submit to executor to avoid blocking
            self._executor.submit(run_async_write)
            
        except RuntimeError:
            # No event loop, use our own
            if self._own_loop and not self._own_loop.is_closed():
                try:
                    self._own_loop.run_until_complete(self._async_backend.write_log(log_entry))
                except Exception as e:
                    debug_logger.error("Failed to write with own loop: %s", e)
                    print(f"Failed to write to Elasticsearch: {e}")
                    print(json.dumps(log_entry, default=str))
            else:
                debug_logger.error("No event loop available for writing")
                print(f"No event loop available, dropping log entry")
                print(json.dumps(log_entry, default=str))
    
    def close(self) -> None:
        """Synchronous close that handles event loop conflicts"""
        debug_logger.debug("Closing sync Elasticsearch wrapper")
        
        if not self._initialized:
            return
        
        try:
            # Check if we're in an async context
            current_loop = asyncio.get_running_loop()
            
            # We're in an async context, run in executor
            def run_async_close():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    new_loop.run_until_complete(self._async_backend.close())
                except Exception as e:
                    debug_logger.error("Failed to close in thread: %s", e)
                finally:
                    new_loop.close()
            
            future = self._executor.submit(run_async_close)
            future.result(timeout=30)  # 30 second timeout for close
            
        except RuntimeError:
            # No event loop, use our own
            if self._own_loop and not self._own_loop.is_closed():
                try:
                    self._own_loop.run_until_complete(self._async_backend.close())
                except Exception as e:
                    debug_logger.error("Failed to close with own loop: %s", e)
                finally:
                    self._own_loop.close()
                    self._own_loop = None
        
        # Shutdown our executor
        self._executor.shutdown(wait=True)
        self._initialized = False
        
        debug_logger.info("Sync Elasticsearch wrapper closed")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        if self._initialized:
            return self._async_backend.get_health_status()
        return {"healthy": False, "initialized": False}