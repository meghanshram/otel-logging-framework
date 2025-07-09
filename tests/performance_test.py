#!/usr/bin/env python3
"""
OTEL Logger Performance Test Suite
Tests sync vs async logging performance across different scenarios:
- Single threaded
- Multi-threaded
- Multi-processed
- Hybrid async/threading
- High throughput scenarios
"""

import asyncio
import threading
import multiprocessing
import time
import statistics
import json
import sys
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import traceback
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import psutil
import tracemalloc

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
    alog_info,
    aotel_span,
)


@dataclass
class PerformanceMetrics:
    """Store performance metrics for a test run"""
    test_name: str
    total_time: float
    logs_per_second: float
    memory_peak_mb: float
    memory_current_mb: float
    cpu_percent: float
    thread_count: int
    log_count: int
    error_count: int
    backend_type: str

class PerformanceTestSuite:
    def __init__(self, log_count: int = 1000, backend_type: str = "filesystem"):
        self.log_count = log_count
        self.backend_type = backend_type
        self.results: List[PerformanceMetrics] = []
        self.process = psutil.Process()
        
        # Configure logger based on backend type
        if backend_type == "filesystem":
            self.logger = configure_logger(
                backend_type="filesystem",
                backend_config={
                    "log_file": f"./performance_test_{int(time.time())}.jsonl",
                    "max_bytes": 100 * 1024 * 1024,  # 100MB
                    "backup_count": 5
                },
                service_name="PerformanceTest",
                service_version="1.0.0",
                log_level="INFO",
                enable_console=False
            )
        elif backend_type == "elasticsearch":
            self.logger = configure_logger(
                config_file="logger_config.yaml",
                service_name="PerformanceTest",
                service_version="1.0.0",
                log_level="INFO",
                enable_console=False
            )
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")

    def get_system_metrics(self) -> Tuple[float, float, float, int]:
        """Get current system metrics"""
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent()
        thread_count = self.process.num_threads()
        
        # Get peak memory if tracemalloc is running
        try:
            current, peak = tracemalloc.get_traced_memory()
            peak_mb = peak / 1024 / 1024
        except:
            peak_mb = memory_mb
            
        return memory_mb, peak_mb, cpu_percent, thread_count

    def log_test_data(self, test_id: str, log_index: int, thread_id: int = 0, process_id: int = 0):
        """Generate test log entry"""
        return {
            "test_id": test_id,
            "log_index": log_index,
            "thread_id": thread_id,
            "process_id": process_id,
            "timestamp": time.time(),
            "data": f"Performance test log entry {log_index}",
            "nested_data": {
                "level1": {"level2": {"value": f"nested_value_{log_index}"}},
                "array": [i for i in range(5)],
                "metadata": {"important": True, "priority": "high"}
            }
        }

    def test_sync_single_thread(self) -> PerformanceMetrics:
        """Test synchronous logging in single thread"""
        print("Running: Sync Single Thread Test...")
        tracemalloc.start()
        
        start_time = time.time()
        error_count = 0
        
        test_id = f"sync_single_{int(time.time())}"
        
        for i in range(self.log_count):
            try:
                log_info(
                    f"Sync single thread log {i}",
                    attributes=self.log_test_data(test_id, i),
                    correlation_id=f"{test_id}_{i}"
                )
            except Exception as e:
                error_count += 1
                print(f"Error in sync single thread: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        logs_per_second = self.log_count / total_time if total_time > 0 else 0
        
        memory_mb, peak_mb, cpu_percent, thread_count = self.get_system_metrics()
        tracemalloc.stop()
        
        return PerformanceMetrics(
            test_name="Sync Single Thread",
            total_time=total_time,
            logs_per_second=logs_per_second,
            memory_peak_mb=peak_mb,
            memory_current_mb=memory_mb,
            cpu_percent=cpu_percent,
            thread_count=thread_count,
            log_count=self.log_count,
            error_count=error_count,
            backend_type=self.backend_type
        )

    async def test_async_single_thread(self) -> PerformanceMetrics:
        """Test asynchronous logging in single thread"""
        print("Running: Async Single Thread Test...")
        tracemalloc.start()
        
        start_time = time.time()
        error_count = 0
        
        test_id = f"async_single_{int(time.time())}"
        
        for i in range(self.log_count):
            try:
                await alog_info(
                    f"Async single thread log {i}",
                    attributes=self.log_test_data(test_id, i),
                    correlation_id=f"{test_id}_{i}"
                )
            except Exception as e:
                error_count += 1
                print(f"Error in async single thread: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        logs_per_second = self.log_count / total_time if total_time > 0 else 0
        
        memory_mb, peak_mb, cpu_percent, thread_count = self.get_system_metrics()
        tracemalloc.stop()
        
        return PerformanceMetrics(
            test_name="Async Single Thread",
            total_time=total_time,
            logs_per_second=logs_per_second,
            memory_peak_mb=peak_mb,
            memory_current_mb=memory_mb,
            cpu_percent=cpu_percent,
            thread_count=thread_count,
            log_count=self.log_count,
            error_count=error_count,
            backend_type=self.backend_type
        )

    def sync_worker_thread(self, thread_id: int, logs_per_thread: int, test_id: str) -> Tuple[int, float]:
        """Worker function for sync multi-threading test"""
        start_time = time.time()
        error_count = 0
        
        for i in range(logs_per_thread):
            try:
                log_info(
                    f"Sync multi-thread log {i} from thread {thread_id}",
                    attributes=self.log_test_data(test_id, i, thread_id),
                    correlation_id=f"{test_id}_{thread_id}_{i}"
                )
            except Exception as e:
                error_count += 1
        
        return error_count, time.time() - start_time

    def test_sync_multi_thread(self, num_threads: int = 4) -> PerformanceMetrics:
        """Test synchronous logging with multiple threads"""
        print(f"Running: Sync Multi-Thread Test ({num_threads} threads)...")
        tracemalloc.start()
        
        start_time = time.time()
        logs_per_thread = self.log_count // num_threads
        test_id = f"sync_multi_{int(time.time())}"
        
        total_errors = 0
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(self.sync_worker_thread, i, logs_per_thread, test_id)
                for i in range(num_threads)
            ]
            
            for future in as_completed(futures):
                errors, _ = future.result()
                total_errors += errors
        
        end_time = time.time()
        total_time = end_time - start_time
        logs_per_second = self.log_count / total_time if total_time > 0 else 0
        
        memory_mb, peak_mb, cpu_percent, thread_count = self.get_system_metrics()
        tracemalloc.stop()
        
        return PerformanceMetrics(
            test_name=f"Sync Multi-Thread ({num_threads})",
            total_time=total_time,
            logs_per_second=logs_per_second,
            memory_peak_mb=peak_mb,
            memory_current_mb=memory_mb,
            cpu_percent=cpu_percent,
            thread_count=thread_count,
            log_count=self.log_count,
            error_count=total_errors,
            backend_type=self.backend_type
        )

    async def async_worker_coroutine(self, worker_id: int, logs_per_worker: int, test_id: str) -> Tuple[int, float]:
        """Worker coroutine for async multi-coroutine test"""
        start_time = time.time()
        error_count = 0
        
        for i in range(logs_per_worker):
            try:
                await alog_info(
                    f"Async multi-coroutine log {i} from worker {worker_id}",
                    attributes=self.log_test_data(test_id, i, worker_id),
                    correlation_id=f"{test_id}_{worker_id}_{i}"
                )
            except Exception as e:
                error_count += 1
        
        return error_count, time.time() - start_time

    async def test_async_multi_coroutine(self, num_workers: int = 10) -> PerformanceMetrics:
        """Test asynchronous logging with multiple coroutines"""
        print(f"Running: Async Multi-Coroutine Test ({num_workers} workers)...")
        tracemalloc.start()
        
        start_time = time.time()
        logs_per_worker = self.log_count // num_workers
        test_id = f"async_multi_{int(time.time())}"
        
        # Create and run coroutines concurrently
        tasks = [
            self.async_worker_coroutine(i, logs_per_worker, test_id)
            for i in range(num_workers)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_errors = sum(result[0] for result in results if not isinstance(result, Exception))
        
        end_time = time.time()
        total_time = end_time - start_time
        logs_per_second = self.log_count / total_time if total_time > 0 else 0
        
        memory_mb, peak_mb, cpu_percent, thread_count = self.get_system_metrics()
        tracemalloc.stop()
        
        return PerformanceMetrics(
            test_name=f"Async Multi-Coroutine ({num_workers})",
            total_time=total_time,
            logs_per_second=logs_per_second,
            memory_peak_mb=peak_mb,
            memory_current_mb=memory_mb,
            cpu_percent=cpu_percent,
            thread_count=thread_count,
            log_count=self.log_count,
            error_count=total_errors,
            backend_type=self.backend_type
        )

    async def test_batch_async_logging(self, batch_size: int = 100) -> PerformanceMetrics:
        """Test batch async logging for high throughput"""
        print(f"Running: Batch Async Logging Test (batch size: {batch_size})...")
        tracemalloc.start()
        
        start_time = time.time()
        error_count = 0
        test_id = f"batch_async_{int(time.time())}"
        
        # Process logs in batches
        for batch_start in range(0, self.log_count, batch_size):
            batch_end = min(batch_start + batch_size, self.log_count)
            batch_tasks = []
            
            for i in range(batch_start, batch_end):
                task = alog_info(
                    f"Batch async log {i}",
                    attributes=self.log_test_data(test_id, i),
                    correlation_id=f"{test_id}_{i}"
                )
                batch_tasks.append(task)
            
            try:
                await asyncio.gather(*batch_tasks, return_exceptions=True)
            except Exception as e:
                error_count += 1
                print(f"Batch error: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        logs_per_second = self.log_count / total_time if total_time > 0 else 0
        
        memory_mb, peak_mb, cpu_percent, thread_count = self.get_system_metrics()
        tracemalloc.stop()
        
        return PerformanceMetrics(
            test_name=f"Batch Async (batch={batch_size})",
            total_time=total_time,
            logs_per_second=logs_per_second,
            memory_peak_mb=peak_mb,
            memory_current_mb=memory_mb,
            cpu_percent=cpu_percent,
            thread_count=thread_count,
            log_count=self.log_count,
            error_count=error_count,
            backend_type=self.backend_type
        )

    def test_span_performance(self) -> PerformanceMetrics:
        """Test span performance with nested operations"""
        print("Running: Span Performance Test...")
        tracemalloc.start()
        
        start_time = time.time()
        error_count = 0
        test_id = f"span_perf_{int(time.time())}"
        
        @otel_trace(operation_name="test_operation", include_args=True, include_result=True)
        def traced_operation(operation_id: int, data: dict) -> dict:
            time.sleep(0.001)  # Simulate some work
            return {"result": f"processed_{operation_id}", "data": data}
        
        span_count = self.log_count // 10  # Fewer spans but more expensive operations
        
        for i in range(span_count):
            try:
                with otel_span(f"performance_span_{i}", {"span_id": i}, f"{test_id}_{i}"):
                    # Simulate nested operations
                    for j in range(10):
                        traced_operation(j, self.log_test_data(test_id, i * 10 + j))
            except Exception as e:
                error_count += 1
                print(f"Span error: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        operations_per_second = self.log_count / total_time if total_time > 0 else 0
        
        memory_mb, peak_mb, cpu_percent, thread_count = self.get_system_metrics()
        tracemalloc.stop()
        
        return PerformanceMetrics(
            test_name="Span Performance",
            total_time=total_time,
            logs_per_second=operations_per_second,
            memory_peak_mb=peak_mb,
            memory_current_mb=memory_mb,
            cpu_percent=cpu_percent,
            thread_count=thread_count,
            log_count=self.log_count,
            error_count=error_count,
            backend_type=self.backend_type
        )

    async def test_hybrid_async_threading(self, num_threads: int = 4, coroutines_per_thread: int = 5) -> PerformanceMetrics:
        """Test hybrid async + threading approach"""
        print(f"Running: Hybrid Async+Threading Test ({num_threads} threads, {coroutines_per_thread} coroutines each)...")
        tracemalloc.start()
        
        start_time = time.time()
        test_id = f"hybrid_{int(time.time())}"
        
        async def thread_async_worker(thread_id: int):
            logs_per_coroutine = self.log_count // (num_threads * coroutines_per_thread)
            
            async def coroutine_worker(coroutine_id: int):
                error_count = 0
                for i in range(logs_per_coroutine):
                    try:
                        await alog_info(
                            f"Hybrid log {i} from thread {thread_id}, coroutine {coroutine_id}",
                            attributes=self.log_test_data(test_id, i, thread_id, coroutine_id),
                            correlation_id=f"{test_id}_{thread_id}_{coroutine_id}_{i}"
                        )
                    except Exception as e:
                        error_count += 1
                return error_count
            
            # Run coroutines within this thread
            tasks = [coroutine_worker(i) for i in range(coroutines_per_thread)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return sum(r for r in results if isinstance(r, int))
        
        # Run async workers in different threads
        def run_async_worker(thread_id: int):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(thread_async_worker(thread_id))
            finally:
                loop.close()
        
        total_errors = 0
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(run_async_worker, i) for i in range(num_threads)]
            for future in as_completed(futures):
                try:
                    errors = future.result()
                    total_errors += errors
                except Exception as e:
                    total_errors += 1
                    print(f"Hybrid worker error: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        logs_per_second = self.log_count / total_time if total_time > 0 else 0
        
        memory_mb, peak_mb, cpu_percent, thread_count = self.get_system_metrics()
        tracemalloc.stop()
        
        return PerformanceMetrics(
            test_name=f"Hybrid Async+Threading ({num_threads}x{coroutines_per_thread})",
            total_time=total_time,
            logs_per_second=logs_per_second,
            memory_peak_mb=peak_mb,
            memory_current_mb=memory_mb,
            cpu_percent=cpu_percent,
            thread_count=thread_count,
            log_count=self.log_count,
            error_count=total_errors,
            backend_type=self.backend_type
        )

    async def run_all_tests(self) -> List[PerformanceMetrics]:
        """Run all performance tests"""
        print(f"Starting Performance Test Suite with {self.log_count} logs per test")
        print(f"Backend: {self.backend_type}")
        print("=" * 60)
        
        # Single threaded tests
        self.results.append(self.test_sync_single_thread())
        self.results.append(await self.test_async_single_thread())
        
        # Multi-threaded/coroutine tests
        self.results.append(self.test_sync_multi_thread(4))
        self.results.append(await self.test_async_multi_coroutine(10))
        
        # High throughput tests
        self.results.append(await self.test_batch_async_logging(50))
        self.results.append(await self.test_batch_async_logging(100))
        
        # Span performance
        self.results.append(self.test_span_performance())
        
        # Hybrid approach
        self.results.append(await self.test_hybrid_async_threading(4, 5))
        
        return self.results

    def print_results(self):
        """Print formatted results"""
        print("\n" + "=" * 80)
        print("PERFORMANCE TEST RESULTS")
        print("=" * 80)
        
        # Sort by logs per second (descending)
        sorted_results = sorted(self.results, key=lambda x: x.logs_per_second, reverse=True)
        
        print(f"{'Test Name':<35} {'Logs/sec':<12} {'Total Time':<12} {'Memory (MB)':<12} {'CPU %':<8} {'Errors':<8}")
        print("-" * 80)
        
        for result in sorted_results:
            print(f"{result.test_name:<35} "
                  f"{result.logs_per_second:<12.1f} "
                  f"{result.total_time:<12.3f} "
                  f"{result.memory_peak_mb:<12.1f} "
                  f"{result.cpu_percent:<8.1f} "
                  f"{result.error_count:<8}")
        
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        
        # Calculate statistics
        throughputs = [r.logs_per_second for r in self.results]
        times = [r.total_time for r in self.results]
        memory_usage = [r.memory_peak_mb for r in self.results]
        
        fastest_test = max(self.results, key=lambda x: x.logs_per_second)
        slowest_test = min(self.results, key=lambda x: x.logs_per_second)
        
        print(f"Fastest Test: {fastest_test.test_name} ({fastest_test.logs_per_second:.1f} logs/sec)")
        print(f"Slowest Test: {slowest_test.test_name} ({slowest_test.logs_per_second:.1f} logs/sec)")
        print(f"Speed Improvement: {fastest_test.logs_per_second / slowest_test.logs_per_second:.1f}x")
        print()
        print(f"Average Throughput: {statistics.mean(throughputs):.1f} logs/sec")
        print(f"Median Throughput: {statistics.median(throughputs):.1f} logs/sec")
        print(f"Throughput Std Dev: {statistics.stdev(throughputs):.1f} logs/sec")
        print()
        print(f"Average Memory Usage: {statistics.mean(memory_usage):.1f} MB")
        print(f"Peak Memory Usage: {max(memory_usage):.1f} MB")
        print()
        print(f"Total Errors: {sum(r.error_count for r in self.results)}")

    def save_results_json(self, filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            filename = f"performance_results_{self.backend_type}_{int(time.time())}.json"
        
        results_data = {
            "test_config": {
                "log_count": self.log_count,
                "backend_type": self.backend_type,
                "timestamp": time.time()
            },
            "results": [
                {
                    "test_name": r.test_name,
                    "total_time": r.total_time,
                    "logs_per_second": r.logs_per_second,
                    "memory_peak_mb": r.memory_peak_mb,
                    "memory_current_mb": r.memory_current_mb,
                    "cpu_percent": r.cpu_percent,
                    "thread_count": r.thread_count,
                    "log_count": r.log_count,
                    "error_count": r.error_count,
                    "backend_type": r.backend_type
                } for r in self.results
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nResults saved to: {filename}")

async def main():
    """Main test execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OTEL Logger Performance Test Suite")
    parser.add_argument("--logs", type=int, default=1000, help="Number of logs per test (default: 1000)")
    parser.add_argument("--backend", choices=["filesystem", "elasticsearch"], default="filesystem", 
                       help="Backend type to test (default: filesystem)")
    parser.add_argument("--quick", action="store_true", help="Run quick test with fewer logs")
    
    args = parser.parse_args()
    
    log_count = 100 if args.quick else args.logs
    
    # Run the test suite
    test_suite = PerformanceTestSuite(log_count=log_count, backend_type=args.backend)
    
    try:
        await test_suite.run_all_tests()
        test_suite.print_results()
        test_suite.save_results_json()
        
        # Close logger
        test_suite.logger.close()
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed with error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())