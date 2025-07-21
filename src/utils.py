"""
Utility functions and logging configuration for the document-based question answering system.
"""

import gc
import logging
import logging.handlers
import os
import time
from functools import wraps
from pathlib import Path
from typing import Any

import psutil


def setup_logging(
    log_level: str = "INFO",
    log_file: str | None = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> None:
    """
    Set up centralized logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        log_format: Log message format
    """
    # Convert string level to logging constant
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(level=level, format=log_format, handlers=[])

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(log_format))

    # Add console handler to root logger
    logging.getLogger().addHandler(console_handler)

    # Add file handler if log_file is specified
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)

    # Set specific logger levels
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    logging.info(f"Logging configured with level: {log_level}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_memory_usage(logger: logging.Logger, context: str = "") -> None:
    """
    Log current memory usage.

    Args:
        logger: Logger instance
        context: Context string for the log message
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024

    logger.info(f"{context} Memory usage: {memory_mb:.1f} MB")


def log_performance(func):
    """
    Decorator to log function performance metrics.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            end_memory = psutil.Process(os.getpid()).memory_info().rss
            duration = end_time - start_time
            memory_diff = (end_memory - start_memory) / 1024 / 1024

            logger.info(
                f"{func.__name__} completed in {duration:.2f}s, "
                f"memory change: {memory_diff:+.1f} MB"
            )

    return wrapper


def batch_process(
    items: list,
    batch_size: int,
    process_func,
    logger: logging.Logger,
    description: str = "Processing",
) -> list:
    """
    Process items in batches with progress logging.

    Args:
        items: List of items to process
        batch_size: Size of each batch
        process_func: Function to apply to each batch
        logger: Logger instance
        description: Description for progress logging

    Returns:
        List of processed results
    """
    results = []
    total_items = len(items)

    logger.info(
        f"Starting {description}: {total_items} items in batches of {batch_size}"
    )

    for i in range(0, total_items, batch_size):
        batch = items[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_items + batch_size - 1) // batch_size

        logger.info(
            f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)"
        )

        batch_start = time.time()
        batch_results = process_func(batch)
        batch_duration = time.time() - batch_start

        results.extend(batch_results)

        logger.info(
            f"Batch {batch_num} completed in {batch_duration:.2f}s, "
            f"processed {len(batch_results)} items"
        )

    logger.info(f"{description} completed: {len(results)} total results")
    return results


def optimize_memory():
    """
    Perform memory optimization operations.
    """
    # Force garbage collection
    gc.collect()

    # Log memory usage after optimization
    logger = get_logger(__name__)
    log_memory_usage(logger, "After memory optimization")


def create_cache_directory(cache_dir: str) -> Path:
    """
    Create and validate cache directory.

    Args:
        cache_dir: Cache directory path

    Returns:
        Path to cache directory
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    logger = get_logger(__name__)
    logger.info(f"Cache directory ready: {cache_path}")

    return cache_path


def get_system_info() -> dict[str, Any]:
    """
    Get system information for logging.

    Returns:
        Dictionary with system information
    """
    return {
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
        "memory_available": psutil.virtual_memory().available
        / 1024
        / 1024
        / 1024,  # GB
        "disk_usage": psutil.disk_usage("/").percent
        if os.name != "nt"
        else psutil.disk_usage("C:\\").percent,
    }


def log_system_info(logger: logging.Logger) -> None:
    """
    Log system information.

    Args:
        logger: Logger instance
    """
    info = get_system_info()
    logger.info(
        f"System info: {info['cpu_count']} CPUs, "
        f"{info['memory_total']:.1f}GB total memory, "
        f"{info['memory_available']:.1f}GB available, "
        f"{info['disk_usage']:.1f}% disk usage"
    )


class ProgressTracker:
    """Track and log progress of long-running operations."""

    def __init__(
        self, total_items: int, logger: logging.Logger, description: str = "Processing"
    ):
        self.total_items = total_items
        self.processed_items = 0
        self.logger = logger
        self.description = description
        self.start_time = time.time()
        self.last_log_time = self.start_time

    def update(self, count: int = 1) -> None:
        """Update progress and log periodically."""
        self.processed_items += count

        current_time = time.time()
        if current_time - self.last_log_time >= 5.0:  # Log every 5 seconds
            progress = (self.processed_items / self.total_items) * 100
            elapsed = current_time - self.start_time
            rate = self.processed_items / elapsed if elapsed > 0 else 0

            self.logger.info(
                f"{self.description}: {progress:.1f}% complete "
                f"({self.processed_items}/{self.total_items}), "
                f"rate: {rate:.1f} items/sec"
            )
            self.last_log_time = current_time

    def finish(self) -> None:
        """Log completion statistics."""
        total_time = time.time() - self.start_time
        avg_rate = self.processed_items / total_time if total_time > 0 else 0

        self.logger.info(
            f"{self.description} completed: {self.processed_items} items "
            f"in {total_time:.2f}s (avg: {avg_rate:.1f} items/sec)"
        )
