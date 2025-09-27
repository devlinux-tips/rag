"""
Batch processing system for large-scale document processing.
Optimized for 300K document scale with configurable memory management and parallelization.
"""

import asyncio
import gc
import os
import time
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import psutil

from ..utils.logging_factory import (
    get_system_logger,
    log_component_end,
    log_component_start,
    log_data_transformation,
    log_error_context,
    log_performance_metric,
)


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing system."""

    enabled: bool
    document_batch_size: int
    embedding_batch_size: int
    vector_insert_batch_size: int
    max_parallel_workers: int
    memory_limit_gb: int
    memory_check_interval: int
    force_gc_interval: int
    checkpoint_interval: int
    progress_report_interval: int
    enable_progress_bar: bool
    max_retry_attempts: int
    retry_delay_seconds: float
    skip_failed_documents: bool
    continue_on_batch_failure: bool
    prefetch_batches: int
    async_processing: bool
    thread_pool_size: int

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BatchProcessingConfig":
        """Create config from configuration dictionary with fail-fast validation."""
        logger = get_system_logger()
        log_component_start("batch_processing_config", "from_config")

        # Direct access - fail if keys missing (no fallbacks)
        batch_config = config["batch_processing"]

        config_obj = cls(
            enabled=batch_config["enabled"],
            document_batch_size=batch_config["document_batch_size"],
            embedding_batch_size=batch_config["embedding_batch_size"],
            vector_insert_batch_size=batch_config["vector_insert_batch_size"],
            max_parallel_workers=batch_config["max_parallel_workers"],
            memory_limit_gb=batch_config["memory_limit_gb"],
            memory_check_interval=batch_config["memory_check_interval"],
            force_gc_interval=batch_config["force_gc_interval"],
            checkpoint_interval=batch_config["checkpoint_interval"],
            progress_report_interval=batch_config["progress_report_interval"],
            enable_progress_bar=batch_config["enable_progress_bar"],
            max_retry_attempts=batch_config["max_retry_attempts"],
            retry_delay_seconds=float(batch_config["retry_delay_seconds"]),
            skip_failed_documents=batch_config["skip_failed_documents"],
            continue_on_batch_failure=batch_config["continue_on_batch_failure"],
            prefetch_batches=batch_config["prefetch_batches"],
            async_processing=batch_config["async_processing"],
            thread_pool_size=batch_config["thread_pool_size"],
        )

        logger.debug(
            "batch_processing_config",
            "from_config",
            f"Batch sizes: docs={config_obj.document_batch_size}, "
            f"embeddings={config_obj.embedding_batch_size}, "
            f"vectors={config_obj.vector_insert_batch_size}",
        )

        logger.debug(
            "batch_processing_config",
            "from_config",
            f"Memory limit: {config_obj.memory_limit_gb}GB, workers: {config_obj.max_parallel_workers}",
        )

        log_component_end("batch_processing_config", "from_config", "Configuration loaded successfully")

        return config_obj

    @classmethod
    def from_validated_config(cls, config: dict[str, Any]) -> "BatchProcessingConfig":
        """Create config from validated configuration dictionary."""
        return cls.from_config(config)


@dataclass
class BatchProcessingResult:
    """Result from batch processing operation."""

    success: bool
    total_documents: int
    processed_documents: int
    failed_documents: int
    total_batches: int
    processed_batches: int
    failed_batches: int
    processing_time_seconds: float
    memory_peak_mb: float
    error_messages: list[str]
    checkpoint_data: dict[str, Any] | None = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_documents == 0:
            return 0.0
        return (self.processed_documents / self.total_documents) * 100.0

    @property
    def throughput_docs_per_second(self) -> float:
        """Calculate document processing throughput."""
        if self.processing_time_seconds == 0:
            return 0.0
        return self.processed_documents / self.processing_time_seconds


@dataclass
class ProcessingCheckpoint:
    """Checkpoint data for resuming processing."""

    processed_count: int
    failed_count: int
    last_processed_file: str
    timestamp: float
    memory_usage_mb: float
    batch_number: int
    metadata: dict[str, Any]


class DocumentProcessor(Protocol):
    """Protocol for document processing functions."""

    def __call__(self, file_path: Path, **kwargs) -> dict[str, Any]:
        """Process a single document."""
        ...


class BatchProcessor:
    """High-performance batch processor for large document collections."""

    def __init__(self, config: BatchProcessingConfig):
        """Initialize batch processor."""
        self.config = config
        self.logger = get_system_logger()

        log_component_start(
            "batch_processor",
            "init",
            enabled=config.enabled,
            document_batch_size=config.document_batch_size,
            memory_limit_gb=config.memory_limit_gb,
        )

        # Initialize memory tracking
        self.process = psutil.Process(os.getpid())
        self.peak_memory_mb = 0.0
        self.batch_counter = 0
        self.processed_documents = 0
        self.failed_documents = 0
        self.error_messages: list[str] = []

        # Initialize progress tracking
        self.start_time = 0.0
        self.last_progress_report = 0.0
        self.last_memory_check = 0
        self.last_gc_force = 0

        self.logger.info(
            "batch_processor", "init", f"Batch processor initialized with {config.max_parallel_workers} workers"
        )

        log_component_end("batch_processor", "init", "Batch processor initialized")

    def process_documents(
        self,
        file_paths: list[Path],
        processor_func: DocumentProcessor,
        progress_callback: Callable[[int, int], None] | None = None,
        checkpoint_callback: Callable[[ProcessingCheckpoint], None] | None = None,
        **processor_kwargs,
    ) -> BatchProcessingResult:
        """
        Process documents in batches with full optimization.

        Args:
            file_paths: List of document file paths to process
            processor_func: Function to process individual documents
            progress_callback: Optional callback for progress updates
            checkpoint_callback: Optional callback for checkpoint data
            **processor_kwargs: Additional arguments for processor function

        Returns:
            BatchProcessingResult with comprehensive metrics
        """
        logger = get_system_logger()
        log_component_start(
            "batch_processor",
            "process_documents",
            total_files=len(file_paths),
            batch_size=self.config.document_batch_size,
        )

        if not self.config.enabled:
            logger.warning(
                "batch_processor", "process_documents", "Batch processing is disabled, processing sequentially"
            )
            return self._process_sequential(file_paths, processor_func, **processor_kwargs)

        # Initialize processing state
        self.start_time = time.time()
        self.processed_documents = 0
        self.failed_documents = 0
        self.error_messages = []
        self.batch_counter = 0
        self.peak_memory_mb = 0.0

        total_files = len(file_paths)
        total_batches = (total_files + self.config.document_batch_size - 1) // self.config.document_batch_size

        logger.info(
            "batch_processor", "process_documents", f"Processing {total_files} files in {total_batches} batches"
        )

        try:
            if self.config.async_processing:
                result = asyncio.run(
                    self._process_async(
                        file_paths, processor_func, progress_callback, checkpoint_callback, **processor_kwargs
                    )
                )
            else:
                result = self._process_threaded(
                    file_paths, processor_func, progress_callback, checkpoint_callback, **processor_kwargs
                )

            processing_time = time.time() - self.start_time

            final_result = BatchProcessingResult(
                success=result["success"],
                total_documents=total_files,
                processed_documents=self.processed_documents,
                failed_documents=self.failed_documents,
                total_batches=total_batches,
                processed_batches=result["processed_batches"],
                failed_batches=result["failed_batches"],
                processing_time_seconds=processing_time,
                memory_peak_mb=self.peak_memory_mb,
                error_messages=self.error_messages,
            )

            log_performance_metric(
                "batch_processor",
                "processing_complete",
                "throughput_docs_per_second",
                final_result.throughput_docs_per_second,
                processed_files=self.processed_documents,
                total_files=total_files,
                processing_time_seconds=processing_time,
            )

            log_component_end(
                "batch_processor",
                "process_documents",
                f"Processing complete: {final_result.success_rate:.1f}% success rate",
            )

            return final_result

        except Exception as e:
            processing_time = time.time() - self.start_time
            error_msg = f"Batch processing failed: {str(e)}"

            logger.error("batch_processor", "process_documents", error_msg)
            log_error_context(
                "batch_processor",
                "process_documents",
                e,
                {"total_files": total_files, "processed": self.processed_documents, "failed": self.failed_documents},
            )

            return BatchProcessingResult(
                success=False,
                total_documents=total_files,
                processed_documents=self.processed_documents,
                failed_documents=self.failed_documents,
                total_batches=total_batches,
                processed_batches=0,
                failed_batches=total_batches,
                processing_time_seconds=processing_time,
                memory_peak_mb=self.peak_memory_mb,
                error_messages=[error_msg],
            )

    async def _process_async(
        self,
        file_paths: list[Path],
        processor_func: DocumentProcessor,
        progress_callback: Callable[[int, int], None] | None,
        checkpoint_callback: Callable[[ProcessingCheckpoint], None] | None,
        **processor_kwargs,
    ) -> dict[str, Any]:
        """Process documents using async/await pattern."""
        logger = get_system_logger()
        log_component_start("batch_processor", "async_processing", total_files=len(file_paths))

        processed_batches = 0
        failed_batches = 0

        try:
            # Create batches
            batches = list(self._create_batches(file_paths, self.config.document_batch_size))

            # Process batches with controlled concurrency
            semaphore = asyncio.Semaphore(self.config.max_parallel_workers)

            async def process_batch_async(batch: list[Path]) -> bool:
                async with semaphore:
                    return await asyncio.to_thread(self._process_batch, batch, processor_func, **processor_kwargs)

            # Execute batches
            tasks = [process_batch_async(batch) for batch in batches]

            for i, task in enumerate(asyncio.as_completed(tasks)):
                try:
                    batch_success = await task
                    if batch_success:
                        processed_batches += 1
                    else:
                        failed_batches += 1

                    # Progress and checkpoint handling
                    if progress_callback:
                        progress_callback(self.processed_documents, len(file_paths))

                    if checkpoint_callback and (i + 1) % self.config.checkpoint_interval == 0:
                        checkpoint = self._create_checkpoint(
                            file_paths[min(i * self.config.document_batch_size, len(file_paths) - 1)]
                        )
                        checkpoint_callback(checkpoint)

                except Exception as e:
                    failed_batches += 1
                    self.error_messages.append(f"Batch {i} failed: {str(e)}")
                    if not self.config.continue_on_batch_failure:
                        break

            log_component_end(
                "batch_processor",
                "async_processing",
                f"Async processing complete: {processed_batches} successful, {failed_batches} failed",
            )

            return {
                "success": failed_batches == 0,
                "processed_batches": processed_batches,
                "failed_batches": failed_batches,
            }

        except Exception as e:
            logger.error("batch_processor", "async_processing", f"Async processing failed: {str(e)}")
            raise

    def _process_threaded(
        self,
        file_paths: list[Path],
        processor_func: DocumentProcessor,
        progress_callback: Callable[[int, int], None] | None,
        checkpoint_callback: Callable[[ProcessingCheckpoint], None] | None,
        **processor_kwargs,
    ) -> dict[str, Any]:
        """Process documents using thread pool."""
        logger = get_system_logger()
        log_component_start("batch_processor", "threaded_processing", total_files=len(file_paths))

        processed_batches = 0
        failed_batches = 0

        try:
            # Create batches
            batches = list(self._create_batches(file_paths, self.config.document_batch_size))

            with ThreadPoolExecutor(max_workers=self.config.max_parallel_workers) as executor:
                # Submit all batch processing tasks
                future_to_batch = {
                    executor.submit(self._process_batch, batch, processor_func, **processor_kwargs): i
                    for i, batch in enumerate(batches)
                }

                # Process completed batches
                for future in as_completed(future_to_batch):
                    batch_index = future_to_batch[future]

                    try:
                        batch_success = future.result()
                        if batch_success:
                            processed_batches += 1
                        else:
                            failed_batches += 1

                        # Memory management
                        self._check_memory_usage()

                        # Progress reporting
                        if progress_callback and self.batch_counter % self.config.progress_report_interval == 0:
                            progress_callback(self.processed_documents, len(file_paths))

                        # Checkpoint creation
                        if checkpoint_callback and self.batch_counter % self.config.checkpoint_interval == 0:
                            last_file_index = min(
                                (batch_index + 1) * self.config.document_batch_size - 1, len(file_paths) - 1
                            )
                            checkpoint = self._create_checkpoint(file_paths[last_file_index])
                            checkpoint_callback(checkpoint)

                    except Exception as e:
                        failed_batches += 1
                        error_msg = f"Batch {batch_index} failed: {str(e)}"
                        self.error_messages.append(error_msg)
                        logger.error("batch_processor", "process_batch", error_msg)

                        if not self.config.continue_on_batch_failure:
                            logger.warning(
                                "batch_processor", "threaded_processing", "Stopping processing due to batch failure"
                            )
                            break

            log_component_end(
                "batch_processor",
                "threaded_processing",
                f"Threaded processing complete: {processed_batches} successful, {failed_batches} failed",
            )

            return {
                "success": failed_batches == 0,
                "processed_batches": processed_batches,
                "failed_batches": failed_batches,
            }

        except Exception as e:
            logger.error("batch_processor", "threaded_processing", f"Threaded processing failed: {str(e)}")
            raise

    def _process_batch(self, batch_files: list[Path], processor_func: DocumentProcessor, **processor_kwargs) -> bool:
        """Process a single batch of documents."""
        logger = get_system_logger()
        self.batch_counter += 1
        batch_id = self.batch_counter

        log_component_start("batch_processor", "process_batch", batch_id=batch_id, batch_size=len(batch_files))

        batch_success = True
        batch_processed = 0
        batch_failed = 0

        try:
            for file_path in batch_files:
                retry_count = 0
                file_processed = False

                while retry_count <= self.config.max_retry_attempts and not file_processed:
                    try:
                        # Process single document
                        result = processor_func(file_path, **processor_kwargs)

                        if result.get("success", True):
                            batch_processed += 1
                            self.processed_documents += 1
                            file_processed = True

                            logger.trace("batch_processor", "process_file", f"Processed: {file_path.name}")
                        else:
                            raise Exception(f"Processor returned failure for {file_path}")

                    except Exception as e:
                        retry_count += 1
                        error_msg = f"File {file_path.name} failed (attempt {retry_count}): {str(e)}"

                        if retry_count <= self.config.max_retry_attempts:
                            logger.warning("batch_processor", "retry_file", error_msg)
                            time.sleep(self.config.retry_delay_seconds)
                        else:
                            batch_failed += 1
                            self.failed_documents += 1
                            self.error_messages.append(error_msg)
                            logger.error("batch_processor", "file_failed", error_msg)

                            if not self.config.skip_failed_documents:
                                batch_success = False
                                break

            # Batch completion logging
            log_data_transformation(
                "batch_processor",
                "batch_complete",
                f"batch[{len(batch_files)}]",
                f"processed[{batch_processed}] failed[{batch_failed}]",
            )

            log_component_end(
                "batch_processor",
                "process_batch",
                f"Batch {batch_id}: {batch_processed} processed, {batch_failed} failed",
            )

            return batch_success

        except Exception as e:
            error_msg = f"Batch {batch_id} processing failed: {str(e)}"
            logger.error("batch_processor", "process_batch", error_msg)
            log_error_context(
                "batch_processor", "process_batch", e, {"batch_id": batch_id, "batch_size": len(batch_files)}
            )
            return False

    def _process_sequential(
        self, file_paths: list[Path], processor_func: DocumentProcessor, **processor_kwargs
    ) -> BatchProcessingResult:
        """Fallback sequential processing when batch processing is disabled."""
        get_system_logger()
        log_component_start("batch_processor", "sequential_processing", total_files=len(file_paths))

        self.start_time = time.time()
        processed = 0
        failed = 0

        for file_path in file_paths:
            try:
                result = processor_func(file_path, **processor_kwargs)
                if result.get("success", True):
                    processed += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                self.error_messages.append(f"Sequential processing failed for {file_path}: {str(e)}")

        processing_time = time.time() - self.start_time

        log_component_end(
            "batch_processor",
            "sequential_processing",
            f"Sequential processing complete: {processed} processed, {failed} failed",
        )

        return BatchProcessingResult(
            success=failed == 0,
            total_documents=len(file_paths),
            processed_documents=processed,
            failed_documents=failed,
            total_batches=1,
            processed_batches=1 if failed == 0 else 0,
            failed_batches=1 if failed > 0 else 0,
            processing_time_seconds=processing_time,
            memory_peak_mb=0.0,
            error_messages=self.error_messages,
        )

    def _create_batches(self, items: list[Path], batch_size: int) -> Generator[list[Path], None, None]:
        """Create batches from list of items."""
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    def _check_memory_usage(self) -> None:
        """Check and manage memory usage."""
        if self.batch_counter % self.config.memory_check_interval == 0:
            memory_info = self.process.memory_info()
            current_memory_mb = memory_info.rss / 1024 / 1024

            # Update peak memory
            if current_memory_mb > self.peak_memory_mb:
                self.peak_memory_mb = current_memory_mb

            # Log memory usage
            log_performance_metric(
                "batch_processor",
                "memory_usage",
                "current_memory_mb",
                current_memory_mb,
                peak_memory_mb=self.peak_memory_mb,
                memory_limit_mb=self.config.memory_limit_gb * 1024,
            )

            # Check memory limit
            if current_memory_mb > (self.config.memory_limit_gb * 1024 * 0.95):  # 95% of limit
                self.logger.warning(
                    "batch_processor",
                    "memory_check",
                    f"Memory usage high: {current_memory_mb:.1f}MB (limit: {self.config.memory_limit_gb * 1024:.1f}MB)",
                )

        # Force garbage collection periodically
        if self.batch_counter % self.config.force_gc_interval == 0:
            gc.collect()
            logger = get_system_logger()
            logger.debug("batch_processor", "memory_management", "Forced garbage collection")

    def _create_checkpoint(self, last_file: Path) -> ProcessingCheckpoint:
        """Create checkpoint data for resume capability."""
        memory_info = self.process.memory_info()
        current_memory_mb = memory_info.rss / 1024 / 1024

        return ProcessingCheckpoint(
            processed_count=self.processed_documents,
            failed_count=self.failed_documents,
            last_processed_file=str(last_file),
            timestamp=time.time(),
            memory_usage_mb=current_memory_mb,
            batch_number=self.batch_counter,
            metadata={
                "peak_memory_mb": self.peak_memory_mb,
                "processing_rate": self.processed_documents / max(time.time() - self.start_time, 1),
                "error_count": len(self.error_messages),
            },
        )


def create_batch_processor(config: dict[str, Any]) -> BatchProcessor:
    """
    Factory function to create batch processor.

    Args:
        config: Configuration dictionary

    Returns:
        BatchProcessor instance
    """
    get_system_logger()
    log_component_start("batch_processor_factory", "create_processor")

    batch_config = BatchProcessingConfig.from_config(config)
    processor = BatchProcessor(batch_config)

    log_component_end(
        "batch_processor_factory",
        "create_processor",
        f"Batch processor created with {batch_config.document_batch_size} document batch size",
    )

    return processor


# Utility functions for batch operations


def estimate_batch_memory_usage(
    document_count: int, avg_document_size_kb: float, embedding_dimensions: int = 1024
) -> float:
    """
    Estimate memory usage for batch processing.

    Args:
        document_count: Number of documents in batch
        avg_document_size_kb: Average document size in KB
        embedding_dimensions: Embedding vector dimensions

    Returns:
        Estimated memory usage in MB
    """
    # Text storage (documents)
    text_memory_mb = (document_count * avg_document_size_kb) / 1024

    # Embedding vectors (float32 = 4 bytes per dimension)
    embedding_memory_mb = (document_count * embedding_dimensions * 4) / (1024 * 1024)

    # Processing overhead (metadata, temporary objects, etc.)
    overhead_factor = 1.5

    total_memory_mb = (text_memory_mb + embedding_memory_mb) * overhead_factor

    return total_memory_mb


def optimize_batch_sizes(
    total_documents: int, available_memory_gb: float, avg_document_size_kb: float = 50, embedding_dimensions: int = 1024
) -> dict[str, int]:
    """
    Calculate optimal batch sizes for given constraints.

    Args:
        total_documents: Total number of documents to process
        available_memory_gb: Available memory in GB
        avg_document_size_kb: Average document size in KB
        embedding_dimensions: Embedding vector dimensions

    Returns:
        Dictionary with optimal batch sizes
    """
    logger = get_system_logger()
    log_component_start(
        "batch_optimizer", "optimize_sizes", total_documents=total_documents, memory_gb=available_memory_gb
    )

    available_memory_mb = available_memory_gb * 1024

    # Start with small batch and scale up
    optimal_document_batch = 10
    for batch_size in [10, 25, 50, 100, 200, 500, 1000]:
        estimated_memory = estimate_batch_memory_usage(batch_size, avg_document_size_kb, embedding_dimensions)

        if estimated_memory < available_memory_mb * 0.8:  # Use 80% of available memory
            optimal_document_batch = batch_size
        else:
            break

    # Calculate related batch sizes
    optimal_embedding_batch = min(optimal_document_batch * 4, 1000)  # Embeddings can be batched larger
    optimal_vector_batch = min(optimal_document_batch * 10, 5000)  # Vector inserts can be even larger

    result = {
        "document_batch_size": optimal_document_batch,
        "embedding_batch_size": optimal_embedding_batch,
        "vector_insert_batch_size": optimal_vector_batch,
        "estimated_batches": (total_documents + optimal_document_batch - 1) // optimal_document_batch,
    }

    logger.info(
        "batch_optimizer",
        "optimize_sizes",
        f"Optimal sizes: docs={optimal_document_batch}, "
        f"embeddings={optimal_embedding_batch}, "
        f"vectors={optimal_vector_batch}",
    )

    log_component_end("batch_optimizer", "optimize_sizes", "Batch sizes optimized")

    return result
