#!/usr/bin/env python3
"""
Incremental Narodne Novine document processor.
Processes documents folder by folder with progress tracking and resumption support.
Enhanced with comprehensive performance monitoring and resource tracking.
"""

import asyncio
import json
import sys
import time
import psutil
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


# Add repository root to Python path
REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "services" / "rag-service"))

from src.utils.factories import create_complete_rag_system
from src.utils.logging_factory import get_system_logger


# Configuration
FEATURE_NAME = "narodne-novine"
LANGUAGE = "hr"
SCOPE = "feature"
BASE_DATA_DIR = Path("/home/rag/src/rag/services/rag-service/data/features/narodne_novine/documents/hr")
PROGRESS_FILE = Path("/home/rag/src/rag/logs/nn_processing_progress.json")
STATS_FILE = Path("/home/rag/src/rag/logs/nn_processing_stats.json")
LOG_FILE = Path("/home/rag/src/rag/logs/nn_incremental_processing.log")


@dataclass
class ResourceSnapshot:
    """System resource snapshot at a specific point in time."""
    timestamp: float
    cpu_percent: float
    ram_used_gb: float
    ram_percent: float
    swap_used_gb: float
    swap_percent: float
    process_cpu_percent: float
    process_memory_mb: float


@dataclass
class PhaseStats:
    """Statistics for a single processing phase."""
    phase_name: str
    start_time: float
    end_time: float
    duration: float
    resources_before: Dict
    resources_after: Dict


@dataclass
class ChunkStats:
    """Statistics about chunks in a folder."""
    total_chunks: int
    min_size: int
    max_size: int
    avg_size: float
    total_chars: int


@dataclass
class FolderStats:
    """Statistics for processing a single folder."""
    folder_id: str
    file_count: int
    document_count: int
    chunk_count: int
    processing_time: float
    docs_per_second: float
    chunks_per_second: float
    timestamp: float
    # Enhanced statistics
    chunk_stats: Optional[Dict] = None
    phase_timings: Optional[Dict] = None  # Dict[phase_name, duration]
    resources: Optional[Dict] = None  # Dict with resource snapshots


@dataclass
class SessionStats:
    """Overall statistics for entire processing session."""
    session_start: float
    session_end: float
    total_duration: float
    folders_processed: int
    total_files: int
    total_documents: int
    total_chunks: int
    avg_docs_per_folder: float
    avg_chunks_per_folder: float
    avg_processing_time_per_folder: float
    overall_docs_per_second: float
    overall_chunks_per_second: float
    folder_stats: List[Dict]


def load_progress():
    """Load processing progress from file."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading progress: {e}")
            return {"processed_folders": [], "last_folder": None, "last_update": None}
    return {"processed_folders": [], "last_folder": None, "last_update": None}


def save_progress(progress):
    """Save processing progress to file."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    progress["last_update"] = time.time()
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)
    print(f"Progress saved: {len(progress['processed_folders'])} folders completed")


def capture_resource_snapshot() -> ResourceSnapshot:
    """Capture current system resource usage."""
    try:
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # Find our process
        current_process = psutil.Process()
        process_cpu = current_process.cpu_percent(interval=0.1)
        process_mem_mb = current_process.memory_info().rss / 1024 / 1024

        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            ram_used_gb=mem.used / 1024**3,
            ram_percent=mem.percent,
            swap_used_gb=swap.used / 1024**3,
            swap_percent=swap.percent,
            process_cpu_percent=process_cpu,
            process_memory_mb=process_mem_mb
        )
    except Exception as e:
        # Fallback if psutil fails
        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=0.0,
            ram_used_gb=0.0,
            ram_percent=0.0,
            swap_used_gb=0.0,
            swap_percent=0.0,
            process_cpu_percent=0.0,
            process_memory_mb=0.0
        )


def log_resource_snapshot(logger, operation: str, snapshot: ResourceSnapshot, folder_id: str = ""):
    """Log resource snapshot in structured format for analysis."""
    logger.info(
        "resource_monitor",
        operation,
        f"RESOURCE_SNAPSHOT | folder={folder_id} | "
        f"cpu={snapshot.cpu_percent:.1f}% | ram={snapshot.ram_used_gb:.1f}GB ({snapshot.ram_percent:.1f}%) | "
        f"swap={snapshot.swap_used_gb:.1f}GB ({snapshot.swap_percent:.1f}%) | "
        f"proc_cpu={snapshot.process_cpu_percent:.1f}% | proc_mem={snapshot.process_memory_mb:.0f}MB"
    )


def get_folders_to_process():
    """Get list of ALL folders across ALL years that need processing.
    Processes from 2025 backwards to get most recent data first.
    """
    progress = load_progress()
    processed = set(progress["processed_folders"])

    all_folders = []

    # Iterate through all year directories in REVERSE order (2025 -> 2012)
    for year_dir in sorted(BASE_DATA_DIR.iterdir(), reverse=True):
        if not year_dir.is_dir():
            continue

        # Get all issue folders in this year (in reverse order - latest first)
        year_folders = sorted([d for d in year_dir.iterdir() if d.is_dir()], reverse=True)
        all_folders.extend(year_folders)

    # Filter out already processed folders
    # Use full path for uniqueness: "2025/121" instead of just "121"
    pending_folders = []
    for folder in all_folders:
        # Create unique ID: "year/folder_name" (e.g., "2025/121", "2024/001")
        folder_id = f"{folder.parent.name}/{folder.name}"
        if folder_id not in processed:
            pending_folders.append(folder)

    print(f"Total folders across all years: {len(all_folders)}")
    print(f"Already processed: {len(processed)}")
    print(f"Pending: {len(pending_folders)}")
    print(f"Processing order: 2025 -> 2012 (newest first)")

    return pending_folders


async def process_folder(rag_system, folder_path: Path, logger) -> FolderStats:
    """Process all documents in a single folder and return comprehensive statistics.

    Enhanced with:
    - Resource monitoring at each phase
    - Phase-level timing breakdown
    - Chunk statistics analysis
    """
    folder_id = f"{folder_path.parent.name}/{folder_path.name}"
    folder_start_time = time.time()

    # Capture initial resource state
    resources_initial = capture_resource_snapshot()
    log_resource_snapshot(logger, "folder_start", resources_initial, folder_id)

    logger.info("incremental_processor", f"process_folder.{folder_path.name}", f"Starting folder: {folder_id}")

    # Get all HTML files in folder
    html_files = sorted(folder_path.glob("*.html"))

    if not html_files:
        logger.warning("incremental_processor", f"process_folder.{folder_path.name}", f"No HTML files found in {folder_id}")
        return FolderStats(
            folder_id=folder_id,
            file_count=0,
            document_count=0,
            chunk_count=0,
            processing_time=0.0,
            docs_per_second=0.0,
            chunks_per_second=0.0,
            timestamp=time.time()
        )

    logger.info("incremental_processor", f"process_folder.{folder_path.name}", f"Found {len(html_files)} HTML files")

    try:
        # Track phase timings
        phase_timings = {}

        # PHASE 1: Document Extraction & Chunking
        phase1_start = time.time()
        resources_phase1_start = capture_resource_snapshot()
        log_resource_snapshot(logger, "phase1_extraction_start", resources_phase1_start, folder_id)

        logger.info("incremental_processor", f"phase1.{folder_path.name}", f"PHASE 1 START: Extraction & Chunking")

        # NOTE: We can't directly instrument RAG system phases without modifying it
        # So we'll track the overall add_documents call and log before/after
        result = await rag_system.add_documents([str(f) for f in html_files])

        resources_phase_end = capture_resource_snapshot()
        processing_time = time.time() - folder_start_time

        # Log completion with resource state
        log_resource_snapshot(logger, "processing_complete", resources_phase_end, folder_id)

        # Calculate statistics
        docs_per_sec = result.processed_documents / processing_time if processing_time > 0 else 0
        chunks_per_sec = result.total_chunks / processing_time if processing_time > 0 else 0

        # Try to extract chunk statistics if available
        chunk_stats_dict = None
        if hasattr(result, 'chunks') and result.chunks:
            chunk_sizes = [len(chunk.content) if hasattr(chunk, 'content') else 0 for chunk in result.chunks]
            if chunk_sizes:
                chunk_stats_dict = {
                    'total_chunks': len(chunk_sizes),
                    'min_size': min(chunk_sizes),
                    'max_size': max(chunk_sizes),
                    'avg_size': sum(chunk_sizes) / len(chunk_sizes),
                    'total_chars': sum(chunk_sizes)
                }
                logger.info(
                    "chunk_stats",
                    folder_path.name,
                    f"CHUNK_STATS | folder={folder_id} | chunks={chunk_stats_dict['total_chunks']} | "
                    f"avg_size={chunk_stats_dict['avg_size']:.0f} | min={chunk_stats_dict['min_size']} | "
                    f"max={chunk_stats_dict['max_size']} | total_chars={chunk_stats_dict['total_chars']}"
                )

        # Build comprehensive statistics
        stats = FolderStats(
            folder_id=folder_id,
            file_count=len(html_files),
            document_count=result.processed_documents,
            chunk_count=result.total_chunks,
            processing_time=processing_time,
            docs_per_second=docs_per_sec,
            chunks_per_second=chunks_per_sec,
            timestamp=time.time(),
            chunk_stats=chunk_stats_dict,
            phase_timings={'total': processing_time},
            resources={
                'initial': asdict(resources_initial),
                'final': asdict(resources_phase_end),
                'ram_delta_mb': (resources_phase_end.ram_used_gb - resources_initial.ram_used_gb) * 1024,
                'cpu_avg': (resources_initial.cpu_percent + resources_phase_end.cpu_percent) / 2
            }
        )

        logger.info(
            "incremental_processor",
            f"process_folder.{folder_path.name}",
            f"COMPLETED: {stats.document_count}/{stats.file_count} docs, "
            f"{stats.chunk_count} chunks in {processing_time:.1f}s "
            f"({docs_per_sec:.2f} docs/s, {chunks_per_sec:.2f} chunks/s)"
        )

        # Log resource deltas
        ram_delta = (resources_phase_end.ram_used_gb - resources_initial.ram_used_gb) * 1024
        logger.info(
            "resource_delta",
            folder_path.name,
            f"RESOURCE_DELTA | folder={folder_id} | ram_delta={ram_delta:.0f}MB | "
            f"cpu_avg={stats.resources['cpu_avg']:.1f}%"
        )

        return stats

    except Exception as e:
        logger.error("incremental_processor", f"process_folder.{folder_path.name}", f"ERROR: {e}")
        # Log resource state at error
        resources_error = capture_resource_snapshot()
        log_resource_snapshot(logger, "error_state", resources_error, folder_id)
        raise


def save_statistics(session_stats: SessionStats):
    """Save session statistics to file."""
    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_FILE, 'w') as f:
        json.dump(asdict(session_stats), f, indent=2)
    print(f"Statistics saved to {STATS_FILE}")


def print_statistics_summary(session_stats: SessionStats, logger):
    """Print human-readable statistics summary."""
    logger.info("incremental_processor", "statistics", "=" * 80)
    logger.info("incremental_processor", "statistics", "SESSION STATISTICS SUMMARY")
    logger.info("incremental_processor", "statistics", "=" * 80)
    logger.info("incremental_processor", "statistics", f"Session Duration: {session_stats.total_duration:.1f}s ({session_stats.total_duration/3600:.2f}h)")
    logger.info("incremental_processor", "statistics", f"Folders Processed: {session_stats.folders_processed}")
    logger.info("incremental_processor", "statistics", f"Total Files: {session_stats.total_files}")
    logger.info("incremental_processor", "statistics", f"Total Documents: {session_stats.total_documents}")
    logger.info("incremental_processor", "statistics", f"Total Chunks: {session_stats.total_chunks}")
    logger.info("incremental_processor", "statistics", "-" * 80)
    logger.info("incremental_processor", "statistics", f"Avg Docs/Folder: {session_stats.avg_docs_per_folder:.1f}")
    logger.info("incremental_processor", "statistics", f"Avg Chunks/Folder: {session_stats.avg_chunks_per_folder:.1f}")
    logger.info("incremental_processor", "statistics", f"Avg Time/Folder: {session_stats.avg_processing_time_per_folder:.1f}s")
    logger.info("incremental_processor", "statistics", "-" * 80)
    logger.info("incremental_processor", "statistics", f"Overall Throughput: {session_stats.overall_docs_per_second:.2f} docs/s")
    logger.info("incremental_processor", "statistics", f"Overall Throughput: {session_stats.overall_chunks_per_second:.2f} chunks/s")
    logger.info("incremental_processor", "statistics", "=" * 80)

    # Find fastest and slowest folders
    if session_stats.folder_stats:
        fastest = max(session_stats.folder_stats, key=lambda x: x['docs_per_second'])
        slowest = min(session_stats.folder_stats, key=lambda x: x['docs_per_second'])
        logger.info("incremental_processor", "statistics", f"Fastest Folder: {fastest['folder_id']} ({fastest['docs_per_second']:.2f} docs/s)")
        logger.info("incremental_processor", "statistics", f"Slowest Folder: {slowest['folder_id']} ({slowest['docs_per_second']:.2f} docs/s)")
        logger.info("incremental_processor", "statistics", "=" * 80)


async def main():
    """Main incremental processing loop."""
    session_start = time.time()
    logger = get_system_logger()
    logger.info("incremental_processor", "startup", "Starting incremental Narodne Novine processor")

    # Get folders to process
    folders = get_folders_to_process()

    if not folders:
        logger.info("incremental_processor", "startup", "No folders to process - all done!")
        return

    logger.info("incremental_processor", "startup", f"Processing {len(folders)} folders")

    # Create RAG system
    logger.info("incremental_processor", "init", "Initializing RAG system...")
    rag_system = create_complete_rag_system(
        language=LANGUAGE,
        scope=SCOPE,
        feature_name=FEATURE_NAME
    )
    await rag_system.initialize()
    logger.info("incremental_processor", "init", "RAG system initialized")

    # Load progress
    progress = load_progress()

    # Statistics tracking
    folder_stats_list: List[FolderStats] = []
    total_files = 0
    total_documents = 0
    total_chunks = 0
    total_folders = len(folders)

    for idx, folder in enumerate(folders, 1):
        folder_id = f"{folder.parent.name}/{folder.name}"

        logger.info(
            "incremental_processor",
            "progress",
            f"Processing folder {idx}/{total_folders}: {folder_id}"
        )

        try:
            # Process folder and collect statistics
            stats = await process_folder(rag_system, folder, logger)
            folder_stats_list.append(stats)

            # Update totals
            total_files += stats.file_count
            total_documents += stats.document_count
            total_chunks += stats.chunk_count

            # Update progress
            progress["processed_folders"].append(folder_id)
            progress["last_folder"] = folder_id
            save_progress(progress)

            # Calculate running averages
            elapsed_time = time.time() - session_start
            avg_time_per_folder = elapsed_time / idx
            estimated_remaining = avg_time_per_folder * (total_folders - idx)

            logger.info(
                "incremental_processor",
                "checkpoint",
                f"Checkpoint: {idx}/{total_folders} folders | "
                f"{total_documents} docs | {total_chunks} chunks | "
                f"Elapsed: {elapsed_time/60:.1f}m | "
                f"ETA: {estimated_remaining/60:.1f}m"
            )

        except Exception as e:
            logger.error("incremental_processor", "error", f"Failed to process {folder_id}: {e}")
            # Save progress and statistics before exiting
            save_progress(progress)
            if folder_stats_list:
                partial_session = SessionStats(
                    session_start=session_start,
                    session_end=time.time(),
                    total_duration=time.time() - session_start,
                    folders_processed=len(folder_stats_list),
                    total_files=total_files,
                    total_documents=total_documents,
                    total_chunks=total_chunks,
                    avg_docs_per_folder=total_documents / len(folder_stats_list),
                    avg_chunks_per_folder=total_chunks / len(folder_stats_list),
                    avg_processing_time_per_folder=sum(s.processing_time for s in folder_stats_list) / len(folder_stats_list),
                    overall_docs_per_second=total_documents / (time.time() - session_start),
                    overall_chunks_per_second=total_chunks / (time.time() - session_start),
                    folder_stats=[asdict(s) for s in folder_stats_list]
                )
                save_statistics(partial_session)
            raise

    # Calculate final statistics
    session_end = time.time()
    total_duration = session_end - session_start

    session_stats = SessionStats(
        session_start=session_start,
        session_end=session_end,
        total_duration=total_duration,
        folders_processed=len(folder_stats_list),
        total_files=total_files,
        total_documents=total_documents,
        total_chunks=total_chunks,
        avg_docs_per_folder=total_documents / len(folder_stats_list) if folder_stats_list else 0,
        avg_chunks_per_folder=total_chunks / len(folder_stats_list) if folder_stats_list else 0,
        avg_processing_time_per_folder=sum(s.processing_time for s in folder_stats_list) / len(folder_stats_list) if folder_stats_list else 0,
        overall_docs_per_second=total_documents / total_duration if total_duration > 0 else 0,
        overall_chunks_per_second=total_chunks / total_duration if total_duration > 0 else 0,
        folder_stats=[asdict(s) for s in folder_stats_list]
    )

    # Save and display statistics
    save_statistics(session_stats)
    print_statistics_summary(session_stats, logger)

    logger.info(
        "incremental_processor",
        "complete",
        f"COMPLETED: {total_folders} folders, {total_documents} documents, {total_chunks} chunks in {total_duration/3600:.2f}h"
    )

    # Cleanup
    await rag_system.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
