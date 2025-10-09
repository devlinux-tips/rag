#!/usr/bin/env python3
"""
Real-time statistics analyzer for Narodne Novine incremental processing.
Parses logs to show current performance while processing is ongoing.
"""

import re
import sys
import psutil
from pathlib import Path
from datetime import timedelta
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FolderStat:
    """Statistics for a single folder."""
    folder_id: str
    file_count: int
    doc_count: int
    chunk_count: int
    processing_time: float
    docs_per_sec: float
    chunks_per_sec: float


def parse_completion_logs(log_file: Path) -> List[FolderStat]:
    """Parse completion logs to extract folder statistics."""
    stats = []

    # Pattern: COMPLETED: 15/15 docs, 30 chunks in 6.1s (2.47 docs/s, 4.95 chunks/s)
    pattern = r'incremental_processor\.process_folder\.(\d+): COMPLETED: (\d+)/(\d+) docs, (\d+) chunks in ([\d.]+)s \(([\d.]+) docs/s, ([\d.]+) chunks/s\)'

    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                folder_num = match.group(1)
                doc_count = int(match.group(2))
                file_count = int(match.group(3))
                chunk_count = int(match.group(4))
                proc_time = float(match.group(5))
                docs_per_sec = float(match.group(6))
                chunks_per_sec = float(match.group(7))

                # Extract year from log line (look for year/folder pattern)
                year_match = re.search(r'(\d{4})/\d+', line)
                year = year_match.group(1) if year_match else "2025"
                folder_id = f"{year}/{folder_num}"

                stats.append(FolderStat(
                    folder_id=folder_id,
                    file_count=file_count,
                    doc_count=doc_count,
                    chunk_count=chunk_count,
                    processing_time=proc_time,
                    docs_per_sec=docs_per_sec,
                    chunks_per_sec=chunks_per_sec
                ))

    return stats


def analyze_stats(stats: List[FolderStat]) -> None:
    """Analyze and display statistics."""
    if not stats:
        print("❌ No statistics found in logs")
        return

    # Overall statistics
    total_folders = len(stats)
    total_docs = sum(s.doc_count for s in stats)
    total_chunks = sum(s.chunk_count for s in stats)
    total_time = sum(s.processing_time for s in stats)

    avg_docs_per_folder = total_docs / total_folders
    avg_chunks_per_folder = total_chunks / total_folders
    avg_time_per_folder = total_time / total_folders
    overall_docs_per_sec = total_docs / total_time if total_time > 0 else 0
    overall_chunks_per_sec = total_chunks / total_time if total_time > 0 else 0

    # Find extremes
    fastest = max(stats, key=lambda s: s.docs_per_sec)
    slowest = min(stats, key=lambda s: s.docs_per_sec)
    most_chunks = max(stats, key=lambda s: s.chunk_count)

    # Display results
    print("=" * 80)
    print("📊 NARODNE NOVINE PROCESSING STATISTICS (REAL-TIME)")
    print("=" * 80)
    print()
    print(f"📁 Folders Processed: {total_folders}")
    print(f"📄 Total Documents:   {total_docs}")
    print(f"📦 Total Chunks:      {total_chunks}")
    print(f"⏱️  Total Time:        {timedelta(seconds=int(total_time))}")
    print()
    print("-" * 80)
    print("AVERAGES")
    print("-" * 80)
    print(f"📄 Docs per folder:   {avg_docs_per_folder:.1f}")
    print(f"📦 Chunks per folder: {avg_chunks_per_folder:.1f}")
    print(f"⏱️  Time per folder:   {avg_time_per_folder:.1f}s")
    print(f"⚡ Throughput:        {overall_docs_per_sec:.2f} docs/s, {overall_chunks_per_sec:.2f} chunks/s")
    print()
    print("-" * 80)
    print("PERFORMANCE ANALYSIS")
    print("-" * 80)
    print(f"🚀 Fastest folder:    {fastest.folder_id}")
    print(f"   └─ {fastest.doc_count} docs in {fastest.processing_time:.1f}s ({fastest.docs_per_sec:.2f} docs/s)")
    print()
    print(f"🐌 Slowest folder:    {slowest.folder_id}")
    print(f"   └─ {slowest.doc_count} docs, {slowest.chunk_count} chunks in {slowest.processing_time:.1f}s ({slowest.docs_per_sec:.2f} docs/s)")
    print()
    print(f"📦 Most chunks:       {most_chunks.folder_id}")
    print(f"   └─ {most_chunks.chunk_count} chunks in {most_chunks.processing_time:.1f}s ({most_chunks.chunks_per_sec:.2f} chunks/s)")
    print()

    # Identify slow folders (bottom 10% by docs/s)
    sorted_by_speed = sorted(stats, key=lambda s: s.docs_per_sec)
    slow_threshold = len(sorted_by_speed) // 10 or 5
    slow_folders = sorted_by_speed[:slow_threshold]

    print("-" * 80)
    print(f"⚠️  SLOWEST {len(slow_folders)} FOLDERS (Performance Bottlenecks)")
    print("-" * 80)
    for i, folder in enumerate(slow_folders, 1):
        print(f"{i}. {folder.folder_id}: {folder.doc_count} docs, {folder.chunk_count} chunks")
        print(f"   └─ {folder.processing_time:.1f}s ({folder.docs_per_sec:.2f} docs/s, {folder.chunks_per_sec:.2f} chunks/s)")

    print()
    print("-" * 80)
    print("📈 PROJECTIONS (Based on current performance)")
    print("-" * 80)

    # Estimate remaining time (assuming 2000 total folders)
    TOTAL_FOLDERS = 2000
    remaining_folders = TOTAL_FOLDERS - total_folders
    estimated_remaining_time = remaining_folders * avg_time_per_folder

    print(f"Remaining folders:    {remaining_folders}")
    print(f"Estimated time left:  {timedelta(seconds=int(estimated_remaining_time))}")
    print(f"Expected completion:  ~{(total_time + estimated_remaining_time) / 3600:.1f} hours total")
    print()
    print("=" * 80)


def get_system_resources() -> dict:
    """Get current system resource usage."""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()

        # Memory usage
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # Disk usage for /home partition
        disk = psutil.disk_usage('/home')

        # Find process_nn_incremental.py process
        process_info = None
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and 'process_nn_incremental.py' in ' '.join(cmdline):
                    process_info = {
                        'pid': proc.info['pid'],
                        'cpu_percent': proc.cpu_percent(interval=0.1),
                        'memory_mb': proc.info['memory_info'].rss / 1024 / 1024,
                        'memory_percent': proc.memory_percent()
                    }
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return {
            'cpu': {
                'percent': cpu_percent,
                'count': cpu_count,
                'freq_mhz': cpu_freq.current if cpu_freq else None
            },
            'memory': {
                'total_gb': mem.total / 1024**3,
                'used_gb': mem.used / 1024**3,
                'available_gb': mem.available / 1024**3,
                'percent': mem.percent
            },
            'swap': {
                'total_gb': swap.total / 1024**3,
                'used_gb': swap.used / 1024**3,
                'percent': swap.percent
            },
            'disk': {
                'total_gb': disk.total / 1024**3,
                'used_gb': disk.used / 1024**3,
                'free_gb': disk.free / 1024**3,
                'percent': disk.percent
            },
            'process': process_info
        }
    except Exception as e:
        return {'error': str(e)}


def display_system_resources(resources: dict) -> None:
    """Display system resource information."""
    print("-" * 80)
    print("💻 SYSTEM RESOURCES (CURRENT)")
    print("-" * 80)

    if 'error' in resources:
        print(f"⚠️  Error getting system info: {resources['error']}")
        return

    # CPU
    cpu = resources['cpu']
    print(f"🔥 CPU Usage:         {cpu['percent']:.1f}% ({cpu['count']} cores", end="")
    if cpu['freq_mhz']:
        print(f" @ {cpu['freq_mhz']:.0f} MHz)", end="")
    print(")")

    # Memory
    mem = resources['memory']
    print(f"💾 RAM:               {mem['used_gb']:.1f}/{mem['total_gb']:.1f} GB ({mem['percent']:.1f}% used)")
    print(f"   └─ Available:      {mem['available_gb']:.1f} GB")

    # Swap
    swap = resources['swap']
    if swap['total_gb'] > 0:
        print(f"💿 Swap:              {swap['used_gb']:.1f}/{swap['total_gb']:.1f} GB ({swap['percent']:.1f}% used)")
        if swap['percent'] > 10:
            print(f"   ⚠️  WARNING: Swap usage high - system may be memory constrained!")

    # Disk
    disk = resources['disk']
    print(f"💽 Disk (/home):      {disk['used_gb']:.1f}/{disk['total_gb']:.1f} GB ({disk['percent']:.1f}% used)")
    print(f"   └─ Free:           {disk['free_gb']:.1f} GB")

    # Process-specific
    proc = resources.get('process')
    if proc:
        print(f"\n🐍 Process (PID {proc['pid']}):")
        print(f"   └─ CPU:            {proc['cpu_percent']:.1f}%")
        print(f"   └─ Memory:         {proc['memory_mb']:.1f} MB ({proc['memory_percent']:.1f}%)")
    else:
        print(f"\n⚠️  Process:          Not running (service stopped)")

    print()


def check_performance_issues(resources: dict, stats: List[FolderStat]) -> None:
    """Analyze potential performance issues."""
    print("-" * 80)
    print("🔍 PERFORMANCE ANALYSIS")
    print("-" * 80)

    issues = []

    # Check memory
    mem = resources.get('memory', {})
    if mem.get('percent', 0) > 90:
        issues.append("🔴 CRITICAL: RAM usage > 90% - severe memory pressure")
    elif mem.get('percent', 0) > 80:
        issues.append("🟡 WARNING: RAM usage > 80% - may cause slowdowns")

    # Check swap
    swap = resources.get('swap', {})
    if swap.get('percent', 0) > 20:
        issues.append("🔴 CRITICAL: High swap usage - system is thrashing")
    elif swap.get('percent', 0) > 5:
        issues.append("🟡 WARNING: Swap being used - RAM may be insufficient")

    # Check CPU
    cpu = resources.get('cpu', {})
    if cpu.get('percent', 0) > 95:
        issues.append("🟡 WARNING: CPU usage > 95% - may indicate thermal throttling")

    # Check disk
    disk = resources.get('disk', {})
    if disk.get('percent', 0) > 90:
        issues.append("🟡 WARNING: Disk > 90% full - may affect performance")

    # Analyze folder timing variance
    if stats:
        times = [s.processing_time for s in stats]
        avg_time = sum(times) / len(times)
        max_time = max(times)

        if max_time > avg_time * 5:  # Outlier > 5x average
            slowest = max(stats, key=lambda s: s.processing_time)
            issues.append(f"⚠️  Outlier detected: {slowest.folder_id} took {slowest.processing_time:.1f}s (avg: {avg_time:.1f}s)")

    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  ✅ No major performance issues detected")

    print()


def main():
    """Main entry point."""
    log_file = Path("/home/rag/src/rag/logs/nn_service.log")

    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        sys.exit(1)

    print("📖 Parsing logs...")
    stats = parse_completion_logs(log_file)

    if not stats:
        print("⚠️  No completed folders found yet. Processing may have just started.")
        sys.exit(0)

    # Get system resources
    resources = get_system_resources()

    # Display analysis
    analyze_stats(stats)
    display_system_resources(resources)
    check_performance_issues(resources, stats)


if __name__ == "__main__":
    main()
