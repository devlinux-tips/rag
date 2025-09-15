#!/usr/bin/env python3
"""
Lines of Code (LOC) Counter for RAG Service

Counts lines of code in services/rag-service/src/** and provides breakdown by folder.
Follows project's fail-fast philosophy and explicit validation principles.
"""

import sys
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class LOCStats:
    """Statistics for lines of code in a directory."""

    total_lines: int
    code_lines: int
    comment_lines: int
    blank_lines: int
    file_count: int


class LOCCounter:
    """Lines of code counter with fail-fast validation."""

    def __init__(self, src_path: str):
        """Initialize with explicit path validation."""
        self.src_path = Path(src_path)
        self._validate_path()

        # Extensions to count (Python and TOML config files)
        self.valid_extensions = {".py", ".toml"}

        # Directories to exclude from counting
        self.exclude_dirs = {
            "__pycache__",
            ".mypy_cache",
            ".claude-flow",
            ".swarm",
            ".pytest_cache",
        }

    def _validate_path(self) -> None:
        """Validate source path exists - fail fast if missing."""
        if not self.src_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {self.src_path}")

        if not self.src_path.is_dir():
            raise NotADirectoryError(f"Source path is not a directory: {self.src_path}")

    def _should_exclude_dir(self, dir_path: Path) -> bool:
        """Check if directory should be excluded from counting."""
        return any(exclude in dir_path.parts for exclude in self.exclude_dirs)

    def _count_lines_in_file(self, file_path: Path) -> LOCStats:
        """Count lines in a single Python or TOML file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except (UnicodeDecodeError, PermissionError) as e:
            print(f"Warning: Could not read {file_path}: {e}")
            return LOCStats(0, 0, 0, 0, 0)

        total_lines = len(lines)
        comment_lines = 0
        blank_lines = 0
        code_lines = 0
        is_python = file_path.suffix == ".py"
        is_toml = file_path.suffix == ".toml"

        in_multiline_comment = False

        for line in lines:
            stripped = line.strip()

            if not stripped:
                blank_lines += 1
            elif is_python and self._is_python_comment(stripped, in_multiline_comment):
                comment_lines += 1
                # Update multiline comment state for Python
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    in_multiline_comment = not in_multiline_comment
            elif is_toml and stripped.startswith("#"):
                comment_lines += 1
            else:
                code_lines += 1

        return LOCStats(
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
            blank_lines=blank_lines,
            file_count=1,
        )

    def _is_python_comment(self, stripped_line: str, in_multiline: bool) -> bool:
        """Check if a Python line is a comment."""
        if in_multiline:
            return True
        if stripped_line.startswith("#"):
            return True
        if (
            stripped_line.startswith('"""')
            and stripped_line.endswith('"""')
            and len(stripped_line) > 6
        ):
            return True
        if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
            return True
        return False

    def _get_source_files(self, directory: Path) -> List[Path]:
        """Get all Python and TOML files in directory, excluding cache directories."""
        source_files = []

        for file_path in directory.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix in self.valid_extensions
                and not self._should_exclude_dir(file_path)
            ):
                source_files.append(file_path)

        return source_files

    def count_directory(self, directory: Path) -> LOCStats:
        """Count LOC for entire directory."""
        source_files = self._get_source_files(directory)

        total_stats = LOCStats(0, 0, 0, 0, 0)

        for file_path in source_files:
            file_stats = self._count_lines_in_file(file_path)
            total_stats.total_lines += file_stats.total_lines
            total_stats.code_lines += file_stats.code_lines
            total_stats.comment_lines += file_stats.comment_lines
            total_stats.blank_lines += file_stats.blank_lines
            total_stats.file_count += file_stats.file_count

        return total_stats

    def get_subdirectory_stats(self) -> Dict[str, LOCStats]:
        """Get LOC statistics for each subdirectory in src."""
        stats_by_dir = {}

        # Get immediate subdirectories
        subdirs = [
            d
            for d in self.src_path.iterdir()
            if d.is_dir() and not self._should_exclude_dir(d)
        ]

        for subdir in subdirs:
            stats_by_dir[subdir.name] = self.count_directory(subdir)

        return stats_by_dir

    def get_total_stats(self) -> LOCStats:
        """Get total LOC statistics for entire src directory."""
        return self.count_directory(self.src_path)

    def get_config_stats(self) -> Dict[str, LOCStats]:
        """Get LOC statistics for TOML config files."""
        config_stats = {}

        # Look for config directory and TOML files
        config_dirs = [
            self.src_path.parent / "config",
            self.src_path.parent.parent / "config",
        ]

        for config_dir in config_dirs:
            if config_dir.exists():
                toml_files = list(config_dir.glob("*.toml"))
                if toml_files:
                    total_config_stats = LOCStats(0, 0, 0, 0, 0)
                    for toml_file in toml_files:
                        file_stats = self._count_lines_in_file(toml_file)
                        total_config_stats.total_lines += file_stats.total_lines
                        total_config_stats.code_lines += file_stats.code_lines
                        total_config_stats.comment_lines += file_stats.comment_lines
                        total_config_stats.blank_lines += file_stats.blank_lines
                        total_config_stats.file_count += file_stats.file_count

                        # Individual file stats
                        config_stats[toml_file.name] = file_stats

                    config_stats["_TOTAL_CONFIG"] = total_config_stats
                break

        return config_stats

    def generate_report(self) -> str:
        """Generate comprehensive LOC report."""
        total_stats = self.get_total_stats()
        dir_stats = self.get_subdirectory_stats()
        config_stats = self.get_config_stats()

        report_lines = [
            "=" * 80,
            "RAG SERVICE - LINES OF CODE REPORT",
            "=" * 80,
            f"Source Directory: {self.src_path}",
            f"Generated: {Path(__file__).name}",
            "",
            "TOTAL PYTHON CODE STATISTICS:",
            f"  Total Files:     {total_stats.file_count:,}",
            f"  Total Lines:     {total_stats.total_lines:,}",
            f"  Code Lines:      {total_stats.code_lines:,}",
            f"  Comment Lines:   {total_stats.comment_lines:,}",
            f"  Blank Lines:     {total_stats.blank_lines:,}",
            "",
        ]

        # Add config summary if found
        if config_stats and "_TOTAL_CONFIG" in config_stats:
            config_total = config_stats["_TOTAL_CONFIG"]
            report_lines.extend(
                [
                    "CONFIGURATION FILES (TOML) SUMMARY:",
                    f"  Config Files:    {config_total.file_count:,}",
                    f"  Total Lines:     {config_total.total_lines:,}",
                    f"  Config Lines:    {config_total.code_lines:,}",
                    f"  Comment Lines:   {config_total.comment_lines:,}",
                    f"  Blank Lines:     {config_total.blank_lines:,}",
                    "",
                ]
            )

        report_lines.extend(
            [
                "BREAKDOWN BY SOURCE DIRECTORY:",
                "-" * 40,
            ]
        )

        # Sort directories by code lines (descending)
        sorted_dirs = sorted(
            dir_stats.items(), key=lambda x: x[1].code_lines, reverse=True
        )

        for dir_name, stats in sorted_dirs:
            report_lines.extend(
                [
                    f"{dir_name}/",
                    f"  Files: {stats.file_count:,} | Total: {stats.total_lines:,} | "
                    f"Code: {stats.code_lines:,} | Comments: {stats.comment_lines:,} | "
                    f"Blank: {stats.blank_lines:,}",
                    "",
                ]
            )

        # Add individual config file breakdown if found
        if config_stats:
            report_lines.extend(
                [
                    "CONFIGURATION FILES BREAKDOWN:",
                    "-" * 40,
                ]
            )
            for file_name, stats in config_stats.items():
                if file_name != "_TOTAL_CONFIG":
                    report_lines.extend(
                        [
                            f"{file_name}",
                            f"  Lines: {stats.total_lines:,} | Config: {stats.code_lines:,} | "
                            f"Comments: {stats.comment_lines:,} | Blank: {stats.blank_lines:,}",
                            "",
                        ]
                    )

        report_lines.extend(
            [
                "=" * 80,
                "Notes:",
                "- Python (.py) and TOML (.toml) files are counted",
                "- Cache directories (__pycache__, .mypy_cache, etc.) are excluded",
                '- Comment detection: # for single-line, """ for Python docstrings',
                "- Config files searched in config/ directory",
                "=" * 80,
            ]
        )

        return "\n".join(report_lines)


def main() -> None:
    """Main execution with fail-fast validation."""
    # Define source path relative to script location
    script_dir = Path(__file__).parent
    src_path = script_dir.parent / "services" / "rag-service" / "src"

    try:
        counter = LOCCounter(str(src_path))
        report = counter.generate_report()
        print(report)

        # Also save to file for reference
        output_file = script_dir / "loc_report.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"\nReport also saved to: {output_file}")

    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
