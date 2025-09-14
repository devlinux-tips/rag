#!/usr/bin/env python3
"""
Dependency Analyzer for RAG Service
Analyzes Python imports and creates dependency graph ordered from low to high dependencies.
"""

import ast
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


class DependencyAnalyzer:
    def __init__(self, src_dir: str):
        self.src_dir = Path(src_dir)
        self.modules: dict[str, dict[str, Any]] = {}  # module_path -> ModuleInfo
        self.dependencies: dict[str, set[str]] = defaultdict(set)  # module -> set of dependencies
        self.reverse_deps: dict[str, set[str]] = defaultdict(set)  # module -> set of dependents

    def analyze(self):
        """Analyze all Python files and build dependency graph."""
        print("🔍 Scanning Python files...")
        self._scan_files()

        print("📊 Building dependency graph...")
        self._build_dependency_graph()

        print("🎯 Computing dependency levels...")
        return self._compute_dependency_levels()

    def _scan_files(self):
        """Scan all Python files and extract import information."""
        for py_file in self.src_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            # Skip legacy modules since they will be deleted
            if "_legacy" in py_file.name:
                print(f"⏭️  Skipping legacy module: {py_file.name}")
                continue

            module_path = self._get_module_path(py_file)
            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)
                imports = self._extract_imports(tree)

                self.modules[module_path] = {
                    "file_path": py_file,
                    "imports": imports,
                    "size": len(content.splitlines()),
                    "is_legacy": "legacy" in py_file.name,
                    "is_provider": "provider" in py_file.name,
                    "category": self._categorize_module(py_file),
                }

            except Exception as e:
                print(f"⚠️  Error processing {py_file}: {e}")

    def _get_module_path(self, file_path: Path) -> str:
        """Convert file path to module path."""
        rel_path = file_path.relative_to(self.src_dir.parent)
        module_parts = rel_path.with_suffix("").parts
        return ".".join(module_parts)

    def _extract_imports(self, tree: ast.AST) -> list[str]:
        """Extract import statements from AST."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Handle relative imports using AST level attribute
                    if node.level > 0:
                        # Construct the relative import string with dots
                        dots = "." * node.level
                        import_name = dots + node.module
                    else:
                        # Absolute import
                        import_name = node.module
                    imports.append(import_name)

        return imports

    def _resolve_relative_import(self, import_name: str, current_module: str) -> str:
        """
        Resolve relative imports to absolute module paths.

        Args:
            import_name: The import string (e.g., "..preprocessing.cleaners")
            current_module: Current module path (e.g., "src.generation.ollama_client")

        Returns:
            Resolved absolute import path
        """
        if not import_name.startswith("."):
            return import_name

        # Count leading dots to determine level
        dot_count = 0
        for char in import_name:
            if char == ".":
                dot_count += 1
            else:
                break

        # Split current module path
        current_parts = current_module.split(".")

        # Go up the hierarchy based on dot count
        # For N dots, we go up N-1 levels from the current package
        base_parts = current_parts[:-1]  # Remove module name, keep package path

        # Go up additional levels based on dots (N dots = go up N-1 levels)
        levels_to_go_up = dot_count - 1  # Single dot = same package (0 levels up)
        for _ in range(levels_to_go_up):
            if len(base_parts) > 1:  # Never remove 'src' - it's our root
                base_parts.pop()  # Add the relative part
        relative_part = import_name[dot_count:]
        if relative_part:
            relative_parts = relative_part.split(".")
            base_parts.extend(relative_parts)

        return ".".join(base_parts)

    def _categorize_module(self, file_path: Path) -> str:
        """Categorize module by its directory."""
        parts = file_path.relative_to(self.src_dir).parts
        if len(parts) > 1:
            return parts[0]  # First directory (utils, models, etc.)
        return "root"

    def _build_dependency_graph(self):
        """Build internal dependency graph (only src dependencies, excluding legacy)."""
        for module, info in self.modules.items():
            for import_name in info["imports"]:
                # Resolve relative imports to absolute paths
                if import_name.startswith("."):
                    resolved_import = self._resolve_relative_import(import_name, module)
                else:
                    resolved_import = import_name

                # Only track internal dependencies (src.* imports)
                if resolved_import.startswith("src."):
                    # Skip legacy dependencies since they will be deleted
                    if "_legacy" in resolved_import:
                        continue
                    # Normalize the import to match our module naming
                    dep_module = resolved_import
                    if dep_module in self.modules:
                        self.dependencies[module].add(dep_module)
                        self.reverse_deps[dep_module].add(module)

    def _compute_dependency_levels(self) -> dict[int, list[str]]:
        """Compute dependency levels using topological sort."""
        levels: dict[int, list[str]] = defaultdict(list)
        in_degree = defaultdict(int)

        # Calculate in-degrees
        for module in self.modules:
            in_degree[module] = len(self.dependencies[module])

        # Start with modules that have no dependencies
        queue = deque([module for module, degree in in_degree.items() if degree == 0])
        level = 0

        while queue:
            current_level_size = len(queue)
            levels[level] = []

            for _ in range(current_level_size):
                module = queue.popleft()
                levels[level].append(module)

                # Update in-degrees of dependents
                for dependent in self.reverse_deps[module]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

            level += 1

        return dict(levels)

    def generate_report(self, levels: dict[int, list[str]]) -> str:
        """Generate comprehensive dependency report."""
        report = []
        report.append("# RAG Service Dependency Analysis Report")
        report.append("=" * 50)
        report.append("")

        # Summary statistics
        total_modules = len(self.modules)
        legacy_count = sum(1 for info in self.modules.values() if info["is_legacy"])
        provider_count = sum(1 for info in self.modules.values() if info["is_provider"])

        report.append("## Summary Statistics")
        report.append(f"- **Total Modules**: {total_modules}")
        report.append(f"- **Legacy Modules**: {legacy_count}")
        report.append(f"- **Provider Modules**: {provider_count}")
        report.append(f"- **Dependency Levels**: {len(levels)}")
        report.append("")

        # Category breakdown
        categories: dict[str, int] = defaultdict(int)
        for info in self.modules.values():
            categories[info["category"]] += 1

        report.append("## Module Categories")
        for category, count in sorted(categories.items()):
            report.append(f"- **{category.title()}**: {count} modules")
        report.append("")

        # Dependency levels (low to high)
        report.append("## Dependency Levels (Low to High Dependencies)")
        report.append("")

        for level, modules in sorted(levels.items()):
            report.append(f"### Level {level} - {len(modules)} modules")
            report.append(
                "*Modules with minimal external dependencies*"
                if level == 0
                else f"*Modules depending on Level {level - 1} and below*"
            )
            report.append("")

            # Sort modules by category and name
            sorted_modules = sorted(modules, key=lambda m: (self.modules[m]["category"], m))

            for module in sorted_modules:
                info = self.modules[module]
                size = info["size"]
                category = info["category"]
                flags = []
                if info["is_legacy"]:
                    flags.append("LEGACY")
                if info["is_provider"]:
                    flags.append("PROVIDER")

                flag_str = f" [{', '.join(flags)}]" if flags else ""

                report.append(f"- **{module}**{flag_str}")
                report.append(f"  - Category: {category.title()}")
                report.append(f"  - Size: {size} lines")

                # Show internal dependencies
                internal_deps = [dep for dep in self.dependencies[module] if dep.startswith("src.")]
                if internal_deps:
                    report.append(f"  - Dependencies: {', '.join(sorted(internal_deps))}")

                report.append("")

            report.append("")

        # High-level dependency summary
        report.append("## High-Level Architecture Overview")
        report.append("")

        # Find modules with most dependencies
        high_dep_modules = sorted(self.modules.keys(), key=lambda m: len(self.dependencies[m]), reverse=True)[:10]

        report.append("### Most Dependent Modules (Top 10)")
        for i, module in enumerate(high_dep_modules, 1):
            dep_count = len(self.dependencies[module])
            if dep_count > 0:
                report.append(f"{i}. **{module}** - {dep_count} dependencies")
        report.append("")

        # Find modules with most dependents
        high_impact_modules = sorted(self.modules.keys(), key=lambda m: len(self.reverse_deps[m]), reverse=True)[:10]

        report.append("### Most Depended Upon Modules (Top 10)")
        for i, module in enumerate(high_impact_modules, 1):
            dependent_count = len(self.reverse_deps[module])
            if dependent_count > 0:
                report.append(f"{i}. **{module}** - {dependent_count} dependents")
        report.append("")

        # Detailed dependency listing
        report.append("## Detailed Dependency Listing")
        report.append("*Ordered from no dependencies to high dependencies*")
        report.append("")
        detailed_listing = self._generate_detailed_dependency_listing(levels)
        report.extend(detailed_listing)

        # Recommendations
        report.append("## Recommendations")
        report.append("")
        report.append("### 🎯 Refactoring Priorities")
        report.append("1. **Legacy Modules**: Consider migrating or removing legacy modules")
        report.append("2. **High Dependency Modules**: Review modules with many dependencies for simplification")
        report.append("3. **Core Dependencies**: Ensure stability of highly depended-upon modules")
        report.append("")

        report.append("### 📐 Architecture Insights")
        report.append("- **Level 0 modules** are foundational and should be most stable")
        report.append("- **Provider modules** implement dependency injection patterns")
        report.append("- **Legacy modules** indicate areas needing modernization")
        report.append("")

        return "\n".join(report)

    def _generate_detailed_dependency_listing(self, levels: dict[int, list[str]]) -> list[str]:
        """Generate detailed listing of each file and its dependencies."""
        listing = []

        # Sort modules by dependency count (low to high)
        all_modules = []
        for level, modules in sorted(levels.items()):
            for module in sorted(modules):
                dep_count = len(self.dependencies[module])
                all_modules.append((dep_count, level, module))

        all_modules.sort(key=lambda x: (x[0], x[1]))  # Sort by dep count, then level

        listing.append("### 📋 Complete Module Dependency Reference")
        listing.append("*Format: `filename.py` → depends on: [`dependency1.py`, `dependency2.py`]*")
        listing.append("")

        current_dep_count = -1
        for dep_count, _level, module in all_modules:
            # Add section headers for different dependency counts
            if dep_count != current_dep_count:
                current_dep_count = dep_count
                if dep_count == 0:
                    listing.append("#### 🟢 Zero Dependencies (Foundation Modules)")
                elif dep_count == 1:
                    listing.append("#### 🟡 Single Dependency (Integration Modules)")
                elif dep_count == 2:
                    listing.append("#### 🟠 Two Dependencies")
                else:
                    listing.append(f"#### 🔴 {dep_count} Dependencies (High Complexity)")
                listing.append("")

            # Get file name without src. prefix
            file_name = module.replace("src.", "").replace(".", "/") + ".py"

            # Get dependencies
            deps = sorted(self.dependencies[module])
            if deps:
                dep_files = [dep.replace("src.", "").replace(".", "/") + ".py" for dep in deps]
                deps_str = ", ".join(f"`{dep}`" for dep in dep_files)
                listing.append(f"- **`{file_name}`** → depends on: [{deps_str}]")
            else:
                listing.append(f"- **`{file_name}`** → depends on: *none*")

            # Add module info
            info = self.modules[module]
            flags = []
            if info["is_provider"]:
                flags.append("PROVIDER")

            flag_str = f" [{', '.join(flags)}]" if flags else ""
            listing.append(f"  - Size: {info['size']} lines | Category: {info['category'].title()}{flag_str}")

            # Show all imports (including external)
            if info["imports"]:
                external_imports = [imp for imp in info["imports"] if not imp.startswith("src.")]
                if external_imports:
                    # Group by common prefixes
                    stdlib_imports = [
                        imp
                        for imp in external_imports
                        if "." not in imp
                        or imp.split(".")[0]
                        in [
                            "os",
                            "sys",
                            "json",
                            "time",
                            "datetime",
                            "pathlib",
                            "typing",
                            "collections",
                            "dataclasses",
                            "enum",
                            "abc",
                            "re",
                            "asyncio",
                            "functools",
                            "itertools",
                        ]
                    ]
                    third_party = [imp for imp in external_imports if imp not in stdlib_imports]

                    ext_imports = []
                    if stdlib_imports:
                        ext_imports.append(f"stdlib: {', '.join(sorted(set(stdlib_imports)))}")
                    if third_party:
                        ext_imports.append(f"3rd-party: {', '.join(sorted(set(third_party)))}")

                    if ext_imports:
                        listing.append(f"  - External imports: {' | '.join(ext_imports)}")

            listing.append("")

        listing.append("---")
        listing.append("")
        listing.append("### 📊 Quick Reference")
        listing.append(f"- **Total files analyzed**: {len(all_modules)}")
        listing.append(f"- **Zero dependencies**: {sum(1 for count, _, _ in all_modules if count == 0)} files")
        listing.append(f"- **Single dependency**: {sum(1 for count, _, _ in all_modules if count == 1)} files")
        listing.append(f"- **Multiple dependencies**: {sum(1 for count, _, _ in all_modules if count > 1)} files")
        listing.append("")

        return listing


def main():
    src_dir = "/home/x/src/rag/learn-rag/services/rag-service/src"

    analyzer = DependencyAnalyzer(src_dir)
    levels = analyzer.analyze()

    print("📝 Generating report...")
    report = analyzer.generate_report(levels)

    # Save report
    report_file = "/home/x/src/rag/learn-rag/services/rag-service/DEPENDENCY_ANALYSIS.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ Report saved to: {report_file}")
    print(f"📊 Analyzed {len(analyzer.modules)} modules across {len(levels)} dependency levels")


if __name__ == "__main__":
    main()
