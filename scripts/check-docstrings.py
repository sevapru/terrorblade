#!/usr/bin/env python3
"""
Check docstring quality across the Terrorblade codebase.

This script analyzes all Python files and reports on docstring coverage
and quality, helping maintain good documentation standards.
"""

import ast
from pathlib import Path


class DocstringChecker:
    """Checks docstring quality and coverage."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.terrorblade_path = project_root / "terrorblade"

    def analyze_file(self, file_path: Path) -> dict[str, any]:
        """Analyze a Python file for docstring quality."""

        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            stats = {
                "file": str(file_path.relative_to(self.project_root)),
                "classes": [],
                "functions": [],
                "module_docstring": ast.get_docstring(tree) is not None
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        "name": node.name,
                        "has_docstring": ast.get_docstring(node) is not None,
                        "docstring_length": len(ast.get_docstring(node) or ""),
                        "line": node.lineno
                    }
                    stats["classes"].append(class_info)

                elif isinstance(node, ast.FunctionDef):
                    # Skip private functions and methods
                    if not node.name.startswith('_') or node.name in ['__init__', '__call__']:
                        func_info = {
                            "name": node.name,
                            "has_docstring": ast.get_docstring(node) is not None,
                            "docstring_length": len(ast.get_docstring(node) or ""),
                            "line": node.lineno,
                            "args_count": len(node.args.args)
                        }
                        stats["functions"].append(func_info)

            return stats

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return {}

    def check_all_files(self) -> dict[str, any]:
        """Check all Python files in the project."""

        total_stats = {
            "files_analyzed": 0,
            "files_with_module_docstring": 0,
            "total_classes": 0,
            "classes_with_docstring": 0,
            "total_functions": 0,
            "functions_with_docstring": 0,
            "detailed_results": []
        }

        for py_file in self.terrorblade_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            stats = self.analyze_file(py_file)
            if not stats:
                continue

            total_stats["files_analyzed"] += 1
            total_stats["detailed_results"].append(stats)

            if stats["module_docstring"]:
                total_stats["files_with_module_docstring"] += 1

            # Count classes
            for cls in stats["classes"]:
                total_stats["total_classes"] += 1
                if cls["has_docstring"]:
                    total_stats["classes_with_docstring"] += 1

            # Count functions
            for func in stats["functions"]:
                total_stats["total_functions"] += 1
                if func["has_docstring"]:
                    total_stats["functions_with_docstring"] += 1

        return total_stats

    def generate_report(self) -> str:
        """Generate a detailed docstring quality report."""

        stats = self.check_all_files()

        # Calculate percentages
        module_coverage = (stats["files_with_module_docstring"] / max(stats["files_analyzed"], 1)) * 100
        class_coverage = (stats["classes_with_docstring"] / max(stats["total_classes"], 1)) * 100
        function_coverage = (stats["functions_with_docstring"] / max(stats["total_functions"], 1)) * 100

        report = f"""# Docstring Quality Report

## Summary

- **Files Analyzed:** {stats['files_analyzed']}
- **Module Docstring Coverage:** {module_coverage:.1f}% ({stats['files_with_module_docstring']}/{stats['files_analyzed']})
- **Class Docstring Coverage:** {class_coverage:.1f}% ({stats['classes_with_docstring']}/{stats['total_classes']})
- **Function Docstring Coverage:** {function_coverage:.1f}% ({stats['functions_with_docstring']}/{stats['total_functions']})

## Overall Grade

"""

        overall_score = (module_coverage + class_coverage + function_coverage) / 3

        if overall_score >= 90:
            grade = "A (Excellent)"
        elif overall_score >= 80:
            grade = "B (Good)"
        elif overall_score >= 70:
            grade = "C (Fair)"
        elif overall_score >= 60:
            grade = "D (Poor)"
        else:
            grade = "F (Very Poor)"

        report += f"**{overall_score:.1f}% - Grade {grade}**\n\n"

        # Add detailed file-by-file results
        report += "## Detailed Results\n\n"

        for file_stats in stats["detailed_results"]:
            file_name = file_stats["file"]
            report += f"### {file_name}\n\n"

            if not file_stats["module_docstring"]:
                report += "⚠️ **Missing module docstring**\n\n"

            # Report missing class docstrings
            missing_class_docs = [cls for cls in file_stats["classes"] if not cls["has_docstring"]]
            if missing_class_docs:
                report += "**Classes missing docstrings:**\n"
                for cls in missing_class_docs:
                    report += f"- `{cls['name']}` (line {cls['line']})\n"
                report += "\n"

            # Report missing function docstrings
            missing_func_docs = [func for func in file_stats["functions"] if not func["has_docstring"]]
            if missing_func_docs:
                report += "**Functions missing docstrings:**\n"
                for func in missing_func_docs:
                    report += f"- `{func['name']}` (line {func['line']})\n"
                report += "\n"

            if not missing_class_docs and not missing_func_docs and file_stats["module_docstring"]:
                report += "✅ **All docstrings present**\n\n"

        return report


def main():
    """Generate and display docstring quality report."""

    project_root = Path(__file__).parent.parent
    checker = DocstringChecker(project_root)

    print("🔍 Analyzing docstring quality across Terrorblade codebase...")

    report = checker.generate_report()

    # Save report to file
    report_path = project_root / "docs-mkdocs" / "docs" / "development" / "docstring-quality.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w') as f:
        f.write(report)

    print(f"📊 Docstring quality report saved to: {report_path}")

    # Also print summary to console
    lines = report.split('\n')
    summary_end = next(i for i, line in enumerate(lines) if line.startswith("## Overall Grade"))

    for line in lines[:summary_end + 3]:
        print(line)


if __name__ == "__main__":
    main()

