#!/usr/bin/env python3
"""
Enhance docstrings by adding examples and usage patterns.

This script analyzes the code structure and adds contextual information
to the generated documentation.
"""

import importlib.util
import inspect
from pathlib import Path
from typing import Any


class DocstringEnhancer:
    """Enhances docstrings with additional context and examples."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.terrorblade_path = project_root / "terrorblade"

    def analyze_module(self, module_path: str) -> dict[str, Any]:
        """Analyze a module and extract enhanced documentation info."""

        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(
                module_path,
                self.project_root / f"{module_path.replace('.', '/')}.py"
            )
            if not spec or not spec.loader:
                return {}

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            info = {
                "classes": [],
                "functions": [],
                "constants": [],
                "usage_examples": []
            }

            # Analyze classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ == module_path:
                    class_info = self._analyze_class(obj)
                    class_info["name"] = name
                    info["classes"].append(class_info)

            # Analyze functions
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if obj.__module__ == module_path:
                    func_info = self._analyze_function(obj)
                    func_info["name"] = name
                    info["functions"].append(func_info)

            # Analyze constants
            for name in dir(module):
                if (not name.startswith('_') and
                    not inspect.isclass(getattr(module, name)) and
                    not inspect.isfunction(getattr(module, name)) and
                    not inspect.ismodule(getattr(module, name))):

                    value = getattr(module, name)
                    if isinstance(value, str | int | float | bool | list | dict | tuple):
                        info["constants"].append({
                            "name": name,
                            "value": str(value)[:100] + "..." if len(str(value)) > 100 else str(value),
                            "type": type(value).__name__
                        })

            return info

        except Exception as e:
            print(f"Warning: Could not analyze {module_path}: {e}")
            return {}

    def _analyze_class(self, cls) -> dict[str, Any]:
        """Analyze a class and extract information."""
        info = {
            "docstring": inspect.getdoc(cls) or "",
            "methods": [],
            "properties": [],
            "inheritance": [base.__name__ for base in cls.__bases__ if base != object]
        }

        # Analyze methods
        for name, method in inspect.getmembers(cls, inspect.ismethod):
            if not name.startswith('_') or name in ['__init__', '__call__']:
                method_info = self._analyze_function(method)
                method_info["name"] = name
                info["methods"].append(method_info)

        # Analyze properties
        for name, prop in inspect.getmembers(cls, lambda x: isinstance(x, property)):
            info["properties"].append({
                "name": name,
                "docstring": inspect.getdoc(prop) or ""
            })

        return info

    def _analyze_function(self, func) -> dict[str, Any]:
        """Analyze a function and extract information."""
        info = {
            "docstring": inspect.getdoc(func) or "",
            "signature": str(inspect.signature(func)),
            "parameters": [],
            "returns": None
        }

        try:
            sig = inspect.signature(func)
            for param_name, param in sig.parameters.items():
                param_info = {
                    "name": param_name,
                    "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else None,
                    "default": str(param.default) if param.default != inspect.Parameter.empty else None
                }
                info["parameters"].append(param_info)

            if sig.return_annotation != inspect.Signature.empty:
                info["returns"] = str(sig.return_annotation)

        except Exception:
            pass

        return info

    def generate_enhanced_docs(self, module_path: str) -> str:
        """Generate enhanced documentation for a module."""

        info = self.analyze_module(module_path)
        if not info:
            return f"# {module_path.split('.')[-1]}\n\n::: {module_path}\n"

        module_name = module_path.split('.')[-1]
        content = f"# {module_name}\n\n"

        # Add module overview
        content += f"::: {module_path}\n    options:\n      show_root_heading: true\n      show_source: false\n\n"

        # Add usage examples if available
        if info.get("usage_examples"):
            content += "## Usage Examples\n\n"
            for example in info["usage_examples"]:
                content += f"```python\n{example}\n```\n\n"

        # Add class summaries
        if info.get("classes"):
            content += "## Classes\n\n"
            for cls in info["classes"]:
                content += f"### {cls['name']}\n\n"
                if cls.get("inheritance"):
                    content += f"**Inherits from:** {', '.join(cls['inheritance'])}\n\n"

                content += f"::: {module_path}.{cls['name']}\n    options:\n      show_root_heading: false\n\n"

        # Add function summaries
        if info.get("functions"):
            content += "## Functions\n\n"
            for func in info["functions"]:
                content += f"### {func['name']}\n\n"
                content += f"::: {module_path}.{func['name']}\n    options:\n      show_root_heading: false\n\n"

        # Add constants
        if info.get("constants"):
            content += "## Constants\n\n"
            for const in info["constants"]:
                content += f"**{const['name']}** (`{const['type']}`): `{const['value']}`\n\n"

        return content


def main():
    """Generate enhanced documentation."""
    project_root = Path(__file__).parent.parent
    DocstringEnhancer(project_root)

    # Example usage - this would be called by the main generation script
    print("DocstringEnhancer initialized and ready for use")


if __name__ == "__main__":
    main()

