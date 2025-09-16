#!/usr/bin/env python3
"""
Generate API documentation from Terrorblade codebase.

This script automatically scans the terrorblade module structure and generates
markdown documentation for each module, organizing it by directory structure.
"""

from pathlib import Path

import mkdocs_gen_files


def get_module_structure(base_path: Path) -> dict[str, list[str]]:
    """
    Scan the terrorblade module and return organized structure.

    Returns:
        Dict mapping category names to lists of module paths
    """
    structure = {
        "Core": [],
        "Data Management": [],
        "Processing": [],
        "User Interfaces": [],
        "Utilities": [],
        "Examples": [],
        "MCP Server": []
    }

    # Define category mappings based on directory structure
    category_mappings = {
        "data/database": "Data Management",
        "data/loaders": "Data Management",
        "data/preprocessing": "Processing",
        "tui": "User Interfaces",
        "examples": "Examples",
        "mcp": "MCP Server",
        "utils": "Utilities"
    }

    terrorblade_path = base_path / "terrorblade"

    for py_file in terrorblade_path.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue

        # Get relative path from terrorblade root
        rel_path = py_file.relative_to(terrorblade_path)

        # Convert to module path
        module_parts = list(rel_path.parts[:-1]) + [rel_path.stem]
        module_path = "terrorblade." + ".".join(module_parts)

        # Determine category
        category = "Core"
        for dir_pattern, cat in category_mappings.items():
            if str(rel_path).startswith(dir_pattern):
                category = cat
                break

        structure[category].append(module_path)

    # Remove empty categories and sort modules
    return {k: sorted(v) for k, v in structure.items() if v}


def generate_module_doc(module_path: str, nav_path: str) -> None:
    """Generate documentation for a single module."""

    module_name = module_path.split('.')[-1]

    # Create enhanced markdown content
    content = f"""# {module_name}

::: {module_path}
    options:
      show_root_heading: true
      show_source: true
      docstring_style: google
      docstring_section_style: table
      heading_level: 2
      members_order: source
      show_submodules: false
      show_signature_annotations: true
      separate_signature: true

## Module Information

**Module Path:** `{module_path}`
**Category:** {get_category_for_module(module_path)}

## Quick Navigation

- [Classes](#classes) - All classes defined in this module
- [Functions](#functions) - All functions with their signatures
- [Constants](#constants) - Module-level constants and variables

---

*This documentation is automatically generated from the source code.*
"""

    # Write the file
    with mkdocs_gen_files.open(nav_path, "w") as f:
        f.write(content)


def get_category_for_module(module_path: str) -> str:
    """Get the category for a module based on its path."""
    category_mappings = {
        "data.database": "Data Management",
        "data.loaders": "Data Management",
        "data.preprocessing": "Processing",
        "tui": "User Interfaces",
        "examples": "Examples",
        "mcp": "MCP Server",
        "utils": "Utilities"
    }

    for pattern, category in category_mappings.items():
        if pattern in module_path:
            return category

    return "Core"


def generate_category_index(category: str, modules: list[str]) -> str:
    """Generate index page for a category."""

    content = f"""# {category}

This section contains the {category.lower()} modules of Terrorblade.

## Modules

"""

    for module in modules:
        module_name = module.split('.')[-1]
        module_file = f"{module.replace('.', '/')}.md"
        content += f"- [{module_name}]({module_file})\n"

    return content


def main():
    """Main function to generate all API documentation."""

    # Get the project root directory
    project_root = Path(__file__).parent.parent

    print("🔍 Scanning Terrorblade module structure...")
    structure = get_module_structure(project_root)

    print(f"📁 Found {sum(len(modules) for modules in structure.values())} modules in {len(structure)} categories")

    # Generate navigation structure
    nav_items = []

    for category, modules in structure.items():
        print(f"📝 Generating documentation for {category} ({len(modules)} modules)")

        # Create category directory
        category_dir = f"api/{category.lower().replace(' ', '-')}"

        # Generate individual module docs
        for module_path in modules:
            module_name = module_path.split('.')[-1]
            nav_path = f"{category_dir}/{module_name}.md"

            generate_module_doc(module_path, nav_path)
            print(f"   ✓ {module_name}")

        # Generate category index
        index_content = generate_category_index(category, modules)
        with mkdocs_gen_files.open(f"{category_dir}/index.md", "w") as f:
            f.write(index_content)

        nav_items.append(f"  - {category}: {category_dir}/index.md")

        # Add individual modules to nav
        for module_path in modules:
            module_name = module_path.split('.')[-1]
            nav_items.append(f"    - {module_name}: {category_dir}/{module_name}.md")

    # Generate main API index
    api_index_content = """# API Reference

Welcome to the Terrorblade API reference. This documentation is automatically generated from the codebase and organized by functional categories.

## Categories

"""

    for category, modules in structure.items():
        category_link = f"api/{category.lower().replace(' ', '-')}/index.md"
        api_index_content += f"- [{category}]({category_link}) - {len(modules)} modules\n"

    api_index_content += """
## Navigation

Use the navigation sidebar to browse through different categories and modules. Each module page contains:

- **Classes**: All classes defined in the module
- **Functions**: All functions with their signatures and docstrings
- **Constants**: Module-level constants and variables
- **Type Hints**: Full type information where available

## Code Examples

Most modules include usage examples in their docstrings. For more comprehensive examples, see the [Examples](../examples/index.md) section.
"""

    with mkdocs_gen_files.open("api/index.md", "w") as f:
        f.write(api_index_content)

    # Generate navigation file for mkdocs
    nav_content = "# API Reference\n\n"
    nav_content += "- [Overview](api/index.md)\n"
    nav_content += "\n".join(nav_items)

    with mkdocs_gen_files.open("api_nav.md", "w") as f:
        f.write(nav_content)

    print(f"✅ Generated API documentation for {sum(len(modules) for modules in structure.values())} modules")
    print("📋 Navigation structure saved to api_nav.md")


if __name__ == "__main__":
    main()
