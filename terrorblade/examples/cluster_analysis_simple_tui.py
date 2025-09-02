"""
Simple Cluster Analysis TUI for Terrorblade

This is a clean, optimized version using modular architecture.
The main implementation is in terrorblade/tui/modules/optimized_tui.py
"""

import sys
from pathlib import Path

# Add terrorblade to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the optimized TUI main function
from terrorblade.tui.modules.optimized_tui import main as optimized_main


def main() -> None:
    """Main entry point - now uses the optimized modular TUI."""
    optimized_main()


if __name__ == "__main__":
    main()
