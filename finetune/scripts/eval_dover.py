"""Compatibility entry point for the unified metric evaluator."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metrics.cli import main


if __name__ == "__main__":
    main(default_metrics="dover")
