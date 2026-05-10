"""Shared pytest fixtures and configuration for CENTINA test suite."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is importable without installation
sys.path.insert(0, str(Path(__file__).parent.parent))
