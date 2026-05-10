"""Remove Python cache files in a Windows/Linux-safe way."""
from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    removed_dirs = 0
    removed_files = 0

    for path in ROOT.rglob("__pycache__"):
        if path.is_dir():
            shutil.rmtree(path)
            removed_dirs += 1

    for pattern in ("*.pyc", "*.pyo"):
        for path in ROOT.rglob(pattern):
            if path.is_file():
                path.unlink()
                removed_files += 1

    print(f"Cleaned {removed_dirs} cache dirs and {removed_files} bytecode files.")


if __name__ == "__main__":
    main()
