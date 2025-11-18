import os
from pathlib import Path
from typing import List

from src.core.models import DataFile


def discover_data_files(data_dir: str) -> List[DataFile]:
    """
    Discover all data files in a directory.

    Args:
        data_dir: Path to the data directory

    Returns:
        List of DataFile objects
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    data_files = []
    for file_path in data_path.rglob("*"):
        if file_path.is_file() and not file_path.name.startswith("."):
            data_files.append(
                DataFile(
                    path=str(file_path),
                    filename=file_path.name,
                    extension=file_path.suffix,
                )
            )

    return sorted(data_files, key=lambda x: x.filename)


def ensure_directory(directory: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory: Path to directory
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def write_file(path: str, content: str) -> None:
    """
    Write content to a file.

    Args:
        path: File path
        content: Content to write
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def read_file(path: str) -> str:
    """
    Read content from a file.

    Args:
        path: File path

    Returns:
        File content
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

