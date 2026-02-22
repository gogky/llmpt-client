"""
Utility functions.
"""

import hashlib
from pathlib import Path
from typing import Optional


def calculate_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
    """
    Calculate hash of a file.

    Args:
        file_path: Path to the file.
        algorithm: Hash algorithm ('sha256', 'sha1', 'md5').

    Returns:
        Hex digest of the file hash.
    """
    hash_obj = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes to human-readable string.

    Args:
        bytes_value: Number of bytes.

    Returns:
        Formatted string (e.g., "1.5 GB").
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def get_optimal_piece_length(file_size: int) -> int:
    """
    Calculate optimal piece length for a torrent based on file size.

    Args:
        file_size: File size in bytes.

    Returns:
        Optimal piece length in bytes.

    Note:
        - Small files (<100MB): 256KB
        - Medium files (100MB-1GB): 1MB
        - Large files (1GB-10GB): 4MB
        - Very large files (>10GB): 16MB
    """
    if file_size < 100 * 1024 * 1024:  # <100MB
        return 256 * 1024  # 256KB
    elif file_size < 1024 * 1024 * 1024:  # <1GB
        return 1024 * 1024  # 1MB
    elif file_size < 10 * 1024 * 1024 * 1024:  # <10GB
        return 4 * 1024 * 1024  # 4MB
    else:  # >10GB
        return 16 * 1024 * 1024  # 16MB
