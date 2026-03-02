"""Software Genomics — data pipeline domain plugin.

Provides DataKernel ABC, MockDataKernel, and data pipeline contracts/genes.
"""
from __future__ import annotations

from pathlib import Path

from sg_data.kernel import DataKernel
from sg_data.mock import MockDataKernel

__all__ = [
    "DataKernel",
    "MockDataKernel",
    "contracts_path",
    "genes_path",
    "fixtures_path",
]


def contracts_path() -> Path:
    """Return the path to the data contracts directory."""
    return Path(__file__).parent.parent / "contracts"


def genes_path() -> Path:
    """Return the path to the data seed genes directory."""
    return Path(__file__).parent.parent / "genes"


def fixtures_path() -> Path:
    """Return the path to the data test fixtures directory."""
    return Path(__file__).parent.parent / "fixtures"
