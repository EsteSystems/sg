"""Software Genomics — network domain plugin.

Provides NetworkKernel ABC, MockNetworkKernel, ProductionNetworkKernel,
and network-specific topology resource mappers.
"""
from __future__ import annotations

from pathlib import Path

from sg_network.kernel import NetworkKernel
from sg_network.mock import MockNetworkKernel
from sg_network.production import ProductionNetworkKernel
from sg_network.mappers import NETWORK_RESOURCE_MAPPERS

__all__ = [
    "NetworkKernel",
    "MockNetworkKernel",
    "ProductionNetworkKernel",
    "NETWORK_RESOURCE_MAPPERS",
    "contracts_path",
    "genes_path",
    "fixtures_path",
]


def contracts_path() -> Path:
    """Return the path to the network contracts directory."""
    return Path(__file__).parent.parent / "contracts"


def genes_path() -> Path:
    """Return the path to the network seed genes directory."""
    return Path(__file__).parent.parent / "genes"


def fixtures_path() -> Path:
    """Return the path to the network test fixtures directory."""
    return Path(__file__).parent.parent / "fixtures"
