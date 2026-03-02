"""StubKernel — minimal domain-agnostic kernel.

Provides no-op implementations of all Kernel ABC methods.
Used as the default kernel when no domain-specific kernel is configured.
"""
from __future__ import annotations

from sg.kernel.base import Kernel


class StubKernel(Kernel):
    """Minimal kernel with no-op resource tracking and no domain operations."""

    def __init__(self) -> None:
        self._tracked: list[tuple[str, str]] = []

    def reset(self) -> None:
        self._tracked.clear()

    def track_resource(self, resource_type: str, name: str) -> None:
        pair = (resource_type, name)
        if pair not in self._tracked:
            self._tracked.append(pair)

    def untrack_resource(self, resource_type: str, name: str) -> None:
        pair = (resource_type, name)
        if pair in self._tracked:
            self._tracked.remove(pair)

    def tracked_resources(self) -> list[tuple[str, str]]:
        return list(self._tracked)

    def domain_name(self) -> str:
        return "stub"

    def create_shadow(self) -> StubKernel:
        return StubKernel()
