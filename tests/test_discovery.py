"""Tests for kernel plugin discovery and StubKernel."""
import importlib.metadata
import pytest

from sg.kernel.base import Kernel
from sg.kernel.stub import StubKernel
from sg_network import MockNetworkKernel
from sg.kernel.discovery import (
    discover_kernels,
    list_kernel_names,
    load_kernel,
    load_kernel_class,
    KernelNotFoundError,
    KernelLoadError,
    _get_entry_points,
)


class TestDiscoverKernels:
    def test_discovers_builtin_kernels(self):
        kernels = discover_kernels()
        assert "stub" in kernels
        assert "mock" in kernels
        assert "network-mock" in kernels
        assert "production" in kernels
        assert "network-production" in kernels

    def test_list_kernel_names_sorted(self):
        names = list_kernel_names()
        assert names == sorted(names)
        assert "stub" in names
        assert "mock" in names

    def test_discover_returns_entry_points(self):
        kernels = discover_kernels()
        for name, ep in kernels.items():
            assert isinstance(ep, importlib.metadata.EntryPoint)
            assert ep.name == name


class TestLoadKernel:
    def test_load_stub(self):
        kernel = load_kernel("stub")
        assert isinstance(kernel, StubKernel)
        assert isinstance(kernel, Kernel)

    def test_load_mock(self):
        kernel = load_kernel("mock")
        assert isinstance(kernel, MockNetworkKernel)

    def test_load_network_mock(self):
        kernel = load_kernel("network-mock")
        assert isinstance(kernel, MockNetworkKernel)

    def test_load_production(self):
        from sg_network import ProductionNetworkKernel
        kernel = load_kernel("production")
        assert isinstance(kernel, ProductionNetworkKernel)

    def test_load_network_production(self):
        from sg_network import ProductionNetworkKernel
        kernel = load_kernel("network-production")
        assert isinstance(kernel, ProductionNetworkKernel)

    def test_load_unknown_raises(self):
        with pytest.raises(KernelNotFoundError, match="unknown kernel"):
            load_kernel("nonexistent-kernel-xyz")

    def test_error_lists_available(self):
        with pytest.raises(KernelNotFoundError, match="stub"):
            load_kernel("nonexistent-kernel-xyz")


class TestLoadKernelClass:
    def test_returns_class_not_instance(self):
        cls = load_kernel_class("stub")
        assert cls is StubKernel

    def test_class_is_kernel_subclass(self):
        cls = load_kernel_class("mock")
        assert issubclass(cls, Kernel)


class TestStubKernel:
    def test_implements_kernel_abc(self):
        k = StubKernel()
        assert isinstance(k, Kernel)

    def test_resource_tracking(self):
        k = StubKernel()
        k.track_resource("thing", "foo")
        assert ("thing", "foo") in k.tracked_resources()
        k.untrack_resource("thing", "foo")
        assert k.tracked_resources() == []

    def test_no_duplicate_tracking(self):
        k = StubKernel()
        k.track_resource("thing", "foo")
        k.track_resource("thing", "foo")
        assert len(k.tracked_resources()) == 1

    def test_reset_clears_tracked(self):
        k = StubKernel()
        k.track_resource("thing", "foo")
        k.reset()
        assert k.tracked_resources() == []

    def test_domain_name(self):
        assert StubKernel().domain_name() == "stub"

    def test_create_shadow(self):
        k = StubKernel()
        k.track_resource("thing", "foo")
        shadow = k.create_shadow()
        assert isinstance(shadow, StubKernel)
        assert shadow.tracked_resources() == []

    def test_describe_operations_empty(self):
        assert StubKernel().describe_operations() == []

    def test_delete_resource(self):
        k = StubKernel()
        k.track_resource("thing", "foo")
        k.delete_resource("thing", "foo")
        assert k.tracked_resources() == []


class TestGetEntryPoints:
    def test_returns_iterable(self):
        eps = _get_entry_points("sg.kernels")
        names = [ep.name for ep in eps]
        assert "stub" in names

    def test_unknown_group_returns_empty(self):
        eps = _get_entry_points("sg.nonexistent.group.12345")
        assert len(list(eps)) == 0
