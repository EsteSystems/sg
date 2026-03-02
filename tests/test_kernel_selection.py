"""Tests for --kernel flag and kernel selection via discovery."""
import argparse
import pytest

from sg.cli import make_kernel
from sg.kernel.base import Kernel
from sg.kernel.stub import StubKernel
from sg_network import MockNetworkKernel


class TestKernelSelection:
    def test_default_kernel_is_mock(self):
        """No --kernel flag defaults to MockNetworkKernel (backward compat)."""
        args = argparse.Namespace(kernel="mock")
        kernel = make_kernel(args)
        assert isinstance(kernel, MockNetworkKernel)

    def test_missing_kernel_attr_defaults_mock(self):
        """If kernel attr missing, defaults to mock."""
        args = argparse.Namespace()
        kernel = make_kernel(args)
        assert isinstance(kernel, MockNetworkKernel)

    def test_stub_kernel(self):
        """--kernel=stub creates StubKernel."""
        args = argparse.Namespace(kernel="stub")
        kernel = make_kernel(args)
        assert isinstance(kernel, StubKernel)

    def test_network_mock_alias(self):
        """--kernel=network-mock creates MockNetworkKernel."""
        args = argparse.Namespace(kernel="network-mock")
        kernel = make_kernel(args)
        assert isinstance(kernel, MockNetworkKernel)

    def test_production_kernel_flag(self):
        """--kernel=production creates ProductionNetworkKernel."""
        args = argparse.Namespace(kernel="production")
        kernel = make_kernel(args)
        from sg_network import ProductionNetworkKernel
        assert isinstance(kernel, ProductionNetworkKernel)

    def test_network_production_alias(self):
        """--kernel=network-production creates ProductionNetworkKernel."""
        args = argparse.Namespace(kernel="network-production")
        kernel = make_kernel(args)
        from sg_network import ProductionNetworkKernel
        assert isinstance(kernel, ProductionNetworkKernel)

    def test_unknown_kernel_exits(self, capsys):
        """Unknown kernel name prints error and exits."""
        args = argparse.Namespace(kernel="bogus-kernel")
        with pytest.raises(SystemExit):
            make_kernel(args)
        captured = capsys.readouterr()
        assert "unknown kernel" in captured.err

    def test_all_kernels_are_kernel_subclass(self):
        """Every discovered kernel is a Kernel subclass."""
        from sg.kernel.discovery import discover_kernels
        for name, ep in discover_kernels().items():
            cls = ep.load()
            assert issubclass(cls, Kernel), f"{name} is not a Kernel subclass"

    def test_production_demo_imports(self):
        """Production demo script imports cleanly."""
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "live_mutation_production",
            Path(__file__).parent.parent / "demo" / "live_mutation_production.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "run_demo")
        assert hasattr(mod, "BROKEN_GENES")
        assert hasattr(mod, "DEFAULT_INPUTS")
