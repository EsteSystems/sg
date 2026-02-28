"""Tests for --kernel flag and kernel selection."""
import argparse
import pytest

from sg.cli import make_kernel
from sg.kernel.mock import MockNetworkKernel


class TestKernelSelection:
    def test_default_kernel_is_mock(self):
        """No --kernel flag defaults to MockNetworkKernel."""
        args = argparse.Namespace(kernel="mock")
        kernel = make_kernel(args)
        assert isinstance(kernel, MockNetworkKernel)

    def test_missing_kernel_attr_defaults_mock(self):
        """If kernel attr missing, defaults to mock."""
        args = argparse.Namespace()
        kernel = make_kernel(args)
        assert isinstance(kernel, MockNetworkKernel)

    def test_production_kernel_flag(self):
        """--kernel=production creates ProductionNetworkKernel."""
        args = argparse.Namespace(kernel="production")
        kernel = make_kernel(args)
        from sg.kernel.production import ProductionNetworkKernel
        assert isinstance(kernel, ProductionNetworkKernel)

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
