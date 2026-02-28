"""Tests for production kernel safety mechanisms."""
import os
import pytest
from unittest.mock import patch


class TestProtectedInterfaces:
    def test_default_protected_interfaces(self):
        """eth0 and lo are always protected."""
        # Re-import to get fresh module state
        import importlib
        import sg.kernel.production as prod
        assert "eth0" in prod.PROTECTED_INTERFACES
        assert "lo" in prod.PROTECTED_INTERFACES

    def test_env_var_adds_interfaces(self):
        """SG_PROTECTED_INTERFACES env var adds extra protected interfaces."""
        with patch.dict(os.environ, {"SG_PROTECTED_INTERFACES": "vswitch0,ens192"}):
            import importlib
            import sg.kernel.production as prod
            importlib.reload(prod)
            assert "vswitch0" in prod.PROTECTED_INTERFACES
            assert "ens192" in prod.PROTECTED_INTERFACES
            assert "eth0" in prod.PROTECTED_INTERFACES  # still protected
            assert "lo" in prod.PROTECTED_INTERFACES  # still protected

    def test_env_var_strips_whitespace(self):
        """Whitespace around interface names is stripped."""
        with patch.dict(os.environ, {"SG_PROTECTED_INTERFACES": " vswitch0 , ens192 "}):
            import importlib
            import sg.kernel.production as prod
            importlib.reload(prod)
            assert "vswitch0" in prod.PROTECTED_INTERFACES
            assert "ens192" in prod.PROTECTED_INTERFACES

    def test_empty_env_var_no_extra(self):
        """Empty SG_PROTECTED_INTERFACES adds nothing extra."""
        with patch.dict(os.environ, {"SG_PROTECTED_INTERFACES": ""}):
            import importlib
            import sg.kernel.production as prod
            importlib.reload(prod)
            assert prod.PROTECTED_INTERFACES == {"eth0", "lo"}


class TestSafetyPrefix:
    def test_safe_name_requires_prefix(self):
        """Resources without sg-test- prefix are rejected."""
        from sg.kernel.production import ProductionNetworkKernel
        kernel = ProductionNetworkKernel(dry_run=True)
        with pytest.raises(ValueError, match="must start with"):
            kernel._safe_name("my-bridge")

    def test_safe_name_accepts_prefix(self):
        """Resources with sg-test- prefix are accepted."""
        from sg.kernel.production import ProductionNetworkKernel
        kernel = ProductionNetworkKernel(dry_run=True)
        assert kernel._safe_name("sg-test-br0") == "sg-test-br0"

    def test_check_protected_blocks_eth0(self):
        """eth0 cannot be modified."""
        from sg.kernel.production import ProductionNetworkKernel
        kernel = ProductionNetworkKernel(dry_run=True)
        with pytest.raises(ValueError, match="protected"):
            kernel._check_protected("eth0")

    def test_check_protected_blocks_lo(self):
        """lo cannot be modified."""
        from sg.kernel.production import ProductionNetworkKernel
        kernel = ProductionNetworkKernel(dry_run=True)
        with pytest.raises(ValueError, match="protected"):
            kernel._check_protected("lo")

    def test_vswitch0_protected_via_env(self):
        """vswitch0 is protected when set via env var."""
        with patch.dict(os.environ, {"SG_PROTECTED_INTERFACES": "vswitch0"}):
            import importlib
            import sg.kernel.production as prod
            importlib.reload(prod)
            kernel = prod.ProductionNetworkKernel(dry_run=True)
            with pytest.raises(ValueError, match="protected"):
                kernel._check_protected("vswitch0")


class TestDryRun:
    def test_dry_run_does_not_execute(self, capsys):
        """Dry run prints commands but doesn't execute."""
        from sg.kernel.production import ProductionNetworkKernel
        kernel = ProductionNetworkKernel(dry_run=True)
        result = kernel._run(["ip", "link", "show"])
        assert result.returncode == 0
        captured = capsys.readouterr()
        assert "[dry-run]" in captured.out
