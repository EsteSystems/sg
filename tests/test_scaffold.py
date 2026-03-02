"""Tests for plugin scaffolding (Phase 12)."""
from __future__ import annotations

import pytest

from sg.scaffold import scaffold_plugin, validate_plugin_name, ScaffoldError


class TestValidatePluginName:
    def test_valid_simple_name(self):
        assert validate_plugin_name("storage") == "storage"

    def test_valid_underscored_name(self):
        assert validate_plugin_name("my_domain") == "my_domain"

    def test_hyphen_normalized_to_underscore(self):
        assert validate_plugin_name("my-domain") == "my_domain"

    def test_empty_name_rejected(self):
        with pytest.raises(ScaffoldError, match="cannot be empty"):
            validate_plugin_name("")

    def test_uppercase_rejected(self):
        with pytest.raises(ScaffoldError, match="invalid plugin name"):
            validate_plugin_name("MyDomain")

    def test_starts_with_digit_rejected(self):
        with pytest.raises(ScaffoldError, match="invalid plugin name"):
            validate_plugin_name("3domain")

    def test_spaces_rejected(self):
        with pytest.raises(ScaffoldError, match="invalid plugin name"):
            validate_plugin_name("my domain")

    def test_python_keyword_rejected(self):
        with pytest.raises(ScaffoldError, match="Python keyword"):
            validate_plugin_name("class")

    def test_reserved_network_rejected(self):
        with pytest.raises(ScaffoldError, match="reserved"):
            validate_plugin_name("network")

    def test_reserved_data_rejected(self):
        with pytest.raises(ScaffoldError, match="reserved"):
            validate_plugin_name("data")


class TestScaffoldPlugin:
    def test_creates_all_directories(self, tmp_path):
        result = scaffold_plugin("storage", tmp_path)
        assert result == tmp_path / "storage"
        assert (result / "sg_storage").is_dir()
        assert (result / "contracts" / "genes").is_dir()
        assert (result / "genes").is_dir()
        assert (result / "fixtures").is_dir()

    def test_creates_all_files(self, tmp_path):
        result = scaffold_plugin("storage", tmp_path)
        assert (result / "pyproject.toml").is_file()
        assert (result / "sg_storage" / "__init__.py").is_file()
        assert (result / "sg_storage" / "kernel.py").is_file()
        assert (result / "sg_storage" / "mock.py").is_file()
        assert (result / "contracts" / "genes" / "example_action.sg").is_file()
        assert (result / "genes" / "example_action_v1.py").is_file()

    def test_pyproject_has_correct_entry_points(self, tmp_path):
        scaffold_plugin("storage", tmp_path)
        content = (tmp_path / "storage" / "pyproject.toml").read_text()
        assert 'name = "sg-storage"' in content
        assert "[project.entry-points.\"sg.kernels\"]" in content
        assert 'storage-mock = "sg_storage.mock:MockStorageKernel"' in content
        assert '"software-genomics>=0.2.0"' in content

    def test_kernel_imports_base(self, tmp_path):
        scaffold_plugin("storage", tmp_path)
        content = (tmp_path / "storage" / "sg_storage" / "kernel.py").read_text()
        assert "from sg.kernel.base import Kernel" in content
        assert "class StorageKernel(Kernel):" in content

    def test_mock_extends_kernel(self, tmp_path):
        scaffold_plugin("storage", tmp_path)
        content = (tmp_path / "storage" / "sg_storage" / "mock.py").read_text()
        assert "from sg_storage.kernel import StorageKernel" in content
        assert "class MockStorageKernel(StorageKernel):" in content

    def test_init_has_path_helpers(self, tmp_path):
        scaffold_plugin("storage", tmp_path)
        content = (tmp_path / "storage" / "sg_storage" / "__init__.py").read_text()
        assert "def contracts_path()" in content
        assert "def genes_path()" in content
        assert "def fixtures_path()" in content

    def test_contract_has_domain_annotation(self, tmp_path):
        scaffold_plugin("storage", tmp_path)
        content = (tmp_path / "storage" / "contracts" / "genes" / "example_action.sg").read_text()
        assert "for storage" in content

    def test_seed_gene_has_execute(self, tmp_path):
        scaffold_plugin("storage", tmp_path)
        content = (tmp_path / "storage" / "genes" / "example_action_v1.py").read_text()
        assert "def execute(input_json: str) -> str:" in content
        assert "gene_sdk" in content

    def test_refuses_existing_directory(self, tmp_path):
        scaffold_plugin("storage", tmp_path)
        with pytest.raises(ScaffoldError, match="already exists"):
            scaffold_plugin("storage", tmp_path)

    def test_hyphenated_name_creates_correct_structure(self, tmp_path):
        result = scaffold_plugin("my-domain", tmp_path)
        assert result == tmp_path / "my-domain"
        assert (result / "sg_my_domain").is_dir()
        content = (result / "pyproject.toml").read_text()
        assert 'name = "sg-my-domain"' in content
        assert 'my-domain-mock = "sg_my_domain.mock:MockMyDomainKernel"' in content

    def test_multi_word_name_title_casing(self, tmp_path):
        scaffold_plugin("my_cool_domain", tmp_path)
        content = (tmp_path / "my_cool_domain" / "sg_my_cool_domain" / "kernel.py").read_text()
        assert "class MyCoolDomainKernel(Kernel):" in content
