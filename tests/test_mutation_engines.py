"""Tests for multi-LLM mutation engine support."""
import json
import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from sg.mutation import (
    MutationEngine, MockMutationEngine, LLMMutationEngine,
    ClaudeMutationEngine, OpenAIMutationEngine, DeepSeekMutationEngine,
    MutationContext,
)
from sg.contracts import ContractStore

import sg_network

CONTRACTS_DIR = sg_network.contracts_path()


@pytest.fixture
def contract_store():
    return ContractStore.open(CONTRACTS_DIR)


# --- Class hierarchy ---

class TestEngineHierarchy:
    def test_claude_is_llm_engine(self):
        assert issubclass(ClaudeMutationEngine, LLMMutationEngine)

    def test_openai_is_llm_engine(self):
        assert issubclass(OpenAIMutationEngine, LLMMutationEngine)

    def test_deepseek_is_openai_engine(self):
        assert issubclass(DeepSeekMutationEngine, OpenAIMutationEngine)

    def test_all_are_mutation_engines(self):
        for cls in [ClaudeMutationEngine, OpenAIMutationEngine, DeepSeekMutationEngine]:
            assert issubclass(cls, MutationEngine)

    def test_mock_not_llm(self):
        assert not issubclass(MockMutationEngine, LLMMutationEngine)


# --- Provider names ---

class TestProviderNames:
    def test_claude_provider(self, contract_store):
        engine = ClaudeMutationEngine("fake-key", contract_store)
        assert engine.provider_name == "claude"

    def test_openai_provider(self, contract_store):
        engine = OpenAIMutationEngine("fake-key", contract_store)
        assert engine.provider_name == "openai"

    def test_deepseek_provider(self, contract_store):
        engine = DeepSeekMutationEngine("fake-key", contract_store)
        assert engine.provider_name == "deepseek"


# --- Default models ---

class TestDefaultModels:
    def test_claude_default_model(self, contract_store):
        engine = ClaudeMutationEngine("key", contract_store)
        assert "claude" in engine.model

    def test_openai_default_model(self, contract_store):
        engine = OpenAIMutationEngine("key", contract_store)
        assert engine.model == "gpt-4o"

    def test_deepseek_default_model(self, contract_store):
        engine = DeepSeekMutationEngine("key", contract_store)
        assert engine.model == "deepseek-chat"

    def test_model_override(self, contract_store):
        engine = OpenAIMutationEngine("key", contract_store, model="gpt-4-turbo")
        assert engine.model == "gpt-4-turbo"

    def test_deepseek_model_override(self, contract_store):
        engine = DeepSeekMutationEngine("key", contract_store, model="deepseek-coder")
        assert engine.model == "deepseek-coder"


# --- Base URLs ---

class TestBaseURLs:
    def test_openai_default_url(self, contract_store):
        engine = OpenAIMutationEngine("key", contract_store)
        assert engine.base_url == "https://api.openai.com/v1"

    def test_deepseek_default_url(self, contract_store):
        engine = DeepSeekMutationEngine("key", contract_store)
        assert engine.base_url == "https://api.deepseek.com"

    def test_custom_base_url(self, contract_store):
        engine = OpenAIMutationEngine("key", contract_store, base_url="https://custom.api.com/v1")
        assert engine.base_url == "https://custom.api.com/v1"


# --- Python extraction (shared across all LLM engines) ---

class TestPythonExtraction:
    def test_extract_from_code_block(self, contract_store):
        engine = OpenAIMutationEngine("key", contract_store)
        text = '```python\ndef execute(x):\n    return "{}"\n```'
        result = engine._extract_python(text)
        assert 'def execute(x):' in result

    def test_extract_from_plain_block(self, contract_store):
        engine = OpenAIMutationEngine("key", contract_store)
        text = '```\ndef execute(x):\n    return "{}"\n```'
        result = engine._extract_python(text)
        assert 'def execute(x):' in result

    def test_extract_plain_text(self, contract_store):
        engine = OpenAIMutationEngine("key", contract_store)
        text = 'def execute(x):\n    return "{}"'
        result = engine._extract_python(text)
        assert 'def execute(x):' in result


# --- Contract prompt building ---

class TestContractPrompt:
    def test_known_locus(self, contract_store):
        engine = OpenAIMutationEngine("key", contract_store)
        prompt = engine._contract_prompt("bridge_create")
        assert "bridge_create" in prompt
        assert "bridge_name" in prompt

    def test_unknown_locus_fallback(self, contract_store):
        engine = OpenAIMutationEngine("key", contract_store)
        prompt = engine._contract_prompt("nonexistent_locus_xyz")
        assert "nonexistent_locus_xyz" in prompt


# --- API call mocking ---

class TestClaudeAPICalls:
    def test_mutate_calls_api(self, contract_store):
        engine = ClaudeMutationEngine("fake-key", contract_store)
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"text": '```python\ndef execute(input_json):\n    return \'{"success": true}\'\n```'}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response) as mock_post:
            ctx = MutationContext(
                gene_source="def execute(x): raise RuntimeError('broken')",
                locus="bridge_create",
                failing_input='{"bridge_name": "br0"}',
                error_message="broken",
            )
            result = engine.mutate(ctx)

        assert "def execute" in result
        call_args = mock_post.call_args
        assert "api.anthropic.com" in call_args[0][0]
        assert call_args[1]["headers"]["x-api-key"] == "fake-key"


class TestOpenAIAPICalls:
    def test_mutate_calls_api(self, contract_store):
        engine = OpenAIMutationEngine("fake-key", contract_store)
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '```python\ndef execute(input_json):\n    return \'{"success": true}\'\n```'}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response) as mock_post:
            ctx = MutationContext(
                gene_source="def execute(x): raise RuntimeError('broken')",
                locus="bridge_create",
                failing_input='{"bridge_name": "br0"}',
                error_message="broken",
            )
            result = engine.mutate(ctx)

        assert "def execute" in result
        call_args = mock_post.call_args
        assert "api.openai.com" in call_args[0][0]
        assert "Bearer fake-key" in call_args[1]["headers"]["Authorization"]


class TestDeepSeekAPICalls:
    def test_mutate_calls_api(self, contract_store):
        engine = DeepSeekMutationEngine("fake-key", contract_store)
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '```python\ndef execute(input_json):\n    return \'{"success": true}\'\n```'}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response) as mock_post:
            ctx = MutationContext(
                gene_source="def execute(x): raise RuntimeError('broken')",
                locus="bridge_create",
                failing_input='{"bridge_name": "br0"}',
                error_message="broken",
            )
            result = engine.mutate(ctx)

        assert "def execute" in result
        call_args = mock_post.call_args
        assert "api.deepseek.com" in call_args[0][0]
        assert call_args[1]["json"]["model"] == "deepseek-chat"


# --- CLI engine selection ---

class TestCLIEngineSelection:
    def test_make_engine_mock(self):
        from sg.cli import make_mutation_engine
        args = MagicMock()
        args.mutation_engine = "mock"
        args.model = None
        root = Path(__file__).parent.parent
        cs = ContractStore.open(root / "contracts")
        engine = make_mutation_engine(args, root, cs)
        assert isinstance(engine, MockMutationEngine)

    def test_make_engine_claude(self):
        from sg.cli import make_mutation_engine
        args = MagicMock()
        args.mutation_engine = "claude"
        args.model = None
        root = Path(__file__).parent.parent
        cs = ContractStore.open(root / "contracts")
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            engine = make_mutation_engine(args, root, cs)
        assert isinstance(engine, ClaudeMutationEngine)

    def test_make_engine_openai(self):
        from sg.cli import make_mutation_engine
        args = MagicMock()
        args.mutation_engine = "openai"
        args.model = None
        root = Path(__file__).parent.parent
        cs = ContractStore.open(root / "contracts")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            engine = make_mutation_engine(args, root, cs)
        assert isinstance(engine, OpenAIMutationEngine)

    def test_make_engine_deepseek(self):
        from sg.cli import make_mutation_engine
        args = MagicMock()
        args.mutation_engine = "deepseek"
        args.model = None
        root = Path(__file__).parent.parent
        cs = ContractStore.open(root / "contracts")
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            engine = make_mutation_engine(args, root, cs)
        assert isinstance(engine, DeepSeekMutationEngine)

    def test_make_engine_auto_prefers_claude(self):
        from sg.cli import make_mutation_engine
        args = MagicMock()
        args.mutation_engine = "auto"
        args.model = None
        root = Path(__file__).parent.parent
        cs = ContractStore.open(root / "contracts")
        with patch.dict("os.environ", {
            "ANTHROPIC_API_KEY": "claude-key",
            "OPENAI_API_KEY": "openai-key",
        }, clear=False):
            engine = make_mutation_engine(args, root, cs)
        assert isinstance(engine, ClaudeMutationEngine)

    def test_make_engine_auto_falls_to_openai(self):
        from sg.cli import make_mutation_engine
        args = MagicMock()
        args.mutation_engine = "auto"
        args.model = None
        root = Path(__file__).parent.parent
        cs = ContractStore.open(root / "contracts")
        env = {"OPENAI_API_KEY": "openai-key"}
        # Clear the other keys to ensure they don't interfere
        with patch.dict("os.environ", env):
            # Remove other API keys if present
            for k in ["ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"]:
                os.environ.pop(k, None)
            engine = make_mutation_engine(args, root, cs)
        assert isinstance(engine, OpenAIMutationEngine)

    def test_make_engine_auto_falls_to_deepseek(self):
        from sg.cli import make_mutation_engine
        args = MagicMock()
        args.mutation_engine = "auto"
        args.model = None
        root = Path(__file__).parent.parent
        cs = ContractStore.open(root / "contracts")
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "ds-key"}):
            for k in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]:
                os.environ.pop(k, None)
            engine = make_mutation_engine(args, root, cs)
        assert isinstance(engine, DeepSeekMutationEngine)

    def test_make_engine_auto_falls_to_mock(self):
        from sg.cli import make_mutation_engine
        args = MagicMock()
        args.mutation_engine = "auto"
        args.model = None
        root = Path(__file__).parent.parent
        cs = ContractStore.open(root / "contracts")
        with patch.dict("os.environ", {}, clear=True):
            engine = make_mutation_engine(args, root, cs)
        assert isinstance(engine, MockMutationEngine)

    def test_model_override_passed_to_engine(self):
        from sg.cli import make_mutation_engine
        args = MagicMock()
        args.mutation_engine = "openai"
        args.model = "gpt-4-turbo"
        root = Path(__file__).parent.parent
        cs = ContractStore.open(root / "contracts")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            engine = make_mutation_engine(args, root, cs)
        assert isinstance(engine, OpenAIMutationEngine)
        assert engine.model == "gpt-4-turbo"
