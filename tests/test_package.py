"""Tests for packaging: version, completions."""
import subprocess
import sys
import pytest


class TestVersion:
    def test_version_available(self):
        """sg.__version__ is a version string."""
        from sg import __version__
        assert isinstance(__version__, str)
        assert "." in __version__

    def test_version_matches_pyproject(self):
        """__version__ matches pyproject.toml."""
        from sg import __version__
        from pathlib import Path
        import re
        pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()
        match = re.search(r'version\s*=\s*"([^"]+)"', pyproject)
        assert match is not None
        assert __version__ == match.group(1)


class TestCompletions:
    def test_completions_bash(self, capsys):
        """bash completions produce valid shell script."""
        from sg.cli import cmd_completions
        import argparse
        args = argparse.Namespace(shell="bash")
        cmd_completions(args)
        out = capsys.readouterr().out
        assert "complete -F" in out
        assert "_sg_completions" in out
        assert "sg" in out

    def test_completions_zsh(self, capsys):
        """zsh completions produce valid compdef."""
        from sg.cli import cmd_completions
        import argparse
        args = argparse.Namespace(shell="zsh")
        cmd_completions(args)
        out = capsys.readouterr().out
        assert "#compdef sg" in out
        assert "_arguments" in out

    def test_completions_fish(self, capsys):
        """fish completions produce complete commands."""
        from sg.cli import cmd_completions
        import argparse
        args = argparse.Namespace(shell="fish")
        cmd_completions(args)
        out = capsys.readouterr().out
        assert "complete -c sg" in out
