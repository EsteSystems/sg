"""Tests for 'sg kernels' command."""
import argparse

from sg.cli import cmd_kernels


class TestCmdKernels:
    def test_lists_builtin_kernels(self, capsys):
        args = argparse.Namespace()
        cmd_kernels(args)
        out = capsys.readouterr().out
        assert "stub" in out
        assert "mock" in out
        assert "production" in out

    def test_shows_domain_info(self, capsys):
        args = argparse.Namespace()
        cmd_kernels(args)
        out = capsys.readouterr().out
        assert "domain:" in out
