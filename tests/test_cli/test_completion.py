"""Tests for the `aquascope completion` subcommand."""

from __future__ import annotations

import sys

import pytest

from aquascope.cli import main


@pytest.mark.parametrize("shell", ("bash", "zsh", "fish"))
def test_completion_prints_a_nonempty_script(shell, capsys, monkeypatch):
    """`aquascope completion <shell>` prints a non-empty activation script."""
    monkeypatch.setattr(sys, "argv", ["aquascope", "completion", shell])
    main()
    out = capsys.readouterr().out
    assert out.strip() != ""
