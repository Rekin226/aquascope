"""`aquascope update` and `--version`: the newest release, how this copy was installed, the right upgrade."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from aquascope import updates


def test_versions_compare_on_the_release() -> None:
    assert updates.is_newer("0.23.0", "0.22.0") and updates.is_newer("1.0.0", "0.99.9")
    assert not updates.is_newer("0.22.0", "0.22.0") and not updates.is_newer("0.21.1", "0.22.0")
    assert updates._release("0.23.0rc1") == (0, 23, 0) and updates._release("1.2") == (1, 2)


def _receipt(prefix: Path, specifier: str | None) -> None:
    spec = f', specifier = "{specifier}"' if specifier else ""
    prefix.mkdir(parents=True, exist_ok=True)
    (prefix / "uv-receipt.toml").write_text(
        "[tool]\n"
        f'requirements = [{{ name = "aquascope", extras = ["all", "gym"]{spec} }}]\n'
        'python = "3.12"\n', encoding="utf-8")


def test_the_install_kind_is_read_off_the_prefix(tmp_path) -> None:
    uv = tmp_path / "uv" / "tools" / "aquascope"
    _receipt(uv, None)
    assert updates.install_kind(uv, "/base") == "uv-tool"
    pipx = tmp_path / ".local" / "pipx" / "venvs" / "aquascope"
    pipx.mkdir(parents=True)
    assert updates.install_kind(pipx, "/base") == "pipx"
    conda = tmp_path / "miniforge" / "envs" / "hydro"
    (conda / "conda-meta").mkdir(parents=True)
    assert updates.install_kind(conda, "/base") == "conda"
    venv = tmp_path / "project" / ".venv"
    venv.mkdir(parents=True)
    assert updates.install_kind(venv, "/base") == "venv"
    assert updates.install_kind(venv, venv) == "system"


def test_a_pinned_uv_tool_is_reinstalled_without_the_pin(tmp_path) -> None:
    _receipt(tmp_path / "free", None)
    assert updates.upgrade_command("uv-tool", prefix=tmp_path / "free") == ["uv", "tool", "upgrade", "aquascope"]
    _receipt(tmp_path / "pinned", "==0.22.0")
    assert updates.upgrade_command("uv-tool", prefix=tmp_path / "pinned") == [
        "uv", "tool", "install", "aquascope[all,gym]", "--force", "--python", "3.12"], (
        "uv tool upgrade never moves an exact pin, so the tool is reinstalled with the same extras and Python")
    assert updates._uv_receipt_loose(tmp_path / "pinned" / "uv-receipt.toml") == {"tool": {
        "requirements": [{"name": "aquascope", "extras": ["all", "gym"], "specifier": "==0.22.0"}],
        "python": "3.12"}}, "Python 3.10 reads the receipt without tomllib"


def test_each_kind_gets_its_own_upgrade() -> None:
    assert updates.upgrade_command("pipx") == ["pipx", "upgrade", "aquascope"]
    assert updates.upgrade_command("venv") == [sys.executable, "-m", "pip", "install", "--upgrade", "aquascope"]
    assert updates.upgrade_command("conda")[:3] == [sys.executable, "-m", "pip"], "conda envs upgrade with their pip"
    assert updates.upgrade_command("editable") is None, "a development checkout is updated with git, not pip"


def test_check_reports_a_newer_release(monkeypatch) -> None:
    monkeypatch.setattr(updates, "installed_version", lambda: "0.22.0")
    monkeypatch.setattr(updates, "latest_version", lambda timeout=10.0: "0.23.0")
    monkeypatch.setattr(updates, "install_kind", lambda: "venv")
    info = updates.check()
    assert info["newer"] and info["latest"] == "0.23.0" and info["command"][-1] == "aquascope"
    assert info["error"] is None


def test_check_says_when_pypi_cannot_be_reached(monkeypatch) -> None:
    def down(timeout=10.0):
        raise OSError("network is unreachable")

    monkeypatch.setattr(updates, "latest_version", down)
    info = updates.check()
    assert info["latest"] is None and "could not reach PyPI" in info["error"]


# ── the CLI ──


def _run(monkeypatch, capsys, info: dict, *argv: str, tty: bool = True) -> tuple[str, str, list]:
    from aquascope.cli import main

    ran: list = []
    base = {"installed": "0.22.0", "latest": "0.23.0", "newer": True, "kind": "uv-tool",
            "command": ["uv", "tool", "upgrade", "aquascope"], "error": None, "location": "/tools/aquascope"}
    monkeypatch.setattr(updates, "check", lambda timeout=10.0: {**base, **info})
    monkeypatch.setattr("subprocess.run", lambda cmd, check=False: ran.append(cmd) or type("R", (), {"returncode": 0}))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: tty, raising=False)
    monkeypatch.setattr(sys, "argv", ["aquascope", "update", *argv])
    main()
    out = capsys.readouterr()
    return out.out, out.err, ran


def test_update_check_names_the_command_and_runs_nothing(monkeypatch, capsys) -> None:
    out, _err, ran = _run(monkeypatch, capsys, {}, "--check")
    assert "0.22.0 -> 0.23.0 available" in out and "uv tool upgrade aquascope" in out and ran == []


def test_update_yes_runs_the_upgrade(monkeypatch, capsys) -> None:
    out, _err, ran = _run(monkeypatch, capsys, {}, "--yes")
    assert ran == [["uv", "tool", "upgrade", "aquascope"]] and "Done" in out


def test_update_asks_first_and_takes_no_for_an_answer(monkeypatch, capsys) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    out, _err, ran = _run(monkeypatch, capsys, {})
    assert ran == [] and "Not upgraded" in out


def test_update_without_a_terminal_needs_yes(monkeypatch, capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        _run(monkeypatch, capsys, {}, tty=False)
    assert exit_info.value.code == 1


def test_update_when_up_to_date_or_in_a_checkout(monkeypatch, capsys) -> None:
    out, _err, ran = _run(monkeypatch, capsys, {"latest": "0.22.0", "newer": False})
    assert "Up to date" in out and ran == []
    out, _err, ran = _run(monkeypatch, capsys, {"kind": "editable", "command": None})
    assert "git pull" in out and ran == []


def test_update_says_when_pypi_is_unreachable(monkeypatch, capsys) -> None:
    with pytest.raises(SystemExit):
        _run(monkeypatch, capsys, {"latest": None, "newer": False, "error": "could not reach PyPI: timeout"})
    assert "could not reach PyPI" in capsys.readouterr().err


def test_version_flag(monkeypatch, capsys) -> None:
    from aquascope import __version__
    from aquascope.cli import main

    monkeypatch.setattr(sys, "argv", ["aquascope", "--version"])
    with pytest.raises(SystemExit):
        main()
    assert capsys.readouterr().out.strip() == f"aquascope {__version__}"
