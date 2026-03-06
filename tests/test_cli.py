import subprocess
import sys


def test_cli_version():
    result = subprocess.run(
        [sys.executable, "-m", "deoldify.cli", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "1.0.0" in result.stdout


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "deoldify.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "colorize" in result.stdout


def test_cli_colorize_help():
    result = subprocess.run(
        [sys.executable, "-m", "deoldify.cli", "colorize", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--model" in result.stdout
    assert "--render-factor" in result.stdout


def test_cli_missing_file():
    result = subprocess.run(
        [sys.executable, "-m", "deoldify.cli", "colorize", "nonexistent.jpg"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
