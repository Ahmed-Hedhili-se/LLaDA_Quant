"""The benchmark scripts must at least parse and expose a working CLI.

Nothing else in the suite imports ``benchmarks/``, so a syntax error there
survives a full green test run and is only discovered on the GPU box, minutes
into a job. That happened once; this is the guard.

These tests deliberately do not *run* the benchmarks — two of them need a GPU,
real weights and the inference repository. Parsing and argument wiring is the
part that can be checked anywhere, and it is the part that broke.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
BENCHMARKS = sorted(
    script
    for directory in ("benchmarks", "tools")
    for script in (ROOT / directory).glob("*.py")
)


def test_there_are_benchmarks_to_check():
    assert BENCHMARKS, "benchmarks/ and tools/ are empty; this guard would silently pass"


@pytest.mark.parametrize("script", BENCHMARKS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_benchmark_parses(script: pathlib.Path):
    try:
        ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    except SyntaxError as exc:
        pytest.fail(f"{script.name} does not parse: line {exc.lineno}: {exc.msg}")


@pytest.mark.parametrize("script", BENCHMARKS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_benchmark_help_works(script: pathlib.Path):
    """``--help`` exercises imports and the whole argparse setup."""
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"{script.name} --help failed:\n{result.stderr[-2000:]}"
    )
    assert "usage" in result.stdout.lower()
