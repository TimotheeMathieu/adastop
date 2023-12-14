import pytest
from click.testing import CliRunner
from adastop.cli import adastop

# we reuse a bit of pytest's own testing machinery, this should eventually come
import subprocess


def test_cli():
    runner = CliRunner()
    result = runner.invoke(adastop, ['reset', 'examples'])
    assert result.exit_code == 0
    result = runner.invoke(adastop, ['compare', 'examples/walker1.csv'])
    assert result.exit_code == 0
    result = runner.invoke(adastop, ['reset', 'examples'])
    assert result.exit_code == 0
