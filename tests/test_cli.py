import pytest
from click.testing import CliRunner
from adastop.cli import adastop

# we reuse a bit of pytest's own testing machinery, this should eventually come
import subprocess


def test_cli():
    runner = CliRunner()
    result = runner.invoke(adastop, ['reset', 'examples'])
    assert result.exit_code == 0
    for j in range(1,6):
        
        result = runner.invoke(adastop, ['compare', 'examples/walker'+str(j)+'.csv'])
        assert result.exit_code == 0
    result = runner.invoke(adastop, ['compare', 'examples/walker3.csv'])
    assert result.exit_code == 1
    result = runner.invoke(adastop, ['plot', 'examples'])
    assert result.exit_code == 0
    result = runner.invoke(adastop, ['reset', 'examples'])
    assert result.exit_code == 0
