import pytest
from click.testing import CliRunner
from adastop.cli import adastop
import os

# we reuse a bit of pytest's own testing machinery, this should eventually come
import subprocess


def test_cli():
    runner = CliRunner()
    test_pdf_path = "test.pdf"

    if os.path.exists(test_pdf_path):
        os.remove(test_pdf_path)

    result = runner.invoke(adastop, ['reset', 'examples'])
    assert result.exit_code == 0
    for j in range(1,6):
        
        result = runner.invoke(adastop, ['compare', "--seed", "42",  'examples/walker'+str(j)+'.csv'])
        assert result.exit_code == 0

    result = runner.invoke(adastop, ['compare',"--seed", "42",  'examples/walker3.csv'])
    assert result.exit_code == 1

    result = runner.invoke(adastop, ['plot', 'examples', test_pdf_path])
    assert result.exit_code == 0
    result = runner.invoke(adastop, ['status', 'examples'])
    assert result.exit_code == 0
    assert os.path.exists(test_pdf_path) == True


    result = runner.invoke(adastop, ['reset', 'examples'])
    assert result.exit_code == 0
        
    result = runner.invoke(adastop, ['compare', "--compare-to-first","--seed", "42",  'examples/walker1.csv'])
    assert result.exit_code == 0



def test_plot_no_comparator_save_file():
    runner = CliRunner()
    runner.invoke(adastop, ['reset', 'examples'])

    result = runner.invoke(adastop, ['plot', 'examples', "test.pdf"])
    assert result.exit_code == 1
    assert result.exception.args[0] == 'Comparator save file not found.'

def test_status_no_comparator_save_file():
    runner = CliRunner()
    runner.invoke(adastop, ['reset', 'examples'])

    result = runner.invoke(adastop, ['status', 'examples'])
    assert result.exit_code == 1
    assert result.exception.args[0] == 'Comparator save file not found.'
