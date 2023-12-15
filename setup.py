from setuptools import find_packages
from distutils.core import setup
import os

ver_file = os.path.join("adastop", "_version.py")
with open(ver_file) as f:
    exec(f.read())

packages = find_packages(exclude=["tests", "examples",])

setup(
    name='adastop',
    version=__version__,
    license="MIT",
    packages=packages,
    include_package_data=True,
    install_requires=[
        'Click', "joblib", "numpy", "matplotlib", "pandas", "tabulate"
    ],
    entry_points={
        'console_scripts': [
            'adastop = adastop.cli:adastop',
        ],
    },
)
