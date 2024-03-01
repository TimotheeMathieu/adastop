from setuptools import find_packages
from distutils.core import setup
import os

ver_file = os.path.join("adastop", "_version.py")
with open(ver_file) as f:
    exec(f.read())

packages = find_packages(exclude=["tests", "examples",])


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='adastop',
    version=__version__,
    license="MIT",
    packages=packages,
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",

    install_requires=[
        'Click', "numpy", "matplotlib", "pandas", "tabulate"
    ],
    entry_points={
        'console_scripts': [
            'adastop = adastop.cli:adastop',
        ],
    },
)
