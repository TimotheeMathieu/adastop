from setuptools import find_packages
from distutils.core import setup

packages = find_packages(exclude=["tests", "examples",])

setup(
    name='adastop',
    version='0.1.0',
    license="MIT",
    packages=packages,
    include_package_data=True,
    install_requires=[
        'Click', "joblib", "numpy", "matplotlib", "pandas", "seaborn"
    ],
    entry_points={
        'console_scripts': [
            'adastop = adastop.cli:adastop',
        ],
    },
)
