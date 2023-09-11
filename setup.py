from setuptools import setup, find_packages

setup(
    name='adastop',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click', "joblib", "numpy", "matplotlib", "pandas", "seaborn", "scipy"
    ],
    entry_points={
        'console_scripts': [
            'adastop = adastop.cli:adastop',
        ],
    },
)
