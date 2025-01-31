# File: setup.py

from setuptools import setup, find_packages

setup(
    name="gdn-resource-thief",       # your package name
    version="0.1.0",
    packages=find_packages(),        # automatically find `envs` and `test` as packages
    include_package_data=True,       # includes any data files
    install_requires=[
        # add your dependencies here
        "numpy",
        "gymnasium",
        "pettingzoo",
        "ray[default]",
        "pygame",
        "matplotlib",
        "pandas"
    ],
    # optional: "scripts" or "entry_points" if you want to create CLI commands
)
